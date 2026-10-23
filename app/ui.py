"""Streamlit front end. Run with: streamlit run app/ui.py"""

from __future__ import annotations

import io
import json
import os
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import importlib
import app.core as core
if (not hasattr(core, "parse_target_drug_query") or
        "selection_method" not in core.fold_decision.__code__.co_varnames or
        not getattr(core, "STRICT_FOLD_CHECKS_EACH_VALUE", False)):
    importlib.reload(core)
from app.core import apply_mutations, environment_report, fetch_uniprot_fasta, parse_target_drug_query, parse_target_query, search_pdb, sequence_identity
import app.workflow as workflow
if "progress_callback" not in workflow.run_rxnflow_generation.__code__.co_varnames:
    importlib.reload(workflow)
from app.workflow import available_ic50_folds, create_job, rxnflow_status, run_command, run_rxnflow_generation, stage_status, update_manifest, vina_python, vina_status
from app.core import fold_decision


@st.cache_data(ttl=60)
def cached_rxnflow_status(python: str, env_dir: str):
    return rxnflow_status(python, env_dir)


@st.cache_data(ttl=60)
def cached_vina_status(python: str):
    return vina_status(python)


st.set_page_config(page_title="Target Screening", layout="wide")
st.title("표적 단백질 기반 후보 우선순위화")
st.caption("연구용 계산 도구입니다. 예측값은 실험적으로 확인된 결합력이나 약효가 아닙니다.")
st.info("실행 모드: 실제 외부 API 조회. 모델, RXNFlow, Vina는 실행 환경과 필요한 설정이 있을 때만 동작하며 demo 결과를 만들지 않습니다.")

with st.expander("실행 환경 및 연결된 자산", expanded=False):
    st.json({"environment": environment_report(), "assets": {p.name: str(p.resolve()) if p.exists() else None
             for p in (ROOT / "vendor").iterdir()}})

st.header("1. PDB 구조 검색")
query = st.text_input("표적·변이·결합 약물 또는 PDB ID", placeholder="EGFR(L858R/T790M) Osimertinib 또는 1BW8")
st.caption("표적과 변이 뒤에 약물명을 함께 입력하면, 해당 약물이 결합된 PDB 복합체를 검색합니다.")
if st.button("PDB 검색", type="primary"):
    try:
        target_query, drug_query = parse_target_drug_query(query)
        target_name, target_mutation = parse_target_query(target_query)
        st.session_state.structures = search_pdb(query)
        st.session_state.query = query.strip()
        st.session_state.target_mutation = target_mutation
        st.session_state.fasta_mutation = target_mutation
        st.session_state.prediction_mutation = target_mutation
    except Exception as exc:
        st.error(f"RCSB 검색 실패: {exc}")

structures = st.session_state.get("structures", [])
if structures:
    st.caption(f"RCSB에서 찾은 구조: {len(structures)}개. 각 PDB ID는 별도 구조로 선택합니다.")
    f1, f2, f3 = st.columns(3)
    species_filter = f1.selectbox("종 필터", ["모든 종", *sorted({c["organism"] for x in structures for c in x["chains"] if c.get("organism")})])
    method_filter = f2.selectbox("구조 결정 방법", ["모든 방법", *sorted({m for x in structures for m in x["method"] if m})])
    sort_by = f3.selectbox("정렬", ["기본순", "해상도 우선", "리간드 보유 우선", "변이 주석 수 우선"])
    structures = [x for x in structures
        if (species_filter == "모든 종" or any(c.get("organism") == species_filter for c in x["chains"]))
        and (method_filter == "모든 방법" or method_filter in x["method"])]
    if sort_by == "해상도 우선":
        structures.sort(key=lambda x: min(x["resolution"]) if x["resolution"] else float("inf"))
    elif sort_by == "리간드 보유 우선":
        structures.sort(key=lambda x: not bool(x["ligands"]))
    elif sort_by == "변이 주석 수 우선":
        structures.sort(key=lambda x: -max([c.get("mutation_count") or 0 for c in x["chains"]] or [0]))
    if not structures:
        st.warning("선택한 필터에 맞는 구조가 없습니다.")
        st.stop()
    st.dataframe(pd.DataFrame([{"PDB": x["pdb_id"], "단백질": x["title"], "체인 단백질": ", ".join(sorted({c["protein"] for c in x["chains"] if c["protein"]})),
        "유전자": ", ".join(sorted({c["gene"] for c in x["chains"] if c["gene"]})), "방법": ", ".join(x["method"] or []),
        "해상도(Å)": ", ".join(map(str, x["resolution"] or [])), "체인": ", ".join(c["chain"] or "?" for c in x["chains"]),
        "종": ", ".join(sorted({c["organism"] or "미상" for c in x["chains"]})),
        "RCSB 변이 수": max([c.get("mutation_count") or 0 for c in x["chains"]] or [0]),
        "결합 리간드": ", ".join(f"{l['name']} ({l['ligand_id']})" for l in x["ligands"])} for x in structures]), use_container_width=True)
    def structure_option_label(pdb_id: str) -> str:
        item = next(x for x in structures if x["pdb_id"] == pdb_id)
        annotation = " ".join([item.get("title") or ""] + [c.get("protein") or "" for c in item["chains"]]).upper()
        variants = list(dict.fromkeys(re.findall(r"(?<![A-Z0-9])[A-Z]\d+[A-Z](?![A-Z0-9])", annotation)))
        variant_text = "/".join(variants) if variants else "변이 표기 없음"
        return f"{variant_text} · {pdb_id} — {item['title']}"

    pdb_id = st.selectbox("구조 변이 / 분석할 PDB ID", [x["pdb_id"] for x in structures],
        format_func=structure_option_label)
    structure = next(x for x in structures if x["pdb_id"] == pdb_id)
    chains = sorted({c["chain"] for c in structure["chains"] if c.get("chain")})
    chain_details = {c["chain"]: c for c in structure["chains"]}
    chain = st.selectbox("분석 체인", chains, format_func=lambda value:
        f"{value} · {chain_details[value]['protein']} · {chain_details[value]['organism'] or '종 미상'} · {chain_details[value]['gene'] or '유전자 미상'} · RCSB 변이 수 {chain_details[value]['mutation_count'] if chain_details[value]['mutation_count'] is not None else '미제공'}") if chains else None
    st.markdown(f"**선택 구조:** `{pdb_id}` · {structure['title']} · {', '.join(structure['method'] or [])} · 해상도 {structure['resolution'] or '미제공'} Å")
    ligand_options = [item for item in structure["ligands"] if item["is_organic_drug_like"]]
    if structure["ligands"] and not ligand_options:
        st.info("이 구조에는 결합된 유기 기준 약물이 없습니다. 물·염·금속 이온은 기준 약물 목록에서 제외했습니다.")
    ligand_choice = st.selectbox("기준 리간드", ["선택 안 함", *range(len(ligand_options))], format_func=lambda i:
        "선택 안 함" if i == "선택 안 함" else f"{ligand_options[i]['name']} ({ligand_options[i]['ligand_id']}) · 체인 {','.join(ligand_options[i]['chains'])}")
    selected_ligand = ligand_options[ligand_choice] if ligand_choice != "선택 안 함" else None
    if selected_ligand:
        st.write({"리간드 ID": selected_ligand["ligand_id"], "이름": selected_ligand["name"],
                  "SMILES": selected_ligand["smiles"] or "RCSB SMILES 미제공", "분자식": selected_ligand["formula"],
                  "분자량": selected_ligand["formula_weight"], "복합체 내 구조 좌표": selected_ligand["structure_available"]})
    st.session_state.selected_structure = structure
    st.session_state.selected_chain = chain
    st.session_state.selected_ligand = selected_ligand
    selection_key = (pdb_id, chain, selected_ligand["ligand_id"] if selected_ligand else None)
    if chain and selection_key != st.session_state.get("_saved_selection"):
        job = create_job(ROOT / "workspaces", st.session_state.get("query", query))
        url = f"https://files.rcsb.org/download/{pdb_id}.cif"
        with urllib.request.urlopen(url, timeout=60) as response:
            (job / f"{pdb_id}.cif").write_bytes(response.read())
        chain_data = next((c for c in structure["chains"] if c["chain"] == chain), {})
        manifest = json.loads((job / "manifest.json").read_text(encoding="utf-8"))
        manifest.update({"pdb_id": pdb_id, "chain": chain, "reference_ligand": selected_ligand,
                         "structure_file": f"{pdb_id}.cif", "structure_is_mutant": bool(chain_data.get("mutation_count")),
                         "structure_mutation_count": chain_data.get("mutation_count"),
                         "stages": {"pdb_search": "complete", "structure_download": "complete"}})
        (job / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        st.session_state.job = str(job)
        st.session_state._saved_selection = selection_key
        st.session_state._uniprot_source_chain = None
    st.caption("선택한 PDB 복합체의 결합 리간드를 DTA 기준 약물로 사용합니다. RxnFlow/Vina pocket center도 이 리간드 좌표에서 계산합니다.")
    st.caption(structure["mutation_annotation"])
    if st.session_state.get("target_mutation"):
        st.info("입력한 변이는 DTA용 UniProt 서열에 적용됩니다. 도킹 구조는 별도로 선택한 PDB이므로, 선택 PDB가 같은 변이를 포함하는지 구조 제목/주석을 확인하세요.")

if st.session_state.get("job"):
    st.success(f"작업 폴더: {st.session_state.job}")

st.header("2. UniProt 서열과 변이 FASTA")
selected_structure = st.session_state.get("selected_structure")
selected_chain = st.session_state.get("selected_chain")
chain_key = (selected_structure.get("pdb_id"), selected_chain) if selected_structure and selected_chain else None
if chain_key and chain_key != st.session_state.get("_uniprot_source_chain"):
    st.session_state.pop("uniprot", None)
    st.session_state.pop("mutant_sequence", None)
    chain_data = next((c for c in selected_structure["chains"] if c["chain"] == selected_chain), {})
    accessions = chain_data.get("uniprot") or []
    annotation = " ".join([selected_structure.get("title") or "", chain_data.get("protein") or ""]).upper()
    annotated_mutations = list(dict.fromkeys(re.findall(r"(?<![A-Z0-9])[A-Z]\d+[A-Z](?![A-Z0-9])", annotation)))
    st.session_state.target_mutation = "/".join(annotated_mutations) or st.session_state.get("target_mutation", "")
    st.session_state.fasta_mutation = st.session_state.target_mutation
    st.session_state.prediction_mutation = st.session_state.target_mutation
    if accessions:
        st.session_state.uniprot_accession = accessions[0]
        try:
            header, sequence = fetch_uniprot_fasta(accessions[0])
            mutation = st.session_state.get("target_mutation", "")
            mutant_sequence = apply_mutations(sequence, mutation) if mutation else sequence
            st.session_state.uniprot = (header, sequence)
            st.session_state.mutant_sequence = mutant_sequence
            st.session_state.uniprot_error = None
        except Exception as exc:
            st.session_state.uniprot_error = str(exc)
    st.session_state._uniprot_source_chain = chain_key
accession = st.session_state.get("uniprot_accession", "")
if st.session_state.get("uniprot_error"):
    st.error(f"UniProt 조회 또는 변이 적용 실패: {st.session_state.uniprot_error}")
if st.session_state.get("uniprot"):
    header, sequence = st.session_state.uniprot
    mutation = st.session_state.get("target_mutation", "")
    mutant_sequence = st.session_state.get("mutant_sequence", sequence)
    st.success(f"UniProt {accession} · {len(sequence)} aa · 선택한 PDB 체인에서 자동 조회")
    if mutation:
        st.caption(f"DTA 입력 서열에 {mutation} 변이를 반영했습니다.")
    with st.expander("UniProt FASTA 확인"):
        st.code(f">{header} {mutation or 'WT'}\n{mutant_sequence}", language=None)

st.header("3. RxnFlow 후보 생성")
job_text = st.session_state.get("job")
selected_structure = st.session_state.get("selected_structure")
selected_chain = st.session_state.get("selected_chain")
selected_ligand = st.session_state.get("selected_ligand")
rxnflow_root = ROOT / "RxnFlow"
rxnflow_python_default = os.environ.get("RXNFLOW_PYTHON", str(rxnflow_root / ".conda" / "bin" / "python"))
rxnflow_env_default = os.environ.get("RXNFLOW_ENV_DIR", str(rxnflow_root / "data" / "envs" / "zincfrag"))
rxnflow_python, rxnflow_env = rxnflow_python_default, rxnflow_env_default
rx_status = cached_rxnflow_status(rxnflow_python, rxnflow_env)
vina_exec = str(Path(vina_python()).expanduser().resolve())
vina_info = cached_vina_status(vina_exec)
vina_ready = vina_info["ready"]
st.caption(f"RxnFlow: {'준비됨' if rx_status['ready'] else '실행 환경 준비 필요'} · Vina: {'준비됨' if vina_ready else '실행 환경 준비 필요'}")
if not rx_status["ready"]:
    st.info("RxnFlow 실행 환경과 building block 파일이 준비된 뒤 생성 버튼을 사용할 수 있습니다. RxnFlow Python은 Python 3.12와 torch-geometric, pmnet_appl이 필요하고, 환경 폴더에는 template.txt, building_block.smi, bb_mask.npy, bb_fp_2_1024.npy, bb_desc.npy가 있어야 합니다.")
if job_text and selected_structure and selected_chain:
    job = Path(job_text)
    manifest_path = job / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cif_path = job / manifest.get("structure_file", "")
    st.caption(f"수용체 입력: {selected_structure['pdb_id']} 체인 {selected_chain} · {cif_path}")
    ligand_id = selected_ligand["ligand_id"] if selected_ligand else ""
    ligand_chain = (selected_ligand.get("chains") or [""])[0] if selected_ligand else ""
    if ligand_id:
        st.caption(f"RxnFlow와 도킹의 pocket center는 선택한 PDB 결합 약물 {ligand_id}의 원자 좌표 중심으로 자동 계산합니다.")
    else:
        st.warning("PDB에 선택 가능한 결합 약물이 없어 RxnFlow 후보 생성을 진행할 수 없습니다. 기준 약물이 결합된 구조를 선택하세요.")
    box_cols = st.columns(3)
    box_size = [box_cols[i].number_input(f"도킹 박스 크기 {'XYZ'[i]} (Å)", min_value=8.0, max_value=40.0, value=20.0, key=f"dock_box_{i}") for i in range(3)]
    num_samples = st.number_input("RxnFlow 후보 수", min_value=1, max_value=50000, value=50000, step=100)
    if rx_status.get("cuda_available", False):
        use_cuda = st.checkbox("RxnFlow GPU 사용", value=True)
    else:
        use_cuda = False
        st.caption("RxnFlow 환경에서 CUDA를 사용할 수 없어 CPU 실행으로 설정했습니다.")
    pipeline_ready = rx_status["ready"] and vina_ready and cif_path.is_file() and bool(ligand_id)
    if st.button("1단계: RxnFlow 후보 생성", type="primary", disabled=not pipeline_ready):
        try:
            stages = dict(manifest.get("stages", {}))
            receptor_dir = job / "receptor"
            prepare_cmd = [vina_exec, str(ROOT / "app" / "vina_pipeline.py"), "prepare",
                "--cif", str(cif_path), "--chain", str(selected_chain), "--ligand-id", ligand_id,
                "--ligand-chain", ligand_chain, "--out-dir", str(receptor_dir)]
            with st.spinner("선택한 PDB 체인으로 수용체를 준비하는 중..."):
                prepared = run_command(prepare_cmd, ROOT, timeout=900)
            if not prepared["ok"]:
                stages["receptor_prepare"] = "failed"
                update_manifest(job, stages=stages, receptor_prepare_stderr=prepared["stderr"][-4000:])
                raise RuntimeError(prepared["stderr"][-4000:] or prepared["stdout"][-4000:])
            receptor_info = json.loads(prepared["stdout"].splitlines()[-1])
            center = receptor_info["center"]
            if not center:
                raise RuntimeError("선택한 PDB 결합 약물의 좌표에서 pocket center를 계산하지 못했습니다.")
            stages["receptor_prepare"] = "complete"
            mutation_count = next((c.get("mutation_count") or 0 for c in selected_structure["chains"]
                if c["chain"] == selected_chain), 0)
            update_manifest(job, receptor_file=str(Path(receptor_info["pdbqt"]).relative_to(job)),
                structure_is_mutant=bool(mutation_count), structure_mutation_count=mutation_count,
                docking_center=center, docking_box_size=box_size, stages=stages)
            candidate_csv = job / "generated_candidates.csv"
            progress_bar = st.progress(0.0, text=f"RxnFlow 생성: 0/{int(num_samples)}개")
            progress_text = st.empty()

            def show_rxnflow_progress(completed: int, total: int) -> None:
                ratio = min(completed / max(total, 1), 1.0)
                progress_bar.progress(ratio, text=f"RxnFlow 생성: {completed:,}/{total:,}개")
                progress_text.caption(f"현재 {completed:,}개 생성 완료 · 목표 {total:,}개")

            generated = run_rxnflow_generation(rxnflow_python, rxnflow_env, receptor_info["pdb"],
                center, str(candidate_csv), int(num_samples), use_cuda,
                progress_callback=show_rxnflow_progress)
            if not generated["ok"]:
                stages["rxnflow_generation"] = "failed"
                update_manifest(job, stages=stages, rxnflow_stderr=generated["stderr"][-4000:])
                raise RuntimeError(generated["stderr"][-4000:] or generated["stdout"][-4000:])
            stages["rxnflow_generation"] = "complete"
            update_manifest(job, rxnflow_candidates=str(candidate_csv.relative_to(job)), stages=stages)
            st.session_state.pop("dta_rows", None)
            st.session_state.pop("dta_passed_csv", None)
            st.success(f"RxnFlow 후보 {int(num_samples)}개를 생성했습니다. 다음 단계에서 기준 약물과 DTA 비교를 진행하세요.")
        except Exception as exc:
            st.error(f"RxnFlow/Vina 파이프라인 실패: {exc}")
elif not job_text:
    st.info("먼저 선택한 PDB 구조를 작업 폴더에 저장하세요.")
else:
    st.info("PDB 구조와 분석 체인을 선택하세요.")

st.header("4. DTA 예측 및 선별")
model_set = st.radio("DTA dataset / measurement", ["DAVIS", "ChEMBL IC50", "ChEMBL Ki", "ChEMBL Kd"], horizontal=True)
status = stage_status(model_set)
st.caption("선택한 DTA 모델의 5개 fold 체크포인트 상태를 확인했습니다." if status["configured"] else "선택한 DTA 모델 체크포인트가 준비되지 않았습니다.")
st.caption("ChEMBL IC50, Ki and Kd use separate 5-fold models. Do not compare predictions across measurement types.")
selection_options = [
    "엄격 기준: 후보 fold 최솟값 > 기준 약물 fold 최댓값",
    "평균 기준: 후보 fold 평균 > 기준 약물 fold 평균",
]
selection_label = st.session_state.get("dta_selection_label", selection_options[0])
selection_method = "mean" if selection_label.startswith("평균") else "strict"
if st.session_state.get("uniprot"):
    st.subheader("선택 서열로 5-fold 예측 (SMILES 직접 입력)")
    generated_path = Path(st.session_state.get("job", "")) / "generated_candidates.csv"
    generated_rows = pd.read_csv(generated_path).to_dict(orient="records") if generated_path.is_file() else []
    reference_drug = st.session_state.get("selected_ligand") or {}
    reference_smiles = reference_drug.get("smiles") or ""
    st.write(f"기준 약물: {reference_drug.get('name') or reference_drug.get('ligand_id') or '미선택'} · RxnFlow 후보: {len(generated_rows)}개")
    if st.button("생성 후보와 기준 약물 DTA 예측", disabled=not generated_rows or not reference_smiles):
        try:
            prediction_sequence = st.session_state.get("mutant_sequence") or st.session_state.uniprot[1]
            candidate_smiles = [str(row.get("SMILES") or row.get("smiles") or "") for row in generated_rows]
            if not reference_smiles or not all(candidate_smiles):
                raise ValueError("기준 리간드 SMILES 또는 생성 후보 SMILES가 없습니다.")
            from app.dta import predict_five_fold
            output_dir = Path(st.session_state.get("job") or create_job(ROOT / "workspaces", accession))
            predictions = predict_five_fold([reference_smiles.strip(), *candidate_smiles], prediction_sequence, model_set, output_dir)
            rows = predictions["rows"]
            reference_folds = [rows[0][f"fold{i}"] for i in range(1, 6)]
            reference_ligand = st.session_state.get("selected_ligand") or {}
            reference_name = reference_ligand.get("name") or reference_ligand.get("ligand_id") or "PDB reference drug"
            reference_id = f"reference_{reference_ligand.get('ligand_id') or 'drug'}"
            screened = [{**rows[0], "candidate_id": reference_name, "fold_mean": sum(reference_folds) / 5,
                         "ID": reference_id, "drug_name": reference_name, "status": "reference"}]
            for i, row in enumerate(rows[1:]):
                decision = fold_decision([row[f"fold{fold}"] for fold in range(1, 6)], reference_folds, selection_method)
                source = generated_rows[i]
                candidate_id = source.get("ID") or source.get("candidate_id") or f"candidate_{i+1}"
                screened.append({**source, **row, "candidate_id": candidate_id, "fold_mean": sum(row[f"fold{fold}"] for fold in range(1, 6)) / 5,
                    "ID": candidate_id, "drug_name": candidate_id, "reference_fold1": reference_folds[0], "reference_fold2": reference_folds[1],
                    "reference_fold3": reference_folds[2], "reference_fold4": reference_folds[3],
                    "reference_fold5": reference_folds[4], **decision})
            result_path = output_dir / "screening_results.csv"
            pd.DataFrame(screened).to_csv(result_path, index=False)
            accepted = [{"ID": row["ID"], "drug_name": row["drug_name"], "SMILES": row["SMILES"]}
                        for row in screened[1:] if row["status"] == "passed"]
            accepted_path = output_dir / "dta_passed_candidates.csv"
            pd.DataFrame(accepted, columns=["ID", "drug_name", "SMILES"]).to_csv(accepted_path, index=False)
            st.session_state.dta_rows = screened
            st.session_state.dta_csv = str(result_path)
            st.session_state.dta_passed_csv = str(accepted_path)
            st.session_state.dta_note = predictions["note"]
        except Exception as exc:
            st.error(f"DTA 예측 실패: {exc}")
if st.session_state.get("dta_rows") is not None:
    screened = list(st.session_state.dta_rows)
    if screened:
        reference_row = screened[0]
        reference_folds = [float(reference_row[f"fold{i}"]) for i in range(1, 6)]
        reference_mean = sum(reference_folds) / 5
        reference_row["fold_mean"] = reference_mean
        reference_row["status"] = "reference"
        for row in screened[1:]:
            candidate_folds = [row.get(f"fold{i}") for i in range(1, 6)]
            decision = fold_decision(candidate_folds, reference_folds, selection_method)
            row.update(decision)
            row["판정"] = "통과" if decision["status"] == "passed" else "제외" if decision["status"] == "rejected" else "예측 실패"
            if decision["status"] == "passed" or decision["status"] == "rejected":
                if selection_method == "strict":
                    row["선별 비교"] = f"최솟값 {decision['candidate_min']:.3f} > 기준 최댓값 {decision['reference_max']:.3f}"
                else:
                    row["선별 비교"] = f"평균 {decision['candidate_mean']:.3f} > 기준 평균 {decision['reference_mean']:.3f}"
            else:
                row["선별 비교"] = decision.get("reason", "5-fold 예측 필요")
            try:
                row["fold_mean"] = sum(float(value) for value in candidate_folds) / 5
            except (TypeError, ValueError):
                row["fold_mean"] = None
        st.session_state.dta_rows = screened

        result_path = Path(st.session_state.get("dta_csv", ""))
        if result_path:
            pd.DataFrame(screened).to_csv(result_path, index=False)
        accepted_rows = [{"ID": row["ID"], "drug_name": row["drug_name"], "SMILES": row["SMILES"]}
                         for row in screened[1:] if row.get("status") == "passed"]
        accepted_path = Path(st.session_state.get("dta_passed_csv", ""))
        if accepted_path:
            pd.DataFrame(accepted_rows, columns=["ID", "drug_name", "SMILES"]).to_csv(accepted_path, index=False)

        def score_cell_styles(row):
            styles = []
            for column in row.index:
                if column in {"status", "판정"}:
                    if row.get("status") == "reference":
                        styles.append("background-color: #dbeafe; font-weight: bold")
                    elif row.get("status") == "passed":
                        styles.append("background-color: #c6efce; color: #006100; font-weight: bold")
                    elif row.get("status") == "rejected":
                        styles.append("background-color: #fce4d6; color: #9c0006")
                    else:
                        styles.append("")
                elif column in [f"fold{i}" for i in range(1, 6)]:
                    if row.get("status") == "reference":
                        styles.append("background-color: #dbeafe; font-weight: bold")
                    elif selection_method == "strict":
                        try:
                            styles.append("background-color: #c6efce; color: #006100" if float(row[column]) > max(reference_folds) else "")
                        except (TypeError, ValueError):
                            styles.append("")
                    else:
                        styles.append("")
                elif column == "fold_mean" and selection_method == "mean":
                    try:
                        styles.append("background-color: #c6efce; color: #006100; font-weight: bold" if float(row[column]) > reference_mean else "")
                    except (TypeError, ValueError):
                        styles.append("")
                else:
                    styles.append("")
            return styles

        candidates_by_mean = sorted(
            screened[1:],
            key=lambda row: float(row["fold_mean"]) if row.get("fold_mean") is not None else float("-inf"),
            reverse=True,
        )
        preview = [reference_row, *candidates_by_mean[:100]]
        preview_df = pd.DataFrame(preview)
        st.subheader("Fold별·평균 DTA 결과")
        st.caption("초록색 셀은 같은 fold의 기준 약물 점수 또는 기준 약물 평균보다 높은 값입니다. 표에는 기준 약물과 평균 상위 100개 후보를 표시합니다.")
        st.dataframe(preview_df.style.apply(score_cell_styles, axis=1).format(precision=3, na_rep="—"),
                     use_container_width=True, height=520)
        st.subheader("후보 선별")
        selection_label = st.radio(
            "선별 기준", selection_options, horizontal=True, key="dta_selection_label",
        )
        selection_method = "mean" if selection_label.startswith("평균") else "strict"
        st.caption("현재 선택 기준으로 판정을 갱신했습니다. 엄격 기준에서는 모든 후보 fold가 기준 약물의 최고 fold보다 높아야 통과합니다. 평균 기준에서는 후보 평균이 기준 약물 평균보다 높아야 통과합니다.")

        if candidates_by_mean:
            chart_candidates = [row for row in candidates_by_mean if row.get("status") == "passed"] or candidates_by_mean
            chart_ids = [str(row.get("ID")) for row in chart_candidates[:100]]
            selected_chart_id = st.selectbox("Fold 그래프로 비교할 후보", chart_ids)
            selected_row = next(row for row in chart_candidates if str(row.get("ID")) == selected_chart_id)
            chart_data = pd.DataFrame({
                "기준 약물": [*reference_folds, reference_mean],
                selected_chart_id: [*(float(selected_row[f"fold{i}"]) for i in range(1, 6)),
                                    float(selected_row["fold_mean"])],
            }, index=["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5", "평균"])
            st.line_chart(chart_data, y_label="DTA 예측값", height=320)

            smiles = str(selected_row.get("SMILES") or selected_row.get("smiles") or "")
            candidate_status = {"passed": "선별 통과", "rejected": "선별 제외", "prediction_failed": "예측 실패"}.get(selected_row.get("status"), "상태 미확인")
            st.markdown(f"**선택 후보:** {selected_row.get('drug_name') or selected_chart_id} · {candidate_status}")
            if smiles:
                try:
                    from rdkit import Chem
                    from rdkit.Chem import Crippen, Descriptors, Draw, rdMolDescriptors

                    molecule = Chem.MolFromSmiles(smiles)
                    if molecule is None:
                        st.warning("선택 후보의 SMILES를 읽을 수 없어 2D 구조를 표시하지 못했습니다.")
                    else:
                        structure_col, info_col = st.columns([1, 2])
                        with structure_col:
                            st.image(Draw.MolToImage(molecule, size=(480, 360)), caption="선택 후보 2D 구조", use_container_width=True)
                        with info_col:
                            st.code(smiles, language=None)
                            prop_cols = st.columns(4)
                            prop_cols[0].metric("분자량", f"{Descriptors.MolWt(molecule):.1f} Da")
                            prop_cols[1].metric("cLogP", f"{Crippen.MolLogP(molecule):.2f}")
                            prop_cols[2].metric("TPSA", f"{rdMolDescriptors.CalcTPSA(molecule):.1f} Å²")
                            prop_cols[3].metric("HBD / HBA", f"{rdMolDescriptors.CalcNumHBD(molecule)} / {rdMolDescriptors.CalcNumHBA(molecule)}")

                        fold_info = pd.DataFrame({
                            "Fold": [f"Fold {i}" for i in range(1, 6)] + ["평균"],
                            "기준 약물": [*reference_folds, reference_mean],
                            "선택 후보": [*(float(selected_row[f"fold{i}"]) for i in range(1, 6)), float(selected_row["fold_mean"])],
                        })
                        st.dataframe(fold_info.style.format({"기준 약물": "{:.3f}", "선택 후보": "{:.3f}"}),
                                     hide_index=True, use_container_width=True)
                except ImportError:
                    st.info("RDKit이 없어 2D 구조와 분자 특성을 표시할 수 없습니다.")
        st.caption(st.session_state.dta_note)
        st.write(f"CSV: {st.session_state.dta_csv}")

st.header("5. DTA 통과 후보 SDF 변환 및 도킹")
job_text = st.session_state.get("job")
passed_csv = st.session_state.get("dta_passed_csv")
if passed_csv and Path(passed_csv).is_file():
    accepted = pd.read_csv(passed_csv)
    st.caption(f"PDB에서 선택한 기준 약물과 DTA 통과 후보 {len(accepted)}개를 같은 조건으로 도킹합니다.")
    if not accepted.empty and st.button("3단계: 통과 후보만 SDF 변환 및 Vina 도킹", type="primary"):
        try:
            job = Path(job_text)
            manifest = json.loads((job / "manifest.json").read_text(encoding="utf-8"))
            center, box_size = manifest.get("docking_center"), manifest.get("docking_box_size", [20, 20, 20])
            reference_row = (st.session_state.get("dta_rows") or [{}])[0]
            reference_smiles = str(reference_row.get("SMILES") or reference_row.get("smiles") or "").strip()
            if not reference_smiles:
                raise ValueError("기준 약물 SMILES가 없어 기준 약물 도킹을 추가할 수 없습니다.")
            docking_input = job / "docking_input_candidates.csv"
            docking_rows = [{
                "ID": reference_row.get("ID") or "reference_drug",
                "drug_name": reference_row.get("drug_name") or "기준 약물",
                "SMILES": reference_smiles,
            }, *accepted.to_dict(orient="records")]
            pd.DataFrame(docking_rows, columns=["ID", "drug_name", "SMILES"]).to_csv(docking_input, index=False)
            dock_cmd = [vina_exec, str(ROOT / "app" / "vina_pipeline.py"), "dock",
                "--receptor", str(job / "receptor" / "receptor.pdbqt"), "--candidates", str(docking_input),
                "--out-dir", str(job), "--center", *map(str, center), "--box-size", *map(str, box_size)]
            with st.spinner("기준 약물과 DTA 통과 후보를 같은 조건으로 Vina 도킹 중..."):
                docked = run_command(dock_cmd, ROOT, timeout=24 * 3600)
            if not docked["ok"]:
                raise RuntimeError(docked["stderr"][-4000:] or docked["stdout"][-4000:])
            dock_info = json.loads(docked["stdout"].splitlines()[-1])
            stages = dict(manifest.get("stages", {}))
            stages["vina_docking"] = "complete"
            update_manifest(job, docking_input_candidates=str(docking_input.relative_to(job)),
                docking_scores=str(Path(dock_info["summary"]).relative_to(job)),
                docking_poses=str(Path(dock_info["poses"]).relative_to(job)), stages=stages)
            st.success(f"도킹 완료: {dock_info['docked']}/{dock_info['rows']}개 성공")
        except Exception as exc:
            st.error(f"SDF 변환/Vina 도킹 실패: {exc}")
elif st.session_state.get("dta_rows") is not None:
    st.info("DTA 통과 후보가 없어 도킹을 진행하지 않습니다.")
if job_text:
    job = Path(job_text)
    receptor_dir = job / "receptor"
    receptor_pdb = receptor_dir / "receptor.pdb"
    receptor_pdbqt = receptor_dir / "receptor.pdbqt"
    if receptor_pdb.is_file() or receptor_pdbqt.is_file():
        st.subheader("구조 파일 다운로드")
        st.caption("원본 PDB에서 분석 체인의 단백질만 남긴 수용체 파일과, 도킹에 사용한 PDBQT를 다운로드할 수 있습니다.")
        protein_cols = st.columns(2)
        manifest = json.loads((job / "manifest.json").read_text(encoding="utf-8"))
        structure_tag = f"{manifest.get('pdb_id', 'structure')}_chain_{manifest.get('chain', 'A')}"
        if receptor_pdb.is_file():
            protein_cols[0].download_button("단백질만 추출한 PDB 다운로드", receptor_pdb.read_bytes(),
                file_name=f"{structure_tag}_protein.pdb", mime="chemical/x-pdb", key="download_protein_pdb")
        if receptor_pdbqt.is_file():
            protein_cols[1].download_button("수용체 PDBQT 다운로드", receptor_pdbqt.read_bytes(),
                file_name=f"{structure_tag}_receptor.pdbqt", mime="chemical/x-pdbqt", key="download_protein_pdbqt")

    if (job / "docking_scores.csv").is_file():
        dock_rows = pd.read_csv(job / "docking_scores.csv")
        reference_row = (st.session_state.get("dta_rows") or [{}])[0]
        reference_id = re.sub(r"[^A-Za-z0-9_.-]", "_", str(reference_row.get("ID") or "reference_drug"))
        reference_dock = dock_rows[dock_rows["ID"].astype(str) == reference_id]
        reference_score = None
        if not reference_dock.empty:
            reference_result = reference_dock.iloc[0]
            try:
                if reference_result.get("status") != "docked":
                    raise ValueError("기준 약물 도킹이 성공하지 않았습니다.")
                reference_score = float(reference_result["Vina"])
                if pd.isna(reference_score):
                    raise ValueError("기준 약물 Vina 점수가 비어 있습니다.")
                st.metric("기준 약물 Vina 점수", f"{reference_score:.3f} kcal/mol")
                st.caption(f"기준 약물: {reference_result['drug_name']} · SMILES에서 3D를 생성해 후보와 같은 조건으로 재도킹한 값입니다. Vina 점수는 더 낮을수록 유리합니다.")
            except (TypeError, ValueError):
                st.warning(f"기준 약물 도킹 실패: {reference_result.get('status', '원인 미상')}")
        else:
            st.info("현재 표시된 기존 도킹 결과에는 기준 약물 점수가 없습니다. 위 버튼을 다시 실행하면 기준 약물도 함께 도킹합니다.")
        candidate_docks = dock_rows[dock_rows["ID"].astype(str) != reference_id]
        st.dataframe(candidate_docks.sort_values("Vina", na_position="last"), use_container_width=True)
        if reference_score is not None:
            candidate_docks = candidate_docks.copy()
            candidate_docks["_vina_numeric"] = pd.to_numeric(candidate_docks["Vina"], errors="coerce")
            better_docks = candidate_docks[
                (candidate_docks.get("status") == "docked") & (candidate_docks["_vina_numeric"] < reference_score)
            ]
            st.caption(f"Vina가 기준 약물보다 유리한 후보: {len(better_docks)}개 · 기준보다 더 낮은(더 음수인) 점수만 묶음에 포함합니다.")
            if not better_docks.empty:
                archive_buffer = io.BytesIO()
                archive_root = "target_screening_docking_bundle"
                with zipfile.ZipFile(archive_buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                    for file_path, archive_name in [
                        (receptor_pdb, f"{archive_root}/receptor/protein_only.pdb"),
                        (receptor_pdbqt, f"{archive_root}/receptor/receptor.pdbqt"),
                        (job / "ligands" / f"{reference_id}.pdbqt", f"{archive_root}/reference/{reference_id}.pdbqt"),
                        (job / "docking_poses" / f"{reference_id}_docked.pdbqt", f"{archive_root}/reference/{reference_id}_docked.pdbqt"),
                    ]:
                        if file_path.is_file():
                            archive.write(file_path, archive_name)
                    for _, dock_row in better_docks.iterrows():
                        candidate_id = re.sub(r"[^A-Za-z0-9_.-]", "_", str(dock_row["ID"]))
                        for file_path, archive_name in [
                            (job / "ligands" / f"{candidate_id}.pdbqt", f"{archive_root}/better_candidates/{candidate_id}.pdbqt"),
                            (job / "docking_poses" / f"{candidate_id}_docked.pdbqt", f"{archive_root}/better_candidates/{candidate_id}_docked.pdbqt"),
                        ]:
                            if file_path.is_file():
                                archive.write(file_path, archive_name)
                    summary_rows = pd.concat([reference_dock, better_docks.drop(columns=["_vina_numeric"])], ignore_index=True)
                    archive.writestr(f"{archive_root}/reference_and_better_scores.csv", summary_rows.to_csv(index=False))
                archive_buffer.seek(0)
                st.download_button(
                    f"기준 약물 + 우수 후보 {len(better_docks)}개 ZIP 다운로드",
                    archive_buffer.getvalue(),
                    file_name="target_screening_reference_and_better_candidates.zip",
                    mime="application/zip", key="download_better_than_reference_bundle",
                )
            else:
                st.info("기준 약물보다 Vina 점수가 더 낮은 성공 후보가 없습니다.")
        ligand_dir = job / "ligands"
        ligand_files = sorted(ligand_dir.glob("*.pdbqt"))
        if ligand_files:
            st.markdown("**약물 PDBQT 다운로드**")
            selected_ligand_name = st.selectbox(
                "다운로드할 약물", [path.name for path in ligand_files],
                format_func=lambda name: Path(name).stem, key="download_ligand_pdbqt_choice"
            )
            selected_ligand_file = ligand_dir / selected_ligand_name
            ligand_col, pose_col = st.columns(2)
            ligand_col.download_button("선택 약물 PDBQT 다운로드", selected_ligand_file.read_bytes(),
                file_name=selected_ligand_file.name, mime="chemical/x-pdbqt", key="download_candidate_ligand_pdbqt")
            pose_file = job / "docking_poses" / f"{selected_ligand_file.stem}_docked.pdbqt"
            if pose_file.is_file():
                pose_col.download_button("선택 약물 도킹 포즈 PDBQT 다운로드", pose_file.read_bytes(),
                    file_name=pose_file.name, mime="chemical/x-pdbqt", key="download_candidate_pose_pdbqt")
