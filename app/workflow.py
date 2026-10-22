"""Stage records and guarded external-tool entry points."""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from .core import fold_decision


def create_job(root: str | Path, target: str) -> Path:
    job = Path(root) / f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    job.mkdir(parents=True)
    manifest = {"target_query": target, "mode": "real", "status": "created", "stages": {},
                "warning": "Predictions and docking scores are computational prioritization only."}
    (job / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return job


def update_manifest(job: str | Path, **values: Any) -> None:
    path = Path(job) / "manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update(values)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def annotate_fold_rows(rows: list[dict[str, Any]], candidate_columns: list[str], reference_columns: list[str]) -> list[dict[str, Any]]:
    if len(candidate_columns) != 5 or len(reference_columns) != 5:
        raise ValueError("Exactly five fold columns are required")
    output = []
    for row in rows:
        result = fold_decision([row.get(k) for k in candidate_columns], [row.get(k) for k in reference_columns])
        output.append({**row, **result})
    return output


def run_command(command: list[str], cwd: str | Path, timeout: int = 3600,
                env: dict[str, str] | None = None,
                on_output: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Run one configured tool, optionally forwarding output lines while it runs."""
    try:
        if on_output is None:
            result = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True, timeout=timeout, check=False)
            return {"ok": result.returncode == 0, "returncode": result.returncode,
                    "command": command, "stdout": result.stdout, "stderr": result.stderr}
        process = subprocess.Popen(command, cwd=cwd, env=env, text=True, bufsize=1,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output = []
        assert process.stdout is not None
        for line in process.stdout:
            output.append(line)
            on_output(line.rstrip("\r\n"))
        try:
            return_code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise
        return {"ok": return_code == 0, "returncode": return_code,
                "command": command, "stdout": "".join(output), "stderr": ""}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "command": command, "stdout": "", "stderr": str(exc)}


def linked_assets() -> dict[str, str | None]:
    root = Path(__file__).resolve().parents[1] / "vendor"
    return {name: (str((root / name).resolve()) if (root / name).exists() else None)
            for name in ["Autodock", "DTA_model", "PLM", "extract_drug_embeddings.py"]}


def available_ic50_folds(model_set: str) -> list[str]:
    root = Path(__file__).resolve().parents[1] / "vendor" / "DTA_model" / "result"
    aliases = {"DAVIS": "DAVIS-complete", "ChEMBL": "IC50", "ChEMBL IC50": "IC50", "ChEMBL Ki": "Ki", "ChEMBL Kd": "Kd"}
    if model_set not in aliases:
        raise ValueError("Choose DAVIS, ChEMBL IC50, ChEMBL Ki, or ChEMBL Kd")
    path = root / aliases[model_set]
    return [str(path / f"fold{i}" / "best.pth") for i in range(1, 6)
            if (path / f"fold{i}" / "best.pth").is_file()]


def download_model_assets(model_set: str) -> tuple[list[str], str]:
    """Use local inference weights, or fetch missing files from the configured HF repo."""
    aliases = {"DAVIS": "DAVIS-complete", "ChEMBL": "IC50", "ChEMBL IC50": "IC50",
               "ChEMBL Ki": "Ki", "ChEMBL Kd": "Kd"}
    if model_set not in aliases:
        raise ValueError("Choose DAVIS, ChEMBL IC50, ChEMBL Ki, or ChEMBL Kd")
    root = Path(__file__).resolve().parents[1] / "vendor"
    fold_root = root / "DTA_model" / "result" / aliases[model_set]
    folds = [fold_root / f"fold{i}" / "best.pth" for i in range(1, 6)]
    plm_checkpoint = (root / "PLM").resolve() / "checkpoint_step500000.pt"
    if all(path.is_file() for path in [*folds, plm_checkpoint]):
        return [str(path) for path in folds], str(plm_checkpoint)

    repo_id = os.environ.get("TARGET_SCREENING_HF_REPO")
    if not repo_id:
        raise FileNotFoundError("Model weights are missing. Set TARGET_SCREENING_HF_REPO to the Hugging Face repo ID.")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub in the DTA environment to download model weights") from exc

    downloaded_paths = []
    for path in folds:
        if path.is_file():
            downloaded_paths.append(str(path))
        else:
            repo_path = f"DTA_model/result/{aliases[model_set]}/{path.parent.name}/best.pth"
            downloaded_paths.append(hf_hub_download(repo_id=repo_id, filename=repo_path))
    if plm_checkpoint.is_file():
        downloaded_plm = str(plm_checkpoint)
    else:
        repo_path = "PLM/checkpoint_step500000.pt"
        downloaded_plm = hf_hub_download(repo_id=repo_id, filename=repo_path)
    return downloaded_paths, downloaded_plm



def stage_status(model_set: str) -> dict[str, Any]:
    import importlib.util
    folds = available_ic50_folds(model_set)
    missing = [module for module in ("torch", "transformers") if importlib.util.find_spec(module) is None]
    return {"model_set": model_set, "fold_checkpoints": folds,
            "configured": len(folds) == 5, "runnable_here": len(folds) == 5 and not missing,
            "missing_runtime": missing}



def rxnflow_status(python: str, env_dir: str) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    python_path = Path(python).expanduser()
    env = Path(env_dir).expanduser()
    if not python_path.is_absolute():
        python_path = Path(__file__).resolve().parents[1] / python_path
    if not env.is_absolute():
        env = Path(__file__).resolve().parents[1] / env
    required_data = ["template.txt", "building_block.smi", "bb_mask.npy",
                     "bb_fp_2_1024.npy", "bb_desc.npy"]
    missing_data = [name for name in required_data if not (env / name).is_file()]
    result = {"python": str(python_path), "environment": str(env), "missing_data": missing_data,
              "ready": False, "missing_runtime": []}
    if not python_path.is_file():
        result["missing_runtime"] = ["RxnFlow Python executable"]
        return result
    source = root / "RxnFlow" / "src"
    code = ("import importlib.util, sys; sys.path.insert(0, " + repr(str(source)) + "); "
            "mods=('torch','torch_geometric','pmnet_appl','rxnflow','rdkit','tqdm','gdown'); "
            "missing=[m for m in mods if importlib.util.find_spec(m) is None]; "
            "print('python-version' if sys.version_info[:2] != (3,12) else ''); "
            "import torch; print('cuda:' + str(torch.cuda.is_available())); "
            "print(chr(10).join(missing))")
    check = subprocess.run([str(python_path), "-c", code], cwd=root / "RxnFlow",
                           text=True, capture_output=True, timeout=30, check=False)
    output_lines = check.stdout.splitlines() if check.returncode == 0 else []
    cuda_line = next((line for line in output_lines if line.startswith("cuda:")), "cuda:False")
    result["cuda_available"] = cuda_line.partition(":")[2] == "True"
    result["missing_runtime"] = [x for x in output_lines if x and not x.startswith("cuda:")] if check.returncode == 0 else [check.stderr.strip()[-1000:]]
    result["ready"] = check.returncode == 0 and not result["missing_runtime"] and not missing_data
    return result


def vina_python() -> str:
    configured = os.environ.get("VINA_PYTHON")
    default = Path("/HDD1/phj318/miniconda3/envs/vina/bin/python")
    return configured or str(default if default.is_file() else Path(__file__).resolve().parents[1] / ".venv-vina/bin/python")


def run_rxnflow_generation(python: str, env_dir: str, receptor_pdb: str, center: list[float],
                           output_csv: str, num_samples: int, use_cuda: bool = False,
                           progress_callback: Callable[[int, int], None] | None = None) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    script = root / "RxnFlow" / "scripts" / "custom_sampling_zeroshot.py"
    python_path = Path(python).expanduser()
    env_path = Path(env_dir).expanduser()
    if not python_path.is_absolute():
        python_path = root / python_path
    if not env_path.is_absolute():
        env_path = root / env_path
    command = [str(python_path), str(script), "--model_path", "qvina-unif-0-64", "--env_dir", str(env_path),
               "--protein", receptor_pdb, "--center", *map(str, center), "--num_samples", str(num_samples),
               "--max_attempts", str(max(10000, num_samples * 100)), "--batch_size", "10", "--out_path", output_csv]
    if use_cuda:
        command.append("--cuda")
    env = os.environ.copy()
    source = str(root / "RxnFlow" / "src")
    env["PYTHONPATH"] = source + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["RXNFLOW_PMNET_CACHE_DIR"] = str(root / "RxnFlow" / "weights" / "pmnet_cache")
    env["RXNFLOW_LOCAL_HOME"] = str(root / "RxnFlow" / "weights" / "pmnet_home")
    def forward_progress(line: str) -> None:
        if progress_callback is None:
            return
        match = re.fullmatch(r"\[PROGRESS\]\s+(\d+)/(\d+)", line)
        if match:
            progress_callback(int(match.group(1)), int(match.group(2)))

    return run_command(command, root / "RxnFlow", timeout=24 * 3600, env=env,
                       on_output=forward_progress if progress_callback else None)



def vina_status(python: str) -> dict[str, Any]:
    python_path = Path(python).expanduser()
    bin_dir = python_path.parent
    missing_commands = [name for name in ("mk_prepare_receptor.py", "mk_prepare_ligand.py")
                        if not (bin_dir / name).is_file()]
    result = {"python": str(python_path), "missing_commands": missing_commands,
              "ready": False, "missing_runtime": []}
    if not python_path.is_file():
        result["missing_runtime"] = ["Vina Python executable"]
        return result
    check = subprocess.run([str(python_path), "-c",
                            "import vina, gemmi, meeko, rdkit"],
                           text=True, capture_output=True, timeout=30, check=False)
    if check.returncode:
        result["missing_runtime"] = [check.stderr.strip()[-1000:]]
    result["ready"] = check.returncode == 0 and not missing_commands
    return result
