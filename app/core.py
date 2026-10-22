"""Small, dependency-light core for target screening workflows."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
import time
from pathlib import Path
from typing import Any

AA = set("ACDEFGHIKLMNPQRSTVWY")
MUTATION = re.compile(r"^([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY])$")


def parse_target_query(query: str) -> tuple[str, str]:
    """Split a target label such as EGFR(L858R/T790M) into gene and FASTA edits."""
    query = query.strip()
    match = re.fullmatch(r"(.+?)\s*\(([^()]*)\)", query)
    if not match:
        return query, ""
    target, notation = match.groups()
    mutations = [item for item in re.split(r"[/,;\s]+", notation.strip().upper()) if item]
    if not target.strip() or not mutations or any(not MUTATION.fullmatch(item) for item in mutations):
        raise ValueError("Use a target like EGFR(L858R) or EGFR(L858R/T790M)")
    return target.strip(), "/".join(mutations)


def parse_target_drug_query(query: str) -> tuple[str, str]:
    """Split EGFR(L858R/T790M) Osimertinib into target/mutations and bound-drug filter."""
    query = query.strip()
    match = re.fullmatch(r"((?:[0-9][A-Z0-9]{3}|.+?\([^()]*\)))(?:\s+(.+))?", query, re.I)
    return (match.group(1).strip(), (match.group(2) or "").strip()) if match else (query, "")


def apply_mutations(sequence: str, notation: str) -> str:
    """Apply 1-based substitutions such as L858R/T790M to the supplied FASTA."""
    seq = "".join(sequence.split()).upper()
    if not seq or set(seq) - AA:
        raise ValueError("FASTA must contain only the 20 standard amino acids")
    edits: dict[int, str] = {}
    for item in filter(None, re.split(r"[/,;\s]+", notation.strip().upper())):
        match = MUTATION.fullmatch(item)
        if not match:
            raise ValueError(f"Invalid mutation notation: {item!r}; use L858R or L858R/T790M")
        original, pos_text, replacement = match.groups()
        pos = int(pos_text)
        if not 1 <= pos <= len(seq):
            raise ValueError(f"{item}: position {pos} is outside FASTA length {len(seq)}")
        if seq[pos - 1] != original:
            raise ValueError(f"{item}: FASTA position {pos} is {seq[pos - 1]}, not {original}")
        if pos in edits:
            raise ValueError(f"Multiple mutations target position {pos}")
        edits[pos] = replacement
    return "".join(edits.get(i + 1, aa) for i, aa in enumerate(seq))


def save_fasta_pair(directory: str | Path, identifier: str, sequence: str, notation: str) -> tuple[Path, Path]:
    mutant = apply_mutations(sequence, notation)
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    wt_path, mutant_path = out / "wt.fasta", out / "mutant.fasta"
    wt_path.write_text(f">{identifier} WT\n{sequence}\n", encoding="utf-8")
    mutant_path.write_text(f">{identifier} {notation} (sequence only; not a mutant structure)\n{mutant}\n", encoding="utf-8")
    return wt_path, mutant_path


def fold_decision(candidate: list[float | None], reference: list[float | None],
                  selection_method: str = "strict") -> dict[str, Any]:
    """Compare five-fold DTA predictions using strict-range or mean separation."""
    result: dict[str, Any] = {"status": "prediction_failed", "delta_robust": None,
                              "candidate_min": None, "candidate_max": None,
                              "reference_min": None, "reference_max": None}
    if len(candidate) != 5 or len(reference) != 5 or any(v is None for v in candidate + reference):
        result["reason"] = "All five candidate and reference fold predictions are required."
        return result
    try:
        cvals, rvals = [float(v) for v in candidate], [float(v) for v in reference]
    except (ValueError, TypeError):
        result["reason"] = "Fold prediction contains a non-numeric value."
        return result
    if not all(map(lambda v: v == v and abs(v) != float("inf"), cvals + rvals)):
        result["reason"] = "Fold prediction contains a non-finite value."
        return result
    if selection_method not in {"strict", "mean"}:
        raise ValueError("selection_method must be 'strict' or 'mean'")
    cmin, cmax, rmin, rmax = min(cvals), max(cvals), min(rvals), max(rvals)
    cmean, rmean = sum(cvals) / 5, sum(rvals) / 5
    if selection_method == "mean":
        delta = cmean - rmean
        passed = delta > 0
        reason = "Candidate mean exceeds reference mean." if passed else "Candidate mean is not greater than reference mean."
    else:
        delta = cmin - rmax
        passed = delta > 0
        reason = "Candidate minimum exceeds reference maximum." if passed else "Candidate minimum is not strictly greater than reference maximum."
    result.update(status="passed" if passed else "rejected", candidate_min=cmin,
                   candidate_max=cmax, reference_min=rmin, reference_max=rmax,
                   candidate_mean=cmean, reference_mean=rmean,
                   delta_robust=delta, selection_method=selection_method, reason=reason)
    return result


def _request_json(url: str, payload: dict[str, Any] | None = None) -> Any:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", "User-Agent": "target-screening/0.1"})
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                raw = response.read()
            if not raw.strip():
                raise ValueError("RCSB returned an empty response")
            return json.loads(raw)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:300]
            last_error = RuntimeError(f"HTTP {exc.code}: {body or exc.reason}")
            if exc.code < 500:
                break
        except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
        if attempt < 2:
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"RCSB API request failed ({url}): {last_error}") from last_error


def search_pdb(query: str, limit: int = 100) -> list[dict[str, Any]]:
    """Search RCSB by gene and require explicit title/description annotation for each requested mutation."""
    target_query, drug_query = parse_target_drug_query(query)
    direct_pdb_id = target_query.upper() if re.fullmatch(r"[0-9][A-Z0-9]{3}", target_query, re.I) else None
    gene_query, mutation_query = ("", "") if direct_pdb_id else parse_target_query(target_query)
    if not direct_pdb_id and not gene_query:
        raise ValueError("Enter a gene, protein name, or four-character PDB ID")
    base = {"return_type": "entry", "request_options": {"paginate": {"start": 0, "rows": max(1, min(limit, 100))}}}
    gene_node = {"type": "terminal", "service": "text", "parameters": {
        "attribute": "rcsb_entity_source_organism.rcsb_gene_name.value", "operator": "exact_match", "value": gene_query}}
    description_node = {"type": "terminal", "service": "text", "parameters": {
        "attribute": "rcsb_polymer_entity.pdbx_description", "operator": "contains_words", "value": gene_query}}
    mutation_nodes = [{"type": "terminal", "service": "text", "parameters": {
        "attribute": "struct.title", "operator": "contains_words", "value": mutation}}
        for mutation in mutation_query.split("/") if mutation_query]
    if direct_pdb_id:
        ids = [direct_pdb_id]
    elif mutation_nodes:
        searches = [
            {"type": "group", "logical_operator": "and", "nodes": [gene_node, *mutation_nodes]},
            {"type": "group", "logical_operator": "and", "nodes": [description_node, *mutation_nodes]},
        ]
    else:
        searches = [gene_node, description_node]
    if not direct_pdb_id:
        search = {"result_set": []}
        for search_query in searches:
            search = _request_json("https://search.rcsb.org/rcsbsearch/v2/query", {**base, "query": search_query})
            if search.get("result_set"):
                break
        ids = [x["identifier"] for x in search.get("result_set", [])]
    if not ids:
        return []
    details = _request_json("https://data.rcsb.org/graphql", {"query": "query($ids:[String!]!){entries(entry_ids:$ids){rcsb_id struct{title} exptl{method} rcsb_entry_info{resolution_combined} polymer_entities{rcsb_entity_source_organism{ncbi_scientific_name rcsb_gene_name{value}} entity_poly{pdbx_seq_one_letter_code_can rcsb_mutation_count rcsb_sample_sequence_length} rcsb_polymer_entity{pdbx_description} rcsb_polymer_entity_container_identifiers{asym_ids uniprot_ids}} nonpolymer_entities{pdbx_entity_nonpoly{name comp_id} rcsb_nonpolymer_entity_container_identifiers{asym_ids}}}}", "variables": {"ids": ids}})
    entries = {e["rcsb_id"]: e for e in details.get("data", {}).get("entries", [])}
    comp_ids = sorted({n.get("pdbx_entity_nonpoly", {}).get("comp_id")
        for e in entries.values() for n in (e.get("nonpolymer_entities") or [])
        if n.get("pdbx_entity_nonpoly", {}).get("comp_id")})
    descriptors = {}
    if comp_ids:
        chem = _request_json("https://data.rcsb.org/graphql", {"query": "query($ids:[String!]!){chem_comps(comp_ids:$ids){chem_comp{id name formula formula_weight} rcsb_chem_comp_descriptor{SMILES SMILES_stereo}}}", "variables": {"ids": comp_ids}})
        descriptors = {c["chem_comp"]["id"]: c for c in chem.get("data", {}).get("chem_comps", [])}
    results = []
    for pdb_id in ids:
        entry = entries.get(pdb_id, {})
        chains = []
        for entity in entry.get("polymer_entities") or []:
            source = entity.get("rcsb_entity_source_organism") or []
            source = source[0] if isinstance(source, list) and source else source
            source = source if isinstance(source, dict) else {}
            seq = (entity.get("entity_poly") or {}).get("pdbx_seq_one_letter_code_can", "")
            desc = (entity.get("rcsb_polymer_entity") or {}).get("pdbx_description", "")
            asym_ids = (entity.get("rcsb_polymer_entity_container_identifiers") or {}).get("asym_ids") or []
            poly = entity.get("entity_poly") or {}
            container = entity.get("rcsb_polymer_entity_container_identifiers") or {}
            chains.extend({"chain": chain, "protein": desc, "sequence": seq,
                           "organism": source.get("ncbi_scientific_name"),
                           "gene": ", ".join(x.get("value", "") for x in (source.get("rcsb_gene_name") or [])),
                           "uniprot": container.get("uniprot_ids") or [],
                           "mutation_count": poly.get("rcsb_mutation_count"),
                           "sequence_length": poly.get("rcsb_sample_sequence_length")}
                          for chain in asym_ids)
        ligands = []
        for nonpoly in entry.get("nonpolymer_entities") or []:
            component = nonpoly.get("pdbx_entity_nonpoly") or {}
            comp_id = component.get("comp_id")
            descriptor = descriptors.get(comp_id, {})
            chem_comp = descriptor.get("chem_comp") or {}
            smiles = descriptor.get("rcsb_chem_comp_descriptor") or {}
            ligand_chains = (nonpoly.get("rcsb_nonpolymer_entity_container_identifiers") or {}).get("asym_ids") or []
            ligands.append({"name": chem_comp.get("name") or component.get("name"), "ligand_id": comp_id,
                            "chains": ligand_chains, "smiles": smiles.get("SMILES_stereo") or smiles.get("SMILES"),
                            "formula": chem_comp.get("formula"), "formula_weight": chem_comp.get("formula_weight"),
                            "is_organic_drug_like": bool(comp_id and "C" in (chem_comp.get("formula") or "") and (chem_comp.get("formula_weight") or 0) >= 100),
                            "structure_available": bool(comp_id)})
        results.append({"pdb_id": pdb_id, "title": (entry.get("struct") or {}).get("title"),
                        "method": [x.get("method") for x in entry.get("exptl") or []],
                        "resolution": (entry.get("rcsb_entry_info") or {}).get("resolution_combined"),
                        "chains": chains, "ligands": ligands,
                        "mutation_annotation": "RCSB polymer mutation count is shown per chain; detailed mutation positions are not inferred from the search term."})
    if mutation_query:
        requested = mutation_query.split("/")
        results = [result for result in results if all(
            re.search(r"(?<![A-Z0-9])" + re.escape(mutation) + r"(?![A-Z0-9])",
                      " ".join([result.get("title") or ""] + [c.get("protein") or "" for c in result.get("chains", [])]).upper())
            for mutation in requested)]
        for result in results:
            result["mutation_annotation"] = f"RCSB title/chain description explicitly annotates {mutation_query}."
    if drug_query:
        term = re.sub(r"[^a-z0-9]", "", drug_query.casefold())
        results = [result for result in results if term in re.sub(
            r"[^a-z0-9]", "", " ".join([result.get("title") or ""] +
                [str(value) for ligand in result.get("ligands", [])
                 for value in (ligand.get("name"), ligand.get("ligand_id")) if value]).casefold())]
    return results


def fetch_uniprot_fasta(accession: str) -> tuple[str, str]:
    accession = accession.strip()
    if not re.fullmatch(r"[A-Z0-9]{6,10}(?:-\d+)?", accession, re.I):
        raise ValueError("Enter a UniProt accession, for example P00533")
    url = "https://rest.uniprot.org/uniprotkb/" + urllib.parse.quote(accession) + ".fasta"
    with urllib.request.urlopen(url, timeout=30) as response:
        text = response.read().decode()
    header, *parts = text.strip().splitlines()
    if not header.startswith(">") or not parts:
        raise ValueError(f"No UniProt FASTA found for {accession}")
    return header[1:], "".join(parts)


def sequence_identity(left: str, right: str) -> dict[str, Any]:
    """Report exact identity over a global alignment using a small Needleman-Wunsch implementation."""
    a, b = left.upper(), right.upper()
    if not a or not b:
        return {"identity": 0.0, "aligned": 0, "differences": []}
    # Match=2, mismatch=-1, gap=-2; ties prefer diagonal, then deletion, then insertion.
    rows, cols = len(a) + 1, len(b) + 1
    score = [[0] * cols for _ in range(rows)]
    trace = [[0] * cols for _ in range(rows)]
    for i in range(1, rows):
        score[i][0], trace[i][0] = -2 * i, 1
    for j in range(1, cols):
        score[0][j], trace[0][j] = -2 * j, 2
    for i in range(1, rows):
        for j in range(1, cols):
            opts = (score[i-1][j-1] + (2 if a[i-1] == b[j-1] else -1), score[i-1][j] - 2, score[i][j-1] - 2)
            best = max(opts)
            score[i][j], trace[i][j] = best, opts.index(best)
    i, j, matches, aligned, diffs = len(a), len(b), 0, 0, []
    while i or j:
        direction = trace[i][j]
        if i and j and direction == 0:
            aligned += 1
            if a[i-1] == b[j-1]:
                matches += 1
            else:
                diffs.append({"left_position": i, "left": a[i-1], "right_position": j, "right": b[j-1]})
            i -= 1; j -= 1
        elif i and (not j or direction == 1):
            aligned += 1
            diffs.append({"left_position": i, "left": a[i-1], "right_position": None, "right": "-"})
            i -= 1
        else:
            aligned += 1
            diffs.append({"left_position": None, "left": "-", "right_position": j, "right": b[j-1]})
            j -= 1
    return {"identity": matches / aligned if aligned else 0.0, "aligned": aligned, "differences": list(reversed(diffs))}


def environment_report() -> dict[str, Any]:
    import importlib.util
    import shutil
    modules = ["torch", "rdkit", "Bio", "streamlit", "vina", "meeko", "gemmi"]
    return {"python_modules": {name: importlib.util.find_spec(name) is not None for name in modules},
            "executables": {name: shutil.which(name) for name in ["vina", "obabel", "mk_prepare_receptor.py", "mk_prepare_ligand.py"]},
            "rxnflow_source": str(Path(__file__).resolve().parents[1] / "RxnFlow"),
            "mode": "real only when each selected stage has its required model and tools; no mock values are emitted"}
