from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import gemmi



def receptor(cif: Path, chain_id: str, ligand_id: str, ligand_chain: str | None, out_dir: Path) -> dict:
    structure = gemmi.read_structure(str(cif))
    if not structure:
        raise ValueError(f"No structure models in {cif}")
    model = structure[0]
    structure.setup_entities()
    center = None
    if ligand_id:
        for chain in model:
            for residue in chain:
                if residue.name != ligand_id:
                    continue
                if ligand_chain and ligand_chain not in (chain.name, residue.subchain):
                    continue
                coords = [atom.pos for atom in residue if not atom.element.is_hydrogen]
                if coords:
                    center = [sum(getattr(pos, axis) for pos in coords) / len(coords) for axis in ("x", "y", "z")]
                    break
            if center:
                break
        if not center:
            raise ValueError(f"Ligand coordinates for {ligand_id} were not found in the selected PDB")
    chain_names = [chain.name for chain in model]
    selected = [chain.name for chain in model if chain.name == chain_id or any(
        residue.subchain == chain_id and residue.entity_type == gemmi.EntityType.Polymer for residue in chain)]
    if not selected:
        raise ValueError(f"Chain {chain_id} not found; available chains: {chain_names}")
    selected_chain_name = selected[0]
    for name in chain_names:
        if name != selected_chain_name:
            model.remove_chain(name)
    model.remove_ligands_and_waters()
    structure.remove_alternative_conformations()
    out_dir.mkdir(parents=True, exist_ok=True)
    pdb = out_dir / "receptor.pdb"
    prefix = out_dir / "receptor"
    structure.write_pdb(str(pdb))
    cmd = [sys.executable, str(Path(sys.executable).parent / "mk_prepare_receptor.py"), "--read_pdb", str(pdb), "-o", str(prefix), "-p", "--default_altloc", "A", "--allow_bad_res"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    pdbqt = out_dir / "receptor.pdbqt"
    if result.returncode or not pdbqt.is_file():
        raise RuntimeError(f"Meeko receptor preparation failed: {result.stderr[-4000:]}")
    return {"pdb": str(pdb), "pdbqt": str(pdbqt), "center": center, "chain": chain_id,
            "ligand_id": ligand_id,
            "meeko_stdout": result.stdout, "meeko_stderr": result.stderr}


def smiles_to_sdf(smi: str, out_path: str | Path, seed: int = 1) -> bool:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return False
    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    cid = AllChem.EmbedMolecule(mol_h, params)
    if cid < 0:
        params.useRandomCoords = True
        params.enforceChirality = False
        cid = AllChem.EmbedMolecule(mol_h, params)
    if cid < 0:
        return False
    AllChem.MMFFOptimizeMolecule(mol_h)
    expected = {idx: label for idx, label in Chem.FindMolChiralCenters(
        mol, includeUnassigned=True, includeCIP=True) if label in {"R", "S"}}
    from_3d = Chem.Mol(mol_h)
    Chem.AssignStereochemistryFrom3D(from_3d, replaceExistingTags=True)
    actual = dict(Chem.FindMolChiralCenters(from_3d, includeUnassigned=True, includeCIP=True))
    if any(actual.get(idx) != label for idx, label in expected.items()):
        return False
    with Chem.SDWriter(str(out_path)) as writer:
        writer.write(Chem.RemoveHs(mol_h))
    return True


def dock(args: argparse.Namespace) -> dict:
    from rdkit import Chem
    from vina import Vina

    out = Path(args.out_dir)
    poses = out / "docking_poses"
    ligands = out / "ligands"
    poses.mkdir(parents=True, exist_ok=True)
    ligands.mkdir(parents=True, exist_ok=True)
    v = Vina(sf_name="vina", cpu=1, seed=args.seed, verbosity=0)
    v.set_receptor(args.receptor)
    v.compute_vina_maps(center=args.center, box_size=args.box_size)
    results = []
    with Path(args.candidates).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            candidate_id = row.get("ID") or row.get("candidate_id") or f"candidate_{len(results)+1}"
            candidate_id = re.sub(r"[^A-Za-z0-9_.-]", "_", candidate_id)
            drug_name = row.get("drug_name") or row.get("name") or candidate_id
            smiles = row.get("SMILES") or row.get("smiles")
            if not smiles:
                results.append({"ID": candidate_id, "drug_name": drug_name, "SMILES": "", "Vina": "", "status": "missing_smiles"})
                continue
            try:
                sdf = ligands / f"{candidate_id}.sdf"
                pdbqt = ligands / f"{candidate_id}.pdbqt"
                if not smiles_to_sdf(smiles, sdf, args.seed):
                    raise ValueError("3D embedding failed")
                molecule = next((mol for mol in Chem.SDMolSupplier(str(sdf), removeHs=False) if mol is not None), None)
                if molecule is None:
                    raise ValueError("Generated SDF could not be read")
                meeko_sdf = ligands / f"{candidate_id}_meeko.sdf"
                with Chem.SDWriter(str(meeko_sdf)) as writer:
                    writer.write(Chem.AddHs(molecule, addCoords=True))

                prep = subprocess.run([sys.executable, str(Path(sys.executable).parent / "mk_prepare_ligand.py"), "-i", str(meeko_sdf), "-o", str(pdbqt)],
                                      capture_output=True, text=True)
                if prep.returncode or not pdbqt.is_file():
                    raise RuntimeError(prep.stderr[-1500:] or "Meeko ligand preparation failed")
                v.set_ligand_from_file(str(pdbqt))
                v.dock(exhaustiveness=args.exhaustiveness, n_poses=args.n_poses)
                pose_path = poses / f"{candidate_id}_docked.pdbqt"
                v.write_poses(str(pose_path), n_poses=args.n_poses, overwrite=True)
                with pose_path.open(encoding="utf-8", errors="replace") as pose_file:
                    score_line = next(line for line in pose_file if "VINA RESULT:" in line)
                score = float(score_line.split()[3])
                results.append({"ID": candidate_id, "drug_name": drug_name, "SMILES": smiles, "Vina": score,
                                "status": "docked", "pose": str(pose_path.relative_to(out))})
            except Exception as exc:
                results.append({"ID": candidate_id, "drug_name": drug_name, "SMILES": smiles, "Vina": "", "status": f"failed: {exc}"})
    summary = out / "docking_scores.csv"
    with summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ID", "drug_name", "SMILES", "Vina", "status", "pose"])
        writer.writeheader()
        writer.writerows(results)
    return {"summary": str(summary), "poses": str(poses), "rows": len(results),
            "docked": sum(row["status"] == "docked" for row in results)}


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--cif", required=True)
    prep.add_argument("--chain", required=True)
    prep.add_argument("--ligand-id", default="")
    prep.add_argument("--ligand-chain", default="")
    prep.add_argument("--out-dir", required=True)
    docking = sub.add_parser("dock")
    docking.add_argument("--receptor", required=True)
    docking.add_argument("--candidates", required=True)
    docking.add_argument("--out-dir", required=True)
    docking.add_argument("--center", nargs=3, type=float, required=True)
    docking.add_argument("--box-size", nargs=3, type=float, default=[20, 20, 20])
    docking.add_argument("--exhaustiveness", type=int, default=8)
    docking.add_argument("--n-poses", type=int, default=10)
    docking.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if args.mode == "prepare":
        result = receptor(Path(args.cif), args.chain, args.ligand_id, args.ligand_chain or None, Path(args.out_dir))
        print(json.dumps(result))
    else:
        print(json.dumps(dock(args)))


if __name__ == "__main__":
    main()
