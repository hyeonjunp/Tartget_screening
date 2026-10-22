import os
import argparse
from pathlib import Path
import csv
import torch
from rdkit import Chem, RDLogger
from tqdm import tqdm

from rxnflow.config import Config, init_empty
from rxnflow.tasks.multi_pocket import ProxySampler
from rxnflow.utils.download import download_pretrained_weight
from rxnflow.tasks.utils.unidock import VinaReward

# Suppress RDKit logging
RDLogger.DisableLog("rdApp.*")

def get_args():
    parser = argparse.ArgumentParser(description="Predict Proxy (Vina) and UniDock scores for molecules.")
    
    # Input/Output
    parser.add_argument("--input_file", type=str, required=True, help="Input file path (SDF, TXT, or CSV)")
    parser.add_argument("--out_path", type=str, required=True, help="Output CSV file path")
    
    # Protein/Pocket
    parser.add_argument("-p", "--protein", type=str, required=True, help="Protein PDB Path")
    parser.add_argument("-l", "--ref_ligand", type=str, help="Reference Ligand Path (required if center is missing)")
    parser.add_argument("-c", "--center", nargs="+", type=float, help="Pocket Center (--center X Y Z)")
    parser.add_argument("-s", "--size", nargs="+", type=float, help="Search Box Size (--size X Y Z)", default=(22.5, 22.5, 22.5))
    
    # Model/Env
    parser.add_argument("--model_path", type=str, help="Checkpoint Path", default="qvina-unif-0-64")
    parser.add_argument("--env_dir", type=str, default="./data/envs/catalog", help="Environment Directory Path")
    parser.add_argument("--cuda", action="store_true", help="CUDA Acceleration")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for prediction (Smaller is better for UniDock)")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")

    return parser.parse_args()

def load_molecules(input_file):
    """Load SMILES and IDs from input file."""
    path = Path(input_file)
    suffix = path.suffix.lower()
    
    smiles_list = []
    ids_list = []
    
    if suffix == ".sdf":
        print(f"[INFO] Reading SDF file: {input_file}")
        suppl = Chem.SDMolSupplier(input_file)
        for i, mol in enumerate(suppl):
            if mol is None:
                continue
            try:
                smi = Chem.MolToSmiles(mol)
                name = mol.GetProp("_Name") if mol.HasProp("_Name") and mol.GetProp("_Name") else f"mol{i}"
                smiles_list.append(smi)
                ids_list.append(name)
            except Exception:
                continue
    elif suffix == ".csv":
        print(f"[INFO] Reading CSV file: {input_file}")
        with open(input_file, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "SMILES" not in reader.fieldnames:
                raise ValueError("CSV must contain a 'SMILES' column.")
            for i, row in enumerate(reader):
                smiles_list.append(row["SMILES"])
                if "ID" in row:
                    ids_list.append(row["ID"])
                else:
                    ids_list.append(f"mol{i}")
    else:
        print(f"[INFO] Reading TXT/Other file: {input_file}")
        with open(input_file, "r") as f:
            for i, line in enumerate(f):
                smi = line.strip()
                if smi:
                    smiles_list.append(smi)
                    ids_list.append(f"mol{i}")
                    
    return smiles_list, ids_list

def main():
    args = get_args()
    
    # 1. Setup Config
    config = init_empty(Config())
    config.seed = args.seed
    config.env_dir = args.env_dir
    config.algo.num_from_policy = args.batch_size
    config.algo.action_subsampling.sampling_ratio = 0.1 

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    # 2. Initialize Proxy Sampler
    print(f"[INFO] Loading model: {args.model_path}")
    model_path = download_pretrained_weight(args.model_path)
    sampler = ProxySampler(config, model_path, device)
    sampler.set_pocket(args.protein, args.center, args.ref_ligand)
    protein_key = sampler.task.protein_key
    
    # 3. Initialize UniDock (VinaReward)
    print(f"[INFO] Initializing UniDock engine...")
    vina = VinaReward(
        args.protein,
        args.center if args.center else sampler.task.center,
        args.ref_ligand,
        args.size,
        search_mode="balance",
        num_workers=4
    )
    
    # 4. Load Molecules
    smiles_list, ids_list = load_molecules(args.input_file)
    print(f"[INFO] Total molecules to evaluate: {len(smiles_list)}")
    
    # 5. Iterative Batch Prediction & Docking
    proxies = []
    unidocks = []
    pbar = tqdm(total=len(smiles_list), desc="Processing")
    
    # Temp directory for UniDock output
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        for i in range(0, len(smiles_list), args.batch_size):
            batch_smiles = smiles_list[i : i + args.batch_size]
            
            # A. Predict Proxy
            with torch.no_grad():
                scores = sampler.task.proxy.scoring_list(protein_key, batch_smiles)
                proxies.extend(scores.cpu().tolist())
            
            # B. Run UniDock
            batch_mols = [Chem.MolFromSmiles(s) for s in batch_smiles]
            valid_indices = [idx for idx, m in enumerate(batch_mols) if m is not None]
            valid_mols = [m for m in batch_mols if m is not None]
            
            batch_unidock = [0.0] * len(batch_smiles)
            if valid_mols:
                sdf_out = Path(tmp_dir) / f"batch_{i}.sdf"
                scores = vina.run_mols(valid_mols, sdf_out)
                for idx, score in zip(valid_indices, scores):
                    batch_unidock[idx] = score
            
            unidocks.extend(batch_unidock)
            pbar.update(len(batch_smiles))
    
    pbar.close()
    
    # 6. Save results
    os.makedirs(os.path.dirname(os.path.abspath(args.out_path)), exist_ok=True)
    with open(args.out_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "SMILES", "Proxy", "UniDock"])
        for mid, msmi, mprox, muni in zip(ids_list, smiles_list, proxies, unidocks):
            writer.writerow([mid, msmi, f"{mprox:.3f}", f"{muni:.3f}"])
            
    print(f"[SUCCESS] Results (Proxy & UniDock) saved to {args.out_path}")

if __name__ == "__main__":
    main()