import os
import tempfile
import time
from argparse import ArgumentParser
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm

from rxnflow.config import Config, init_empty
from rxnflow.tasks.unidock_vina import VinaSampler

# Suppress RDKit logging
RDLogger.DisableLog("rdApp.*")

def parse_args():
    parser = ArgumentParser("RxnFlow", description="Inference Sampling with RxnFlow")
    run_cfg = parser.add_argument_group("Operation Config")
    run_cfg.add_argument("-m", "--model_path", type=str, required=True, help="Model Checkpoitn Path")
    run_cfg.add_argument("-n", "--num_samples", type=int, required=True, help="Number of Samples")
    run_cfg.add_argument(
        "-o",
        "--out_path",
        type=str,
        required=True,
        help="Output Path (.csv | .smi). If csv, the docking score is calculated.",
    )
    run_cfg.add_argument("--env_dir", type=str, help="Environment Directory Path")
    run_cfg.add_argument(
        "--subsampling_ratio",
        type=float,
        default=0.1,
        help="Action Subsampling Ratio. Memory-efficiency trade-off (Higher ratio increase samplinge efficiency; default: 0.1)",
    )
    run_cfg.add_argument("--cuda", action="store_true", help="CUDA Acceleration")
    run_cfg.add_argument("--batch_size", type=int, default=10, help="Batch size for each UniDock sampling step")

    opt_cfg = parser.add_argument_group("Protein Config (overwrite training setting)")
    opt_cfg.add_argument("-p", "--protein", type=str, help="Protein PDB Path")
    opt_cfg.add_argument("-c", "--center", nargs="+", type=float, help="Pocket Center (--center X Y Z)")
    opt_cfg.add_argument("-l", "--ref_ligand", type=str, help="Reference Ligand Path (required if center is missing)")
    opt_cfg.add_argument(
        "-s", "--size", nargs="+", type=float, help="Search Box Size (--size X Y Z)", default=(22.5, 22.5, 22.5)
    )
    return parser.parse_args()


def get_inchi_key(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
            
        # Check if Murcko Scaffold extraction is possible
        scaffold = MurckoScaffold.GetMurckoScaffoldForMol(mol)
        if scaffold is None or scaffold.GetNumAtoms() == 0:
            return None
            
        return Chem.MolToInchiKey(mol)
    except Exception:
        return None


def run(args):
    ckpt_path = Path(args.model_path)

    # change config from training
    config = init_empty(Config())

    # low subsampling ratio: force exploration
    # high subsampling ratio: more exploitation
    config.algo.action_subsampling.sampling_ratio = args.subsampling_ratio

    if args.env_dir is not None:
        config.env_dir = args.env_dir
    if args.protein is not None:
        config.task.docking.protein_path = args.protein
    if args.ref_ligand_path is not None:
        config.task.docking.ref_ligand_path = args.ref_ligand
    if args.center is not None:
        config.task.docking.center = args.center
    if args.size is not None:
        config.task.docking.size = args.size

    device = "cuda" if args.cuda else "cpu"
    save_reward = os.path.splitext(args.out_path)[1] == ".csv"

    success_count = 0
    unique_inchi_keys = set()
    final_res = []

    print(f"[INFO] Generating {args.num_samples} valid and unique molecules using UniDock.")
    pbar = tqdm(total=args.num_samples, desc="UniDock Sampling")

    # NOTE: Run
    with tempfile.TemporaryDirectory() as tempdir:
        config.log_dir = tempdir
        sampler = VinaSampler(config, ckpt_path, device)
        
        tick_st = time.time()
        while success_count < args.num_samples:
            current_batch_size = min(args.batch_size, args.num_samples - success_count)
            # UniDock batch sampling
            batch_res = sampler.sample(current_batch_size, calc_reward=save_reward)
            
            for sample in batch_res:
                smiles = sample["smiles"]
                if not smiles:
                    continue
                
                inchi_key = get_inchi_key(smiles)
                # Validity and Uniqueness Check
                if inchi_key and inchi_key not in unique_inchi_keys:
                    unique_inchi_keys.add(inchi_key)
                    final_res.append(sample)
                    success_count += 1
                    pbar.update(1)
                    
                    if success_count >= args.num_samples:
                        break
        tick_end = time.time()
        
    pbar.close()
    print(f"Sampling: {tick_end - tick_st:.3f} sec")
    print(f"Generated Unique Molecules: {len(final_res)}")

    if save_reward:
        with open(args.out_path, "w") as w:
            w.write("ID,SMILES,Vina\n")
            for idx, sample in enumerate(final_res):
                smiles = sample["smiles"]
                vina = sample["info"]["reward"][0] * -1
                w.write(f"sample{idx},{smiles},{vina:.3f}\n")
    else:
        with open(args.out_path, "w") as w:
            for idx, sample in enumerate(final_res):
                w.write(f"{sample['smiles']}\tsample{idx}\n")


if __name__ == "__main__":
    args = parse_args()
    run(args)
