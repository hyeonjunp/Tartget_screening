import csv
import os
import time
from argparse import ArgumentParser
from pathlib import Path

import torch
from rdkit import Chem, RDLogger
from tqdm import tqdm

from rxnflow.config import Config, init_empty
from rxnflow.tasks.multi_pocket import ProxySampler
from rxnflow.utils.download import download_pretrained_weight

# Suppress RDKit logging
RDLogger.DisableLog("rdApp.*")


def parse_args():
    parser = ArgumentParser("RxnFlow", description="(QED - Docking Proxy) Custom Zero Shot Sampling with RxnFlow")
    opt_cfg = parser.add_argument_group("Protein Config")
    opt_cfg.add_argument("-p", "--protein", type=str, required=True, help="Protein PDB Path")
    opt_cfg.add_argument("-l", "--ref_ligand", type=str, help="Reference Ligand Path (required if center is missing)")
    opt_cfg.add_argument("-c", "--center", nargs="+", type=float, help="Pocket Center (--center X Y Z)")

    run_cfg = parser.add_argument_group("Operation Config")
    run_cfg.add_argument("--model_path", type=str, help="Checkpoint Path", default="qvina-unif-0-64")
    run_cfg.add_argument("-n", "--num_samples", type=int, default=100, help="Total number of valid/unique samples")
    run_cfg.add_argument("-o", "--out_path", type=str, required=True, help="Output CSV path")
    run_cfg.add_argument("--env_dir", type=str, default="./data/envs/catalog", help="Environment Directory Path")
    run_cfg.add_argument(
        "--subsampling_ratio",
        type=float,
        default=0.1,
        help="Action Subsampling Ratio. Memory-variance trade-off; default: 0.1",
    )
    run_cfg.add_argument(
        "--temperature",
        type=str,
        default="uniform-16-64",
        help="temperature setting (e.g., uniform-16-64, uniform-32-64, ...)",
    )
    run_cfg.add_argument("--cuda", action="store_true", help="CUDA Acceleration")
    run_cfg.add_argument("--seed", type=int, help="seed", default=1)
    run_cfg.add_argument("--batch_size", type=int, default=100, help="Batch size for each sampling step")
    run_cfg.add_argument("--max_attempts", type=int, default=10000, help="Maximum sampled molecules before stopping")
    return parser.parse_args()


def get_inchi_key(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToInchiKey(mol) if mol is not None else None
    except Exception:
        return None


def run(args):
    from _utils import parse_temperature

    # change config from training
    config = init_empty(Config())
    config.seed = args.seed
    config.env_dir = args.env_dir
    config.algo.num_from_policy = args.batch_size
    config.algo.action_subsampling.sampling_ratio = args.subsampling_ratio

    device = "cuda" if args.cuda else "cpu"
    # User requested single CSV output
    save_reward = True

    # create sampler
    # Keep PharmacoNet downloads inside this repository, not in Path.home().
    from pmnet_appl.base.proxy import BaseProxy
    from pmnet_appl.tacogfn_reward.proxy import TacoGFN_Proxy

    local_home = Path(os.environ.get("RXNFLOW_LOCAL_HOME", Path(__file__).resolve().parents[1] / "weights" / "pmnet_home")).resolve()
    local_home.mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = str(local_home)
    cache_root = Path(os.environ.get("RXNFLOW_PMNET_CACHE_DIR", Path(__file__).resolve().parents[1] / "weights" / "pmnet_cache"))
    BaseProxy.root_dir = cache_root / "base_proxy"
    TacoGFN_Proxy.root_dir = cache_root / "tacogfn_proxy"

    model_path = download_pretrained_weight(args.model_path)
    sampler = ProxySampler(config, model_path, device)
    sample_dist, dist_params = parse_temperature(args.temperature)
    sampler.update_temperature(sample_dist, dist_params)

    # set binding site
    sampler.set_pocket(args.protein, args.center, args.ref_ligand)

    success_count = 0
    attempts = 0
    unique_inchi_keys = set()
    final_res = []

    print(f"[INFO] Generating {args.num_samples} valid and unique molecules.")
    pbar = tqdm(total=args.num_samples, desc="Sampling")

    tick = time.time()
    while success_count < args.num_samples and attempts < args.max_attempts:
        # Increase batch size slightly to account for invalid/duplicate samples
        current_batch_size = min(args.batch_size, (args.num_samples - success_count) * 2)
        current_batch_size = max(min(current_batch_size, args.max_attempts - attempts), 1)
        
        batch_res = sampler.sample(current_batch_size, calc_reward=save_reward)
        attempts += len(batch_res)
        
        for sample in batch_res:
            smiles = sample["smiles"]
            if not smiles:
                continue
                
            inchi_key = get_inchi_key(smiles)
            # Both validity (via RDKit) and uniqueness check
            if inchi_key and inchi_key not in unique_inchi_keys:
                unique_inchi_keys.add(inchi_key)
                final_res.append(sample)
                success_count += 1
                pbar.update(1)
                
                if success_count >= args.num_samples:
                    break

        print(f"[PROGRESS] {success_count}/{args.num_samples}", flush=True)

    pbar.close()
    elapsed = time.time() - tick
    print(f"Sampling finished: {elapsed:.3f} sec")
    print(f"Total Unique/Valid Molecules: {len(final_res)}")
    # Save to CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.out_path)), exist_ok=True)
    with open(args.out_path, "w", newline="") as w:
        writer = csv.writer(w)
        writer.writerow(["ID", "SMILES", "QED", "Proxy"])
        for idx, sample in enumerate(final_res):
            writer.writerow([f"sample{idx}", sample["smiles"], float(sample["info"]["reward_qed"]),
                             float(sample["info"]["reward_vina"])])
            
    print(f"[SUCCESS] Saved results to {args.out_path}")
    if success_count < args.num_samples:
        raise RuntimeError(f"Only generated {success_count}/{args.num_samples} unique valid molecules in {attempts} attempts")



if __name__ == "__main__":
    args = parse_args()
    run(args)
