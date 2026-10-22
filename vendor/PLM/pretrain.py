import os
import gc
import yaml
import argparse
import torch
from torch.utils.data import DataLoader
from Bio import SeqIO
from itertools import cycle
from tqdm import tqdm

from model import ProteinMLM
from Trainer import ProteinMLMTrainer
from dataprocessing import (
    ProteinDataset,
    ProteinDatasetVal,
    vocab,
    load_distance_dict_from_matrix_csv,
)

# --------------------------------------------------
# Utils
# --------------------------------------------------
def load_fasta_sequences(path):
    return [str(r.seq).upper() for r in SeqIO.parse(path, "fasta")]

def leq(seqs, n):
    return [s for s in seqs if len(s) <= n]

def between(seqs, lo, hi):
    return [s for s in seqs if lo < len(s) <= hi]

def concat_batches(*batches):
    return {k: torch.cat([b[k] for b in batches], dim=0) for k in batches[0]}

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    run_dir = os.path.join(cfg["save_dir"], cfg["run_name"])
    os.makedirs(run_dir, exist_ok=True)

    # =========================
    # Load sequences
    # =========================
    wt_seqs = load_fasta_sequences(cfg["wt_fasta"])
    mut_seqs = load_fasta_sequences(cfg["mut_fasta"])
    val_wt_seqs = load_fasta_sequences(cfg["val_wt_fasta"])
    val_mut_seqs = load_fasta_sequences(cfg["val_mut_fasta"])

    wt_512 = leq(wt_seqs, 512)
    wt_1024 = between(wt_seqs, 512, 1024)
    mut_512 = leq(mut_seqs, 512)
    mut_1024 = between(mut_seqs, 512, 1024)

    val_wt_512 = leq(val_wt_seqs, 512)
    val_wt_1024 = between(val_wt_seqs, 512, 1024)
    val_mut_512 = leq(val_mut_seqs, 512)
    val_mut_1024 = between(val_mut_seqs, 512, 1024)

    distance_dict = load_distance_dict_from_matrix_csv(cfg["distance_csv"])

    print(f"WT ≤512: {len(wt_512)}, WT 512–1024: {len(wt_1024)}")
    print(f"Mut ≤512: {len(mut_512)}, Mut 512–1024: {len(mut_1024)}")

    # =========================
    # Dataloaders
    # =========================
    bs = cfg["batch_size"]
    Phase1WT = bs // 4
    Phase1Mut = 3*bs // 4
    Phase2WT = bs // 8
    Phase2Mut = 3*bs // 8
    # Phase 1
    P1_wt512_loader = DataLoader(
        ProteinDataset(wt_512, vocab, distance_dict, max_len=512),
        batch_size=Phase1WT, shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=True
    )
    P1_mut512_loader = DataLoader(
        ProteinDataset(mut_512, vocab, distance_dict, max_len=512),
        batch_size=Phase1Mut, shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=True
    )

    P1_val_wt_loader_512 = DataLoader(
        ProteinDatasetVal(val_wt_512, vocab, distance_dict, 512, seed=1024),
        batch_size=cfg["valid_batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=True
    )
    P1_val_mut_loader_512 = DataLoader(
        ProteinDatasetVal(val_mut_512, vocab, distance_dict, 512, seed=1024),
        batch_size=cfg["valid_batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=True
    )

    # Phase 2
    P2_wt512_loader = DataLoader(
        ProteinDataset(wt_512, vocab, distance_dict, 1024),
        batch_size=Phase2WT, shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=True
    )
    P2_wt1024_loader = DataLoader(
        ProteinDataset(wt_1024, vocab, distance_dict, 1024),
        batch_size=Phase2WT, shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=True
    )
    P2_mut512_loader = DataLoader(
        ProteinDataset(mut_512, vocab, distance_dict, 1024),
        batch_size=Phase2Mut, shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=True
    )
    P2_mut1024_loader = DataLoader(
        ProteinDataset(mut_1024, vocab, distance_dict, 1024),
        batch_size=Phase2Mut, shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=True
    )

    P2_val_wt_loader_512 = DataLoader(
        ProteinDatasetVal(val_wt_512, vocab, distance_dict, 1024, seed=1024),
        batch_size=cfg["valid_batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=True
    )
    P2_val_wt_loader_1024 = DataLoader(
        ProteinDatasetVal(val_wt_1024, vocab, distance_dict, 1024, seed=1024),
        batch_size=cfg["valid_batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=True
    )
    P2_val_mut_loader_512 = DataLoader(
        ProteinDatasetVal(val_mut_512, vocab, distance_dict, 1024, seed=1024),
        batch_size=cfg["valid_batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=True
    )
    P2_val_mut_loader_1024 = DataLoader(
        ProteinDatasetVal(val_mut_1024, vocab, distance_dict, 1024, seed=1024),
        batch_size=cfg["valid_batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=True
    )

    # =========================
    # Model / Trainer
    # =========================
    mcfg = cfg["model"]
    model = ProteinMLM(
        vocab_size=len(vocab),
        embed_dim=mcfg["embed_dim"],
        num_layers=mcfg["num_layers"],
        num_heads=mcfg["num_heads"],
        ffn_embed_dim=mcfg["ffn_embed_dim"],
        padding_idx=vocab["[PAD]"],
        dropout=mcfg["dropout"],
    )

    trainer = ProteinMLMTrainer(
        model=model,
        vocab=vocab,
        lr=cfg["lr"],
        max_steps=cfg["training_steps"],
        eval_interval=cfg["eval_interval"],
    )

    if cfg.get("resume", False):
        trainer.load_latest_checkpoint(run_dir)

    # =========================
    # Phase schedule
    # =========================
    phase1_end = cfg["phase1_steps"]
    phase2_steps = cfg["phase2_steps"]
    training_steps = cfg["training_steps"]

    # =========================
    # Phase 1 (resume-safe)
    # =========================
    if trainer.global_step < phase1_end:
        print(" Phase 1: WT (512) : Mut (512) = 1 : 3")

        wt_it = cycle(P1_wt512_loader)
        mut_it = cycle(P1_mut512_loader)

        pbar = tqdm(
            total=phase1_end,
            initial=trainer.global_step,
            desc="Training (Phase 1 | WT512 : Mut512 = 1:3)",
            dynamic_ncols=True,
        )

        while trainer.global_step < phase1_end:
            batch = concat_batches(next(wt_it), next(mut_it))
            loss = trainer.train_step(batch)

            pbar.update(1)
            pbar.set_postfix(loss=f"{loss:.4f}")

            if trainer.global_step % cfg["eval_interval"] == 0:
                l, p = trainer.validate(P1_val_wt_loader_512, "phase1_Val512_wt")
                trainer.log_to_csv(run_dir, trainer.global_step, "phase1_Val512_wt", l, p)
                
                l, p = trainer.validate(P1_val_mut_loader_512, "phase1_Val512_mut")
                trainer.log_to_csv(run_dir, trainer.global_step, "phase1_Val512_mut", l, p)
                
                trainer.save_checkpoint(run_dir)

        pbar.close()

    print(f" Switch to Phase 2 at step {trainer.global_step}")

    del P1_wt512_loader, P1_mut512_loader
    del P1_val_wt_loader_512, P1_val_mut_loader_512
    gc.collect()
    torch.cuda.empty_cache()

    # =========================
    # Phase 2 (resume-safe)
    # =========================
    wt512_it = cycle(P2_wt512_loader)
    wt1024_it = cycle(P2_wt1024_loader)
    mut512_it = cycle(P2_mut512_loader)
    mut1024_it = cycle(P2_mut1024_loader)

    phase2_done = max(0, trainer.global_step - phase1_end)

    pbar = tqdm(
        total=phase2_steps,
        initial=phase2_done,
        desc="Training (Phase 2 | WT512 : Mut512 : WT1024 : Mut1024 = 2:2:6:6)",
        dynamic_ncols=True,
    )

    while trainer.global_step < training_steps:
        batch = concat_batches(
            next(wt512_it),
            next(wt1024_it),
            next(mut512_it),
            next(mut1024_it),
        )

        loss = trainer.train_step(batch)

        pbar.update(1)
        pbar.set_postfix(loss=f"{loss:.4f}")

        if trainer.global_step % cfg["eval_interval"] == 0:
            l, p = trainer.validate(P2_val_wt_loader_512, "phase2_Val512_wt")
            trainer.log_to_csv(run_dir, trainer.global_step, "phase2_Val512_wt", l, p)
            
            l, p = trainer.validate(P2_val_mut_loader_512, "phase2_Val512_mut")
            trainer.log_to_csv(run_dir, trainer.global_step, "phase2_Val512_mut", l, p)
            
            l, p = trainer.validate(P2_val_wt_loader_1024, "phase2_Val1024_wt")
            trainer.log_to_csv(run_dir, trainer.global_step, "phase2_Val1024_wt", l, p)
            
            l, p = trainer.validate(P2_val_mut_loader_1024, "phase2_Val1024_mut")
            trainer.log_to_csv(run_dir, trainer.global_step, "phase2_Val1024_mut", l, p)
            
            trainer.save_checkpoint(run_dir)

    pbar.close()
    print(" Training finished")

if __name__ == "__main__":
    main()
