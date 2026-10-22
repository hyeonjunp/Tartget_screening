import os
import sys
import random
import logging
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import Integration
from process_data import DTAData
from train_and_test import train

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


# ----------------------------
# Seed utils
# ----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------
# Global config
# ----------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256
TEST_BATCH_SIZE = 512
MAX_SMILES_LEN = 128
max_len= 1024

# ----------------------------
# One fold run
# ----------------------------
def run_fold(
    fold,
    seed,
    data_dir,
    result_dir,
    # pe_type,
    # max_len,
    resume=False
):
    # =========================
    # Data path
    # =========================
    fold_dir = os.path.join(data_dir, f"fold{fold+1}")

    # =========================
    # Result save path
    # DAVIS_result/int_sincos/len1024/fold1
    # =========================
    save_path = os.path.join(
        result_dir,
        # pe_type,
        # f"len{max_len}",
        f"fold{fold+1}"
    )
    os.makedirs(save_path, exist_ok=True)

    MODEL_NAME = os.path.join(save_path)

    # =========================
    # Logging
    # =========================
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        filename=f"{MODEL_NAME}.log",
        level=logging.DEBUG
    )

    writer = SummaryWriter(log_dir=os.path.join(save_path, "runs"))

    # =========================
    # Load data
    # =========================
    df_train = np.load(os.path.join(fold_dir, "train_data.npz"))
    df_val   = np.load(os.path.join(fold_dir, "valid_data.npz"))
    df_test  = np.load(os.path.join(fold_dir, "test_data.npz"))

    def _maybe_load(name):
        path = os.path.join(fold_dir, name)
        return np.load(path) if os.path.isfile(path) else None

    df_test_wild = np.load(os.path.join(fold_dir, "test_wild_data.npz"))
    df_test_mutant = np.load(os.path.join(fold_dir, "test_mutant_data.npz"))
    # DAVIS 등 일부 데이터셋에는 non5(학습 데이터에 5개 미만 등장하는 단백질) 분할이 없을 수 있음 → 있으면만 평가
    df_test_non5 = _maybe_load("test_non5_data.npz")
    df_test_wild_non5 = _maybe_load("test_wild_non5_data.npz")
    df_test_mutant_non5 = _maybe_load("test_mutant_non5_data.npz")

    # =========================
    # Dataset
    # =========================
    train_set = DTAData(
        df_train["drug_embedding"], df_train["protein_embedding"],df_train["protein_max_embedding"], df_train["affinity"],
        device, MAX_SMILES_LEN, max_len
    )
    val_set = DTAData(
        df_val["drug_embedding"], df_val["protein_embedding"],df_val["protein_max_embedding"], df_val["affinity"],
        device, MAX_SMILES_LEN, max_len
    )
    test_set = DTAData(
        df_test["drug_embedding"], df_test["protein_embedding"],df_test["protein_max_embedding"], df_test["affinity"],
        device, MAX_SMILES_LEN, max_len
    )

    test_wild_set = DTAData(
        df_test_wild["drug_embedding"], df_test_wild["protein_embedding"],df_test_wild["protein_max_embedding"], df_test_wild["affinity"],
        device, MAX_SMILES_LEN, max_len
    )
    test_mutant_set = DTAData(
        df_test_mutant["drug_embedding"], df_test_mutant["protein_embedding"],df_test_mutant["protein_max_embedding"], df_test_mutant["affinity"],
        device, MAX_SMILES_LEN, max_len
    )
    def _maybe_set(df):
        if df is None:
            return None
        return DTAData(
            df["drug_embedding"], df["protein_embedding"], df["protein_max_embedding"], df["affinity"],
            device, MAX_SMILES_LEN, max_len
        )

    test_non5_set = _maybe_set(df_test_non5)
    test_wild_non5_set = _maybe_set(df_test_wild_non5)
    test_mutant_non5_set = _maybe_set(df_test_mutant_non5)

    # =========================
    # DataLoader
    # =========================
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        generator=g, num_workers=8, pin_memory=True, persistent_workers=True
    )
    valid_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False,
        generator=g, num_workers=8, pin_memory=True, persistent_workers=True
    )
    test_loader = DataLoader(
        test_set, batch_size=TEST_BATCH_SIZE, shuffle=False,
        generator=g, num_workers=8, pin_memory=True, persistent_workers=True
    )

    test_wild_loader = DataLoader(test_wild_set, batch_size=TEST_BATCH_SIZE, shuffle=False,
                                  generator=g, num_workers=8, pin_memory=True, persistent_workers=True)
    test_mutant_loader = DataLoader(test_mutant_set, batch_size=TEST_BATCH_SIZE, shuffle=False,
                                    generator=g, num_workers=8, pin_memory=True, persistent_workers=True)
    def _maybe_loader(dataset):
        if dataset is None:
            return None
        return DataLoader(dataset, batch_size=TEST_BATCH_SIZE, shuffle=False,
                           generator=g, num_workers=8, pin_memory=True, persistent_workers=True)

    test_non5_loader = _maybe_loader(test_non5_set)
    test_wild_non5_loader = _maybe_loader(test_wild_non5_set)
    test_mutant_non5_loader = _maybe_loader(test_mutant_non5_set)

    # =========================
    # Model
    # =========================
    import torch.nn as nn
    model = Integration(
        drug_model=nn.Identity(),
        protein_model=nn.Identity(),
        drug_out_dim=384,
        protein_dim=1024,
        hidden_sizes=[1024, 512, 256, 1],
        dropout=0.1,
    ).to(device)

    # =========================
    # Train
    # =========================
    train(
        model,
        train_loader, valid_loader,
        test_loader,
        test_wild_loader, test_mutant_loader,
        test_non5_loader, test_wild_non5_loader, test_mutant_non5_loader,
        writer,
        MODEL_NAME,
        save_path,
        lr=1e-4,
        epoch=1000,
        resume=resume
    )


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default=None,
                        choices=["DAVIS", "IC50", "Kd", "Ki"],
                        help="데이터셋 이름만 주면 data_dir/result_dir을 자동으로 정하고, "
                             "npz가 아직 없으면 preprocess_dataset.py로 먼저 만든 뒤 학습한다.")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Embedding dataset dir (contains fold1~5). --dataset 대신 직접 지정할 때 사용")
    parser.add_argument("--result_dir", type=str, default=None,
                        help="Root dir to save results. --dataset 대신 직접 지정할 때 사용")

    # parser.add_argument("--pe_type", type=str, required=True,
    #                     help="e.g. int_sincos / float_sincos")
    #parser.add_argument("--max_len", type=int, choices=[1024, 2048], required=True)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    if args.dataset:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, repo_root)
        import preprocess_dataset as pp

        cfg = pp.DATASETS[args.dataset]
        group = cfg["group"]
        args.data_dir = args.data_dir or os.path.join(repo_root, "DTA_model_npz", group)
        args.result_dir = args.result_dir or os.path.join(repo_root, "DTA_model", "result", group)

        train_npz_exists = all(
            os.path.isfile(os.path.join(args.data_dir, f"fold{i}", "train_data.npz"))
            for i in range(1, 6)
        )
        if not train_npz_exists:
            print(f"[{args.dataset}] 학습용 npz가 없어서 preprocess_dataset.py로 먼저 만듭니다...")
            pp.ensure_unique_lists(cfg, force=False)
            pp.ensure_protein_embedding(cfg, os.path.join(repo_root, "PLM", "checkpoint_step500000.pt"), force=False)
            pp.ensure_drug_embedding(cfg, batch_size=256, force=False)
            pp.build_all_npz(cfg, force=False)
    elif not args.data_dir or not args.result_dir:
        parser.error("--dataset 또는 (--data_dir, --result_dir) 조합 중 하나는 필요합니다.")

    set_seed(args.seed)

    for fold in range(5):
        print(f"\n🚀 Starting Fold {fold+1}")
        run_fold(
            fold,
            args.seed,
            args.data_dir,
            args.result_dir,
            #args.pe_type,
            #args.max_len,
            resume=args.resume
        )


if __name__ == "__main__":
    main()
