import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import Integration
from process_data import DTAData


BATCH_SIZE = 512
MAX_SMILES_LEN = 128
MAX_LEN = 1024

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def build_model():
    return Integration(
        drug_model=nn.Identity(),
        protein_model=nn.Identity(),
        drug_out_dim=384,
        protein_dim=1024,
        hidden_sizes=[1024, 512, 256, 1],
        dropout=0.1,
    ).to(device)


def load_model(ckpt_path):
    model = build_model()
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@torch.no_grad()
def predict(model, loader):
    preds = []
    for drug_emb, fasta, fasta_max_emb, _ in loader:
        drug_emb      = drug_emb.to(device, non_blocking=True)
        fasta         = fasta.to(device, non_blocking=True)
        fasta_max_emb = fasta_max_emb.to(device, non_blocking=True)
        out = model(drug_emb, fasta, fasta_max_emb).view(-1)
        preds.extend(out.cpu().tolist())
    return np.array(preds)


def predict_npz(npz_path, models, save_dir):
    data = np.load(npz_path, allow_pickle=True)

    dataset = DTAData(
        data["drug_embedding"],
        data["protein_embedding"],
        data["protein_max_embedding"],
        data["affinity"],
        device, MAX_SMILES_LEN, MAX_LEN,
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    fold_preds = []
    for fold_idx, model in enumerate(models, start=1):
        preds = predict(model, loader)
        fold_preds.append(preds)
        print(f"  fold{fold_idx} done ({len(preds)} samples)")

    fold_preds = np.stack(fold_preds, axis=0)  # (5, N)

    df = pd.DataFrame({
        "ID":            data["ID"],
        "SMILES":        data["SMILES"],
        "protein":       data["protein"],
        "true_affinity": data["affinity"].astype(float),
    })
    for i, col_preds in enumerate(fold_preds, start=1):
        df[f"pred_fold{i}"] = col_preds

    df["pred_mean"] = fold_preds.mean(axis=0)
    df["pred_std"]  = fold_preds.std(axis=0)

    os.makedirs(save_dir, exist_ok=True)
    out_name = os.path.splitext(os.path.basename(npz_path))[0] + "_pred.csv"
    out_path = os.path.join(save_dir, out_name)
    df.to_csv(out_path, index=False)
    print(f"  Saved → {out_path}")
    return df


def load_models_from_dir(model_dir):
    models = []
    for fold in range(1, 6):
        ckpt = os.path.join(model_dir, f"fold{fold}", "best.pth")
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"체크포인트 없음: {ckpt}")
        models.append(load_model(ckpt))
        print(f"  Loaded fold{fold}: {ckpt}")
    return models


def run_dataset(model_dir, dataset_name, npz_files, save_root):
    print(f"\n{'='*60}")
    print(f"[{dataset_name}] Loading 5-fold models from: {model_dir}")
    models = load_models_from_dir(model_dir)

    save_dir = os.path.join(save_root, dataset_name)
    all_dfs = []
    for npz_path in npz_files:
        print(f"\n  Predicting: {os.path.basename(npz_path)}")
        df = predict_npz(npz_path, models, save_dir)
        all_dfs.append(df)

    if len(all_dfs) > 1:
        merged = pd.concat(all_dfs, ignore_index=True)
        merged_path = os.path.join(save_dir, "all_predictions.csv")
        merged.to_csv(merged_path, index=False)
        print(f"  Merged CSV → {merged_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_files", type=str, nargs="+", required=True,
                        help="예측할 npz 파일 경로 목록")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="결과 저장 루트 디렉토리 (데이터셋별 하위폴더 자동 생성)")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model_dir", type=str,
                       help="단일 모델 디렉토리 (fold1~5/best.pth 포함)")
    group.add_argument("--model_root", type=str,
                       help="모든 데이터셋 결과가 있는 루트 (DAVIS-complete/Kd/IC50/Ki 자동 탐색)")

    parser.add_argument("--datasets", type=str, nargs="+",
                        default=["DAVIS-complete", "Kd", "IC50", "Ki"],
                        help="--model_root 사용 시 처리할 데이터셋 이름 목록")
    args = parser.parse_args()

    if args.model_root:
        for dataset in args.datasets:
            model_dir = os.path.join(args.model_root, dataset)
            if not os.path.isdir(model_dir):
                print(f"[Skip] 디렉토리 없음: {model_dir}")
                continue
            run_dataset(model_dir, dataset, args.npz_files, args.save_dir)
    else:
        dataset_name = os.path.basename(args.model_dir.rstrip("/"))
        run_dataset(args.model_dir, dataset_name, args.npz_files, args.save_dir)


if __name__ == "__main__":
    main()

