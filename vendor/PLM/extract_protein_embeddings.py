import os
import argparse
import torch
import pandas as pd
import numpy as np

from model import ProteinMLM
from dataprocessing import (
    Bert_input_data,
    vocab,
    load_distance_dict_from_matrix_csv,
)


def build_model(ckpt_path, embed_dim, num_layers, num_heads, ffn_embed_dim, dropout, device):
    model = ProteinMLM(
        vocab_size=len(vocab),
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ffn_embed_dim=ffn_embed_dim,
        padding_idx=vocab["[PAD]"],
        dropout=dropout,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def extract(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    distance_dict = load_distance_dict_from_matrix_csv(args.distance_csv)
    model = build_model(
        args.ckpt, args.embed_dim, args.num_layers, args.num_heads,
        args.ffn_embed_dim, args.dropout, device,
    )

    df = pd.read_csv(args.csv)
    df = df.drop_duplicates(subset=[args.id_col]).reset_index(drop=True)
    ids = df[args.id_col].astype(str).tolist()
    sequences = df[args.seq_col].tolist()
    print(f"▶ {len(sequences)} unique sequences ({args.id_col} 기준)")

    id_list, mean_list, max_list, seq_list = [], [], [], []

    for i, (pid, seq) in enumerate(zip(ids, sequences)):
        data = Bert_input_data(
            seq=seq, vocab=vocab, distance_dict=distance_dict, max_len=args.max_len
        )
        input_ids = data["input_ids"].unsqueeze(0).to(device)
        position_ids = data["position_ids"].unsqueeze(0).to(device)
        attention_mask = data["attention_mask"].unsqueeze(0).to(device)

        _, token_reps = model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
        token_reps = token_reps.squeeze(0)  # (T, C)

        tokens_len = int(attention_mask.sum().item())
        actual = token_reps[1:tokens_len - 1]  # CLS/SEP 제거

        mean_rep = actual.mean(dim=0)
        max_rep = actual.max(dim=0)[0]

        id_list.append(pid)
        mean_list.append(mean_rep.cpu().numpy())
        max_list.append(max_rep.cpu().numpy())

        if args.save_tokens:
            fixed = torch.zeros((args.max_len - 2, args.embed_dim))
            fixed[: actual.size(0)] = actual.cpu()
            seq_list.append(fixed.numpy())

        if (i + 1) % 50 == 0 or (i + 1) == len(sequences):
            print(f"✓ Processed {i + 1}/{len(sequences)}")

    out_dir = os.path.dirname(os.path.abspath(args.save_path))
    os.makedirs(out_dir, exist_ok=True)

    save_kwargs = dict(
        protein=np.array(id_list),
        mean=np.stack(mean_list, axis=0).astype(np.float32),
        max=np.stack(max_list, axis=0).astype(np.float32),
    )
    if args.save_tokens:
        save_kwargs["seq"] = np.stack(seq_list, axis=0).astype(np.float32)

    np.savez_compressed(args.save_path, **save_kwargs)
    print(
        f"\n✅ Saved protein embeddings → {args.save_path}\n"
        f"   - proteins: {len(id_list)}\n"
        f"   - mean/max dim: {mean_list[0].shape[0]}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="학습된 PLM(MutPLM) 체크포인트로 단백질 서열 임베딩(mean/max pooling)을 추출해 npz로 저장한다."
    )
    parser.add_argument("--csv", required=True, help="id_col, seq_col 컬럼을 포함한 csv (예: unique_fasta.csv)")
    parser.add_argument("--id_col", default="target_accession", help="단백질 식별자 컬럼명")
    parser.add_argument("--seq_col", default="FASTA", help="아미노산 서열 컬럼명")
    parser.add_argument("--ckpt", default="checkpoint_step500000.pt", help="PLM 체크포인트 경로")
    parser.add_argument("--distance_csv", default="six_aa_pairwise_distance.csv")
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--embed_dim", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=16)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--ffn_embed_dim", type=int, default=4096)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--save_path", required=True, help="출력 npz 경로")
    parser.add_argument(
        "--save_tokens", action="store_true",
        help="토큰별(seq) 임베딩까지 저장 (기본은 mean/max만 저장, seq는 용량이 매우 큼)",
    )
    args = parser.parse_args()
    extract(args)


if __name__ == "__main__":
    main()

# 사용 예시 (PLM/ 디렉토리에서 실행)
# python extract_protein_embeddings.py \
#   --csv ../DTA_dataset/chembl_35/IC50/unique_fasta.csv \
#   --id_col target_accession --seq_col FASTA \
#   --ckpt checkpoint_step500000.pt \
#   --save_path ../protein_emb/IC50_protein.npz
