import os
import argparse
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


@torch.no_grad()
def get_embeddings(smiles_list, model, tokenizer, device, batch_size=32, max_length=512):
    model.eval()
    embeddings = []

    for i in tqdm(range(0, len(smiles_list), batch_size)):
        batch_smiles = smiles_list[i: i + batch_size]

        inputs = tokenizer(
            batch_smiles, return_tensors="pt",
            padding=True, truncation=True, max_length=max_length,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(**inputs)

        # Masked mean pooling
        last_hidden_state = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = torch.sum(last_hidden_state * mask, 1)
        counts = torch.clamp(mask.sum(1), min=1e-9)
        batch_embeddings = summed / counts

        embeddings.append(batch_embeddings.cpu().numpy())

    return np.vstack(embeddings)


def main():
    parser = argparse.ArgumentParser(
        description="ChemBERTa로 SMILES 임베딩(mean pooling)을 추출해 npz로 저장한다."
    )
    parser.add_argument("--csv", required=True, help="smiles_col 컬럼을 포함한 csv")
    parser.add_argument("--smiles_col", default="SMILES")
    parser.add_argument(
        "--id_col", default=None,
        help="선택: SMILES와 함께 저장할 ID 컬럼 (예: compound_chembl_id, drug_name). 지정 시 drug_id 키로 저장",
    )
    parser.add_argument("--model_name", default="DeepChem/ChemBERTa-100M-MLM")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--save_path", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = pd.read_csv(args.csv)
    df = df.drop_duplicates(subset=[args.smiles_col]).reset_index(drop=True)
    smiles_list = df[args.smiles_col].astype(str).tolist()
    print(f"▶ {len(smiles_list)} unique SMILES ({args.smiles_col} 기준)")

    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)

    # RoBERTa 계열은 position id를 padding_idx+1부터 매기므로, position embedding
    # 테이블(max_position_embeddings)을 넘지 않으려면 실제 토큰 길이는 그보다 2 작아야 한다.
    # 이 여유를 두지 않으면 긴 SMILES에서 position embedding lookup이 범위를 벗어나
    # CUDA device-side assert(indexSelectLargeIndex)로 죽는다.
    max_length = model.config.max_position_embeddings - 2
    embeddings = get_embeddings(smiles_list, model, tokenizer, device, args.batch_size, max_length)

    save_kwargs = dict(
        SMILES=np.array(smiles_list),
        drug_embedding=embeddings.astype(np.float32),
    )
    if args.id_col:
        save_kwargs["drug_id"] = df[args.id_col].astype(str).to_numpy()

    out_dir = os.path.dirname(os.path.abspath(args.save_path))
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(args.save_path, **save_kwargs)

    print(
        f"\n✅ Saved drug embeddings → {args.save_path}\n"
        f"   - molecules: {len(smiles_list)}\n"
        f"   - embedding dim: {embeddings.shape[1]}"
    )


if __name__ == "__main__":
    main()

# 사용 예시
# python DLM/extract_drug_embeddings.py \
#   --csv DTA_dataset/chembl_35/IC50/unique_smiles.csv \
#   --smiles_col SMILES --id_col compound_chembl_id \
#   --save_path drug_emb/IC50_drug.npz
