import torch
import pandas as pd
from torch.utils.data import Dataset
from mask_tokens import mask_tokens
def load_distance_dict_from_matrix_csv(csv_path):
    df = pd.read_csv(csv_path, index_col=0)

    distance_dict = {}
    for a1 in df.index:
        distance_dict[a1] = {}
        for a2 in df.columns:
            distance_dict[a1][a2] = float(df.loc[a1, a2])

    return distance_dict

distance_dict = load_distance_dict_from_matrix_csv("six_aa_pairwise_distance.csv")

def positional_embedding(chars, max_len, distance_dict):
    X = torch.zeros(max_len, dtype=torch.float32)
    total = 1.0

    for i in range(min(len(chars), max_len)):
        if i > 0:
            prev = chars[i - 1]
            curr = chars[i]
            total += distance_dict.get(prev, {}).get(curr, 0.0)

        X[i] = total
    X = torch.round(X).long()
    return X
AA_VOCAB = [
    "A","C","D","E","F","G","H","I","K","L",
    "M","N","P","Q","R","S","T","V","W","Y",
    "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
]

vocab = {tok: i for i, tok in enumerate(AA_VOCAB)}
inv_vocab = {i: tok for tok, i in vocab.items()}


def Bert_input_data(seq, vocab, distance_dict, max_len=1024):
    """
    seq: protein sequence (string)
    vocab: token -> id dict
    distance_dict: AA pairwise distance dict
    """

    # ----------------------------
    # 1. Tokenize
    # ----------------------------
    aa_tokens = list(seq.strip().upper())
    max_aa_len = max_len - 2
    aa_tokens = aa_tokens[:max_aa_len]
    
    tokens = ["[CLS]"] + aa_tokens + ["[SEP]"]

    input_ids = [vocab.get(t, vocab["[UNK]"]) for t in tokens]
    attention_mask = [1] * len(input_ids)

    # ----------------------------
    # 2. Positional Embedding (AA-based)
    # ----------------------------
    pe = torch.zeros(max_len, dtype=torch.float)

    aa_pe = positional_embedding(
        chars=aa_tokens,
        max_len=max_len - 2,  # CLS, SEP 제외
        distance_dict=distance_dict
    )

    # CLS = 0, AA PE 채우기
    pe[1:1 + len(aa_pe)] = aa_pe[:max_len - 2]

    # ----------------------------
    # 3. Padding
    # ----------------------------
    while len(input_ids) < max_len:
        input_ids.append(vocab["[PAD]"])
        attention_mask.append(0)

    input_ids = input_ids[:max_len]
    attention_mask = attention_mask[:max_len]

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "position_ids": pe
    }


class ProteinDataset(Dataset):
    def __init__(
        self,
        sequences,
        vocab,
        distance_dict,
        max_len=1024,
        mlm_probability=0.15,
    ):
        self.sequences = sequences
        self.vocab = vocab
        self.distance_dict = distance_dict
        self.max_len = max_len
        self.mlm_probability = mlm_probability

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        data = Bert_input_data(
            seq=seq,
            vocab=self.vocab,
            distance_dict=self.distance_dict,
            max_len=self.max_len,
        )

        input_ids = data["input_ids"]
        pos = data["position_ids"]
        attn = data["attention_mask"]

        masked, labels = mask_tokens(
            input_ids.clone(),
            self.vocab,
            mlm_probability=self.mlm_probability,
        )

        return {
            "input_ids": masked,
            "labels": labels,
            "position_ids": pos,
            "attention_mask": attn,
        }

class ProteinDatasetVal(Dataset):
    def __init__(
        self,
        sequences,
        vocab,
        distance_dict,
        max_len=1024,
        mlm_probability=0.15,
        seed=1234,
    ):
        self.sequences = sequences
        self.vocab = vocab
        self.distance_dict = distance_dict
        self.max_len = max_len
        self.mlm_probability = mlm_probability
        self.seed = seed

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]

        # Deterministic masking per sample based on seed + index
        gen = torch.Generator().manual_seed(self.seed + idx)

        data = Bert_input_data(
            seq=seq,
            vocab=self.vocab,
            distance_dict=self.distance_dict,
            max_len=self.max_len,
        )

        input_ids = data["input_ids"]
        pos = data["position_ids"]
        attn = data["attention_mask"]

        masked, labels = mask_tokens(
            input_ids.clone(),
            self.vocab,
            mlm_probability=self.mlm_probability,
            generator=gen,
        )

        return {
            "input_ids": masked,
            "labels": labels,
            "position_ids": pos,
            "attention_mask": attn,
        }
