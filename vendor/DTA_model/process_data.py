from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd


def to_tensor(arr):
    """numpy ndarray / list / pd.Series / torch.Tensor → float32 Tensor"""
    if isinstance(arr, pd.Series):
        arr = np.stack(arr.values)
    if isinstance(arr, list):
        arr = np.stack(arr)
    if isinstance(arr, np.ndarray):
        return torch.from_numpy(arr).float()
    if torch.is_tensor(arr):
        return arr.float()
    raise TypeError(f"Unsupported type: {type(arr)}")


class DTAData(Dataset):
    """
    Drug embedding + Protein embedding → affinity 예측용 Dataset.
    SMILES → graph 변환 없이 drug embedding을 직접 사용.
    """
    def __init__(self, drug_embeddings, fasta_embeddings, fasta_max_embeddings, affinity,
                 device, max_smiles_len=128, max_fasta_len=1024):

        self.drug_embeddings  = to_tensor(drug_embeddings)   # (N, D_drug)
        self.fasta_embeddings = to_tensor(fasta_embeddings)  # (N, D_prot)
        self.fasta_max_embeddings = to_tensor(fasta_max_embeddings)  # (N, D_prot)

        # affinity
        if isinstance(affinity, (pd.Series, list, np.ndarray)):
            self.affinity = torch.tensor(np.array(affinity), dtype=torch.float32)
        else:
            self.affinity = affinity.float()

        self.device = device
        self.max_smiles_len = max_smiles_len
        self.max_fasta_len  = max_fasta_len

    def __len__(self):
        return len(self.affinity)

    def __getitem__(self, idx):
        drug_emb  = self.drug_embeddings[idx]   # torch.Tensor (D_drug,)
        fasta_emb = self.fasta_embeddings[idx]  # torch.Tensor (D_prot,)
        fasta_max_emb = self.fasta_max_embeddings[idx]  # torch.Tensor (D_prot,)
        y         = self.affinity[idx]           # torch.Tensor ()

        return drug_emb, fasta_emb, fasta_max_emb, y
