"""Inference adapter for the provided MutPLM + ChemBERTa 5-fold DTA checkpoints."""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any



def predict_five_fold(smiles: list[str], sequence: str, model_set: str, save_dir: str | Path) -> dict[str, Any]:
    """Predict each molecule against one sequence using the selected dataset's five folds.

    Requires the MutPLM_DTA environment (PyTorch, Transformers, Pandas, RDKit-compatible
    ChemBERTa dependencies). Uses the repository's own PLM preprocessing and weights.
    """
    if not smiles or not sequence.strip():
        raise ValueError("At least one SMILES and one protein sequence are required")
    from . import workflow
    if not hasattr(workflow, "download_model_assets"):
        importlib.reload(workflow)
    checkpoint_paths, protein_checkpoint = workflow.download_model_assets(model_set)
    if len(checkpoint_paths) != 5:
        raise FileNotFoundError(f"{model_set} requires all five best.pth checkpoints")

    try:
        import numpy as np
        import pandas as pd
        import torch
        from transformers import AutoModel, AutoTokenizer
        from rdkit import Chem
    except ImportError as exc:
        raise RuntimeError("Install the MutPLM_DTA environment (including RDKit and Transformers) before DTA prediction") from exc

    invalid = [index + 1 for index, value in enumerate(smiles) if Chem.MolFromSmiles(value) is None]
    if invalid:
        raise ValueError(f"Invalid SMILES at input rows: {invalid}")

    original_sequence_length = len(sequence)
    sequence = sequence[:1022]

    root = Path(__file__).resolve().parents[1]
    vendor = root / "vendor"
    plm_dir = (vendor / "PLM").resolve()
    dta_dir = (vendor / "DTA_model").resolve()
    sys.path.insert(0, str(plm_dir))
    try:
        # dataprocessing.py reads its distance matrix relative to the PLM working directory.
        previous_cwd = Path.cwd()
        os.chdir(plm_dir)
        try:
            import dataprocessing
            spec = importlib.util.spec_from_file_location("mutplm_embedding_model", plm_dir / "model.py")
            plm_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(plm_module)
        finally:
            os.chdir(previous_cwd)
    finally:
        sys.path.remove(str(plm_dir))

    torch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    protein_model = plm_module.ProteinMLM(vocab_size=len(dataprocessing.vocab), embed_dim=1024,
        num_layers=16, num_heads=16, ffn_embed_dim=4096,
        padding_idx=dataprocessing.vocab["[PAD]"], dropout=0.1)
    protein_ckpt = _torch_load(torch, protein_checkpoint, map_location="cpu")
    protein_model.load_state_dict(protein_ckpt["model_state"])
    protein_model.to(torch_device).eval()
    encoded = dataprocessing.Bert_input_data(sequence, dataprocessing.vocab,
        dataprocessing.load_distance_dict_from_matrix_csv(str(plm_dir / "six_aa_pairwise_distance.csv")), 1024)
    with torch.no_grad():
        _, token_reps = protein_model(input_ids=encoded["input_ids"].unsqueeze(0).to(torch_device),
            position_ids=encoded["position_ids"].unsqueeze(0).to(torch_device),
            attention_mask=encoded["attention_mask"].unsqueeze(0).to(torch_device))
        length = int(encoded["attention_mask"].sum())
        actual = token_reps[0, 1:length - 1]
        protein_mean = actual.mean(0).unsqueeze(0)
        protein_max = actual.max(0).values.unsqueeze(0)

    # The trained DTA checkpoints expect 384-wide vectors; their original DLM extractor uses ChemBERTa-77M.
    tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
    drug_model = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MLM").to(torch_device).eval()
    inputs = tokenizer(smiles, return_tensors="pt", padding=True, truncation=True,
        max_length=drug_model.config.max_position_embeddings - 2)
    inputs = {key: value.to(torch_device) for key, value in inputs.items()}
    with torch.no_grad():
        hidden = drug_model(**inputs).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).expand_as(hidden).float()
        drug_emb = ((hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)).float()

    sys.path.insert(0, str(dta_dir))
    try:
        import model as dta_model
    finally:
        sys.path.remove(str(dta_dir))
    predicted: list[list[float]] = []
    for checkpoint in checkpoint_paths:
        model = dta_model.Integration(drug_model=torch.nn.Identity(), protein_model=torch.nn.Identity(),
            drug_out_dim=384, protein_dim=1024, hidden_sizes=[1024, 512, 256, 1], dropout=0.1).to(torch_device)
        state = _torch_load(torch, checkpoint, map_location=torch_device)
        model.load_state_dict(state["model"])
        model.eval()
        with torch.no_grad():
            preds = model(drug_emb, protein_mean.expand(len(smiles), -1),
                          protein_max.expand(len(smiles), -1)).view(-1)
        predicted.append(preds.cpu().numpy().astype(float).tolist())

    folds = np.asarray(predicted)
    df = pd.DataFrame({"candidate_id": [f"candidate_{i+1}" for i in range(len(smiles))], "SMILES": smiles})
    for i, fold in enumerate(folds, 1):
        df[f"fold{i}"] = fold
    df["fold_min"], df["fold_max"] = folds.min(axis=0), folds.max(axis=0)
    destination = Path(save_dir)
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / f"{model_set.lower()}_predictions.csv"
    df.to_csv(output, index=False)
    return {"model_set": model_set, "dataset_target": {"DAVIS": "pIC50", "ChEMBL": "ChEMBL IC50 affinity scale", "ChEMBL IC50": "ChEMBL IC50 affinity scale", "ChEMBL Ki": "ChEMBL Ki affinity scale", "ChEMBL Kd": "ChEMBL Kd affinity scale"}[model_set],
            "device": str(torch_device), "rows": df.to_dict(orient="records"), "csv": str(output),
            "note": (f"Protein sequence was truncated from {original_sequence_length} to 1022 residues. " if original_sequence_length > 1022 else "") +
                    (f"Model output is in the training target scale ({model_set}); do not interpret Ki/Kd as pIC50." if model_set.startswith("ChEMBL") else "Model output is in the training target scale (pIC50).")}


def _torch_load(torch: Any, path: str | Path, **kwargs: Any) -> Any:
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)
