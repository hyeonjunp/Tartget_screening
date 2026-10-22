import os
import copy
import logging
import torch
import pandas as pd
from tqdm import tqdm
from metrics import get_cindex, get_rm2, pearson, spearman


# ======================
# Checkpoint (단일 모델)
# ======================
def save_checkpoint(path, epoch, model, optimizer,
                    best_val_loss=None, extra=None):
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_val_loss": best_val_loss if best_val_loss is not None else float("inf"),
        "extra": extra or {},
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(ckpt, path)


def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    start_epoch   = ckpt.get("epoch", -1) + 1
    best_val_loss = ckpt.get("best_val_loss", float("inf"))
    extra         = ckpt.get("extra", {})
    return start_epoch, best_val_loss, extra


# ======================
# Evaluation (FP32)
# ======================
@torch.no_grad()
def test(data_loader, model, loss_fn,device, save_path=None):
    model.eval()
    y_true, y_pred = [], []
    smiles_list, fasta_list = [], []

    total_se = 0.0
    total_samples = 0

    for drug_emb, fasta, fasta_max_emb, label in data_loader:
        drug_emb = drug_emb.to(device, non_blocking=True)
        fasta = fasta.to(device, non_blocking=True)
        fasta_max_emb = fasta_max_emb.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        
        pred_affinity_logits = model(drug_emb, fasta, fasta_max_emb).view(-1)
        batch_size = label.size(0)

        if loss_fn is not None:
            loss = loss_fn(pred_affinity_logits, label)
            total_se += loss.item() * batch_size

        total_samples += batch_size
        y_true.extend(label.cpu().tolist())
        y_pred.extend(pred_affinity_logits.cpu().tolist())

    avg_loss = total_se / total_samples if loss_fn is not None else 0.0
    ci = get_cindex(y_true, y_pred)
    rm2 = get_rm2(y_true, y_pred)
    pearson_r = pearson(y_true, y_pred)
    spearman_r = spearman(y_true, y_pred)
    return avg_loss, ci, rm2, pearson_r, spearman_r

@torch.no_grad()
def valid(data_loader, model, loss_fn,device, save_path=None):
    model.eval()
    y_true, y_pred = [], []
    smiles_list, fasta_list = [], []

    total_se = 0.0
    total_samples = 0

    for drug_emb, fasta, fasta_max_emb, label in data_loader:
        drug_emb = drug_emb.to(device, non_blocking=True)
        fasta = fasta.to(device, non_blocking=True)
        fasta_max_emb = fasta_max_emb.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        
        pred_affinity_logits = model(drug_emb, fasta, fasta_max_emb).view(-1)
        batch_size = label.size(0)

        if loss_fn is not None:
            loss = loss_fn(pred_affinity_logits, label)
            total_se += loss.item() * batch_size

        total_samples += batch_size
        y_true.extend(label.cpu().tolist())
        y_pred.extend(pred_affinity_logits.cpu().tolist())

    avg_loss = total_se / total_samples if loss_fn is not None else 0.0
    return avg_loss

# =================================
# Training (수정 완료된 최종 버전)
# =================================
def train(model, train_loader, val_loader,
          test_loader, test_wild_loader, test_mutant_loader,
          test_non5_loader, test_wild_non5_loader, test_mutant_non5_loader,
          writer, NAME, save_csv, lr=1e-4, epoch=1, resume=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=20,
        min_lr=1e-6,
    )
    loss_fn = torch.nn.MSELoss(reduction='mean')

    os.makedirs(save_csv, exist_ok=True)
    best_path = os.path.join(save_csv, "best.pth")
    last_path = os.path.join(save_csv, "last.pth")

    model_best = copy.deepcopy(model)
    min_score = float('inf')
    start_epoch = 0

    # =========================
    #  Early stopping 설정
    # =========================
    patience = 80
    early_stop_counter = 0

    if resume and os.path.isfile(last_path):
        start_epoch, min_score, _ = load_checkpoint(last_path, model, optimizer, map_location=device)
        model_best = copy.deepcopy(model)
        logging.info(f"[Resume] start_epoch={start_epoch}, best_val_score={min_score:.6f}")

    for epo in range(start_epoch, epoch):

        model.train()
        total_loss = 0.0
        total_samples = 0

        for drug_emb, fasta, fasta_max_emb, label in tqdm(train_loader, desc=f"Epoch {epo+1}/{epoch}"):
            drug_emb = drug_emb.to(device, non_blocking=True)
            fasta = fasta.to(device, non_blocking=True)
            fasta_max_emb = fasta_max_emb.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred_aff = model(drug_emb, fasta, fasta_max_emb).view(-1)
            loss = loss_fn(pred_aff, label)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * len(label)
            total_samples += len(label)

        avg_train_loss = total_loss / total_samples

        if writer is not None:
            writer.add_scalar(f'Loss/Train_Affinity_{NAME}', avg_train_loss, epo)

        logging.info(f'Epoch {epo+1} - Train Loss: {avg_train_loss:.4f}')

        # =========================
        # Validation
        # =========================
        val_loss = valid(val_loader, model, loss_fn, device)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        logging.info(f"Epoch {epo+1} - Valid Loss: {val_loss:.4f} | LR: {current_lr:.6e}")

        if writer is not None:
            writer.add_scalar(f'Loss/Valid_{NAME}', val_loss, epo)

        logging.info(f'Epoch {epo+1} - Valid Loss: {val_loss:.4f}')

        # =========================
        # BEST 갱신 여부 확인
        # =========================
        if val_loss < min_score:
            min_score = val_loss
            model_best = copy.deepcopy(model)
            save_checkpoint(best_path, epo, model_best, optimizer, best_val_loss=min_score)

            early_stop_counter = 0   # 리셋
            logging.info(f"[Saved] New BEST @ epoch {epo+1} -> {best_path} (score: {val_loss:.4f})")

        else:
            early_stop_counter += 1
            logging.info(f"[EarlyStop] Counter: {early_stop_counter}/{patience}")

        # 항상 last 저장
        save_checkpoint(last_path, epo, model, optimizer, best_val_loss=min_score)

        # =========================
        # Early stopping 조건
        # =========================
        if early_stop_counter >= patience:
            logging.info(f"[STOP] No improvement for {patience} epochs. Early stopping at epoch {epo+1}.")
            break

    # 최종 평가
    if os.path.isfile(best_path):
        load_checkpoint(best_path, model_best, optimizer=None, map_location=device)
        logging.info("[Loaded] BEST checkpoint for final evaluation.")

    model_best.eval()

    # =========================
    # Test summary (fold 단위)
    # =========================
    summary_rows = []

    for split_name, loader in [
        ("test", test_loader),
        ("test_wild", test_wild_loader),
        ("test_mutant", test_mutant_loader),
        ("test_non5", test_non5_loader),
        ("test_wild_non5", test_wild_non5_loader),
        ("test_mutant_non5", test_mutant_non5_loader),
    ]:
        if loader is None:
            continue

        avg_loss, ci, rm2, pearson_r, spearman_r = test(
            loader, model_best, loss_fn, device
        )

        logging.info(
            f'{split_name:17s} - '
            f'Loss {avg_loss:.4f}, CI {ci:.4f}, RM2 {rm2:.4f}, '
            f'Pearson {pearson_r:.4f}, Spearman {spearman_r:.4f}'
        )

        summary_rows.append({
            "split": split_name,
            "loss": avg_loss,
            "ci": ci,
            "rm2": rm2,
            "pearson": pearson_r,
            "spearman": spearman_r
        })

    # =========================
    # CSV 저장 (fold당 1개)
    # =========================
    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(save_csv, "test_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)