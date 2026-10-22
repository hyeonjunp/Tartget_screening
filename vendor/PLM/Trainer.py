# Trainer.py
import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from mask_tokens import mask_tokens


class ProteinMLMTrainer:
    def __init__(
        self,
        model,
        vocab,
        lr,
        max_steps,
        train_loader=None,
        val_loader=None,
        eval_interval=10000,
    ):
        self.model = model
        self.vocab = vocab
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.max_steps = int(max_steps)
        self.eval_interval = int(eval_interval)
        self.global_step = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optim = AdamW(
            self.model.parameters(),
            lr=float(lr),
            weight_decay=0.01,
        )
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    # --------------------------------------------------
    # Checkpoint (step-based)
    # --------------------------------------------------
    def save_checkpoint(self, run_dir):
        path = os.path.join(run_dir, f"checkpoint_step{self.global_step}.pt")
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optim.state_dict(),
                "global_step": self.global_step,
            },
            path,
        )
        print(f" Saved checkpoint at step {self.global_step}")

    def load_latest_checkpoint(self, run_dir):
        if not os.path.exists(run_dir):
            return

        ckpts = [
            f for f in os.listdir(run_dir)
            if f.startswith("checkpoint_step")
        ]
        if not ckpts:
            return

        latest = sorted(
            ckpts,
            key=lambda x: int(x.split("step")[1].split(".")[0])
        )[-1]

        path = os.path.join(run_dir, latest)
        ckpt = torch.load(path, map_location=self.device)

        self.model.load_state_dict(ckpt["model_state"])
        self.optim.load_state_dict(ckpt["optimizer_state"])
        self.global_step = ckpt["global_step"]

        print(f" Resumed from {latest} (step={self.global_step})")

    # --------------------------------------------------
    # Step-based training (epoch 없음)
    # --------------------------------------------------
    def train_step(self, batch):
        self.model.train()

        ids = batch["input_ids"].to(self.device)
        pos = batch["position_ids"].to(self.device)
        mask = batch["attention_mask"].to(self.device)

        masked, labels = mask_tokens(ids.clone(), self.vocab)
        masked = masked.to(self.device)
        labels = labels.to(self.device)

        logits, _ = self.model(masked, pos, mask)
        loss = self.loss_fn(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        )

        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optim.step()

        self.global_step += 1
        return loss.item()

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------
    @torch.no_grad()
    def validate(self, loader=None, name="val"):
        loader = self.val_loader if loader is None else loader
        if loader is None:
            return
    
        self.model.eval()
    
        total_loss = 0.0
        n = 0
    
        for batch in tqdm(
            loader,
            desc=f"Validate [{name}] step={self.global_step}",
            leave=False,
        ):
            # 🔹 그대로 사용 (이미 mask 고정됨)
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            pos = batch["position_ids"].to(self.device)
            mask = batch["attention_mask"].to(self.device)
    
            logits, _ = self.model(input_ids, pos, mask)

            loss = self.loss_fn(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )
    
            total_loss += loss.item()
            n += 1
    
        if n == 0:
            print(f"[step {self.global_step}] {name}: no batches")
            self.model.train()
            return
    
        avg_loss = total_loss / n
        ppl = torch.exp(torch.tensor(avg_loss, device=self.device))
    
        print(
            f"[step {self.global_step}] "
            f"{name} | loss: {avg_loss:.4f} | PPL: {ppl:.2f}"
        )
        
        self.model.train()
        
        # CSV Logging
        # Determine the directory (assuming run_dir is passed or stored)
        # Here we don't have run_dir in validate(), so we'll need to modify validate() signature or store run_dir in __init__
        # For a quick fix without changing validate() calls everywhere, we can try to infer or pass it.
        # But wait, validate() calls in pretrain.py pass (loader, name).
        # We can just return the metrics and let pretrain.py log them, OR modify validate() to accept run_dir.
        # However, pretrain.py calls validate() multiple times (for wt, mut, etc), so we might want to log rows with 'name' column.
        
        return avg_loss, ppl.item()

    # --------------------------------------------------
    # Checkpoint
    # --------------------------------------------------
    def save_checkpoint(self, run_dir):
        path = os.path.join(run_dir, f"checkpoint_step{self.global_step}.pt")
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optim.state_dict(),
                "global_step": self.global_step,
            },
            path,
        )
        print(f" Saved checkpoint at step {self.global_step}")
        
    def log_to_csv(self, run_dir, step, name, loss, ppl):
        import csv
        log_path = os.path.join(run_dir, "validation_log.csv")
        file_exists = os.path.isfile(log_path)
        
        with open(log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["step", "name", "loss", "ppl"])
            writer.writerow([step, name, loss, ppl])