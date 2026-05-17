"""
Training script – LSTM (model 1) and Transformer (model 3).

Usage:
    python train_seq.py --model lstm  --epochs 30
    python train_seq.py --model transformer --epochs 30

Both models are trained identically (teacher forcing + cross-entropy).
Checkpoints saved to model/weights/<model_name>_best.pt
"""

from __future__ import annotations
import argparse
import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.dataset import DatesDataset
from utils.tokenizer import validate_date, tokens_to_date, decode_conditions

from model.model_lstm        import ConditionalSeq2SeqLSTM
from model.model_transformer import ConditionalTransformerDecoder


def resolve_project_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    cwd_path = os.path.abspath(path)
    if os.path.exists(cwd_path):
        return cwd_path
    return os.path.join(PROJECT_ROOT, path)


def get_model(name: str) -> nn.Module:
    if name == "lstm":
        return ConditionalSeq2SeqLSTM()
    elif name == "transformer":
        return ConditionalTransformerDecoder()
    else:
        raise ValueError(f"Unknown model: {name}")


def train_one_epoch(model, loader, optimizer, criterion, device, clip=1.0):
    model.train()
    total_loss = 0.0
    for day, mon, leap, dec, inp, tgt in loader:
        day, mon, leap, dec = day.to(device), mon.to(device), leap.to(device), dec.to(device)
        inp, tgt = inp.to(device), tgt.to(device)

        optimizer.zero_grad()
        logits = model(day, mon, leap, dec, inp)   # (B, T, V)
        B, T, V = logits.shape
        loss = criterion(logits.view(B * T, V), tgt.view(B * T))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n_valid, n_total = 0, 0

    for day, mon, leap, dec, inp, tgt in loader:
        day, mon, leap, dec = day.to(device), mon.to(device), leap.to(device), dec.to(device)
        inp, tgt = inp.to(device), tgt.to(device)

        logits = model(day, mon, leap, dec, inp)
        B, T, V = logits.shape
        loss = criterion(logits.view(B * T, V), tgt.view(B * T))
        total_loss += loss.item()

        # Validity check: autoregressive decode
        seqs = model.generate(day, mon, leap, dec)
        for b in range(B):
            date_str = tokens_to_date(seqs[b])
            d, m, l, decade = decode_conditions(
                day[b].item(), mon[b].item(), leap[b].item(), dec[b].item()
            )
            if date_str and validate_date(date_str, d, m, l, decade):
                n_valid += 1
            n_total += 1

    avg_loss = total_loss / len(loader)
    validity = n_valid / n_total if n_total > 0 else 0.0
    return avg_loss, validity


def train(model_name: str, data_path: str, epochs: int, batch_size: int,
          lr: float, seed: int, weights_dir: str):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{model_name}] device={device}")
    data_path = resolve_project_path(data_path)
    weights_dir = resolve_project_path(weights_dir)

    train_ds = DatesDataset(data_path, split="train", seed=seed)
    val_ds   = DatesDataset(data_path, split="val",   seed=seed)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"  train={len(train_ds)}  val={len(val_ds)}")

    model = get_model(model_name).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # PAD_IDX=0

    os.makedirs(weights_dir, exist_ok=True)
    best_path   = os.path.join(weights_dir, f"{model_name}_best.pt")
    best_loss   = float("inf")
    history     = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss           = train_one_epoch(model, train_dl, optimizer, criterion, device)
        val_loss, validity = evaluate(model, val_dl, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        history.append({"epoch": epoch, "train_loss": tr_loss,
                         "val_loss": val_loss, "validity": validity})
        print(f"  Epoch {epoch:3d}/{epochs}  "
              f"train={tr_loss:.4f}  val={val_loss:.4f}  "
              f"validity={validity:.3f}  ({elapsed:.1f}s)")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({"epoch": epoch,
                        "model_state": model.state_dict(),
                        "history": history}, best_path)
            print(f"    ✓ saved best checkpoint (val_loss={best_loss:.4f})")

    print(f"\nTraining complete. Best val_loss={best_loss:.4f}")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       default="lstm",
                        choices=["lstm", "transformer"])
    parser.add_argument("--data",        default="data/data.txt")
    parser.add_argument("--epochs",      type=int,   default=30)
    parser.add_argument("--batch_size",  type=int,   default=512)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--weights_dir", default="model/weights")
    args = parser.parse_args()

    train(
        model_name  = args.model,
        data_path   = args.data,
        epochs      = args.epochs,
        batch_size  = args.batch_size,
        lr          = args.lr,
        seed        = args.seed,
        weights_dir = args.weights_dir,
    )
