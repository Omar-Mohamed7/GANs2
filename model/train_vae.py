"""
Training script – Conditional VAE  (Model 4)

Usage:
    python train_vae.py --epochs 40

Checkpoint: model/weights/vae_best.pt
"""

from __future__ import annotations
import argparse
import os
import sys
import time
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.dataset   import DatesDataset
from utils.tokenizer import (
    validate_date, tokens_to_date, decode_conditions, MAX_DATE_LEN
)
from model.model_vae import ConditionalVAE


def resolve_project_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    cwd_path = os.path.abspath(path)
    if os.path.exists(cwd_path):
        return cwd_path
    return os.path.join(PROJECT_ROOT, path)


def train_vae(data_path: str, epochs: int, batch_size: int,
              lr: float, beta: float, seed: int, weights_dir: str):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[VAE] device={device}")
    data_path = resolve_project_path(data_path)
    weights_dir = resolve_project_path(weights_dir)

    train_ds = DatesDataset(data_path, split="train", seed=seed)
    val_ds   = DatesDataset(data_path, split="val",   seed=seed)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"  train={len(train_ds)}  val={len(val_ds)}")

    model = ConditionalVAE().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    os.makedirs(weights_dir, exist_ok=True)
    best_path    = os.path.join(weights_dir, "vae_best.pt")
    best_validity = 0.0
    history       = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        tr_recon, tr_kl = 0.0, 0.0
        n_steps = 0

        for day, mon, leap, dec, inp, tgt in train_dl:
            day  = day.to(device); mon  = mon.to(device)
            leap = leap.to(device); dec = dec.to(device)
            # date chars: tgt[:, :MAX_DATE_LEN]  (shift of inp removes BOS)
            date_chars = tgt[:, :MAX_DATE_LEN].to(device)   # (B, 10)

            optimizer.zero_grad()
            logits, mu, logvar = model(day, mon, leap, dec, date_chars)
            loss, info = ConditionalVAE.loss(logits, date_chars, mu, logvar, beta)
            loss.backward()
            optimizer.step()

            tr_recon += info["recon"]
            tr_kl    += info["kl"]
            n_steps  += 1

        scheduler.step()

        # ── Validation ─────────────────────────────────────────────────────
        model.eval()
        n_valid, n_total = 0, 0
        with torch.no_grad():
            for day, mon, leap, dec, _, _ in val_dl:
                day  = day.to(device); mon  = mon.to(device)
                leap = leap.to(device); dec = dec.to(device)
                seqs = model.generate_tokens(day, mon, leap, dec)
                for b in range(day.size(0)):
                    date_str = tokens_to_date(seqs[b])
                    d, m, l, decade = decode_conditions(
                        day[b].item(), mon[b].item(), leap[b].item(), dec[b].item()
                    )
                    if date_str and validate_date(date_str, d, m, l, decade):
                        n_valid += 1
                    n_total += 1

        validity = n_valid / n_total if n_total > 0 else 0.0
        elapsed  = time.time() - t0
        history.append({
            "epoch": epoch,
            "recon": tr_recon / n_steps,
            "kl":    tr_kl    / n_steps,
            "validity": validity,
        })
        print(f"  Epoch {epoch:3d}/{epochs}  "
              f"recon={tr_recon/n_steps:.4f}  kl={tr_kl/n_steps:.4f}  "
              f"validity={validity:.3f}  ({elapsed:.1f}s)")

        if validity >= best_validity:
            best_validity = validity
            torch.save({
                "epoch":      epoch,
                "model_state": model.state_dict(),
                "history":    history,
            }, best_path)
            print(f"    ✓ saved best VAE checkpoint (validity={best_validity:.3f})")

    print(f"\nVAE training complete. Best validity={best_validity:.3f}")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",        default="data/data.txt")
    parser.add_argument("--epochs",      type=int,   default=40)
    parser.add_argument("--batch_size",  type=int,   default=512)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--beta",        type=float, default=0.5,
                        help="KL weight in ELBO loss")
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--weights_dir", default="model/weights")
    args = parser.parse_args()

    train_vae(
        data_path   = args.data,
        epochs      = args.epochs,
        batch_size  = args.batch_size,
        lr          = args.lr,
        beta        = args.beta,
        seed        = args.seed,
        weights_dir = args.weights_dir,
    )
