"""
Training script – Conditional GAN  (Model 2)

Usage:
    python train_gan.py --epochs 50

The GAN is trained with the standard minimax objective + label smoothing.
A validation metric: percentage of generated dates that pass all conditions.
Checkpoint: model/weights/gan_best.pt
"""

from __future__ import annotations
import argparse
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.dataset  import DatesDataset
from utils.tokenizer import (
    validate_date, tokens_to_date, decode_conditions,
    MAX_DATE_LEN, N_DATE_TOKENS
)
from model.model_gan import (
    ConditionEncoder, Generator, Discriminator, make_real_date_tensor
)


def resolve_project_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    cwd_path = os.path.abspath(path)
    if os.path.exists(cwd_path):
        return cwd_path
    return os.path.join(PROJECT_ROOT, path)


def train_gan(data_path: str, epochs: int, batch_size: int,
              lr_g: float, lr_d: float, seed: int, weights_dir: str):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[GAN] device={device}")
    data_path = resolve_project_path(data_path)
    weights_dir = resolve_project_path(weights_dir)

    train_ds = DatesDataset(data_path, split="train", seed=seed)
    val_ds   = DatesDataset(data_path, split="val",   seed=seed)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"  train={len(train_ds)}  val={len(val_ds)}")

    # Shared condition encoder weights
    cond_enc = ConditionEncoder().to(device)
    G = Generator(noise_dim=64, hidden=256, cond_enc=cond_enc).to(device)
    D = Discriminator(hidden=256, cond_enc=cond_enc).to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=lr_g, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=lr_d, betas=(0.5, 0.999))

    os.makedirs(weights_dir, exist_ok=True)
    best_path    = os.path.join(weights_dir, "gan_best.pt")
    best_validity = 0.0
    history       = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        G.train(); D.train()
        g_losses, d_losses = [], []

        for day, mon, leap, dec, inp, tgt in train_dl:
            day  = day.to(device); mon  = mon.to(device)
            leap = leap.to(device); dec = dec.to(device)
            # tgt: (B, 11) — BOS included; we only need the 10 date chars
            # inp[:,1:11] or tgt[:,0:10] = date chars without BOS/EOS
            # tgt is shifted: positions 0..9 → date chars (after BOS)
            date_chars = tgt[:, :MAX_DATE_LEN].to(device)   # (B, 10) date tokens
            B = day.size(0)

            real_flat = make_real_date_tensor(date_chars)    # (B, 10*V)

            # ── Train Discriminator ──────────────────────────────────────
            opt_D.zero_grad()
            real_labels = torch.full((B,), 0.9, device=device)   # label smoothing
            fake_labels = torch.zeros(B, device=device)

            d_real = D(day, mon, leap, dec, real_flat)
            d_real_loss = F.binary_cross_entropy_with_logits(d_real, real_labels)

            fake_soft = G(day, mon, leap, dec)               # (B,T,V) Gumbel-soft
            fake_flat = fake_soft.view(B, -1).detach()       # no G gradients
            d_fake = D(day, mon, leap, dec, fake_flat)
            d_fake_loss = F.binary_cross_entropy_with_logits(d_fake, fake_labels)

            d_loss = (d_real_loss + d_fake_loss) * 0.5
            d_loss.backward()
            opt_D.step()
            d_losses.append(d_loss.item())

            # ── Train Generator ───────────────────────────────────────────
            opt_G.zero_grad()
            fake_soft = G(day, mon, leap, dec)
            fake_flat = fake_soft.view(B, -1)
            d_fake_for_g = D(day, mon, leap, dec, fake_flat)
            g_loss = F.binary_cross_entropy_with_logits(
                d_fake_for_g, torch.ones(B, device=device)
            )
            g_loss.backward()
            opt_G.step()
            g_losses.append(g_loss.item())

        # ── Validation: validity rate ─────────────────────────────────────
        G.eval()
        n_valid, n_total = 0, 0
        with torch.no_grad():
            for day, mon, leap, dec, _, _ in val_dl:
                day  = day.to(device); mon  = mon.to(device)
                leap = leap.to(device); dec = dec.to(device)
                seqs = G.generate_tokens(day, mon, leap, dec)  # list of (T,) lists
                for b in range(day.size(0)):
                    toks = seqs[b]
                    date_str = tokens_to_date(toks)
                    d, m, l, decade = decode_conditions(
                        day[b].item(), mon[b].item(), leap[b].item(), dec[b].item()
                    )
                    if date_str and validate_date(date_str, d, m, l, decade):
                        n_valid += 1
                    n_total += 1

        validity   = n_valid / n_total if n_total > 0 else 0.0
        avg_g_loss = sum(g_losses) / len(g_losses)
        avg_d_loss = sum(d_losses) / len(d_losses)
        elapsed    = time.time() - t0
        history.append({"epoch": epoch, "g_loss": avg_g_loss,
                         "d_loss": avg_d_loss, "validity": validity})
        print(f"  Epoch {epoch:3d}/{epochs}  "
              f"G={avg_g_loss:.4f}  D={avg_d_loss:.4f}  "
              f"validity={validity:.3f}  ({elapsed:.1f}s)")

        if validity >= best_validity:
            best_validity = validity
            torch.save({
                "epoch": epoch,
                "G_state": G.state_dict(),
                "D_state": D.state_dict(),
                "history": history,
            }, best_path)
            print(f"    ✓ saved best GAN checkpoint (validity={best_validity:.3f})")

    print(f"\nGAN training complete. Best validity={best_validity:.3f}")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",        default="data/data.txt")
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--batch_size",  type=int,   default=512)
    parser.add_argument("--lr_g",        type=float, default=2e-4)
    parser.add_argument("--lr_d",        type=float, default=2e-4)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--weights_dir", default="model/weights")
    args = parser.parse_args()

    train_gan(
        data_path   = args.data,
        epochs      = args.epochs,
        batch_size  = args.batch_size,
        lr_g        = args.lr_g,
        lr_d        = args.lr_d,
        seed        = args.seed,
        weights_dir = args.weights_dir,
    )
