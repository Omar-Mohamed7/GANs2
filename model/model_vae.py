"""
Model 4 – Conditional Variational Autoencoder  (outside-course)
===============================================================
Architecture
  Encoder  : takes the real date (as char embeddings) + condition embeddings
             → MLP → μ and log σ² in a latent space of dimension `latent_dim`.
  Decoder  : takes a sampled latent z + condition embeddings
             → MLP → logits for each of the 10 date character positions.

Training loss : reconstruction loss (cross-entropy per char) + KL divergence.

At inference the encoder is discarded; z is sampled from N(0,I) and the
decoder generates a date from the conditions alone — making this a proper
generative model that can produce diverse valid dates.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.tokenizer import (
    N_DAYS, N_MONTHS, N_LEAPS, N_DECADES,
    N_DATE_TOKENS, MAX_DATE_LEN,
    BOS_IDX, EOS_IDX, PAD_IDX,
)


# ── shared condition encoder ──────────────────────────────────────────────────
class CondEmb(nn.Module):
    def __init__(self, embed_day=16, embed_month=16, embed_leap=4, embed_decade=32):
        super().__init__()
        self.emb_day    = nn.Embedding(N_DAYS,    embed_day)
        self.emb_month  = nn.Embedding(N_MONTHS,  embed_month)
        self.emb_leap   = nn.Embedding(N_LEAPS,   embed_leap)
        self.emb_decade = nn.Embedding(N_DECADES, embed_decade)
        self.out_dim    = embed_day + embed_month + embed_leap + embed_decade

    def forward(self, day, month, leap, decade):
        return torch.cat([
            self.emb_day(day), self.emb_month(month),
            self.emb_leap(leap), self.emb_decade(decade),
        ], dim=-1)


class ConditionalVAE(nn.Module):
    def __init__(
        self,
        latent_dim: int  = 64,
        hidden: int      = 256,
        embed_char: int  = 32,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_emb   = CondEmb()
        cond_dim        = self.cond_emb.out_dim

        # Date character embeddings (for encoder)
        self.emb_char = nn.Embedding(N_DATE_TOKENS, embed_char, padding_idx=PAD_IDX)
        date_flat_dim = MAX_DATE_LEN * embed_char

        # ── Encoder q(z | x, c) ──────────────────────────────────────────
        enc_in = date_flat_dim + cond_dim
        self.encoder = nn.Sequential(
            nn.Linear(enc_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
        )
        self.fc_mu     = nn.Linear(hidden // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden // 2, latent_dim)

        # ── Decoder p(x | z, c) ──────────────────────────────────────────
        dec_in = latent_dim + cond_dim
        self.decoder = nn.Sequential(
            nn.Linear(dec_in, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, MAX_DATE_LEN * N_DATE_TOKENS),
        )

    # ── reparameterization trick ─────────────────────────────────────────────
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # ── forward (training) ───────────────────────────────────────────────────
    def forward(
        self,
        day, month, leap, decade,
        date_tokens: torch.Tensor,   # (B, MAX_DATE_LEN) — date chars only, no BOS/EOS
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (logits, mu, logvar).
        logits : (B, MAX_DATE_LEN, N_DATE_TOKENS)
        """
        cond = self.cond_emb(day, month, leap, decade)  # (B, cond_dim)

        # Encode
        char_emb  = self.emb_char(date_tokens)          # (B, T, E)
        B, T, E   = char_emb.shape
        date_flat = char_emb.view(B, T * E)             # (B, T*E)
        enc_in    = torch.cat([date_flat, cond], dim=-1)
        h         = self.encoder(enc_in)
        mu        = self.fc_mu(h)
        logvar    = self.fc_logvar(h)
        z         = self.reparameterize(mu, logvar)

        # Decode
        logits = self._decode(z, cond)
        return logits, mu, logvar

    def _decode(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        dec_in = torch.cat([z, cond], dim=-1)
        out    = self.decoder(dec_in)
        B      = z.size(0)
        return out.view(B, MAX_DATE_LEN, N_DATE_TOKENS)

    # ── VAE loss ─────────────────────────────────────────────────────────────
    @staticmethod
    def loss(logits: torch.Tensor, targets: torch.Tensor,
             mu: torch.Tensor, logvar: torch.Tensor,
             beta: float = 1.0) -> tuple[torch.Tensor, dict]:
        """
        logits  : (B, T, V)
        targets : (B, T)  — integer class indices
        """
        B, T, V = logits.shape
        recon = F.cross_entropy(
            logits.reshape(B * T, V), targets.reshape(B * T),
            ignore_index=PAD_IDX
        )
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total = recon + beta * kl
        return total, {"recon": recon.item(), "kl": kl.item()}

    # ── prior sampling (inference) ────────────────────────────────────────────
    @torch.no_grad()
    def generate_tokens(self, day, month, leap, decade) -> list[list[int]]:
        """Sample z ~ N(0,I), decode, argmax → token ids."""
        B    = day.size(0)
        z    = torch.randn(B, self.latent_dim, device=day.device)
        cond = self.cond_emb(day, month, leap, decade)
        logits = self._decode(z, cond)          # (B, T, V)
        ids    = logits.argmax(dim=-1)          # (B, T)
        return ids.tolist()
