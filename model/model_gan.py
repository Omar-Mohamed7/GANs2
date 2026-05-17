"""
Model 2 – Conditional GAN  (in-course, required)
=================================================
Architecture
  Generator   : takes a noise vector z and the condition embeddings → MLP → 10
                logits (one per date character position), each over the char vocab.
                Uses Gumbel-Softmax (straight-through) so the output is
                differentiable yet discrete-like.
  Discriminator: takes the condition embeddings + a flat one-hot date encoding
                → MLP → real/fake score.

Training scheme : standard GAN minimax with label smoothing.

Note : the Generator noise makes this a *generative* model — the same conditions
       can produce different valid dates, which is exactly what the problem asks for.
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


# ── shared condition encoder (used by both G and D) ──────────────────────────
class ConditionEncoder(nn.Module):
    def __init__(
        self,
        embed_day: int    = 16,
        embed_month: int  = 16,
        embed_leap: int   = 4,
        embed_decade: int = 32,
    ):
        super().__init__()
        self.emb_day    = nn.Embedding(N_DAYS,    embed_day)
        self.emb_month  = nn.Embedding(N_MONTHS,  embed_month)
        self.emb_leap   = nn.Embedding(N_LEAPS,   embed_leap)
        self.emb_decade = nn.Embedding(N_DECADES, embed_decade)
        self.out_dim = embed_day + embed_month + embed_leap + embed_decade

    def forward(self, day, month, leap, decade):
        return torch.cat([
            self.emb_day(day),
            self.emb_month(month),
            self.emb_leap(leap),
            self.emb_decade(decade),
        ], dim=-1)   # (B, out_dim)


# ── Generator ────────────────────────────────────────────────────────────────
class Generator(nn.Module):
    """
    z (noise) + condition → Gumbel-Softmax date tokens
    Output shape: (B, MAX_DATE_LEN, N_DATE_TOKENS)  — soft one-hots
    """

    def __init__(
        self,
        noise_dim: int = 64,
        hidden: int    = 256,
        cond_enc: ConditionEncoder = None,
        tau: float = 1.0,
    ):
        super().__init__()
        self.noise_dim = noise_dim
        self.tau = tau
        self.cond_enc = cond_enc or ConditionEncoder()
        in_dim = noise_dim + self.cond_enc.out_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, MAX_DATE_LEN * N_DATE_TOKENS),
        )

    def forward(self, day, month, leap, decade, z: torch.Tensor | None = None):
        B = day.size(0)
        if z is None:
            z = torch.randn(B, self.noise_dim, device=day.device)
        cond = self.cond_enc(day, month, leap, decade)   # (B, cond_dim)
        x    = torch.cat([z, cond], dim=-1)              # (B, noise+cond)
        out  = self.net(x)                               # (B, T*V)
        logits = out.view(B, MAX_DATE_LEN, N_DATE_TOKENS)  # (B, T, V)
        # Gumbel-Softmax straight-through for differentiable discrete output
        soft = F.gumbel_softmax(logits, tau=self.tau, hard=False)
        return soft  # (B, T, V)

    @torch.no_grad()
    def generate_tokens(self, day, month, leap, decade) -> list[list[int]]:
        """Hard argmax decode — used at inference."""
        soft = self.forward(day, month, leap, decade)
        ids = soft.argmax(dim=-1)   # (B, T)
        return ids.tolist()


# ── Discriminator ─────────────────────────────────────────────────────────────
class Discriminator(nn.Module):
    """
    Condition + flat date (one-hot or soft) → real/fake logit.
    """

    def __init__(self, hidden: int = 256, cond_enc: ConditionEncoder = None):
        super().__init__()
        self.cond_enc = cond_enc or ConditionEncoder()
        date_dim = MAX_DATE_LEN * N_DATE_TOKENS
        in_dim   = self.cond_enc.out_dim + date_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, day, month, leap, decade, date_flat: torch.Tensor):
        """date_flat: (B, T*V) — either one-hot or soft from G."""
        cond  = self.cond_enc(day, month, leap, decade)
        x     = torch.cat([cond, date_flat], dim=-1)
        return self.net(x).squeeze(-1)   # (B,)


# ── convenience: make real date tensor ───────────────────────────────────────
def make_real_date_tensor(target_tokens: torch.Tensor) -> torch.Tensor:
    """
    target_tokens : (B, T)  — token ids for the date chars (no BOS, no EOS)
                              T = MAX_DATE_LEN = 10
    Returns (B, T*V) one-hot float.
    """
    B, T = target_tokens.shape
    oh   = F.one_hot(target_tokens, num_classes=N_DATE_TOKENS).float()  # (B,T,V)
    return oh.view(B, -1)   # (B, T*V)
