"""
Model 3 – Conditional Transformer Decoder  (outside-course)
============================================================
Architecture
  The four condition tokens are projected to the transformer's model dimension
  and used as a "prefix" (4 conditioning tokens).  The decoder then auto-
  regressively generates the date characters using causal self-attention.

  Condition prefix  : [day_emb, month_emb, leap_emb, decade_emb]  → 4 tokens
  Decoder sequence  : [BOS, d, d, -, m, m, -, y, y, y, y]        → 11 tokens
  Combined sequence : prefix (4) + decoder (11) = 15 tokens total

  During training, we only compute the loss on the decoder part (positions 4-14).
  At inference we autoregressively append one character at a time.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
from utils.tokenizer import (
    N_DAYS, N_MONTHS, N_LEAPS, N_DECADES,
    N_DATE_TOKENS, MAX_DATE_LEN,
    BOS_IDX, EOS_IDX, PAD_IDX,
)

N_COND_TOKENS = 4  # prefix length


class ConditionalTransformerDecoder(nn.Module):
    def __init__(
        self,
        d_model: int    = 128,
        nhead: int      = 4,
        num_layers: int = 3,
        ffn_dim: int    = 256,
        dropout: float  = 0.1,
        max_seq_len: int = N_COND_TOKENS + MAX_DATE_LEN + 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_cond  = N_COND_TOKENS

        # Condition embeddings → d_model
        self.emb_day    = nn.Embedding(N_DAYS,    d_model)
        self.emb_month  = nn.Embedding(N_MONTHS,  d_model)
        self.emb_leap   = nn.Embedding(N_LEAPS,   d_model)
        self.emb_decade = nn.Embedding(N_DECADES, d_model)

        # Character embeddings for date tokens
        self.emb_char = nn.Embedding(N_DATE_TOKENS, d_model, padding_idx=PAD_IDX)

        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model, dropout, max_seq_len)

        # Transformer decoder stack (decoder-only = causal)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ffn_dim,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

        self.fc_out = nn.Linear(d_model, N_DATE_TOKENS)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ── build causal (look-ahead) mask ───────────────────────────────────────
    @staticmethod
    def _causal_mask(T: int, device) -> torch.Tensor:
        """Upper-triangular mask; True = ignore."""
        return torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()

    # ── forward ──────────────────────────────────────────────────────────────
    def forward(
        self,
        day_idx:  torch.Tensor,   # (B,)
        mon_idx:  torch.Tensor,
        leap_idx: torch.Tensor,
        dec_idx:  torch.Tensor,
        input_tokens: torch.Tensor,  # (B, T)  BOS + date chars
    ) -> torch.Tensor:
        """Returns logits (B, T, V) for the date-token positions only."""
        B, T = input_tokens.shape
        device = day_idx.device

        # Build condition prefix: (B, 4, d_model)
        cond_prefix = torch.stack([
            self.emb_day(day_idx),
            self.emb_month(mon_idx),
            self.emb_leap(leap_idx),
            self.emb_decade(dec_idx),
        ], dim=1)   # (B, 4, d_model)

        # Build date token embeddings: (B, T, d_model)
        char_emb = self.emb_char(input_tokens)

        # Concatenate prefix + date: (B, 4+T, d_model)
        seq = torch.cat([cond_prefix, char_emb], dim=1)
        seq = self.pos_enc(seq)

        # Causal mask for full sequence
        full_T = N_COND_TOKENS + T
        mask   = self._causal_mask(full_T, device)

        out = self.transformer(seq, mask=mask)  # (B, 4+T, d_model)

        # Return logits only for the date-token positions
        date_out = out[:, N_COND_TOKENS:, :]    # (B, T, d_model)
        return self.fc_out(date_out)             # (B, T, V)

    # ── autoregressive generation ─────────────────────────────────────────
    @torch.no_grad()
    def generate(
        self,
        day_idx:  torch.Tensor,
        mon_idx:  torch.Tensor,
        leap_idx: torch.Tensor,
        dec_idx:  torch.Tensor,
        max_len:  int = MAX_DATE_LEN + 2,
    ) -> list[list[int]]:
        B = day_idx.size(0)
        device = day_idx.device
        generated = torch.full((B, 1), BOS_IDX, dtype=torch.long, device=device)
        sequences = [[] for _ in range(B)]
        done = [False] * B

        for _ in range(max_len):
            logits = self.forward(
                day_idx, mon_idx, leap_idx, dec_idx, generated
            )                            # (B, T_cur, V)
            next_tok = logits[:, -1, :].argmax(dim=-1)   # (B,)
            for b in range(B):
                if not done[b]:
                    tok = next_tok[b].item()
                    if tok == EOS_IDX:
                        done[b] = True
                    else:
                        sequences[b].append(tok)
            if all(done):
                break
            generated = torch.cat(
                [generated, next_tok.unsqueeze(1)], dim=1
            )
        return sequences


# ── Positional Encoding ───────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 64):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
