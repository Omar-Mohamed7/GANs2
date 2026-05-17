"""
Model 1 – Conditional Seq2Seq LSTM  (in-course architecture)
============================================================
Architecture
  Encoder : four separate embedding tables for (day, month, leap, decade).
            Their vectors are concatenated → linear projection → encoder hidden state.
  Decoder : single-layer LSTM that generates the date character by character.
            At each step the decoder input is the previous character token embedding
            concatenated with a condition context vector (from the encoder).

Formulation
  Input  : (day, month, leap, decade) conditions
  Output : char-level token sequence "dd-mm-yyyy"  (always 10 chars)
  Loss   : cross-entropy over the 10 output positions (+ EOS), ignoring PAD.
  Train  : teacher forcing.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from utils.tokenizer import (
    N_DAYS, N_MONTHS, N_LEAPS, N_DECADES,
    N_DATE_TOKENS, BOS_IDX, EOS_IDX, PAD_IDX, MAX_DATE_LEN
)


class ConditionalSeq2SeqLSTM(nn.Module):
    def __init__(
        self,
        embed_day: int    = 16,
        embed_month: int  = 16,
        embed_leap: int   = 4,
        embed_decade: int = 32,
        embed_char: int   = 32,
        hidden: int       = 256,
        dropout: float    = 0.2,
    ):
        super().__init__()
        cond_dim = embed_day + embed_month + embed_leap + embed_decade

        # Condition embeddings
        self.emb_day    = nn.Embedding(N_DAYS,    embed_day)
        self.emb_month  = nn.Embedding(N_MONTHS,  embed_month)
        self.emb_leap   = nn.Embedding(N_LEAPS,   embed_leap)
        self.emb_decade = nn.Embedding(N_DECADES, embed_decade)

        # Project condition vector → LSTM h0, c0
        self.cond2h = nn.Linear(cond_dim, hidden)
        self.cond2c = nn.Linear(cond_dim, hidden)

        # Character embedding for decoder
        self.emb_char = nn.Embedding(N_DATE_TOKENS, embed_char, padding_idx=PAD_IDX)

        # Decoder LSTM: input = char_emb cat cond_ctx
        self.decoder = nn.LSTM(
            input_size=embed_char + cond_dim,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc_out  = nn.Linear(hidden, N_DATE_TOKENS)

    # ── condition encoding ────────────────────────────────────────────────
    def encode_conditions(
        self,
        day_idx: torch.Tensor,   # (B,)
        mon_idx: torch.Tensor,
        leap_idx: torch.Tensor,
        dec_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (cond_ctx, h0, c0); shapes (B, cond_dim) and (1,B,hidden)."""
        e_day   = self.emb_day(day_idx)
        e_month = self.emb_month(mon_idx)
        e_leap  = self.emb_leap(leap_idx)
        e_dec   = self.emb_decade(dec_idx)
        ctx = torch.cat([e_day, e_month, e_leap, e_dec], dim=-1)   # (B, cond_dim)
        h0  = torch.tanh(self.cond2h(ctx)).unsqueeze(0)            # (1, B, H)
        c0  = torch.tanh(self.cond2c(ctx)).unsqueeze(0)
        return ctx, h0, c0

    # ── forward (teacher forcing) ─────────────────────────────────────────
    def forward(
        self,
        day_idx:  torch.Tensor,  # (B,)
        mon_idx:  torch.Tensor,
        leap_idx: torch.Tensor,
        dec_idx:  torch.Tensor,
        input_tokens: torch.Tensor,   # (B, T) BOS + date chars
    ) -> torch.Tensor:
        """Returns logits (B, T, V)."""
        ctx, h, c = self.encode_conditions(day_idx, mon_idx, leap_idx, dec_idx)
        B, T = input_tokens.shape

        char_emb = self.emb_char(input_tokens)            # (B, T, embed_char)
        ctx_rep  = ctx.unsqueeze(1).expand(B, T, -1)      # (B, T, cond_dim)
        dec_in   = torch.cat([char_emb, ctx_rep], dim=-1) # (B, T, embed_char+cond_dim)

        out, _ = self.decoder(dec_in, (h, c))             # (B, T, H)
        logits  = self.fc_out(self.dropout(out))          # (B, T, V)
        return logits

    # ── greedy autoregressive decode ───────────────────────────────────────
    @torch.no_grad()
    def generate(
        self,
        day_idx:  torch.Tensor,
        mon_idx:  torch.Tensor,
        leap_idx: torch.Tensor,
        dec_idx:  torch.Tensor,
        max_len:  int = MAX_DATE_LEN + 2,
    ) -> list[list[int]]:
        """Greedy decoding, returns list of token-id lists per sample."""
        ctx, h, c = self.encode_conditions(day_idx, mon_idx, leap_idx, dec_idx)
        B = day_idx.size(0)
        device = day_idx.device

        prev = torch.full((B, 1), BOS_IDX, dtype=torch.long, device=device)
        sequences = [[] for _ in range(B)]
        done = [False] * B

        for _ in range(max_len):
            char_emb = self.emb_char(prev)                        # (B,1,E)
            ctx_rep  = ctx.unsqueeze(1)                           # (B,1,C)
            dec_in   = torch.cat([char_emb, ctx_rep], dim=-1)
            out, (h, c) = self.decoder(dec_in, (h, c))
            logits = self.fc_out(out.squeeze(1))                  # (B, V)
            pred   = logits.argmax(dim=-1)                        # (B,)

            for b in range(B):
                if not done[b]:
                    tok = pred[b].item()
                    if tok == EOS_IDX:
                        done[b] = True
                    else:
                        sequences[b].append(tok)
            if all(done):
                break
            prev = pred.unsqueeze(1)

        return sequences
