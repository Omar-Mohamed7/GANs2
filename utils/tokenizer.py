"""
Custom tokenizer for the Dates Generator problem.
Input conditions → tokens → model → output date tokens
"""

from __future__ import annotations
import re
from typing import List, Tuple, Optional
import datetime


# ── vocabulary ──────────────────────────────────────────────────────────────
DAYS   = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
          "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
DECADES = [str(d) for d in range(180, 221)]          # 180 … 220
LEAPS  = ["False", "True"]

MONTH_TO_NUM = {m: i+1 for i, m in enumerate(MONTHS)}
NUM_TO_MONTH = {v: k for k, v in MONTH_TO_NUM.items()}
DAY_TO_IDX   = {d: i for i, d in enumerate(DAYS)}
IDX_TO_DAY   = {v: k for k, v in DAY_TO_IDX.items()}

# Date digit vocabulary: digits 0-9 plus '-' separator
DATE_VOCAB   = [str(i) for i in range(10)] + ["-"]
DATE_PAD     = "<PAD>"
DATE_BOS     = "<BOS>"
DATE_EOS     = "<EOS>"
DATE_UNK     = "<UNK>"
DATE_SPECIAL = [DATE_PAD, DATE_BOS, DATE_EOS, DATE_UNK]
DATE_FULL_VOCAB = DATE_SPECIAL + DATE_VOCAB          # indices 0-14

D2I = {t: i for i, t in enumerate(DATE_FULL_VOCAB)}
I2D = {v: k for k, v in D2I.items()}

PAD_IDX = D2I[DATE_PAD]
BOS_IDX = D2I[DATE_BOS]
EOS_IDX = D2I[DATE_EOS]
UNK_IDX = D2I[DATE_UNK]

# Condition vocab sizes (for embedding tables)
N_DAYS    = len(DAYS)    # 7
N_MONTHS  = len(MONTHS)  # 12
N_LEAPS   = len(LEAPS)   # 2
N_DECADES = len(DECADES) # 41
N_DATE_TOKENS = len(DATE_FULL_VOCAB)  # 15

MAX_DATE_LEN = 10   # "dd-mm-yyyy" = 10 chars


# ── condition encoders ───────────────────────────────────────────────────────
def encode_conditions(day: str, month: str, leap: str, decade: str
                      ) -> Tuple[int, int, int, int]:
    """Return integer indices for each condition."""
    day_idx    = DAY_TO_IDX[day]
    month_idx  = MONTH_TO_NUM[month] - 1          # 0-based
    leap_idx   = 0 if leap == "False" else 1
    decade_idx = int(decade) - 180                # 0-based offset
    return day_idx, month_idx, leap_idx, decade_idx


def decode_conditions(day_idx: int, month_idx: int, leap_idx: int,
                      decade_idx: int) -> Tuple[str, str, str, str]:
    day    = IDX_TO_DAY[day_idx]
    month  = NUM_TO_MONTH[month_idx + 1]
    leap   = LEAPS[leap_idx]
    decade = str(decade_idx + 180)
    return day, month, leap, decade


# ── date encoders ────────────────────────────────────────────────────────────
def date_to_tokens(date_str: str) -> List[int]:
    """'3-12-1962' → list of token ids (char-level, with BOS/EOS)."""
    # Pad day and month to 2 digits, year to 4: canonical = dd-mm-yyyy
    parts = date_str.strip().split("-")
    d, m, y = parts
    canonical = f"{int(d):02d}-{int(m):02d}-{int(y):04d}"   # 10 chars always
    tokens = [BOS_IDX]
    for ch in canonical:
        tokens.append(D2I.get(ch, UNK_IDX))
    tokens.append(EOS_IDX)
    return tokens


def tokens_to_date(token_ids: List[int]) -> Optional[str]:
    """Convert token ids back to a date string 'dd-mm-yyyy'."""
    chars = []
    for tid in token_ids:
        tok = I2D.get(tid, DATE_UNK)
        if tok in DATE_SPECIAL:
            continue
        chars.append(tok)
    s = "".join(chars)
    if len(s) == 10 and s[2] == "-" and s[5] == "-":
        return s
    return None


# ── line parser ──────────────────────────────────────────────────────────────
LINE_RE = re.compile(
    r"\[(\w+)\]\s+\[(\w+)\]\s+\[(\w+)\]\s+\[(\d+)\](?:\s+(\S+))?"
)

def parse_line(line: str) -> Optional[dict]:
    m = LINE_RE.match(line.strip())
    if not m:
        return None
    day, month, leap, decade, date = m.groups()
    return {"day": day, "month": month, "leap": leap,
            "decade": decade, "date": date}


def format_output_line(day: str, month: str, leap: str, decade: str,
                        date_str: str) -> str:
    return f"[{day}] [{month}] [{leap}] [{decade}] {date_str}"


# ── date validation ──────────────────────────────────────────────────────────
def is_leap_year(year: int) -> bool:
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


DAYS_IN_MONTH = {
    1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
    7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31
}

WEEKDAY_NAMES = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]

def validate_date(date_str: str, day: str, month: str,
                  leap: str, decade: str) -> bool:
    """Return True if the date string satisfies all four conditions."""
    try:
        parts = date_str.strip().split("-")
        if len(parts) != 3:
            return False
        d, mo, y = int(parts[0]), int(parts[1]), int(parts[2])
        if not (1 <= d <= 31 and 1 <= mo <= 12 and 1800 <= y <= 2200):
            return False
        # month days
        max_d = DAYS_IN_MONTH[mo]
        if mo == 2 and is_leap_year(y):
            max_d = 29
        if d > max_d:
            return False
        dt = datetime.date(y, mo, d)
        # conditions
        if WEEKDAY_NAMES[dt.weekday()] != day:
            return False
        if NUM_TO_MONTH[mo] != month:
            return False
        actual_leap = "True" if is_leap_year(y) else "False"
        if actual_leap != leap:
            return False
        if str(y // 10) != decade:
            return False
        return True
    except Exception:
        return False
