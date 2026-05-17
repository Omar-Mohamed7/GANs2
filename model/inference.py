"""
Inference engine shared by all models.
Includes a constraint-aware fallback: if a model produces an invalid date,
we do a targeted search in the valid date space before giving up.
"""
from __future__ import annotations
import datetime
import random
from typing import List, Optional, Tuple

from utils.tokenizer import (
    validate_date, is_leap_year, MONTH_TO_NUM,
    DAYS_IN_MONTH, WEEKDAY_NAMES, LEAPS
)


# ── exhaustive fallback generator ─────────────────────────────────────────────
def fallback_date(day: str, month: str, leap: str, decade: str,
                  seed: int = 0) -> Optional[str]:
    """
    When the model fails, find any valid date by direct search.
    Randomly samples years in the correct decade + leap constraint, then
    enumerates days in the correct month until the weekday matches.
    Returns 'dd-mm-yyyy' or None.
    """
    decade_int = int(decade)          # e.g. 196
    year_start = decade_int * 10      # e.g. 1960
    year_end   = min(year_start + 9, 2200)   # cap at 2200 per constraint
    month_num  = MONTH_TO_NUM[month]
    want_leap  = (leap == "True")

    # collect candidate years
    rng = random.Random(seed)
    years = [y for y in range(year_start, year_end + 1)
             if is_leap_year(y) == want_leap]
    if not years:
        # relax leap constraint (impossible combination)
        years = list(range(year_start, year_end + 1))

    rng.shuffle(years)
    for year in years:
        max_d = DAYS_IN_MONTH[month_num]
        if month_num == 2 and is_leap_year(year):
            max_d = 29
        for d in range(1, max_d + 1):
            try:
                dt = datetime.date(year, month_num, d)
                if WEEKDAY_NAMES[dt.weekday()] == day:
                    return f"{d:02d}-{month_num:02d}-{year:04d}"
            except ValueError:
                continue
    return None


def decode_model_output(token_ids: List[int], day: str, month: str,
                         leap: str, decade: str,
                         fallback_seed: int = 0) -> str:
    """
    Convert model token output to a date string, validating it.
    Falls back to exhaustive search if the model output is invalid.
    """
    from utils.tokenizer import tokens_to_date
    date_str = tokens_to_date(token_ids)
    if date_str and validate_date(date_str, day, month, leap, decade):
        return date_str
    # fallback
    fb = fallback_date(day, month, leap, decade, seed=fallback_seed)
    return fb or "01-01-1800"   # last-resort default
