"""
Generate the calendar dataset used by the date models.

Outputs:
    data/data.txt
    data/example_input.txt
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.tokenizer import MONTHS, WEEKDAY_NAMES, is_leap_year


def format_condition_line(dt: datetime.date, include_date: bool) -> str:
    day = WEEKDAY_NAMES[dt.weekday()]
    month = MONTHS[dt.month - 1]
    leap = "True" if is_leap_year(dt.year) else "False"
    decade = str(dt.year // 10)
    prefix = f"[{day}] [{month}] [{leap}] [{decade}]"
    if include_date:
        return f"{prefix} {dt:%d-%m-%Y}"
    return prefix


def generate_dataset(output_dir: str, start_year: int, end_year: int,
                     example_count: int) -> tuple[str, str, int]:
    os.makedirs(output_dir, exist_ok=True)
    data_path = os.path.join(output_dir, "data.txt")
    example_path = os.path.join(output_dir, "example_input.txt")

    rows = []
    current = datetime.date(start_year, 1, 1)
    final = datetime.date(end_year, 12, 31)
    step = datetime.timedelta(days=1)

    while current <= final:
        rows.append(format_condition_line(current, include_date=True))
        current += step

    with open(data_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(row + "\n")

    if example_count > len(rows):
        example_count = len(rows)
    stride = max(1, len(rows) // example_count)
    example_rows = rows[::stride][:example_count]

    with open(example_path, "w", encoding="utf-8") as f:
        for row in example_rows:
            f.write(row.rsplit(" ", 1)[0] + "\n")

    return data_path, example_path, len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate calendar data files.")
    parser.add_argument("--output_dir", default=os.path.join(PROJECT_ROOT, "data"))
    parser.add_argument("--start_year", type=int, default=1800)
    parser.add_argument("--end_year", type=int, default=2200)
    parser.add_argument("--example_count", type=int, default=32)
    args = parser.parse_args()

    data_path, example_path, count = generate_dataset(
        output_dir=args.output_dir,
        start_year=args.start_year,
        end_year=args.end_year,
        example_count=args.example_count,
    )
    print(f"Wrote {count} rows to {data_path}")
    print(f"Wrote example inputs to {example_path}")


if __name__ == "__main__":
    main()
