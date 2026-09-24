"""Convert a knowledge-base CSV export to a JSON document, not a questions file."""

import argparse
import csv
import json
from pathlib import Path


def convert(source: Path, target: Path) -> int:
    with source.open(encoding="utf-8-sig", newline="") as file:
        rows = [{key: value for key, value in row.items() if key} for row in csv.DictReader(file)]
    with target.open("x", encoding="utf-8") as file:
        json.dump(rows, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return len(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument(
        "target", type=Path, help="New output path; existing files are not overwritten"
    )
    args = parser.parse_args()
    print(f"Converted {convert(args.source, args.target)} document records.")
