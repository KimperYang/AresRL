#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_INPUT_PATH = (
    "/home/azureuser/cloudfiles/code/Users/jingbo.yang/TBAMA/rl/data/"
    "apigen_with_instruction_full.json.with_gt.filtered.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help="Input JSON file.")
    parser.add_argument("--index", type=int, default=0, help="Index of the record to print.")
    parser.add_argument(
        "--random",
        action="store_true",
        help="Print a random record instead of --index.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indentation for pretty-printed JSON.",
    )
    return parser.parse_args()


def load_records(path: str) -> List[Dict[str, Any]]:
    with Path(path).open() as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected input JSON to be a list of records.")
    return data


def select_index(total: int, args: argparse.Namespace) -> int:
    if total <= 0:
        raise ValueError("Input JSON is empty.")
    if args.random:
        rng = random.Random(args.seed)
        return rng.randrange(total)
    if args.index < 0:
        return max(0, total + args.index)
    return min(args.index, total - 1)


def main() -> None:
    args = parse_args()
    records = load_records(args.input)
    idx = select_index(len(records), args)
    print(json.dumps(records[idx], ensure_ascii=True, indent=args.indent))


if __name__ == "__main__":
    main()
