#!/usr/bin/env python3
import argparse
import json
import math
import os
from typing import Any, Dict, List, Tuple


DEFAULT_INPUT = "outputs/router_rollouts/sft_router_verify.actions_agree.both_success.metrics.jsonl"
DEFAULT_OUTPUT = (
    "outputs/router_rollouts/sft_router_verify.actions_agree.both_success.metrics.filtered.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input metrics JSONL.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output filtered JSONL.")
    parser.add_argument(
        "--top_pct",
        type=float,
        default=30.0,
        help="Top percentile for SR=1.0 prompts within each domain (default 30).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    records: List[Dict[str, Any]] = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    # Split by domain
    by_domain: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        domain = rec.get("domain") or "unknown"
        by_domain.setdefault(domain, []).append(rec)

    kept: List[Dict[str, Any]] = []
    stats = {}
    top_frac = max(0.0, min(1.0, args.top_pct / 100.0))

    for domain, items in by_domain.items():
        if domain == "unknown":
            stats[domain] = {
                "total": len(items),
                "sr_lt_1": 0,
                "sr_eq_1": 0,
                "sr_eq_1_with_var": 0,
                "sr_eq_1_kept_top": 0,
                "kept_total": 0,
                "skipped": True,
            }
            continue
        sr_one = []
        sr_lt = []
        for rec in items:
            sr = rec.get("sr")
            if isinstance(sr, (int, float)) and sr >= 1.0:
                sr_one.append(rec)
            else:
                sr_lt.append(rec)

        # Rank SR=1.0 by success_cost_variance (descending). Ignore None.
        sr_one_with_var = [
            rec for rec in sr_one if isinstance(rec.get("success_cost_variance"), (int, float))
        ]
        sr_one_with_var.sort(key=lambda r: r["success_cost_variance"], reverse=True)

        top_n = int(math.ceil(len(sr_one_with_var) * top_frac))
        top_n = max(0, min(len(sr_one_with_var), top_n))

        kept_domain = sr_lt + sr_one_with_var[:top_n]
        kept.extend(kept_domain)

        stats[domain] = {
            "total": len(items),
            "sr_lt_1": len(sr_lt),
            "sr_eq_1": len(sr_one),
            "sr_eq_1_with_var": len(sr_one_with_var),
            "sr_eq_1_kept_top": top_n,
            "kept_total": len(kept_domain),
        }

    with open(args.output, "w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(json.dumps({"input": args.input, "output": args.output, "stats": stats}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
