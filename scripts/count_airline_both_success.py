#!/usr/bin/env python3
import argparse
import json
import os
from typing import Any

AIRLINE_ONLY_TOOLS = {
    "book_reservation",
    "cancel_reservation",
    "get_reservation_details",
    "list_all_airports",
    "search_direct_flight",
    "search_onestop_flight",
    "send_certificate",
    "update_reservation_baggages",
    "update_reservation_flights",
    "update_reservation_passengers",
}

RETAIL_ONLY_TOOLS = {
    "cancel_pending_order",
    "exchange_delivered_order_items",
    "find_user_id_by_email",
    "find_user_id_by_name_zip",
    "get_order_details",
    "get_product_details",
    "list_all_product_types",
    "modify_pending_order_address",
    "modify_pending_order_items",
    "modify_pending_order_payment",
    "modify_user_address",
    "return_delivered_order_items",
}


def _infer_domain(tools: Any) -> str:
    names = set()
    if isinstance(tools, str):
        try:
            tools = json.loads(tools)
        except Exception:
            tools = []
    if isinstance(tools, list):
        for item in tools:
            if not isinstance(item, dict):
                continue
            func = item.get("function") if "function" in item else item
            if isinstance(func, dict):
                name = func.get("name")
                if name:
                    names.add(name)
    if names & AIRLINE_ONLY_TOOLS:
        return "airline"
    if names & RETAIL_ONLY_TOOLS:
        return "retail"
    return "airline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jsonl",
        default="outputs/router_rollouts/rollout_verify.jsonl",
        help="rollout_verify.jsonl path",
    )
    parser.add_argument(
        "--dataset",
        default="data/apigen_with_instruction_full.json.with_gt.filtered.json",
        help="dataset JSON path",
    )
    parser.add_argument(
        "--domain",
        choices=("airline", "retail"),
        default="airline",
        help="Which domain to count.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output JSON file of filtered tasks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.jsonl):
        raise FileNotFoundError(args.jsonl)
    if not os.path.exists(args.dataset):
        raise FileNotFoundError(args.dataset)

    with open(args.dataset, "r", encoding="utf-8") as f:
        data = json.load(f)

    target_domain = args.domain
    domain_indices = {
        i for i, rec in enumerate(data) if _infer_domain(rec.get("tools", [])) == target_domain
    }

    success_any = set()
    with open(args.jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            idx = rec.get("index")
            if not isinstance(idx, int) or idx not in domain_indices:
                continue
            verify = rec.get("verify") or {}
            llm = (verify.get("llm_judge") or {}).get("success")
            tau = (verify.get("tau_bench") or {}).get("success")
            if llm is True and tau is True:
                success_any.add(idx)

    print(f"{target_domain} tasks total", len(domain_indices))
    print("tasks with >=1 rollout both success", len(success_any))

    if args.output:
        filtered = [data[idx] for idx in sorted(success_any)]
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(filtered, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
