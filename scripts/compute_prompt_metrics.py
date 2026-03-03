#!/usr/bin/env python3
import argparse
import json
import os
from typing import Any, Dict, List, Optional


DEFAULT_INPUT = "outputs/router_rollouts/sft_router_verify.actions_agree.both_success.jsonl"
DEFAULT_OUTPUT = "outputs/router_rollouts/sft_router_verify.actions_agree.both_success.metrics.jsonl"

STEP_COST = {
    "low": 0.02,
    "medium": 0.06,
    "high": 0.12,
}

AIRLINE_TOOLS = {
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

RETAIL_TOOLS = {
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


def _rollout_cost(obj: Dict[str, Any]) -> Optional[float]:
    efforts = obj.get("efforts")
    if isinstance(efforts, list) and efforts:
        total = 0.0
        for e in efforts:
            c = STEP_COST.get(str(e).lower())
            if c is None:
                continue
            total += c
        return total

    # Fallback to turn_records if efforts missing.
    total = 0.0
    found = False
    for rec in obj.get("turn_records", []) or []:
        effort = rec.get("effort")
        c = STEP_COST.get(str(effort).lower()) if effort is not None else None
        if c is None:
            continue
        total += c
        found = True
    return total if found else None


def _is_success(obj: Dict[str, Any], method: str) -> bool:
    v = obj.get("verify")
    if not isinstance(v, dict):
        return False
    block = v.get(method)
    if not isinstance(block, dict):
        return False
    return block.get("success") is True


def _infer_domain_from_trajectory(traj: Any) -> str:
    tool_names = set()
    for item in traj or []:
        if not isinstance(item, dict):
            continue
        if item.get("from") != "function_call":
            continue
        raw = item.get("value")
        if isinstance(raw, str):
            try:
                payload = json.loads(raw)
            except Exception:
                payload = {}
        elif isinstance(raw, dict):
            payload = raw
        else:
            payload = {}
        name = payload.get("name")
        if isinstance(name, str) and name:
            tool_names.add(name)
    if tool_names & AIRLINE_TOOLS:
        return "airline"
    if tool_names & RETAIL_TOOLS:
        return "retail"
    return "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input rollouts JSONL.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output metrics JSONL.")
    parser.add_argument(
        "--success-method",
        default="tau_bench",
        choices=("tau_bench", "llm_judge"),
        help="Which verify method defines success for SR and variance.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Group by index
    groups: Dict[int, Dict[str, Any]] = {}
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            idx = obj.get("index")
            if not isinstance(idx, int):
                continue
            g = groups.setdefault(idx, {"rollouts": [], "instruction": None})
            g["rollouts"].append(obj)
            if g["instruction"] is None and isinstance(obj.get("instruction"), str):
                g["instruction"] = obj.get("instruction")
            if g.get("domain") in (None, "unknown"):
                domain = _infer_domain_from_trajectory(obj.get("trajectory"))
                g["domain"] = domain

    with open(args.output, "w", encoding="utf-8") as out_f:
        for idx, g in sorted(groups.items()):
            rollouts = g["rollouts"]
            total_rollouts = len(rollouts)
            # SR
            success_flags = [(_is_success(r, args.success_method)) for r in rollouts]
            success_count = sum(1 for s in success_flags if s)
            sr = success_count / 8.0

            # Step Cost (avg cost across all rollouts)
            costs: List[float] = []
            success_costs: List[float] = []
            for r, s in zip(rollouts, success_flags):
                cost = _rollout_cost(r)
                if cost is None:
                    continue
                costs.append(cost)
                if s:
                    success_costs.append(cost)

            avg_cost = sum(costs) / len(costs) if costs else None

            # Cost variance among successful rollouts (population variance).
            if len(success_costs) >= 2:
                mean = sum(success_costs) / len(success_costs)
                var = sum((c - mean) ** 2 for c in success_costs) / len(success_costs)
            else:
                var = None

            out_f.write(
                json.dumps(
                    {
                        "index": idx,
                        "instruction": g.get("instruction"),
                        "domain": g.get("domain") or "unknown",
                        "rollouts": total_rollouts,
                        "success_method": args.success_method,
                        "success_count": success_count,
                        "sr": sr,
                        "avg_step_cost": avg_cost,
                        "success_cost_variance": var,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
