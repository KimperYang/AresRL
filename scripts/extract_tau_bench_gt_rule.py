#!/usr/bin/env python3
import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_INPUT_PATH = "data/apigen_with_instruction_full.json.with_gt.filtered.json"


def _safe_json_loads(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return None
    return value


def _is_failed_observation(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, dict):
        lower_keys = {str(k).lower() for k in value.keys()}
        if any(k in lower_keys for k in ("error", "exception", "traceback")):
            return True
        status = value.get("status")
        if isinstance(status, str):
            status_l = status.lower()
            if any(tok in status_l for tok in ("error", "exception", "traceback", "failed", "failure")):
                return True
        # Fall through to string check for nested text.
        value = json.dumps(value, ensure_ascii=True)
    elif isinstance(value, list):
        value = json.dumps(value, ensure_ascii=True)
    elif not isinstance(value, str):
        value = str(value)

    text = value.lower()
    return any(tok in text for tok in ("error", "exception", "traceback", "failed", "failure"))


def _parse_function_call(value: Any) -> Optional[Tuple[str, Dict[str, Any]]]:
    payload = _safe_json_loads(value)
    if not isinstance(payload, dict):
        return None
    name = payload.get("name")
    if not isinstance(name, str) or not name:
        return None
    arguments = payload.get("arguments")
    if arguments is None and "kwargs" in payload:
        arguments = payload.get("kwargs")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except Exception:
            arguments = {}
    if not isinstance(arguments, dict):
        arguments = {}
    return name, arguments


def extract_actions_from_conversation(conversation: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    convo = list(conversation)
    actions: List[Dict[str, Any]] = []
    for idx, turn in enumerate(convo):
        if turn.get("from") != "function_call":
            continue
        parsed = _parse_function_call(turn.get("value"))
        if not parsed:
            continue
        name, arguments = parsed
        if name == "think":
            continue
        obs = None
        if idx + 1 < len(convo) and convo[idx + 1].get("from") == "observation":
            obs = convo[idx + 1].get("value")
        if _is_failed_observation(obs):
            continue
        actions.append({"name": name, "arguments": arguments})
    return actions


def normalize_actions(raw_actions: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw_actions, list):
        return []
    normalized: List[Dict[str, Any]] = []
    for action in raw_actions:
        if not isinstance(action, dict):
            continue
        name = action.get("name")
        if not isinstance(name, str):
            continue
        arguments = action.get("arguments")
        if arguments is None and "kwargs" in action:
            arguments = action.get("kwargs")
        if not isinstance(arguments, dict):
            arguments = {}
        normalized.append({"name": name, "arguments": arguments})
    return normalized


def _canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _canonicalize(value[k]) for k in sorted(value.keys())}
    if isinstance(value, list):
        return [_canonicalize(v) for v in value]
    return value


def _canonical_actions(actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{"name": a.get("name"), "arguments": _canonicalize(a.get("arguments", {}))} for a in actions]


def load_records(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected input JSON to be a list of records.")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help="Input JSON file.")
    parser.add_argument("--output", default="", help="Optional output JSONL path for rule actions.")
    parser.add_argument("--start", type=int, default=0, help="Start index (inclusive).")
    parser.add_argument("--end", type=int, default=-1, help="End index (exclusive). -1 means end of file.")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare rule actions against existing record['actions'] and print summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_records(args.input)
    end = len(records) if args.end == -1 else min(len(records), args.end)
    start = max(0, args.start)
    if start >= end:
        raise ValueError("Invalid start/end range.")

    out_f = None
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        out_f = open(args.output, "w", encoding="utf-8")

    stats = {
        "total": 0,
        "exact_match": 0,
        "same_names": 0,
        "len_match": 0,
        "both_empty": 0,
        "gpt_empty_rule_nonempty": 0,
        "rule_empty_gpt_nonempty": 0,
        "args_mismatch": 0,
        "names_mismatch_same_len": 0,
        "len_mismatch": 0,
    }

    for idx in range(start, end):
        record = records[idx]
        conversation = record.get("conversations") or []
        if not isinstance(conversation, list):
            conversation = []
        rule_actions = extract_actions_from_conversation(conversation)

        if out_f:
            out_f.write(json.dumps({"index": idx, "actions_rule": rule_actions}, ensure_ascii=True) + "\n")

        if args.compare:
            gpt_actions = normalize_actions(record.get("actions"))
            stats["total"] += 1

            rule_c = _canonical_actions(rule_actions)
            gpt_c = _canonical_actions(gpt_actions)

            if not rule_c and not gpt_c:
                stats["both_empty"] += 1
                stats["exact_match"] += 1
                stats["same_names"] += 1
                stats["len_match"] += 1
                continue
            if not gpt_c and rule_c:
                stats["gpt_empty_rule_nonempty"] += 1
            if gpt_c and not rule_c:
                stats["rule_empty_gpt_nonempty"] += 1

            rule_names = [a["name"] for a in rule_c]
            gpt_names = [a["name"] for a in gpt_c]
            if len(rule_c) == len(gpt_c):
                stats["len_match"] += 1
            else:
                stats["len_mismatch"] += 1

            if rule_names == gpt_names:
                stats["same_names"] += 1
                if rule_c == gpt_c:
                    stats["exact_match"] += 1
                else:
                    stats["args_mismatch"] += 1
            else:
                if len(rule_c) == len(gpt_c):
                    stats["names_mismatch_same_len"] += 1

    if out_f:
        out_f.close()

    if args.compare:
        print(json.dumps(stats, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
