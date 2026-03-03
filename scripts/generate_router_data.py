import argparse
import json
import os
import sys
from typing import Any

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import datasets

from router_env import ROUTER_SYSTEM_PROMPT, build_initial_messages, format_observation


def _load_entries(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError("Expected a list of entries in the dataset JSON.")
    return payload


def _parse_tools(entry: dict[str, Any]) -> list[dict[str, Any]]:
    tools = entry.get("tools", [])
    if isinstance(tools, str):
        try:
            tools = json.loads(tools)
        except Exception:
            tools = []
    if not isinstance(tools, list):
        return []
    normalized = []
    for item in tools:
        if not isinstance(item, dict):
            continue
        if "type" in item and "function" in item:
            normalized.append(item)
        elif "name" in item:
            normalized.append({"type": "function", "function": item})
    return normalized


def _extract_conversation(entry: dict[str, Any]) -> list[dict[str, Any]]:
    convo = entry.get("conversations") or entry.get("conversation")
    if not isinstance(convo, list):
        raise ValueError("Entry missing conversation list.")
    return convo


def _first_human_message(conversation: list[dict[str, Any]]) -> str:
    for item in conversation:
        if item.get("from") == "human":
            return item.get("value", "")
    return ""


def _count_assistant_turns(conversation: list[dict[str, Any]]) -> int:
    return sum(1 for item in conversation if item.get("from") == "gpt")


def build_sample(entry: dict[str, Any], index: int, split: str) -> dict[str, Any]:
    conversation = _extract_conversation(entry)
    instruction = entry.get("instruction", "")
    system_prompt = entry.get("system", "")
    tools = _parse_tools(entry)
    initial_user_message = _first_human_message(conversation) or "Hello."
    max_turns = entry.get("max_turns") or max(1, _count_assistant_turns(conversation) + 2)

    initial_messages = build_initial_messages(system_prompt, initial_user_message)
    user_prompt = format_observation(
        initial_messages,
        turn=0,
        max_turns=max_turns,
        last_action="none",
        valid_action=None,
        status="start",
        max_history_chars=None,
    )

    return {
        "data_source": "apigen_router",
        "prompt": [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "instruction": instruction,
        "system": system_prompt,
        "tools": tools,
        "golden_conversation": conversation,
        "initial_user_message": initial_user_message,
        "max_turns": max_turns,
        "extra_info": {
            "split": split,
            "index": index,
            "interaction_kwargs": {
                "name": "router",
                "instruction": instruction,
                "system_prompt": system_prompt,
                "tools": tools,
                "golden_conversation": conversation,
                "initial_user_message": initial_user_message,
                "max_turns": max_turns,
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_json",
        default="data/apigen_with_instruction_10.json",
        help="Path to APIGen JSON data.",
    )
    parser.add_argument(
        "--out_dir",
        default="data/apigen",
        help="Output directory for parquet files.",
    )
    parser.add_argument(
        "--val_count",
        type=int,
        default=1,
        help="How many samples to reserve for validation.",
    )
    args = parser.parse_args()

    entries = _load_entries(os.path.join(ROOT_DIR, args.source_json))
    if not entries:
        raise ValueError("No entries found in source JSON.")
    val_count = max(1, min(len(entries) - 1, args.val_count))
    train_entries = entries[:-val_count]
    val_entries = entries[-val_count:]

    train_rows = [build_sample(entry, idx, "train") for idx, entry in enumerate(train_entries)]
    val_rows = [build_sample(entry, idx, "val") for idx, entry in enumerate(val_entries)]

    os.makedirs(os.path.join(ROOT_DIR, args.out_dir), exist_ok=True)
    datasets.Dataset.from_list(train_rows).to_parquet(
        os.path.join(ROOT_DIR, args.out_dir, "train.parquet")
    )
    datasets.Dataset.from_list(val_rows).to_parquet(
        os.path.join(ROOT_DIR, args.out_dir, "val.parquet")
    )


if __name__ == "__main__":
    main()
