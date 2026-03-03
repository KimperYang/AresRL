import argparse
import json
import os
import sys
from typing import Any, Dict, List

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import datasets


ROUTER_SYSTEM_PROMPT = (
    "You are a router that selects the reasoning_effort (low, medium, or high) for the OSS model's *next* response.\n\n"
    "Context: The conversation involves an agent helping a user with tool calls. The OSS model charges more cost and latency at higher reasoning levels.\n"
    "- Choose HIGH if the next step likely needs careful reasoning, or complex tool sequencing.\n"
    "- Choose MEDIUM if the next step is moderately complex.\n"
    "- Choose LOW if the next step is straightforward and the risk is low.\n\n"
    "Produce exactly one word: low, medium, or high."
)

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


def normalize_tools(tools: List[Dict[str, Any]] | None) -> List[Dict[str, Any]]:
    if not tools:
        return []
    normalized = []
    for tool in tools:
        if "type" in tool and "function" in tool:
            normalized.append(tool)
        else:
            normalized.append({"type": "function", "function": tool})
    return normalized


def normalize_actions(raw_actions: Any) -> List[Dict[str, Any]]:
    if raw_actions is None:
        return []
    if isinstance(raw_actions, str):
        try:
            raw_actions = json.loads(raw_actions)
        except Exception:
            return []
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


def normalize_outputs(raw_outputs: Any) -> List[str]:
    if raw_outputs is None:
        return []
    if isinstance(raw_outputs, str):
        try:
            raw_outputs = json.loads(raw_outputs)
        except Exception:
            raw_outputs = [raw_outputs]
    if not isinstance(raw_outputs, list):
        raw_outputs = [raw_outputs]
    outputs: List[str] = []
    for item in raw_outputs:
        if item is None:
            continue
        outputs.append(str(item))
    return outputs


def extract_initial_user_message(conversations: List[Dict[str, Any]]) -> str:
    for item in conversations:
        if item.get("from") in ("human", "user"):
            value = item.get("value", "")
            return value if isinstance(value, str) else str(value)
    return ""


def infer_tool_domain(tools: List[Dict[str, Any]]) -> str:
    names = set()
    for tool in tools:
        function = tool.get("function", {})
        if isinstance(function, dict):
            name = function.get("name")
            if name:
                names.add(name)
    if names & AIRLINE_ONLY_TOOLS:
        return "airline"
    if names & RETAIL_ONLY_TOOLS:
        return "retail"
    return "airline"


def build_router_history(messages: List[Dict[str, Any]], max_chars: int = 8000) -> str:
    parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            continue
        if not isinstance(content, str):
            continue
        parts.append(f"{role}: {content}")
    joined = "\n".join(parts)
    return joined[-max_chars:] if max_chars and len(joined) > max_chars else joined


def load_records(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.hf_dataset:
        ds = datasets.load_dataset(args.hf_dataset, split=args.hf_split)
        if args.max_samples:
            ds = ds.select(range(args.max_samples))
        return list(ds)

    with open(args.input_path) as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError("Expected input JSON to be a list of records.")
    if args.max_samples:
        records = records[: args.max_samples]
    return records


def build_sample(record: Dict[str, Any], index: int, split: str) -> Dict[str, Any]:
    conversations = record.get("conversations") or []
    if not isinstance(conversations, list):
        conversations = []
    tools_raw = record.get("tools") or []
    if isinstance(tools_raw, str):
        try:
            tools_raw = json.loads(tools_raw)
        except Exception:
            tools_raw = []
    tools = normalize_tools(tools_raw)
    tools_json = json.dumps(tools, ensure_ascii=True)
    tool_domain = infer_tool_domain(tools)
    initial_user_message = extract_initial_user_message(conversations)
    system_prompt = record.get("system", "")
    actions = normalize_actions(record.get("actions"))
    outputs = normalize_outputs(record.get("outputs"))
    prompt = [
        {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_router_history(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": initial_user_message},
                ]
            ),
        },
    ]
    return {
        "data_source": "apigen_router",
        "prompt": prompt,
        "extra_info": {
            "split": split,
            "index": index,
            "interaction_kwargs": {
                "name": "apigen_router",
                "instruction": record.get("instruction", ""),
                "system": record.get("system", ""),
                "tools": tools_json,
                "golden_conversation": conversations,
                "initial_user_message": initial_user_message,
                "tool_domain": tool_domain,
                "actions": actions,
                "outputs": outputs,
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="data/apigen_router", help="Output directory for parquet files.")
    parser.add_argument("--input_path", default="data/apigen_with_instruction_full.json")
    parser.add_argument("--hf_dataset", default="", help="Optional HF dataset name, e.g. Salesforce/APIGen-MT-5k")
    parser.add_argument("--hf_split", default="train")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--val_count", type=int, default=1, help="How many samples to use for validation.")
    args = parser.parse_args()

    records = load_records(args)
    if not records:
        raise ValueError("No records found to build dataset.")

    os.makedirs(args.out_dir, exist_ok=True)

    val_count = max(1, min(len(records) - 1, args.val_count))
    train_records = records[:-val_count]
    val_records = records[-val_count:]

    train_rows = [build_sample(rec, idx, "train") for idx, rec in enumerate(train_records)]
    val_rows = [build_sample(rec, idx, "val") for idx, rec in enumerate(val_records)]

    datasets.Dataset.from_list(train_rows).to_parquet(os.path.join(args.out_dir, "train.parquet"))
    datasets.Dataset.from_list(val_rows).to_parquet(os.path.join(args.out_dir, "val.parquet"))


if __name__ == "__main__":
    main()
