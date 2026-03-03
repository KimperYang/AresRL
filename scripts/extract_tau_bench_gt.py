#!/usr/bin/env python3
import argparse
import asyncio
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

from openai import AsyncAzureOpenAI
from tqdm import tqdm


DEFAULT_INPUT_PATH = (
    "/home/azureuser/cloudfiles/code/Users/jingbo.yang/TBAMA/rl/data/"
    "apigen_with_instruction_full.json"
)

SYSTEM_PROMPT = """You are an expert annotator for tau-bench tasks.
You will receive a user instruction and a full conversation between a user and an agent.
The conversation includes tool calls written as `function_call` entries with JSON arguments, and `observation` entries with tool results.

Your job: extract the ground-truth action sequence and the optional outputs list used for tau-bench evaluation.

Rules:
- Only include successful tool calls that actually happened in the conversation, in order.
- Exclude any tool call that failed (e.g., observation contains Error/Exception/Traceback or states failure).
- Exclude tool calls to "think".
- Each action must be in tau-bench format: {"name": "<tool_name>", "arguments": { ... }}.
- Outputs are literal strings (usually numbers) that must appear in the agent's replies to verify success.
- Only include outputs that are explicitly stated by the agent and required by the instruction; if unsure, return [].
- Outputs should be strings without commas or currency symbols (e.g., "$1,234.50" -> "1234.50").

Return ONLY a JSON object with this schema:
{"actions": [ {"name": "...", "arguments": {...}}, ... ], "outputs": ["..."] }

Examples:

Instruction:
Your user id is linda_chen_1234. You want to change reservation AB12CD to a later flight on the same day and upgrade to business. If there is a price difference, tell me the amount. Use payment method credit_card_998877.

Conversation:
human: Hi, I need to change my reservation AB12CD to a later flight today and upgrade to business.
gpt: Sure. Please provide your user id and the payment method to charge any fare difference.
human: User id linda_chen_1234. Use credit_card_998877.
function_call: {"name": "get_reservation_details", "arguments": {"reservation_id": "AB12CD"}}
observation: {"reservation_id":"AB12CD","origin":"SFO","destination":"JFK","flights":[{"flight_number":"HAT100","date":"2024-06-01"}],"cabin":"economy"}
function_call: {"name": "search_direct_flight", "arguments": {"origin": "SFO", "destination": "JFK", "date": "2024-06-01"}}
observation: {"flights":[{"flight_number":"HAT220","date":"2024-06-01","price":800},{"flight_number":"HAT240","date":"2024-06-01","price":900}]}
function_call: {"name": "update_reservation_flights", "arguments": {"reservation_id": "AB12CD", "cabin": "business", "flights": [{"flight_number": "HAT240", "date": "2024-06-01"}], "payment_id": "credit_card_998877"}}
observation: {"status":"success","price_difference":327}
gpt: Done. Your reservation is updated and the price difference is $327.

Output:
{"actions": [{"name": "get_reservation_details", "arguments": {"reservation_id": "AB12CD"}}, {"name": "search_direct_flight", "arguments": {"origin": "SFO", "destination": "JFK", "date": "2024-06-01"}}, {"name": "update_reservation_flights", "arguments": {"reservation_id": "AB12CD", "cabin": "business", "flights": [{"flight_number": "HAT240", "date": "2024-06-01"}], "payment_id": "credit_card_998877"}}], "outputs": ["327"]}

Instruction:
You want to exchange items in delivered order W12345: swap hiking boots to size 9 and return the gloves.

Conversation:
human: I want to exchange order W12345. Swap the hiking boots to size 9 and return the gloves.
gpt: I can help with that. Please confirm you want to proceed with the exchange/return.
human: Yes, please proceed.
function_call: {"name": "exchange_delivered_order_items", "arguments": {"order_id": "W12345", "items": [{"item_id": "boots_4455", "new_option": {"size": "9"}}, {"item_id": "gloves_1002", "action": "return"}]}}
observation: Error: return window closed
gpt: I'm sorry, the return window has closed for that order.

Output:
{"actions": [], "outputs": []}
"""

TASK_BACKGROUND = """You are labeling tau-bench ground-truth actions and outputs.
The agent can call tools; tool calls appear as `function_call` with JSON arguments, followed by `observation`.
Use the user instruction and the conversation to determine which tool calls succeeded and which output values are required."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help="Input JSON file.")
    parser.add_argument(
        "--output",
        default="",
        help="Output JSON file. Default: <input>.with_gt.json",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="Azure OpenAI deployment name (defaults to gpt-4o).",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=8,
        help="Maximum concurrent requests.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index (inclusive).",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=-1,
        help="End index (exclusive). -1 means end of file.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip records that already have actions/outputs.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="Save intermediate results every N processed records.",
    )
    parser.add_argument(
        "--include-system",
        action="store_true",
        help="Include the record's system policy in the prompt.",
    )
    parser.add_argument(
        "--azure-endpoint",
        default="",
        help="Override AZURE_OPENAI_ENDPOINT.",
    )
    parser.add_argument(
        "--azure-api-key",
        default="",
        help="Override AZURE_OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--azure-api-version",
        default="",
        help="Override AZURE_OPENAI_API_VERSION (default 2024-02-15-preview).",
    )
    parser.add_argument(
        "--azure-deployment",
        default="",
        help="Override AZURE_OPENAI_DEPLOYMENT (defaults to --model).",
    )
    parser.add_argument(
        "--error-log",
        default="",
        help="Optional path to write errors as JSONL.",
    )
    return parser.parse_args()


def load_records(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError("Expected input JSON to be a list of records.")
    return records


def format_conversation(conversations: Iterable[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for turn in conversations:
        role = turn.get("from", "")
        value = turn.get("value", "")
        if isinstance(value, (dict, list)):
            value_str = json.dumps(value, ensure_ascii=True)
        elif value is None:
            value_str = ""
        else:
            value_str = str(value)
        lines.append(f"{role}: {value_str}")
    return "\n".join(lines)


def build_messages(record: Dict[str, Any], include_system: bool) -> List[Dict[str, str]]:
    instruction = record.get("instruction", "")
    conversation = record.get("conversations") or []
    if not isinstance(conversation, list):
        conversation = []
    convo_text = format_conversation(conversation)

    system_policy = record.get("system", "") if include_system else ""
    policy_block = f"\nSystem policy:\n{system_policy}\n" if system_policy else ""

    user_content = (
        f"{TASK_BACKGROUND}\n\n"
        f"Instruction:\n{instruction}\n"
        f"{policy_block}\n"
        f"Conversation:\n{convo_text}\n"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


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
        if name == "think":
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
    if not isinstance(raw_outputs, list):
        raw_outputs = [raw_outputs]
    cleaned: List[str] = []
    for item in raw_outputs:
        if item is None:
            continue
        value = str(item).strip()
        if not value:
            continue
        value = value.replace(",", "")
        if value.startswith("$"):
            value = value[1:]
        if value.lower().startswith("usd "):
            value = value[4:]
        cleaned.append(value)
    return cleaned


def should_drop_record(record: Dict[str, Any]) -> bool:
    actions = record.get("actions")
    outputs = record.get("outputs")
    return (
        isinstance(actions, list)
        and isinstance(outputs, list)
        and not actions
        and not outputs
    )


def filtered_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [record for record in records if not should_drop_record(record)]


async def annotate_record(
    client: AsyncAzureOpenAI,
    model: str,
    record: Dict[str, Any],
    include_system: bool,
    temperature: float,
    max_retries: int,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    messages = build_messages(record, include_system=include_system)
    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or "{}"
            data = json.loads(content)
            actions = normalize_actions(data.get("actions"))
            outputs = normalize_outputs(data.get("outputs"))
            return actions, outputs
        except Exception as exc:
            last_error = exc
            backoff = min(2 ** attempt, 8)
            await asyncio.sleep(backoff)
    raise RuntimeError(f"Failed after {max_retries} attempts: {last_error}")


def build_client(args: argparse.Namespace) -> Tuple[AsyncAzureOpenAI, str]:
    endpoint = args.azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = args.azure_api_key or os.getenv("AZURE_OPENAI_API_KEY")
    api_version = (
        args.azure_api_version
        or os.getenv("AZURE_OPENAI_API_VERSION")
        or "2024-02-15-preview"
    )
    if not endpoint:
        raise ValueError("Missing AZURE_OPENAI_ENDPOINT.")
    if not api_key:
        raise ValueError("Missing AZURE_OPENAI_API_KEY.")
    deployment = args.azure_deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT") or args.model
    client = AsyncAzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
    )
    return client, deployment


async def run_async(args: argparse.Namespace) -> None:
    records = load_records(args.input)
    if not records:
        raise ValueError("Input JSON file is empty.")

    output_path = args.output or f"{args.input}.with_gt.json"
    end_index = len(records) if args.end == -1 else min(len(records), args.end)
    start_index = max(0, args.start)
    if start_index >= end_index:
        raise ValueError("Invalid start/end index range.")

    client, deployment = build_client(args)
    semaphore = asyncio.Semaphore(max(1, args.max_concurrency))
    pending: List[asyncio.Task[Tuple[int, Optional[str]]]] = []
    error_log = open(args.error_log, "a") if args.error_log else None

    async def process_one(index: int) -> Tuple[int, Optional[str]]:
        record = records[index]
        if args.skip_existing and ("actions" in record or "outputs" in record):
            return index, None
        async with semaphore:
            try:
                actions, outputs = await annotate_record(
                    client=client,
                    model=deployment,
                    record=record,
                    include_system=args.include_system,
                    temperature=args.temperature,
                    max_retries=args.max_retries,
                )
                record["actions"] = actions
                record["outputs"] = outputs
                return index, None
            except Exception as exc:
                return index, str(exc)

    for i in range(start_index, end_index):
        pending.append(asyncio.create_task(process_one(i)))

    processed = 0
    total = end_index - start_index
    pbar = tqdm(
        total=total,
        desc="Extracting",
        unit="task",
        dynamic_ncols=True,
        ascii=True,
    )
    for task in asyncio.as_completed(pending):
        idx, err = await task
        processed += 1
        if err:
            msg = {"index": idx, "error": err}
            if error_log:
                error_log.write(json.dumps(msg, ensure_ascii=True) + "\n")
                error_log.flush()
            else:
                tqdm.write(f"[error] index={idx}: {err}", file=sys.stderr)
        if args.save_every and processed % args.save_every == 0:
            with open(output_path, "w") as f:
                json.dump(filtered_records(records), f, ensure_ascii=True)
        pbar.update(1)
    pbar.close()

    if error_log:
        error_log.close()

    with open(output_path, "w") as f:
        json.dump(filtered_records(records), f, ensure_ascii=True)


def main() -> None:
    args = parse_args()
    asyncio.run(run_async(args))


if __name__ == "__main__":
    main()
