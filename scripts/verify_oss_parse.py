#!/usr/bin/env python3
import concurrent.futures
import json
import os
import re

from litellm import completion
from litellm import exceptions as litellm_exceptions

THINK_START_TOKEN = "<|channel|>analysis<|message|>"
THINK_END_TOKEN = "<|end|><|start|>assistant<|channel|>final<|message|>"

_COMPLETION_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=2)


def configure_oss_env() -> None:
    base_url = os.getenv("OSS_OPENAI_BASE_URL") or "http://127.0.0.1:30000/v1"
    api_key = os.getenv("OSS_OPENAI_API_KEY", "EMPTY")
    timeout = os.getenv("OSS_OPENAI_TIMEOUT")

    if base_url:
        os.environ.setdefault("LITELLM_API_BASE", base_url)
        os.environ.setdefault("OPENAI_API_BASE", base_url)
    if api_key:
        os.environ.setdefault("OPENAI_API_KEY", api_key)
        os.environ.setdefault("LITELLM_API_KEY", api_key)
    if timeout:
        os.environ.setdefault("LITELLM_REQUEST_TIMEOUT", timeout)


def _call_completion_with_timeout(call_kwargs: dict, timeout_s: float | None = None):
    if timeout_s is None:
        timeout_s = call_kwargs.get("timeout")
    if timeout_s is not None:
        try:
            timeout_s = float(timeout_s)
        except (TypeError, ValueError):
            timeout_s = None
        if timeout_s is not None and timeout_s <= 0:
            timeout_s = None
    if timeout_s is None:
        return completion(**call_kwargs)
    future = _COMPLETION_EXECUTOR.submit(completion, **call_kwargs)
    try:
        return future.result(timeout=timeout_s)
    except concurrent.futures.TimeoutError as exc:
        future.cancel()
        raise litellm_exceptions.Timeout(f"completion timed out after {timeout_s}s") from exc


def strip_oss_reasoning(content: str) -> tuple[str, str]:
    if not content:
        return "", ""

    reasoning_raw = ""
    final_raw = content

    if THINK_START_TOKEN in content:
        after_start = content.split(THINK_START_TOKEN, 1)[1]
        if "<|end|>" in after_start:
            reasoning_raw, final_raw = after_start.split("<|end|>", 1)
        else:
            reasoning_raw = after_start
            final_raw = ""

    msg_idx = content.rfind("<|message|>")
    if msg_idx != -1:
        after_msg = content[msg_idx + len("<|message|>") :]
        if "<|call|>" in after_msg:
            final_raw = after_msg.split("<|call|>", 1)[0]
        elif "<|end|>" in after_msg:
            final_raw = after_msg.split("<|end|>", 1)[0]
        else:
            final_raw = after_msg

    reasoning = re.sub(r"<\|.*?\|>", "", reasoning_raw).strip()
    final = re.sub(r"<\|.*?\|>", "", final_raw).strip()
    return final, reasoning


def sanitize_messages_for_oss(messages: list[dict]) -> list[dict]:
    def _remove_all_tags(text: str) -> str:
        return re.sub(r"<\|.*?\|>", "", text)

    sanitized = []
    for msg in messages:
        new_msg = dict(msg)
        content = new_msg.get("content")
        if isinstance(content, str):
            cleaned, _ = strip_oss_reasoning(content)
            cleaned = _remove_all_tags(cleaned)
            fallback = _remove_all_tags(content)
            new_msg["content"] = cleaned if cleaned else fallback
        sanitized.append(new_msg)
    return sanitized


def parse_oss_tool_call(text: str) -> dict | None:
    to_idx = text.rfind("to=")
    if to_idx == -1:
        return None
    after_to = text[to_idx + 3 :]
    func_segment = after_to.split("<|", 1)[0].strip()
    func_token = func_segment.split()[0] if func_segment else ""
    func_token = func_token.split("{", 1)[0]
    if not func_token:
        return None
    func = func_token.split(".")[-1]

    start = text.find("{", to_idx)
    if start == -1:
        return None
    brace = 0
    end_idx = None
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            brace += 1
        elif ch == "}":
            brace -= 1
            if brace == 0:
                end_idx = i + 1
                break
    if end_idx is None:
        return None
    json_blob = text[start:end_idx]
    try:
        args = json.loads(json_blob)
    except Exception:
        return None
    return {
        "id": "oss-tool-call",
        "type": "function",
        "function": {
            "name": func,
            "arguments": json.dumps(args),
        },
    }


def _is_gpt5_model(model_name: str) -> bool:
    return "gpt-5" in (model_name or "").lower()


def call_like_router(*, messages, tools, model, provider, effort, temperature, max_tokens, timeout):
    if provider == "oss":
        messages = sanitize_messages_for_oss(messages)

    api_provider = "openai" if provider == "oss" else provider
    call_kwargs = {
        "messages": messages,
        "model": model,
        "custom_llm_provider": api_provider,
        "tools": tools,
    }
    if not _is_gpt5_model(model):
        call_kwargs["temperature"] = temperature
    if effort and provider == "oss":
        call_kwargs["extra_body"] = {"reasoning_effort": effort}
    if max_tokens is not None:
        call_kwargs["max_tokens"] = max_tokens
    if timeout is not None:
        call_kwargs["timeout"] = timeout
    res = _call_completion_with_timeout(call_kwargs)
    raw_content = res.choices[0].message.content or ""
    message = res.choices[0].message.model_dump()

    tool_call = None
    if message.get("tool_calls"):
        tool_call = message["tool_calls"][0]
    elif provider == "oss":
        tool_call = parse_oss_tool_call(raw_content)

    cleaned_message, _ = strip_oss_reasoning(raw_content)
    cleaned_message = cleaned_message or message.get("content") or ""
    if tool_call is not None:
        cleaned_message = ""

    return {
        "raw_content": raw_content,
        "message": message,
        "tool_call": tool_call,
        "cleaned_message": cleaned_message,
    }


def print_result(title, result):
    print("=" * 80)
    print(title)
    print("raw_content:")
    print(result["raw_content"])
    print("\nparsed tool_call:")
    print(result["tool_call"])
    print("\ncleaned_message:")
    print(result["cleaned_message"])


def main():
    configure_oss_env()

    model = os.getenv("OSS_OPENAI_MODEL", "openai/gpt-oss-20b")
    provider = "oss"

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                    },
                    "required": ["city"],
                },
            },
        }
    ]

    tool_call_messages = [
        {
            "role": "system",
            "content": "You must call the tool get_weather and nothing else.",
        },
        {
            "role": "user",
            "content": "What's the weather in Paris?",
        },
    ]

    message_only_messages = [
        {
            "role": "system",
            "content": "Do not call any tools. Reply with a short sentence.",
        },
        {
            "role": "user",
            "content": "Say hello.",
        },
    ]

    tool_call_result = call_like_router(
        messages=tool_call_messages,
        tools=tools,
        model=model,
        provider=provider,
        effort="high",
        temperature=0.0,
        max_tokens=4096,
        timeout=60,
    )

    message_result = call_like_router(
        messages=message_only_messages,
        tools=tools,
        model=model,
        provider=provider,
        effort="high",
        temperature=0.0,
        max_tokens=4096,
        timeout=60,
    )

    print_result("TOOL CALL CASE", tool_call_result)
    print_result("MESSAGE ONLY CASE", message_result)


if __name__ == "__main__":
    main()
