from __future__ import annotations

import concurrent.futures
import copy
import json
import os
import random
import re
import sys
import time
from hashlib import sha256
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import uuid4

from litellm import completion
from litellm import exceptions as litellm_exceptions

from verl.interactions.base import BaseInteraction


THINK_START_TOKEN = "<|channel|>analysis<|message|>"
THINK_END_TOKEN = "<|end|><|start|>assistant<|channel|>final<|message|>"
QWEN_THINK_START_TOKEN = "<think>"
QWEN_THINK_END_TOKEN = "</think>"

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

RETRYABLE_LITELLM_EXCEPTIONS = (
    litellm_exceptions.RateLimitError,
    litellm_exceptions.APIError,
    litellm_exceptions.Timeout,
    litellm_exceptions.APIConnectionError,
    litellm_exceptions.InternalServerError,
    litellm_exceptions.BadGatewayError,
)

_COMPLETION_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=8)


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


def configure_azure_openai_env() -> None:
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("AZURE_ENDPOINT")
    azure_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_API_KEY")
    azure_version = os.getenv("AZURE_OPENAI_API_VERSION") or os.getenv("AZURE_API_VERSION")

    if azure_endpoint and not os.getenv("AZURE_API_BASE"):
        os.environ["AZURE_API_BASE"] = azure_endpoint
    if azure_key and not os.getenv("AZURE_API_KEY"):
        os.environ["AZURE_API_KEY"] = azure_key
    if (azure_key or azure_endpoint) and not azure_version:
        os.environ["AZURE_API_VERSION"] = "2024-02-15-preview"
    if azure_endpoint and not os.getenv("AZURE_API_TYPE"):
        os.environ["AZURE_API_TYPE"] = "azure"


def _maybe_add_azure_params(call_kwargs: dict[str, Any], provider: str) -> None:
    if provider != "azure":
        return
    api_base = (
        os.getenv("AZURE_OPENAI_ENDPOINT")
        or os.getenv("AZURE_API_BASE")
        or os.getenv("AZURE_ENDPOINT")
    )
    api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION") or os.getenv("AZURE_API_VERSION")
    if api_base:
        call_kwargs["api_base"] = api_base
    if api_key:
        call_kwargs["api_key"] = api_key
    if api_version:
        call_kwargs["api_version"] = api_version


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


def sanitize_messages_for_oss(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
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


def _infer_tool_domain(tools: list[dict[str, Any]], fallback: str) -> str:
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
    if fallback in (None, "", "auto"):
        return "airline"
    return fallback


def _to_hashable(item: Any):
    if isinstance(item, dict):
        return tuple((key, _to_hashable(value)) for key, value in sorted(item.items()))
    if isinstance(item, list):
        return tuple(_to_hashable(element) for element in item)
    if isinstance(item, set):
        return tuple(sorted(_to_hashable(element) for element in item))
    return item


def _consistent_hash(value: Any) -> str:
    return sha256(str(value).encode("utf-8")).hexdigest()


def _data_hash(data: dict[str, Any]) -> str:
    return _consistent_hash(_to_hashable(data))


def _normalize_gt_actions(raw_actions: Any) -> list[dict[str, Any]]:
    if raw_actions is None:
        return []
    if isinstance(raw_actions, str):
        try:
            raw_actions = json.loads(raw_actions)
        except Exception:
            return []
    if not isinstance(raw_actions, list):
        return []
    normalized: list[dict[str, Any]] = []
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


def _normalize_gt_outputs(raw_outputs: Any) -> list[str]:
    if raw_outputs is None:
        return []
    if isinstance(raw_outputs, str):
        try:
            raw_outputs = json.loads(raw_outputs)
        except Exception:
            raw_outputs = [raw_outputs]
    if not isinstance(raw_outputs, list):
        raw_outputs = [raw_outputs]
    outputs: list[str] = []
    for item in raw_outputs:
        if item is None:
            continue
        outputs.append(str(item))
    return outputs


def _is_retryable_exception(exc: Exception) -> bool:
    if isinstance(exc, RETRYABLE_LITELLM_EXCEPTIONS):
        return True
    status_code = getattr(exc, "status_code", None)
    if status_code in (408, 429, 500, 502, 503, 504):
        return True
    message = str(exc).lower()
    return any(token in message for token in ("rate limit", "timeout", "temporarily", "overloaded"))


def _call_completion_with_timeout(call_kwargs: dict[str, Any], timeout_s: float | None = None):
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


def _call_completion_with_retry(
    call_kwargs: dict[str, Any],
    *,
    max_attempts: int,
    base_delay: float,
    max_delay: float,
):
    attempt = 0
    while True:
        attempt += 1
        try:
            return _call_completion_with_timeout(call_kwargs)
        except Exception as exc:
            if attempt >= max_attempts or not _is_retryable_exception(exc):
                raise
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            delay += random.uniform(0.0, min(1.0, base_delay))
            time.sleep(delay)


def parse_oss_tool_call(text: str) -> dict[str, Any] | None:
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


def _sanitize_schema(value: Any) -> Any:
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray)):
        try:
            value = value.tolist()
        except Exception:
            pass
    if isinstance(value, dict):
        cleaned = {}
        for k, v in value.items():
            if v is None:
                continue
            cleaned[k] = _sanitize_schema(v)
        return cleaned
    if isinstance(value, list):
        return [_sanitize_schema(v) for v in value]
    return value


def _normalize_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if not tools:
        return []
    normalized = []
    for tool in tools:
        if "type" in tool and "function" in tool:
            normalized.append(tool)
        else:
            normalized.append({"type": "function", "function": tool})
    for tool in normalized:
        func = tool.get("function")
        if isinstance(func, dict) and "parameters" in func:
            func["parameters"] = _sanitize_schema(func["parameters"])
    return normalized


def _tool_names(tools: list[dict[str, Any]]) -> list[str]:
    names = []
    for tool in tools:
        if "function" in tool and isinstance(tool["function"], dict):
            name = tool["function"].get("name")
        else:
            name = tool.get("name")
        if name:
            names.append(name)
    return names


def _format_conversation_for_router(messages: list[dict[str, Any]], max_chars: int) -> str:
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            continue
        if not isinstance(content, str):
            continue
        parts.append(f"{role}: {content}")
    joined = "\n".join(parts)
    if max_chars and len(joined) > max_chars:
        return joined[-max_chars:]
    return joined


def _format_conversation_for_judge(conversation: list[dict[str, Any]], max_chars: int) -> str:
    lines = []
    for item in conversation:
        role = item.get("from", "")
        value = item.get("value", "")
        if role == "human":
            prefix = "User"
        elif role == "gpt":
            prefix = "Assistant"
        elif role == "function_call":
            prefix = "Tool Call"
        elif role == "observation":
            prefix = "Tool Result"
        else:
            prefix = role or "Unknown"
        lines.append(f"{prefix}: {value}")
    text = "\n".join(lines)
    if max_chars and len(text) > max_chars:
        text = text[-max_chars:]
    return text


def _strip_thinking_content(text: str) -> str:
    cleaned = text
    if not QWEN_THINK_START_TOKEN or not QWEN_THINK_END_TOKEN:
        return cleaned.strip()
    pattern = re.escape(QWEN_THINK_START_TOKEN) + r".*?" + re.escape(QWEN_THINK_END_TOKEN)
    while QWEN_THINK_START_TOKEN in cleaned and QWEN_THINK_END_TOKEN in cleaned:
        cleaned = re.sub(pattern, "", cleaned, flags=re.S)
    cleaned = cleaned.replace(QWEN_THINK_END_TOKEN, "")
    return cleaned.strip()


def _parse_router_choice(
    text: str | None,
    *,
    thinking_model: bool = False,
) -> tuple[str, bool]:
    if not text:
        return "invalid", False
    candidate = text
    if thinking_model:
        candidate = _strip_thinking_content(text)
    lowered = candidate.strip().lower()
    if lowered in {"low", "medium", "high"}:
        return lowered, True
    return "invalid", False


class LLMUserSimulator:
    def __init__(
        self,
        *,
        model: str,
        provider: str,
        instruction: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        initial_user_message: Optional[str] = None,
        request_timeout: Optional[float] = None,
        retry_max_attempts: int = 5,
        retry_base_delay: float = 2.0,
        retry_max_delay: float = 60.0,
    ) -> None:
        configure_azure_openai_env()
        configure_oss_env()
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        if request_timeout is not None:
            try:
                request_timeout = float(request_timeout)
            except (TypeError, ValueError):
                request_timeout = None
        if request_timeout is not None and request_timeout <= 0:
            request_timeout = None
        self.request_timeout = request_timeout
        self.retry_max_attempts = retry_max_attempts
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay
        self.total_cost = 0.0
        self.messages: list[dict[str, Any]] = []
        self.system_prompt = self._build_system_prompt(instruction)
        self._init_messages(initial_user_message)

    def _build_system_prompt(self, instruction: str) -> str:
        instruction_display = f"\n\nInstruction: {instruction}\n" if instruction else ""
        return (
            "You are a user interacting with an agent."
            f"{instruction_display}"
            "\nRules:\n"
            "- Just generate one line at a time to simulate the user's message.\n"
            "- Do not give away all the instruction at once. Only provide the information that is necessary for the current step.\n"
            "- Do not hallucinate information that is not provided in the instruction. For example, if the agent asks for the order id but it is not mentioned in the instruction, do not make up an order id, just say you do not remember or have it.\n"
            "- If the instruction goal is satisified, generate '###STOP###' as a standalone message without anything else to end the conversation.\n"
            "- Do not repeat the exact instruction in the conversation. Instead, use your own words to convey the same information.\n"
            "- Try to make the conversation as natural as possible, and stick to the personalities in the instruction."
        )

    def _init_messages(self, initial_user_message: Optional[str]) -> None:
        self.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": "Hi! How can I help you today?"},
        ]
        if initial_user_message:
            self.messages.append({"role": "assistant", "content": initial_user_message})
            self.initial_message = initial_user_message
        else:
            self.initial_message = self._generate_next_message()

    def _generate_next_message(self) -> str:
        provider = "openai" if self.provider == "oss" else self.provider
        req_messages = sanitize_messages_for_oss(self.messages) if self.provider == "oss" else self.messages
        call_kwargs: dict[str, Any] = {
            "model": self.model,
            "custom_llm_provider": provider,
            "messages": req_messages,
        }
        if not _is_gpt5_model(self.model):
            call_kwargs["temperature"] = self.temperature
        if self.max_tokens is not None:
            call_kwargs["max_tokens"] = self.max_tokens
        if self.request_timeout is not None:
            call_kwargs["timeout"] = self.request_timeout
        _maybe_add_azure_params(call_kwargs, provider)
        try:
            if provider == "azure":
                res = _call_completion_with_retry(
                    call_kwargs,
                    max_attempts=self.retry_max_attempts,
                    base_delay=self.retry_base_delay,
                    max_delay=self.retry_max_delay,
                )
            else:
                res = _call_completion_with_timeout(call_kwargs)
        except litellm_exceptions.ContentPolicyViolationError:
            fallback = "###STOP###"
            self.messages.append({"role": "assistant", "content": fallback})
            return fallback
        message = res.choices[0].message
        content, _ = strip_oss_reasoning(message.content or "")
        cleaned_message = message.model_dump()
        cleaned_message["content"] = content or cleaned_message.get("content")
        self.messages.append(cleaned_message)
        self.total_cost += res._hidden_params.get("response_cost") or 0.0
        return cleaned_message["content"]

    def step(self, agent_message: str) -> str:
        self.messages.append({"role": "user", "content": agent_message})
        return self._generate_next_message()


@dataclass
class RouterEpisode:
    instruction: str
    golden_conversation: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    system_prompt: str
    tool_domain: str
    tool_map: dict[str, Any]
    data: dict[str, Any]
    user_sim: LLMUserSimulator
    gt_actions: list[dict[str, Any]] = field(default_factory=list)
    gt_outputs: list[str] = field(default_factory=list)
    agent_messages: list[dict[str, Any]] = field(default_factory=list)
    conversation_log: list[dict[str, Any]] = field(default_factory=list)
    reasoning_log: list[dict[str, Any]] = field(default_factory=list)
    turn_records: list[dict[str, Any]] = field(default_factory=list)
    turn_index: int = 0
    agent_steps_in_turn: int = 0
    user_turn_index: int = 0
    done: bool = False


class APIGenRouterInteraction(BaseInteraction):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self._instances: dict[str, RouterEpisode] = {}

        self.tau_bench_root = config.get(
            "tau_bench_root", "/home/azureuser/cloudfiles/code/Users/jingbo.yang/tau-bench"
        )
        self.tool_domain = config.get("tool_domain", "airline")
        self.oss_model = config.get("oss_model", os.getenv("OSS_OPENAI_MODEL", "openai/gpt-oss-20b"))
        self.oss_provider = config.get("oss_provider", "oss")
        self.oss_temperature = float(config.get("oss_temperature", 0.0))
        self.oss_max_tokens = config.get("oss_max_tokens")
        self.user_model = config.get("user_model", "gpt-4o")
        self.user_provider = config.get("user_provider", "azure")
        self.user_temperature = float(config.get("user_temperature", 0.0))
        self.user_max_tokens = config.get("user_max_tokens")
        self.judge_model = config.get("judge_model", "gpt-5")
        self.judge_provider = config.get("judge_provider", "azure")
        self.judge_temperature = float(config.get("judge_temperature", 0.0))
        self.judge_max_tokens = int(config.get("judge_max_tokens", 8))
        self.azure_retry_max_attempts = int(config.get("azure_retry_max_attempts", 5))
        self.azure_retry_base_delay = float(config.get("azure_retry_base_delay", 2.0))
        self.azure_retry_max_delay = float(config.get("azure_retry_max_delay", 60.0))
        base_timeout = config.get("request_timeout")
        if base_timeout is not None:
            try:
                base_timeout = float(base_timeout)
            except (TypeError, ValueError):
                base_timeout = None
        if base_timeout is not None and base_timeout <= 0:
            base_timeout = None
        self.request_timeout = base_timeout
        self.oss_timeout = config.get("oss_timeout", base_timeout)
        self.user_timeout = config.get("user_timeout", base_timeout)
        self.judge_timeout = config.get("judge_timeout", base_timeout)
        for attr in ("oss_timeout", "user_timeout", "judge_timeout"):
            value = getattr(self, attr)
            if value is None:
                value = base_timeout
            else:
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    value = base_timeout
            if value is not None and value <= 0:
                value = None
            setattr(self, attr, value)

        self.max_turns = int(config.get("max_turns", 24))
        self.max_agent_steps = int(config.get("max_agent_steps", 6))
        raw_max_total_steps = config.get("max_total_steps")
        if raw_max_total_steps is None:
            self.max_total_steps = None
        else:
            try:
                self.max_total_steps = int(raw_max_total_steps)
            except (TypeError, ValueError):
                self.max_total_steps = None
            if self.max_total_steps is not None and self.max_total_steps <= 0:
                self.max_total_steps = None
        self.max_history_chars = int(config.get("max_history_chars", 8000))
        self.max_judge_chars = int(config.get("max_judge_chars", 12000))
        self.success_reward = float(config.get("success_reward", 10.0))
        try:
            self.failure_reward = float(config.get("failure_reward", 0.0))
        except (TypeError, ValueError):
            self.failure_reward = 0.0
        self.penalties = config.get("penalties", {"low": -1.0, "medium": -2.0, "high": -3.0})
        self.thinking_model = bool(config.get("thinking_model", False))
        try:
            self.format_penalty = float(config.get("format_penalty", -0.1))
        except (TypeError, ValueError):
            self.format_penalty = -0.1
        self.step_penalty_on_success_only = bool(config.get("step_penalty_on_success_only", False))
        raw_penalty_agg = config.get("step_penalty_aggregation", "sum")
        self.step_penalty_aggregation = str(raw_penalty_agg or "sum").lower()
        if self.step_penalty_aggregation not in ("sum", "mean"):
            raise ValueError(
                "Unsupported step_penalty_aggregation: "
                f"{self.step_penalty_aggregation}. Use 'sum' or 'mean'."
            )
        raw_verify_method = config.get("verify_method") or "llm_judge"
        self.verify_method = str(raw_verify_method).lower()
        if self.verify_method not in ("llm_judge", "tau_bench"):
            raise ValueError(f"Unsupported verify_method: {self.verify_method}")
        self.terminate_tools = set(config.get("terminate_tools", ["transfer_to_human_agents"]))

        self.log_dir = config.get("log_dir")
        self.log_every_n = int(config.get("log_every_n", 1))
        self.print_rollouts = bool(config.get("print_rollouts", False))
        self.print_rollout_summary = bool(config.get("print_rollout_summary", True))
        self.print_max_chars = int(config.get("print_max_chars", 4000))
        self._log_counter = 0
        self._log_path = None
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            self._log_path = os.path.join(self.log_dir, "router_rollouts.jsonl")

        self._tool_maps: dict[str, dict[str, Any]] = {}
        self._data_load_funcs: dict[str, Any] = {}

    def _ensure_tau_bench(self, tool_domain: str) -> None:
        if self.tau_bench_root and self.tau_bench_root not in sys.path:
            sys.path.insert(0, self.tau_bench_root)
        configure_oss_env()
        configure_azure_openai_env()
        if tool_domain in self._tool_maps and tool_domain in self._data_load_funcs:
            return

        if tool_domain == "airline":
            from tau_bench.envs.airline.data import load_data
            from tau_bench.envs.airline.tools import ALL_TOOLS

            self._data_load_funcs[tool_domain] = load_data
            self._tool_maps[tool_domain] = {tool.get_info()["function"]["name"]: tool for tool in ALL_TOOLS}
        elif tool_domain == "retail":
            from tau_bench.envs.retail.data import load_data
            from tau_bench.envs.retail.tools import ALL_TOOLS

            self._data_load_funcs[tool_domain] = load_data
            self._tool_maps[tool_domain] = {tool.get_info()["function"]["name"]: tool for tool in ALL_TOOLS}
        else:
            raise ValueError(f"Unsupported tool domain: {tool_domain}")

    async def start_interaction(
        self,
        instance_id: Optional[str] = None,
        *,
        instruction: str,
        tools: list[dict[str, Any]] | str,
        system: str,
        golden_conversation: list[dict[str, Any]],
        initial_user_message: Optional[str] = None,
        actions: Optional[list[dict[str, Any]]] = None,
        outputs: Optional[list[str]] = None,
        **kwargs,
    ) -> str:
        if instance_id is None:
            instance_id = str(uuid4())

        tools_list = json.loads(tools) if isinstance(tools, str) else tools
        normalized_tools = _normalize_tools(tools_list)
        tool_domain = kwargs.get("tool_domain") or self.tool_domain
        tool_domain = _infer_tool_domain(normalized_tools, tool_domain)
        self._ensure_tau_bench(tool_domain)
        if not isinstance(golden_conversation, list):
            golden_conversation = []
        instruction = instruction if isinstance(instruction, str) else str(instruction)
        system = system if isinstance(system, str) else str(system)
        data = copy.deepcopy(self._data_load_funcs[tool_domain]())
        gt_actions = _normalize_gt_actions(actions)
        gt_outputs = _normalize_gt_outputs(outputs)

        user_sim = LLMUserSimulator(
            model=self.user_model,
            provider=self.user_provider,
            instruction=instruction,
            temperature=self.user_temperature,
            max_tokens=self.user_max_tokens,
            initial_user_message=initial_user_message,
            request_timeout=self.user_timeout,
            retry_max_attempts=self.azure_retry_max_attempts,
            retry_base_delay=self.azure_retry_base_delay,
            retry_max_delay=self.azure_retry_max_delay,
        )
        first_user_message = initial_user_message or user_sim.initial_message

        agent_messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": first_user_message},
        ]
        conversation_log = [{"from": "human", "value": first_user_message}]

        self._instances[instance_id] = RouterEpisode(
            instruction=instruction,
            golden_conversation=golden_conversation,
            tools=normalized_tools,
            system_prompt=system,
            tool_domain=tool_domain,
            tool_map=self._tool_maps[tool_domain],
            data=data,
            gt_actions=gt_actions,
            gt_outputs=gt_outputs,
            user_sim=user_sim,
            agent_messages=agent_messages,
            conversation_log=conversation_log,
        )
        return instance_id

    async def generate_response(
        self, instance_id: str, messages: list[dict[str, Any]], **kwargs
    ) -> tuple[bool, str, float, dict[str, Any]]:
        episode = self._instances[instance_id]
        if episode.done:
            observation = _format_conversation_for_router(episode.agent_messages, self.max_history_chars)
            return True, observation, 0.0, {}

        effort, format_valid = _parse_router_choice(
            _last_assistant_content(messages),
            thinking_model=self.thinking_model,
        )
        if not format_valid:
            turn_idx = episode.turn_index
            penalty = float(self.penalties.get(effort, self.penalties.get("low", -1.0)))
            reward = penalty + self.format_penalty + self.failure_reward
            total_penalty = penalty + sum(record.get("penalty", 0.0) for record in episode.turn_records)
            if self.step_penalty_on_success_only:
                reward -= total_penalty
            elif self.step_penalty_aggregation == "mean":
                total_steps = len(episode.turn_records) + 1
                if total_steps > 0:
                    reward += (total_penalty / total_steps) - total_penalty
            episode.turn_index += 1
            episode.done = True
            episode.turn_records.append(
                {
                    "turn": turn_idx,
                    "effort": "invalid",
                    "penalty": penalty,
                    "format_valid": False,
                    "format_penalty": self.format_penalty,
                    "reward": reward,
                    "done": True,
                }
            )
            self._maybe_log_episode(
                episode,
                judge_success=None,
                judge_output=None,
                verify_success=False,
                verify_detail="invalid_format",
            )
            observation = _format_conversation_for_router(episode.agent_messages, self.max_history_chars)
            return True, observation, reward, {
                "effort": "invalid",
                "judge_success": None,
                "verify_method": self.verify_method,
                "verify_success": False,
                "verify_detail": "invalid_format",
                "format_penalty": self.format_penalty,
                "format_valid": False,
            }
        penalty = float(self.penalties.get(effort, self.penalties.get("low", -1.0)))
        format_penalty = 0.0 if format_valid else self.format_penalty
        turn_idx = episode.turn_index
        done = False
        judge_result = None
        judge_output = None

        agent_reply, did_call_tool, terminated_by_tool = self._run_agent_step(episode, effort)
        if did_call_tool:
            done = terminated_by_tool
        else:
            user_reply = episode.user_sim.step(agent_reply or "")
            episode.conversation_log.append({"from": "human", "value": user_reply})
            episode.agent_messages.append({"role": "user", "content": user_reply})
            done = "###STOP###" in user_reply

        episode.turn_index += 1
        if not done and self.max_total_steps is not None and episode.turn_index >= self.max_total_steps:
            done = True

        reward = penalty + format_penalty
        verify_success = None
        verify_detail = None
        if done:
            if self.verify_method == "llm_judge":
                judge_result, judge_output = self._judge_episode(episode)
                verify_success = judge_result
                verify_detail = judge_output
            else:
                verify_success, verify_detail = self._verify_episode_tau_bench(episode)
            if verify_success:
                reward += self.success_reward
            else:
                reward += self.failure_reward
            if verify_success:
                if self.step_penalty_aggregation == "mean":
                    total_penalty = penalty + sum(
                        record.get("penalty", 0.0) for record in episode.turn_records
                    )
                    total_steps = len(episode.turn_records) + 1
                    if total_steps > 0:
                        reward += (total_penalty / total_steps) - total_penalty
            else:
                if self.step_penalty_on_success_only:
                    total_penalty = penalty + sum(
                        record.get("penalty", 0.0) for record in episode.turn_records
                    )
                    reward -= total_penalty
                elif self.step_penalty_aggregation == "mean":
                    total_penalty = penalty + sum(
                        record.get("penalty", 0.0) for record in episode.turn_records
                    )
                    total_steps = len(episode.turn_records) + 1
                    if total_steps > 0:
                        reward += (total_penalty / total_steps) - total_penalty
            episode.done = True
        episode.turn_records.append(
            {
                "turn": turn_idx,
                "effort": effort,
                "penalty": penalty,
                "format_valid": format_valid,
                "format_penalty": format_penalty,
                "reward": reward,
                "done": done,
            }
        )
        if done:
            self._maybe_log_episode(episode, judge_result, judge_output, verify_success, verify_detail)

        observation = _format_conversation_for_router(episode.agent_messages, self.max_history_chars)
        return done, observation, reward, {
            "effort": effort,
            "judge_success": judge_result,
            "verify_method": self.verify_method,
            "verify_success": verify_success,
            "verify_detail": verify_detail,
            "format_penalty": format_penalty,
            "format_valid": format_valid,
        }

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        self._instances.pop(instance_id, None)

    def _run_agent_step(self, episode: RouterEpisode, effort: str) -> tuple[str, bool, bool]:
        message, tool_call, reasoning = self._call_oss_agent(episode, effort)
        episode.reasoning_log.append(
            {
                "turn": episode.turn_index,
                "effort": effort,
                "reasoning": reasoning,
                "has_tool_call": tool_call is not None,
            }
        )
        if tool_call:
            tool_name = tool_call["function"]["name"]
            raw_args = tool_call["function"].get("arguments", "{}")
            if isinstance(raw_args, str):
                try:
                    tool_args = json.loads(raw_args)
                except Exception:
                    tool_args = {}
            elif isinstance(raw_args, dict):
                tool_args = raw_args
            else:
                tool_args = {}
            if not isinstance(raw_args, str):
                tool_call = dict(tool_call)
                tool_call["function"] = dict(tool_call["function"])
                tool_call["function"]["arguments"] = json.dumps(tool_args)
            tool_result = self._execute_tool(episode, tool_name, tool_args)

            episode.conversation_log.append(
                {"from": "function_call", "value": json.dumps({"name": tool_name, "arguments": tool_args})}
            )
            episode.conversation_log.append({"from": "observation", "value": tool_result})

            assistant_entry = {
                "role": "assistant",
                "tool_calls": [tool_call],
            }
            if message:
                assistant_entry["content"] = message
            episode.agent_messages.append(assistant_entry)
            episode.agent_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.get("id", "tool-call"),
                    "name": tool_name,
                    "content": tool_result,
                }
            )

            terminated_by_tool = tool_name in self.terminate_tools
            return "", True, terminated_by_tool

        if not message:
            message = "I'm unable to continue with this request."
        episode.conversation_log.append({"from": "gpt", "value": message})
        episode.agent_messages.append({"role": "assistant", "content": message})
        return message, False, False

    def _call_oss_agent(
        self, episode: RouterEpisode, effort: str
    ) -> tuple[str, Optional[dict[str, Any]], str]:
        provider = "openai" if self.oss_provider == "oss" else self.oss_provider
        req_messages = (
            sanitize_messages_for_oss(episode.agent_messages)
            if self.oss_provider == "oss"
            else episode.agent_messages
        )
        call_kwargs: dict[str, Any] = {
            "messages": req_messages,
            "model": self.oss_model,
            "custom_llm_provider": provider,
            "tools": episode.tools,
        }
        if not _is_gpt5_model(self.oss_model):
            call_kwargs["temperature"] = self.oss_temperature
        if effort and self.oss_provider == "oss":
            call_kwargs["extra_body"] = {"reasoning_effort": effort}
        if self.oss_max_tokens is not None:
            call_kwargs["max_tokens"] = self.oss_max_tokens
        if self.oss_timeout is not None:
            call_kwargs["timeout"] = self.oss_timeout
        _maybe_add_azure_params(call_kwargs, provider)
        if provider == "azure":
            res = _call_completion_with_retry(
                call_kwargs,
                max_attempts=self.azure_retry_max_attempts,
                base_delay=self.azure_retry_base_delay,
                max_delay=self.azure_retry_max_delay,
            )
        else:
            res = _call_completion_with_timeout(call_kwargs)
        raw_content = res.choices[0].message.content or ""
        message = res.choices[0].message.model_dump()
        tool_call = None
        if message.get("tool_calls"):
            tool_call = message["tool_calls"][0]
        elif self.oss_provider == "oss":
            tool_call = parse_oss_tool_call(raw_content)
        content, reasoning = strip_oss_reasoning(raw_content)
        cleaned_message = content or message.get("content") or ""
        if tool_call is not None:
            cleaned_message = ""
        return cleaned_message, tool_call, reasoning

    def _execute_tool(self, episode: RouterEpisode, name: str, args: dict[str, Any]) -> str:
        tool_cls = episode.tool_map.get(name)
        if not tool_cls:
            return f"Error: tool not found: {name}"
        try:
            return tool_cls.invoke(episode.data, **args)
        except Exception as exc:
            return f"Error: {exc}"

    def _apply_gt_actions(self, data: dict[str, Any], episode: RouterEpisode) -> None:
        for action in episode.gt_actions:
            name = action.get("name")
            if not name or name in self.terminate_tools:
                continue
            arguments = action.get("arguments", {})
            if not isinstance(arguments, dict):
                arguments = {}
            tool_cls = episode.tool_map.get(name)
            if not tool_cls:
                continue
            try:
                tool_cls.invoke(data, **arguments)
            except Exception:
                continue

    def _verify_outputs(self, outputs: list[str], conversation: list[dict[str, Any]]) -> tuple[bool, list[str]]:
        if not outputs:
            return True, []
        missing: list[str] = []
        for output in outputs:
            needle = str(output).lower()
            if not needle:
                continue
            found = False
            for item in conversation:
                if item.get("from") != "gpt":
                    continue
                content = str(item.get("value", "")).lower().replace(",", "")
                if needle and needle in content:
                    found = True
                    break
            if not found:
                missing.append(str(output))
        return len(missing) == 0, missing

    def _verify_episode_tau_bench(self, episode: RouterEpisode) -> tuple[bool, str]:
        if episode.gt_actions is None or episode.gt_outputs is None:
            return False, "missing_gt_fields"

        current_hash = _data_hash(episode.data)
        gt_data = copy.deepcopy(self._data_load_funcs[episode.tool_domain]())
        self._apply_gt_actions(gt_data, episode)
        gt_hash = _data_hash(gt_data)
        actions_ok = current_hash == gt_hash

        outputs_ok, missing_outputs = self._verify_outputs(
            episode.gt_outputs, episode.conversation_log
        )
        success = actions_ok and outputs_ok
        detail = {
            "actions_ok": actions_ok,
            "outputs_ok": outputs_ok,
            "missing_outputs": missing_outputs,
        }
        if not actions_ok:
            detail["current_hash"] = current_hash
            detail["gt_hash"] = gt_hash
        return success, json.dumps(detail, ensure_ascii=True)

    def _maybe_log_episode(
        self,
        episode: RouterEpisode,
        judge_success: Optional[bool],
        judge_output: Optional[str],
        verify_success: Optional[bool],
        verify_detail: Optional[str],
    ) -> None:
        if not self._log_path and not (self.print_rollouts or self.print_rollout_summary):
            return
        self._log_counter += 1
        if self.log_every_n > 1 and self._log_counter % self.log_every_n != 0:
            return

        total_reward = sum(record.get("reward", 0.0) for record in episode.turn_records)
        record = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "instruction": episode.instruction,
            "judge_success": judge_success,
            "judge_output": judge_output,
            "verify_method": self.verify_method,
            "verify_success": verify_success,
            "verify_detail": verify_detail,
            "total_reward": total_reward,
            "turn_records": episode.turn_records,
            "conversation": episode.conversation_log,
        }

        if self._log_path:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=True) + "\n")

        if self.print_rollout_summary:
            print(f"[router_rollout] turns={len(episode.turn_records)} reward={total_reward:.2f}")
        if self.print_rollouts:
            if judge_output is not None:
                print(f"[router_rollout] judge_success={judge_success} judge_output={judge_output!r}")
            if verify_detail is not None:
                print(
                    f"[router_rollout] verify_method={self.verify_method} "
                    f"verify_success={verify_success} verify_detail={verify_detail!r}"
                )
            conversation_text = _format_conversation_for_judge(
                episode.conversation_log, self.print_max_chars
            )
            print(conversation_text)

    def _judge_episode(self, episode: RouterEpisode) -> tuple[bool, str]:
        reference = _format_conversation_for_judge(episode.golden_conversation, self.max_judge_chars)
        candidate = _format_conversation_for_judge(episode.conversation_log, self.max_judge_chars)
        prompt = (
            "You are an evaluator judging whether a candidate conversation successfully fulfills the user's request.\n"
            "Use the instruction as the ground truth. Compare the candidate conversation against the reference conversation.\n"
            "Ignore stylistic differences, but be strict about task completion, correctness, and required constraints.\n"
            "Answer with exactly one word: yes or no.\n\n"
            f"Instruction:\n{episode.instruction}\n\n"
            f"Reference Conversation:\n{reference}\n\n"
            f"Candidate Conversation:\n{candidate}\n\n"
            "Answer:"
        )
        provider = "openai" if self.judge_provider == "oss" else self.judge_provider
        call_kwargs: dict[str, Any] = {
            "model": self.judge_model,
            "custom_llm_provider": provider,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.judge_max_tokens,
        }
        if not _is_gpt5_model(self.judge_model):
            call_kwargs["temperature"] = self.judge_temperature
        if self.judge_timeout is not None:
            call_kwargs["timeout"] = self.judge_timeout
        _maybe_add_azure_params(call_kwargs, provider)
        if provider == "azure":
            try:
                res = _call_completion_with_retry(
                    call_kwargs,
                    max_attempts=self.azure_retry_max_attempts,
                    base_delay=self.azure_retry_base_delay,
                    max_delay=self.azure_retry_max_delay,
                )
            except litellm_exceptions.ContentPolicyViolationError:
                return False, "content_filter"
        else:
            res = _call_completion_with_timeout(call_kwargs)
        content = res.choices[0].message.content or ""
        cleaned, _ = strip_oss_reasoning(content)
        verdict = (cleaned or content).strip().lower()
        if verdict.startswith("yes"):
            return True, cleaned or content
        if verdict.startswith("no"):
            return False, cleaned or content
        return ("yes" in verdict and "no" not in verdict), cleaned or content


def _last_assistant_content(messages: list[dict[str, Any]]) -> str:
    for item in reversed(messages):
        if item.get("role") == "assistant":
            content = item.get("content", "")
            return content if isinstance(content, str) else str(content)
    return ""


def _is_gpt5_model(model_name: str) -> bool:
    return "gpt-5" in (model_name or "").lower()
