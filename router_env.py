from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

import requests

ROUTER_SYSTEM_PROMPT = (
    "You are a router that selects the reasoning_effort (low, medium, or high) for the OSS model's *next* response.\n\n"
    "Context: The conversation involves an agent helping a user with tool calls. The OSS model charges more cost and latency at higher reasoning levels.\n"
    "- Choose HIGH if the next step likely needs careful reasoning, or complex tool sequencing.\n"
    "- Choose MEDIUM if the next step is moderately complex.\n"
    "- Choose LOW if the next step is straightforward and the risk is low.\n\n"
    "Produce exactly one word: low, medium, or high."
)

VALID_ROUTER_ACTIONS = ("low", "medium", "high")
DEFAULT_STEP_PENALTIES = {"low": -1.0, "medium": -2.0, "high": -3.0}

THINK_START_TOKEN = "<|channel|>analysis<|message|>"
THINK_END_TOKEN = "<|end|><|start|>assistant<|channel|>final<|message|>"


def parse_router_action(text: str | None) -> Optional[str]:
    if not text:
        return None
    match = re.search(r"\b(low|medium|high)\b", text, flags=re.IGNORECASE)
    return match.group(1).lower() if match else None


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
    def _remove_tags(text: str) -> str:
        return re.sub(r"<\|.*?\|>", "", text)

    sanitized: list[dict[str, Any]] = []
    for msg in messages:
        new_msg = dict(msg)
        content = new_msg.get("content")
        if isinstance(content, str):
            cleaned, _ = strip_oss_reasoning(content)
            cleaned = _remove_tags(cleaned)
            fallback = _remove_tags(content)
            new_msg["content"] = cleaned if cleaned else fallback
        sanitized.append(new_msg)
    return sanitized


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


def parse_json_tool_call(text: str) -> dict[str, Any] | None:
    raw = text.strip()
    if not raw.startswith("{"):
        return None
    try:
        payload = json.loads(raw)
    except Exception:
        return None
    name = payload.get("name")
    if not name:
        return None
    args = payload.get("arguments", {})
    if isinstance(args, str):
        args_json = args
    else:
        args_json = json.dumps(args)
    return {
        "id": "json-tool-call",
        "type": "function",
        "function": {"name": name, "arguments": args_json},
    }


def extract_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        return tool_calls
    content = message.get("content") or ""
    if not isinstance(content, str):
        return []
    parsed = parse_oss_tool_call(content) or parse_json_tool_call(content)
    return [parsed] if parsed else []


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float) -> dict[str, Any]:
    response = requests.post(url, json=payload, headers=headers, timeout=timeout)
    if not response.ok:
        raise RuntimeError(f"HTTP {response.status_code} from {url}: {response.text[:500]}")
    return response.json()


def openai_chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    tools: Optional[list[dict[str, Any]]] = None,
    temperature: Optional[float] = 0.0,
    max_tokens: Optional[int] = None,
    extra_body: Optional[dict[str, Any]] = None,
    timeout: float = 60.0,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    payload: dict[str, Any] = {"model": model, "messages": messages}
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if tools:
        payload["tools"] = tools
    if extra_body:
        payload.update(extra_body)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    return _post_json(url, payload, headers, timeout)


def azure_chat_completion(
    *,
    endpoint: str,
    api_key: str,
    deployment: str,
    api_version: str,
    messages: list[dict[str, Any]],
    temperature: Optional[float] = 0.0,
    max_tokens: Optional[int] = None,
    timeout: float = 60.0,
) -> dict[str, Any]:
    url = endpoint.rstrip("/") + f"/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
    payload: dict[str, Any] = {"messages": messages}
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    headers = {"api-key": api_key, "Content-Type": "application/json"}
    return _post_json(url, payload, headers, timeout)


def build_initial_messages(system_prompt: str, first_user_message: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": first_user_message},
    ]


def _format_messages(messages: list[dict[str, Any]], max_chars: Optional[int]) -> str:
    lines: list[str] = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "assistant" and msg.get("tool_calls"):
            for tool_call in msg.get("tool_calls", []):
                func = tool_call.get("function", {})
                name = func.get("name", "unknown_tool")
                args = func.get("arguments", "")
                lines.append(f"assistant -> tool_call: {name}({args})")
            if isinstance(content, str) and content.strip():
                lines.append(f"assistant: {content}")
        elif role == "tool":
            name = msg.get("name", "tool")
            lines.append(f"tool ({name}): {content}")
        else:
            lines.append(f"{role}: {content}")
    joined = "\n".join(lines)
    if max_chars is not None and max_chars > 0 and len(joined) > max_chars:
        return joined[-max_chars:]
    return joined


def _is_gpt5_model(model_name: str) -> bool:
    return "gpt-5" in (model_name or "").lower()


def format_observation(
    messages: list[dict[str, Any]],
    *,
    turn: int,
    max_turns: int,
    last_action: Optional[str] = None,
    valid_action: Optional[bool] = None,
    status: Optional[str] = None,
    max_history_chars: Optional[int] = None,
) -> str:
    lines = [
        "Router task:",
        f"turn = {turn}/{max_turns}",
    ]
    if last_action is not None:
        if valid_action is None:
            validity = "unknown"
        else:
            validity = "valid" if valid_action else "invalid"
        lines.append(f"last_action = {last_action} ({validity})")
    if status:
        lines.append(f"status = {status}")
    lines.append("conversation:")
    lines.append(_format_messages(messages, max_history_chars))
    lines.append("reply with one of: low, medium, high")
    return "\n".join(lines)


def _safe_json_loads(payload: Any) -> Any:
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except Exception:
            return payload
    return payload


def _normalize_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in tools:
        if not isinstance(item, dict):
            continue
        if "type" in item and "function" in item:
            normalized.append(item)
        elif "name" in item:
            normalized.append({"type": "function", "function": item})
    return normalized


def _conversation_tool_pairs(conversation: list[dict[str, Any]]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for idx, item in enumerate(conversation):
        if item.get("from") != "function_call":
            continue
        value = item.get("value", "{}")
        try:
            payload = json.loads(value)
        except Exception:
            payload = {}
        tool_name = payload.get("name", "unknown_tool")
        observation = ""
        if idx + 1 < len(conversation) and conversation[idx + 1].get("from") == "observation":
            observation = conversation[idx + 1].get("value", "")
        pairs.append((tool_name, observation))
    return pairs


def _conversation_human_messages(conversation: list[dict[str, Any]]) -> list[str]:
    return [item.get("value", "") for item in conversation if item.get("from") == "human"]


class ToolOracle:
    def __init__(self, conversation: list[dict[str, Any]]) -> None:
        self._pairs = _conversation_tool_pairs(conversation)
        self._cursor = 0

    def call(self, name: str, arguments: Any) -> tuple[str, bool]:
        if self._cursor < len(self._pairs):
            expected_name, output = self._pairs[self._cursor]
            if expected_name == name:
                self._cursor += 1
                return output, True
        error = json.dumps(
            {"error": "tool_output_not_found", "tool": name, "arguments": _safe_json_loads(arguments)}
        )
        return error, False


class ScriptedUserSimulator:
    def __init__(self, messages: list[str]) -> None:
        self._messages = messages
        self._cursor = 0

    def reset(self) -> str:
        self._cursor = 0
        if not self._messages:
            return "###STOP###"
        msg = self._messages[0]
        self._cursor = 1
        return msg

    def has_next(self) -> bool:
        return self._cursor < len(self._messages)

    def step(self, _agent_content: str) -> str:
        if not self.has_next():
            return "###STOP###"
        msg = self._messages[self._cursor]
        self._cursor += 1
        return msg


class LLMUserSimulator:
    def __init__(
        self,
        *,
        model: str,
        provider: str,
        base_url: Optional[str],
        api_key: Optional[str],
        endpoint: Optional[str],
        api_version: Optional[str],
        temperature: float,
        max_tokens: int,
        instruction: str,
        initial_user_message: Optional[str] = None,
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self.provider = provider
        self.base_url = base_url
        self.api_key = api_key
        self.endpoint = endpoint
        self.api_version = api_version
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.instruction = instruction
        self.initial_user_message = initial_user_message
        self.timeout = timeout
        self.messages: list[dict[str, Any]] = []

    def build_system_prompt(self) -> str:
        instruction_display = f"\n\nInstruction: {self.instruction}\n" if self.instruction else ""
        return (
            "You are a user interacting with an agent."
            f"{instruction_display}"
            "\nRules:\n"
            "- Just generate one line at a time to simulate the user's message.\n"
            "- Do not give away all the instruction at once. Only provide the information that is necessary for the current step.\n"
            "- Do not hallucinate information that is not provided in the instruction.\n"
            "- If the instruction goal is satisified, generate '###STOP###' as a standalone message without anything else to end the conversation.\n"
            "- Do not repeat the exact instruction in the conversation. Instead, use your own words to convey the same information.\n"
            "- Try to make the conversation as natural as possible, and stick to the personalities in the instruction."
        )

    def reset(self) -> str:
        self.messages = [
            {"role": "system", "content": self.build_system_prompt()},
            {"role": "user", "content": "Hi! How can I help you today?"},
        ]
        if self.initial_user_message:
            self.messages.append({"role": "assistant", "content": self.initial_user_message})
            return self.initial_user_message
        return self._generate_next_message()

    def _generate_next_message(self) -> str:
        if self.provider == "azure":
            response = azure_chat_completion(
                endpoint=self.endpoint or os.getenv("AZURE_OPENAI_ENDPOINT", ""),
                api_key=self.api_key or os.getenv("AZURE_OPENAI_API_KEY", ""),
                deployment=self.model,
                api_version=self.api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
                messages=self.messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
            )
        else:
            response = openai_chat_completion(
                base_url=self.base_url or os.getenv("OSS_OPENAI_BASE_URL", "http://127.0.0.1:30000/v1"),
                api_key=self.api_key or os.getenv("OSS_OPENAI_API_KEY", "EMPTY"),
                model=self.model,
                messages=sanitize_messages_for_oss(self.messages),
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
            )
        message = response["choices"][0]["message"]
        content = message.get("content", "") or ""
        cleaned, _ = strip_oss_reasoning(content)
        final_content = cleaned or content
        self.messages.append({"role": "assistant", "content": final_content})
        return final_content

    def step(self, agent_content: str) -> str:
        self.messages.append({"role": "user", "content": agent_content})
        return self._generate_next_message()


class LLMJudge:
    def __init__(
        self,
        *,
        model: str,
        endpoint: Optional[str],
        api_key: Optional[str],
        api_version: Optional[str],
        temperature: float,
        max_tokens: int,
        timeout: float,
        enabled: bool = True,
    ) -> None:
        self.model = model
        self.endpoint = endpoint
        self.api_key = api_key
        self.api_version = api_version
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.enabled = enabled

    def _format_conversation(self, conversation: list[dict[str, Any]]) -> str:
        lines = []
        for item in conversation:
            role = item.get("from", "unknown")
            value = item.get("value", "")
            lines.append(f"{role}: {value}")
        return "\n".join(lines)

    def evaluate(
        self,
        *,
        instruction: str,
        golden_conversation: list[dict[str, Any]],
        candidate_conversation: list[dict[str, Any]],
    ) -> tuple[bool, str]:
        if not self.enabled:
            return False, "judge_disabled"
        endpoint = self.endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = self.api_key or os.getenv("AZURE_OPENAI_API_KEY")
        if not endpoint or not api_key:
            return False, "missing_azure_env"
        prompt = (
            "You are a strict evaluator. Compare the candidate conversation to the gold conversation and the user instruction.\n"
            "Return 'true' only if the candidate successfully satisfies the user instruction and matches the key outcomes of the gold conversation.\n"
            "Return 'false' otherwise. Output exactly one word: true or false.\n\n"
            f"Instruction:\n{instruction}\n\n"
            f"Gold Conversation:\n{self._format_conversation(golden_conversation)}\n\n"
            f"Candidate Conversation:\n{self._format_conversation(candidate_conversation)}\n\n"
            "Classification:"
        )
        temperature = None if _is_gpt5_model(self.model) else self.temperature
        response = azure_chat_completion(
            endpoint=endpoint,
            api_key=api_key,
            deployment=self.model,
            api_version=self.api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
        )
        content = response["choices"][0]["message"].get("content", "") or ""
        normalized = content.strip().lower()
        return "true" in normalized, content.strip()


class RouterEnv:
    def __init__(
        self,
        *,
        config: dict[str, Any],
        instruction: str,
        system_prompt: str,
        tools: list[dict[str, Any]],
        golden_conversation: list[dict[str, Any]],
        initial_user_message: Optional[str] = None,
        max_turns: Optional[int] = None,
    ) -> None:
        self.config = config
        self.instruction = instruction
        self.system_prompt = system_prompt
        self.tools = _normalize_tools(tools)
        self.golden_conversation = golden_conversation
        self.step_penalties = config.get("step_penalties", DEFAULT_STEP_PENALTIES)
        self.success_reward = float(config.get("success_reward", 10.0))
        self.max_turns = int(max_turns or config.get("max_turns", 12))
        self.max_history_chars = config.get("max_history_chars", 8000)
        self.max_tool_calls = int(config.get("max_tool_calls", 4))
        self._tool_oracle = ToolOracle(golden_conversation)
        self._user_mode = (config.get("user", {}).get("mode") or "scripted").lower()
        self._initialize_user_simulator(initial_user_message)
        self._judge = self._build_judge()
        self.turns = 0
        self.messages: list[dict[str, Any]] = []
        self.transcript: list[dict[str, Any]] = []
        self._reset_state(initial_user_message)

    def _initialize_user_simulator(self, initial_user_message: Optional[str]) -> None:
        user_cfg = self.config.get("user", {})
        if self._user_mode == "llm":
            self._user_sim = LLMUserSimulator(
                model=user_cfg.get("model", "gpt-4o"),
                provider=user_cfg.get("provider", "azure"),
                base_url=user_cfg.get("base_url"),
                api_key=user_cfg.get("api_key"),
                endpoint=user_cfg.get("endpoint"),
                api_version=user_cfg.get("api_version"),
                temperature=float(user_cfg.get("temperature", 0.7)),
                max_tokens=int(user_cfg.get("max_tokens", 128)),
                instruction=self.instruction,
                initial_user_message=initial_user_message,
                timeout=float(user_cfg.get("timeout", 60.0)),
            )
        else:
            scripted_messages = _conversation_human_messages(self.golden_conversation)
            self._user_sim = ScriptedUserSimulator(scripted_messages)

    def _build_judge(self) -> LLMJudge:
        judge_cfg = self.config.get("judge", {})
        return LLMJudge(
            model=judge_cfg.get("model", "gpt-5"),
            endpoint=judge_cfg.get("endpoint"),
            api_key=judge_cfg.get("api_key"),
            api_version=judge_cfg.get("api_version"),
            temperature=float(judge_cfg.get("temperature", 0.0)),
            max_tokens=int(judge_cfg.get("max_tokens", 8)),
            timeout=float(judge_cfg.get("timeout", 60.0)),
            enabled=bool(judge_cfg.get("enabled", True)),
        )

    def _reset_state(self, initial_user_message: Optional[str]) -> None:
        self.turns = 0
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.transcript = []
        first_user = self._user_sim.reset()
        if initial_user_message and self._user_mode != "llm":
            first_user = initial_user_message
        self.messages.append({"role": "user", "content": first_user})
        self.transcript.append({"from": "human", "value": first_user})

    def _call_agent(self, reasoning_effort: str) -> str:
        agent_cfg = self.config.get("agent", {})
        base_url = agent_cfg.get("base_url") or os.getenv("OSS_OPENAI_BASE_URL", "http://127.0.0.1:30000/v1")
        api_key = agent_cfg.get("api_key") or os.getenv("OSS_OPENAI_API_KEY", "EMPTY")
        model = agent_cfg.get("model") or os.getenv("OSS_OPENAI_MODEL", "default")
        temperature = float(agent_cfg.get("temperature", 0.0))
        max_tokens = int(agent_cfg.get("max_tokens", 256))
        timeout = float(agent_cfg.get("timeout", 60.0))
        extra_body = {"reasoning_effort": reasoning_effort}

        for _ in range(self.max_tool_calls + 1):
            response = openai_chat_completion(
                base_url=base_url,
                api_key=api_key,
                model=model,
                messages=sanitize_messages_for_oss(self.messages),
                tools=self.tools,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=extra_body,
                timeout=timeout,
            )
            message = response["choices"][0]["message"]
            content = message.get("content", "") or ""
            cleaned, _ = strip_oss_reasoning(content)
            content = cleaned or content
            tool_calls = extract_tool_calls(message)

            assistant_msg: dict[str, Any] = {"role": "assistant", "content": content}
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            self.messages.append(assistant_msg)

            if tool_calls:
                tool_call = tool_calls[0]
                func = tool_call.get("function", {})
                name = func.get("name", "unknown_tool")
                arguments = func.get("arguments", "{}")
                tool_output, _matched = self._tool_oracle.call(name, arguments)
                self.transcript.append(
                    {"from": "function_call", "value": json.dumps({"name": name, "arguments": _safe_json_loads(arguments)})}
                )
                self.messages.append(
                    {
                        "role": "tool",
                        "content": tool_output,
                        "name": name,
                        "tool_call_id": tool_call.get("id", "tool-call"),
                    }
                )
                self.transcript.append({"from": "observation", "value": tool_output})
                continue

            if content:
                self.transcript.append({"from": "gpt", "value": content})
            return content
        return ""

    def step(self, action: Optional[str]) -> tuple[str, float, bool, dict[str, Any]]:
        self.turns += 1
        parsed_action = action if action in VALID_ROUTER_ACTIONS else None
        valid_action = parsed_action is not None
        effort = parsed_action or "low"
        step_penalty = float(self.step_penalties.get(effort, -1.0))

        assistant_reply = self._call_agent(effort)

        done = False
        user_msg = ""
        if isinstance(self._user_sim, ScriptedUserSimulator):
            if self._user_sim.has_next():
                user_msg = self._user_sim.step(assistant_reply)
                if user_msg and user_msg != "###STOP###":
                    self.messages.append({"role": "user", "content": user_msg})
                    self.transcript.append({"from": "human", "value": user_msg})
            else:
                done = True
        else:
            user_msg = self._user_sim.step(assistant_reply)
            if user_msg and user_msg != "###STOP###":
                self.messages.append({"role": "user", "content": user_msg})
                self.transcript.append({"from": "human", "value": user_msg})
            if "###STOP###" in user_msg:
                done = True

        if self.turns >= self.max_turns:
            done = True

        success = False
        judge_reason = ""
        reward = step_penalty
        if done:
            success, judge_reason = self._judge.evaluate(
                instruction=self.instruction,
                golden_conversation=self.golden_conversation,
                candidate_conversation=self.transcript,
            )
            if success:
                reward += self.success_reward

        observation = format_observation(
            self.messages,
            turn=self.turns,
            max_turns=self.max_turns,
            last_action=action or "none",
            valid_action=valid_action,
            status="done" if done else "continue",
            max_history_chars=self.max_history_chars,
        )
        info = {
            "action": action,
            "valid_action": valid_action,
            "effort_used": effort,
            "turn": self.turns,
            "done": done,
            "success": success,
            "judge_reason": judge_reason,
        }
        return observation, float(reward), done, info
