#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import sys
import time
import re
import types
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Optional
from uuid import uuid4
import threading
import random

from tqdm import tqdm
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import router_interaction as ri
from router_interaction import APIGenRouterInteraction
from router_env import ROUTER_SYSTEM_PROMPT, openai_chat_completion


@dataclass(frozen=True)
class RolloutJob:
    index: int
    rollout_id: int
    record: dict[str, Any]


def _load_json(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError("Expected input JSON to be a list of records.")
    return payload


def _load_interaction_config(path: str) -> dict[str, Any]:
    config: dict[str, Any]
    try:
        from omegaconf import OmegaConf  # type: ignore

        cfg = OmegaConf.load(path)
        data = OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Invalid interaction config format.")
    if "interaction" in data:
        interactions = data.get("interaction") or []
        if not isinstance(interactions, list) or not interactions:
            raise ValueError("Interaction config missing entries.")
        selected = None
        for item in interactions:
            if isinstance(item, dict) and item.get("name") == "apigen_router":
                selected = item
                break
        if selected is None:
            selected = interactions[0]
        config = selected.get("config") or {}
    else:
        config = data
    if not isinstance(config, dict):
        raise ValueError("Interaction config is not a dict.")
    return config


def _load_router_sampling_config(path: str) -> dict[str, Any]:
    try:
        from omegaconf import OmegaConf  # type: ignore

        cfg = OmegaConf.load(path)
        data = OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    if not isinstance(data, dict):
        return {}
    rollout = (
        data.get("actor_rollout_ref", {})
        or {}
    ).get("rollout", {}) or {}
    model = (data.get("actor_rollout_ref", {}) or {}).get("model", {}) or {}
    return {
        "temperature": rollout.get("temperature"),
        "top_p": rollout.get("top_p"),
        "top_k": rollout.get("top_k"),
        "response_length": rollout.get("response_length"),
        "model_path": model.get("path"),
    }


def _router_full_history(messages: list[dict[str, Any]], max_chars: int = 8000) -> str:
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if not isinstance(content, str):
            continue
        parts.append(f"{role}: {content}")
    joined = "\n".join(parts)
    return joined[-max_chars:] if max_chars and len(joined) > max_chars else joined


def _strip_qwen_thinking(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"</think>", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("<|im_end|>", "")
    return cleaned


def _parse_qwen_router_choice(choice: str) -> Optional[str]:
    cleaned = _strip_qwen_thinking(choice).strip()
    lowered = cleaned.lower()
    if lowered in {"low", "medium", "high"}:
        return lowered
    return None


def _judge_episode_with_reasoning(
    self: APIGenRouterInteraction,
    episode: Any,
) -> tuple[bool, Any]:
    reference = ri._format_conversation_for_judge(
        episode.golden_conversation, self.max_judge_chars
    )
    candidate = ri._format_conversation_for_judge(
        episode.conversation_log, self.max_judge_chars
    )
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
    if not ri._is_gpt5_model(self.judge_model):
        call_kwargs["temperature"] = self.judge_temperature
    if self.judge_timeout is not None:
        call_kwargs["timeout"] = self.judge_timeout
    ri._maybe_add_azure_params(call_kwargs, provider)
    if provider == "azure":
        try:
            res = ri._call_completion_with_retry(
                call_kwargs,
                max_attempts=self.azure_retry_max_attempts,
                base_delay=self.azure_retry_base_delay,
                max_delay=self.azure_retry_max_delay,
            )
        except ri.litellm_exceptions.ContentPolicyViolationError:
            return False, {"verdict": "content_filter", "reasoning": ""}
    else:
        res = ri._call_completion_with_timeout(call_kwargs)
    content = res.choices[0].message.content or ""
    cleaned, reasoning = ri.strip_oss_reasoning(content)
    verdict = (cleaned or content).strip().lower()
    detail: dict[str, Any] = {"verdict": cleaned or content}
    if reasoning:
        detail["reasoning"] = reasoning
    if verdict.startswith("yes"):
        return True, detail
    if verdict.startswith("no"):
        return False, detail
    return ("yes" in verdict and "no" not in verdict), detail


def _first_human_message(conversation: list[dict[str, Any]]) -> Optional[str]:
    for item in conversation:
        if item.get("from") == "human":
            value = item.get("value", "")
            return value if isinstance(value, str) else str(value)
    return None


def _normalize_tools(raw_tools: Any) -> list[dict[str, Any]] | str:
    if isinstance(raw_tools, str):
        return raw_tools
    if not isinstance(raw_tools, list):
        return []
    return raw_tools


def _choose_effort(
    *,
    mode: str,
    fixed_effort: str,
    seed: Optional[int],
    uid: str,
    turn_index: int,
) -> str:
    choices = ("low", "medium", "high")
    if mode == "fixed":
        return fixed_effort
    if seed is None:
        return random.choice(choices)
    seed_payload = f"{seed}:{uid}:{turn_index}"
    seed_hash = sha256(seed_payload.encode("utf-8")).hexdigest()
    seed_int = int(seed_hash[:16], 16)
    return choices[seed_int % len(choices)]


def _call_router(
    *,
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    observation: str,
    temperature: Optional[float],
    top_p: Optional[float],
    top_k: Optional[int],
    max_tokens: Optional[int],
    timeout: Optional[float],
    max_retries: int,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": observation},
    ]
    extra_body: dict[str, Any] = {}
    if top_p is not None:
        extra_body["top_p"] = top_p
    if top_k is not None:
        extra_body["top_k"] = top_k

    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            response = openai_chat_completion(
                base_url=base_url,
                api_key=api_key,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=extra_body or None,
                timeout=timeout or 60.0,
            )
            content = response["choices"][0]["message"]["content"]
            return content if isinstance(content, str) else str(content)
        except Exception as exc:
            last_error = exc
            if attempt == max_retries - 1:
                break
            time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"router call failed: {last_error}")


_THREAD_LOCAL: dict[int, APIGenRouterInteraction] = {}


def _get_interaction(config: dict[str, Any]) -> APIGenRouterInteraction:
    tid = threading.get_ident()
    interaction = _THREAD_LOCAL.get(tid)
    if interaction is None:
        interaction = APIGenRouterInteraction(copy.deepcopy(config))
        # Override judge to include reasoning in detail for verify outputs.
        interaction._judge_episode = types.MethodType(_judge_episode_with_reasoning, interaction)
        _THREAD_LOCAL[tid] = interaction
    return interaction


async def _run_rollout_async(
    job: RolloutJob,
    *,
    config: dict[str, Any],
    effort_mode: str,
    fixed_effort: str,
    effort_seed: Optional[int],
    router_settings: dict[str, Any],
) -> dict[str, Any]:
    record = job.record
    instruction = record.get("instruction", "")
    system_prompt = record.get("system", "")
    golden_conversation = record.get("conversations") or record.get("conversation") or []
    if not isinstance(golden_conversation, list):
        golden_conversation = []
    tools = _normalize_tools(record.get("tools", []))
    initial_user_message = _first_human_message(golden_conversation)
    actions = record.get("actions")
    outputs = record.get("outputs")

    interaction = _get_interaction(config)
    instance_id = uuid4().hex
    await interaction.start_interaction(
        instance_id,
        instruction=instruction,
        tools=tools,
        system=system_prompt,
        golden_conversation=golden_conversation,
        initial_user_message=initial_user_message,
        actions=actions,
        outputs=outputs,
    )

    try:
        episode = interaction._instances[instance_id]
        efforts: list[str] = []
        done = False
        turn_index = 0
        last_extra_info: dict[str, Any] | None = None
        while not done:
            if effort_mode == "router":
                observation = _router_full_history(episode.agent_messages, max_chars=8000)
                try:
                    raw_effort = _call_router(
                        base_url=router_settings["base_url"],
                        api_key=router_settings["api_key"],
                        model=router_settings["model"],
                        system_prompt=ROUTER_SYSTEM_PROMPT,
                        observation=observation,
                        temperature=router_settings.get("temperature"),
                        top_p=router_settings.get("top_p"),
                        top_k=router_settings.get("top_k"),
                        max_tokens=router_settings.get("max_tokens"),
                        timeout=router_settings.get("timeout"),
                        max_retries=router_settings.get("max_retries", 3),
                    )
                    effort = _parse_qwen_router_choice(raw_effort) or "low"
                except Exception:
                    effort = "low"
                efforts.append(effort)
                messages = [{"role": "assistant", "content": effort}]
            else:
                effort = _choose_effort(
                    mode=effort_mode,
                    fixed_effort=fixed_effort,
                    seed=effort_seed,
                    uid=f"{job.index}:{job.rollout_id}",
                    turn_index=turn_index,
                )
                efforts.append(effort)
                messages = [{"role": "assistant", "content": effort}]
            done, _observation, _reward, extra_info = await interaction.generate_response(
                instance_id, messages
            )
            last_extra_info = extra_info or {}
            turn_index += 1

        episode = interaction._instances[instance_id]
        verify_method = interaction.verify_method

        if verify_method == "llm_judge":
            llm_success = last_extra_info.get("verify_success")
            llm_detail = last_extra_info.get("verify_detail")
            tau_success, tau_detail = interaction._verify_episode_tau_bench(episode)
        else:
            tau_success = last_extra_info.get("verify_success")
            tau_detail = last_extra_info.get("verify_detail")
            llm_success, llm_detail = interaction._judge_episode(episode)

        total_reward = sum(record.get("reward", 0.0) for record in episode.turn_records)
        result = {
            "index": job.index,
            "rollout_id": job.rollout_id,
            "uid": f"{job.index}:{job.rollout_id}:{instance_id}",
            "timestamp": time.time(),
            "instruction": instruction,
            "effort_mode": effort_mode,
            "efforts": efforts,
            "verify_method": verify_method,
            "verify": {
                "llm_judge": {"success": llm_success, "detail": llm_detail},
                "tau_bench": {"success": tau_success, "detail": tau_detail},
            },
            "total_reward": total_reward,
            "turn_records": episode.turn_records,
            "trajectory": episode.conversation_log,
            "reasoning_log": episode.reasoning_log,
        }
        return result
    finally:
        await interaction.finalize_interaction(instance_id)


def _run_rollout_sync(
    job: RolloutJob,
    *,
    config: dict[str, Any],
    effort_mode: str,
    fixed_effort: str,
    effort_seed: Optional[int],
    router_settings: dict[str, Any],
) -> dict[str, Any]:
    return asyncio.run(
        _run_rollout_async(
            job,
            config=config,
            effort_mode=effort_mode,
            fixed_effort=fixed_effort,
            effort_seed=effort_seed,
            router_settings=router_settings,
        )
    )


def _write_jsonl(out_file, record: dict[str, Any]) -> None:
    out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    out_file.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/apigen_with_instruction_full.json.with_gt.filtered.json",
        help="Input JSON file.",
    )
    parser.add_argument(
        "--output",
        default="outputs/router_rollouts/rollout_verify.jsonl",
        help="Output JSONL file.",
    )
    parser.add_argument(
        "--interaction-config",
        default="config/apigen_router_interaction_grpo.yaml",
        help="Interaction config path.",
    )
    parser.add_argument("--n", type=int, default=4, help="Rollouts per task.")
    parser.add_argument("--workers", type=int, default=4, help="Concurrency.")
    parser.add_argument("--start", type=int, default=0, help="Start index (inclusive).")
    parser.add_argument("--end", type=int, default=-1, help="End index (exclusive).")
    parser.add_argument(
        "--effort-mode",
        choices=("random", "fixed", "router"),
        default="random",
        help="How to choose router effort.",
    )
    parser.add_argument(
        "--effort-seed",
        type=int,
        default=42,
        help="Seed for random effort (use -1 to disable).",
    )
    parser.add_argument(
        "--fixed-effort",
        choices=("low", "medium", "high"),
        default="medium",
        help="Fixed effort when --effort-mode fixed.",
    )
    parser.add_argument(
        "--router-base-url",
        default="",
        help="Router server base URL (defaults to QWEN_ROUTER_BASE_URL or http://127.0.0.1:30001/v1).",
    )
    parser.add_argument(
        "--router-api-key",
        default="",
        help="Router server API key (defaults to QWEN_ROUTER_API_KEY or EMPTY).",
    )
    parser.add_argument(
        "--router-model",
        default="",
        help="Router model name (defaults to QWEN_ROUTER_MODEL or 'default').",
    )
    parser.add_argument(
        "--router-config",
        default="",
        help="Optional config path to read router sampling params.",
    )
    parser.add_argument("--router-temperature", type=float, default=None)
    parser.add_argument("--router-top-p", type=float, default=None)
    parser.add_argument("--router-top-k", type=int, default=None)
    parser.add_argument("--router-max-tokens", type=int, default=None)
    parser.add_argument("--router-timeout", type=float, default=None)
    parser.add_argument("--router-max-retries", type=int, default=3)
    parser.add_argument("--user-model", default="gpt-4o", help="Override user model.")
    parser.add_argument("--user-provider", default="azure", help="Override user provider.")
    parser.add_argument("--oss-temperature", type=float, default=None, help="Override OSS agent temperature.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip rollouts already present in the output JSONL.",
    )
    parser.add_argument(
        "--enable-interaction-log",
        action="store_true",
        help="Enable interaction logging (disabled by default to avoid concurrent writes).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = os.path.join(ROOT_DIR, args.input)
    output_path = os.path.join(ROOT_DIR, args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    records = _load_json(input_path)
    end_index = len(records) if args.end == -1 else min(len(records), args.end)
    start_index = max(0, args.start)
    if start_index >= end_index:
        raise ValueError("Invalid start/end range.")

    config = _load_interaction_config(os.path.join(ROOT_DIR, args.interaction_config))
    if args.user_model:
        config["user_model"] = args.user_model
    if args.user_provider:
        config["user_provider"] = args.user_provider
    if args.oss_temperature is not None:
        config["oss_temperature"] = float(args.oss_temperature)
    if not args.enable_interaction_log:
        config["log_dir"] = None
        config["print_rollout_summary"] = False
        config["print_rollouts"] = False

    effort_seed = None if args.effort_seed is None or args.effort_seed < 0 else args.effort_seed

    router_sampling = {}
    if args.effort_mode == "router" and args.router_config:
        router_sampling = _load_router_sampling_config(os.path.join(ROOT_DIR, args.router_config))

    router_base_url = args.router_base_url or os.getenv(
        "QWEN_ROUTER_BASE_URL", "http://127.0.0.1:30001/v1"
    )
    router_api_key = args.router_api_key or os.getenv("QWEN_ROUTER_API_KEY", "EMPTY")
    router_model = args.router_model or os.getenv("QWEN_ROUTER_MODEL") or router_sampling.get("model_path") or "default"
    env_temp = os.getenv("QWEN_ROUTER_TEMPERATURE")
    env_temp_val = None
    if env_temp is not None:
        try:
            env_temp_val = float(env_temp)
        except ValueError:
            env_temp_val = None
    env_max_tokens = os.getenv("QWEN_ROUTER_MAX_TOKENS")
    router_temperature = (
        args.router_temperature
        if args.router_temperature is not None
        else env_temp_val if env_temp_val is not None else router_sampling.get("temperature")
    )
    router_top_p = args.router_top_p if args.router_top_p is not None else router_sampling.get("top_p")
    router_top_k = args.router_top_k if args.router_top_k is not None else router_sampling.get("top_k")
    router_max_tokens = args.router_max_tokens
    if router_max_tokens is None:
        if env_max_tokens is not None:
            try:
                router_max_tokens = int(env_max_tokens)
            except ValueError:
                router_max_tokens = None
        if router_max_tokens is None:
            resp_len = router_sampling.get("response_length")
            if isinstance(resp_len, int) and resp_len > 0:
                router_max_tokens = resp_len
            else:
                router_max_tokens = 8
    router_settings = {
        "base_url": router_base_url,
        "api_key": router_api_key,
        "model": router_model,
        "temperature": router_temperature,
        "top_p": router_top_p,
        "top_k": router_top_k,
        "max_tokens": router_max_tokens,
        "timeout": args.router_timeout,
        "max_retries": args.router_max_retries,
    }

    existing: set[tuple[int, int]] = set()
    if args.skip_existing and os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if not isinstance(obj, dict):
                    continue
                idx = obj.get("index")
                rid = obj.get("rollout_id")
                if isinstance(idx, int) and isinstance(rid, int):
                    existing.add((idx, rid))

    jobs: list[RolloutJob] = []
    for idx in range(start_index, end_index):
        record = records[idx]
        for rollout_id in range(args.n):
            if existing and (idx, rollout_id) in existing:
                continue
            jobs.append(RolloutJob(index=idx, rollout_id=rollout_id, record=record))

    total_jobs = len(jobs)
    completed = 0

    with open(output_path, "a", encoding="utf-8") as out_file:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            in_flight = {}
            job_iter = iter(jobs)

            def submit_next():
                try:
                    job = next(job_iter)
                except StopIteration:
                    return False
                future = executor.submit(
                    _run_rollout_sync,
                    job,
                    config=config,
                    effort_mode=args.effort_mode,
                    fixed_effort=args.fixed_effort,
                    effort_seed=effort_seed,
                    router_settings=router_settings,
                )
                in_flight[future] = job
                return True

            for _ in range(min(args.workers, total_jobs)):
                if not submit_next():
                    break

            pbar = tqdm(
                total=total_jobs,
                desc="rollout_verify",
                unit="rollout",
                dynamic_ncols=True,
                ascii=True,
            )
            while in_flight:
                done, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED)
                for fut in done:
                    job = in_flight.pop(fut)
                    try:
                        result = fut.result()
                    except Exception as exc:
                        result = {
                            "index": job.index,
                            "rollout_id": job.rollout_id,
                            "uid": f"{job.index}:{job.rollout_id}",
                            "timestamp": time.time(),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    _write_jsonl(out_file, result)
                    completed += 1
                    pbar.update(1)
                    submit_next()
            pbar.close()


if __name__ == "__main__":
    main()
