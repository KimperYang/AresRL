from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import random
import sys
import time
from typing import Any
from uuid import uuid4

import hydra
import numpy as np
import ray
from omegaconf import OmegaConf
from tqdm import tqdm

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
    AgentLoopWorker,
    AgentLoopManager,
    DictConfigWrap,
    get_trajectory_info,
    _agent_loop_registry,
)
from verl.experimental.agent_loop.utils import resolve_config_path
from verl.interactions.base import BaseInteraction
from verl.interactions.utils.interaction_registry import initialize_interactions_from_config
from verl.protocol import DataProto
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import RolloutTraceConfig, rollout_trace_attr


class RouterStepAgentLoop(AgentLoopBase):
    """Run router decisions with a fresh context each turn (tau-bench style)."""

    def __init__(
        self,
        trainer_config: DictConfigWrap,
        server_manager,
        tokenizer,
        processor,
        dataset_cls,
        dataset_config,
        **kwargs,
    ):
        super().__init__(
            trainer_config,
            server_manager,
            tokenizer,
            processor,
            dataset_cls,
            dataset_config,
            **kwargs,
        )
        config = trainer_config.config
        self.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        self.response_length = config.actor_rollout_ref.rollout.response_length
        self.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
        self.interaction_config_file = config.actor_rollout_ref.rollout.multi_turn.interaction_config_path
        rollout_custom = config.actor_rollout_ref.rollout.get("custom", {}) or {}
        self.generate_timeout_s = rollout_custom.get("generate_timeout_s")
        if self.generate_timeout_s is not None:
            try:
                self.generate_timeout_s = float(self.generate_timeout_s)
            except (TypeError, ValueError):
                self.generate_timeout_s = None
        if self.generate_timeout_s is not None and self.generate_timeout_s <= 0:
            self.generate_timeout_s = None
        self.interaction_map: dict[str, BaseInteraction] = {}
        if self.interaction_config_file:
            self.interaction_map = initialize_interactions_from_config(self.interaction_config_file)

    def _empty_response_debug_path(self) -> str:
        debug_root = None
        try:
            debug_root = self.config.trainer.default_local_dir
        except Exception:
            debug_root = None
        if not debug_root:
            debug_root = "/tmp"
        debug_dir = os.path.join(debug_root, "debug")
        os.makedirs(debug_dir, exist_ok=True)
        return os.path.join(debug_dir, "empty_responses.jsonl")

    async def _log_empty_response(
        self,
        *,
        prompt_ids: list[int],
        messages: list[dict[str, Any]],
        sampling_params: dict[str, Any],
        turn_index: int,
        **kwargs,
    ) -> None:
        prompt_text = None
        decode_error = None
        try:
            prompt_text = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(prompt_ids, skip_special_tokens=True)
            )
        except Exception as exc:
            decode_error = f"{type(exc).__name__}: {exc}"

        record = {
            "ts": time.time(),
            "pid": os.getpid(),
            "uid": kwargs.get("uid"),
            "turn_index": turn_index,
            "prompt_len": len(prompt_ids),
            "response_len": 0,
            "max_model_len": self.config.actor_rollout_ref.rollout.get("max_model_len"),
            "prompt_length_cfg": self.prompt_length,
            "response_length_cfg": self.response_length,
            "sampling_params": sampling_params,
            "prompt_text_head": prompt_text[:1000] if prompt_text else None,
            "prompt_text_tail": prompt_text[-1000:] if prompt_text else None,
            "messages": messages,
            "decode_error": decode_error,
            "global_steps": kwargs.get("global_steps"),
        }

        try:
            path = self._empty_response_debug_path()
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:
            print(
                f"[router_debug] failed to write empty response log: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> list[AgentLoopOutput]:
        request_id = uuid4().hex
        raw_prompt = kwargs["raw_prompt"]
        messages = copy.deepcopy(raw_prompt) if isinstance(raw_prompt, list) else list(raw_prompt)
        system_prompt = ""
        for msg in messages:
            if msg.get("role") == "system" and isinstance(msg.get("content"), str):
                system_prompt = msg["content"]
                break

        interaction = None
        interaction_kwargs = {}
        if self.interaction_config_file:
            interaction_kwargs = kwargs["extra_info"]["interaction_kwargs"]
            name = interaction_kwargs.get("name")
            if not name:
                raise ValueError("'name' key is required in interaction_kwargs")
            if name not in self.interaction_map:
                raise ValueError(
                    f"Interaction '{name}' not found. Available: {list(self.interaction_map.keys())}"
                )
            interaction = self.interaction_map[name]
            await interaction.start_interaction(request_id, **interaction_kwargs)

        baseline_effort_mode = kwargs.get("baseline_effort_mode")
        baseline_effort_seed = kwargs.get("baseline_effort_seed")
        random_effort_choices = ("low", "medium", "high")

        outputs: list[AgentLoopOutput] = []
        turn_rewards: list[float] = []
        turn_index = 0
        done = False

        while not done:
            metrics: dict[str, Any] = {}
            prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )
            if len(prompt_ids) > self.prompt_length:
                prompt_ids = prompt_ids[-self.prompt_length :]

            step_sampling = dict(sampling_params)
            step_sampling.setdefault("max_tokens", self.response_length)

            if baseline_effort_mode == "random":
                seed_value = None
                if baseline_effort_seed is not None:
                    try:
                        seed_value = int(baseline_effort_seed)
                    except (TypeError, ValueError):
                        seed_value = None
                if seed_value is None:
                    assistant_message = random.choice(random_effort_choices)
                else:
                    uid_value = kwargs.get("uid", "")
                    seed_payload = f"{seed_value}:{uid_value}:{turn_index}"
                    seed_hash = hashlib.sha256(seed_payload.encode("utf-8")).hexdigest()
                    seed_int = int(seed_hash[:16], 16)
                    rng = random.Random(seed_int)
                    assistant_message = rng.choice(random_effort_choices)
                response_ids = self.tokenizer.encode(assistant_message, add_special_tokens=False)
                if len(response_ids) > self.response_length:
                    response_ids = response_ids[: self.response_length]
                response_logprobs = None
            else:
                with simple_timer("generate_sequences", metrics):
                    if self.generate_timeout_s is not None:
                        token_output = await asyncio.wait_for(
                            self.server_manager.generate(
                                request_id=f"{request_id}_{turn_index}",
                                prompt_ids=prompt_ids,
                                sampling_params=step_sampling,
                                image_data=None,
                            ),
                            timeout=self.generate_timeout_s,
                        )
                    else:
                        token_output = await self.server_manager.generate(
                            request_id=f"{request_id}_{turn_index}",
                            prompt_ids=prompt_ids,
                            sampling_params=step_sampling,
                            image_data=None,
                        )

                response_ids = token_output.token_ids
                response_logprobs = list(token_output.log_probs) if token_output.log_probs is not None else None
                if len(response_ids) > self.response_length:
                    response_ids = response_ids[: self.response_length]
                    if response_logprobs is not None:
                        response_logprobs = response_logprobs[: self.response_length]
                if response_logprobs is not None and len(response_logprobs) > len(response_ids):
                    response_logprobs = response_logprobs[: len(response_ids)]
                assistant_message = await self.loop.run_in_executor(
                    None, lambda: self.tokenizer.decode(response_ids, skip_special_tokens=True)
                )

            if len(response_ids) == 0:
                await self._log_empty_response(
                    prompt_ids=prompt_ids,
                    messages=messages,
                    sampling_params=step_sampling,
                    turn_index=turn_index,
                    **kwargs,
                )

            if interaction is None:
                raise ValueError("interaction_config_path must be set for router training.")

            interaction_messages = messages + [{"role": "assistant", "content": assistant_message}]
            done, observation, reward, extra_info = await interaction.generate_response(
                request_id, interaction_messages, **interaction_kwargs
            )

            reward_value = float(reward or 0.0)
            turn_rewards.append(reward_value)

            extra_info = extra_info or {}
            response_mask = [1] * len(response_ids)
            critic_response_mask = [0] * len(response_ids)
            if critic_response_mask:
                critic_response_mask[-1] = 1
            outputs.append(
                AgentLoopOutput(
                    prompt_ids=prompt_ids,
                    response_ids=response_ids,
                    response_mask=response_mask,
                    critic_response_mask=critic_response_mask,
                    response_logprobs=response_logprobs,
                    multi_modal_data={},
                    reward_score=None,
                    num_turns=0,
                    metrics=AgentLoopMetrics(**metrics),
                    extra_fields={
                        "turn_index": turn_index,
                        "turn_reward": reward_value,
                        "format_penalty": extra_info.get("format_penalty"),
                        "format_valid": extra_info.get("format_valid"),
                        "effort": extra_info.get("effort"),
                        "judge_success": extra_info.get("judge_success"),
                        "verify_method": extra_info.get("verify_method"),
                        "verify_success": extra_info.get("verify_success"),
                        "episode_done": done,
                    }
                )
            )

            if done:
                break

            observation_text = observation if isinstance(observation, str) else str(observation)
            if system_prompt:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": observation_text},
                ]
            else:
                messages = [{"role": "user", "content": observation_text}]
            turn_index += 1

        returns: list[float] = []
        running = 0.0
        for reward_value in reversed(turn_rewards):
            running += reward_value
            returns.append(running)
        returns.reverse()

        total_turns = len(outputs)
        for idx, output in enumerate(outputs):
            output.reward_score = returns[idx]
            output.num_turns = total_turns
            output.extra_fields["return_to_go"] = returns[idx]

        if interaction is not None:
            await interaction.finalize_interaction(request_id)

        return outputs


class RouterStepAgentLoopTotalReward(RouterStepAgentLoop):
    """Broadcast total episode reward to every turn (GRPO-style)."""

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> list[AgentLoopOutput]:
        outputs = await super().run(sampling_params, **kwargs)
        if not outputs:
            return outputs
        total_reward = float(
            sum((output.extra_fields.get("turn_reward") or 0.0) for output in outputs)
        )
        for output in outputs:
            output.reward_score = total_reward
            output.extra_fields["return_to_go"] = total_reward
            output.extra_fields["total_reward"] = total_reward
        return outputs


@ray.remote
class RouterStepAgentLoopWorker(AgentLoopWorker):
    """Agent loop worker that flattens per-turn outputs into a batch."""

    def _get_worker_tag(self) -> str:
        tag = getattr(self, "_worker_tag", None)
        if tag is not None:
            return tag
        try:
            actor_id = ray.get_runtime_context().get_actor_id()
            tag = actor_id.hex() if hasattr(actor_id, "hex") else str(actor_id)
        except Exception:
            tag = "unknown"
        self._worker_tag = tag
        return tag

    def _ensure_agent_loop_registry(self, agent_name: str) -> None:
        if agent_name in _agent_loop_registry:
            return

        agent_loop_config_path = self.config.actor_rollout_ref.rollout.agent.agent_loop_config_path
        load_error = None
        if agent_loop_config_path:
            try:
                resolved_path = resolve_config_path(agent_loop_config_path)
                agent_loop_configs = OmegaConf.load(resolved_path)
                if OmegaConf.is_dict(agent_loop_configs):
                    agent_loop_configs = [agent_loop_configs]
                for agent_loop_config in agent_loop_configs:
                    name = agent_loop_config.get("name") if hasattr(agent_loop_config, "get") else None
                    if name:
                        _agent_loop_registry[name] = agent_loop_config
            except Exception as exc:
                load_error = exc

        if agent_name not in _agent_loop_registry:
            default_agent_loop = self.config.actor_rollout_ref.rollout.agent.default_agent_loop
            if agent_name == default_agent_loop:
                _agent_loop_registry[agent_name] = {"_target_": "router_step_agent_loop.RouterStepAgentLoop"}
            else:
                if load_error is not None:
                    raise KeyError(
                        "Agent loop config load failed for "
                        f"{agent_loop_config_path}: {load_error}"
                    ) from load_error
                raise KeyError(
                    f"Agent loop '{agent_name}' not registered. Available: {list(_agent_loop_registry.keys())}"
                )

    async def generate_sequences(self, batch: DataProto) -> DataProto:
        config = self.config.actor_rollout_ref.rollout
        temperature = config.temperature
        top_p = config.top_p
        top_k = config.top_k
        do_sample = config.do_sample

        if batch.meta_info.get("validate", False):
            temperature = config.val_kwargs.temperature
            top_p = config.val_kwargs.top_p
            top_k = config.val_kwargs.top_k
            do_sample = config.val_kwargs.do_sample

        if "temperature" in batch.meta_info:
            temperature = batch.meta_info["temperature"]
        if "top_p" in batch.meta_info:
            top_p = batch.meta_info["top_p"]
        if "top_k" in batch.meta_info:
            top_k = batch.meta_info["top_k"]
        if "do_sample" in batch.meta_info:
            do_sample = batch.meta_info["do_sample"]

        if not do_sample:
            temperature = 0.0
            top_p = 1.0
            top_k = -1

        sampling_params = dict(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )

        if "agent_name" not in batch.non_tensor_batch:
            default_agent_loop = config.agent.default_agent_loop
            batch.non_tensor_batch["agent_name"] = np.array([default_agent_loop] * len(batch), dtype=object)

        if self.config.algorithm.get("adv_estimator") == "grpo_rollout":
            if "index" not in batch.non_tensor_batch and "uid" in batch.non_tensor_batch:
                batch.non_tensor_batch["index"] = np.array(batch.non_tensor_batch["uid"], dtype=object)
        index = batch.non_tensor_batch.get("index", np.arange(len(batch)))

        max_samples_per_worker = RolloutTraceConfig.get_instance().max_samples_per_step_per_worker
        if max_samples_per_worker is not None:
            unique_sample_indices = np.unique(index)
            if max_samples_per_worker < len(unique_sample_indices):
                selected_samples = set(
                    np.random.choice(unique_sample_indices, max_samples_per_worker, replace=False).tolist()
                )
                traced_indices = set(i for i in range(len(batch)) if index[i] in selected_samples)
            else:
                traced_indices = set(range(len(batch)))
        else:
            traced_indices = set(range(len(batch)))

        trajectory_info = await get_trajectory_info(
            batch.meta_info.get("global_steps", -1), index.tolist(), batch.meta_info.get("validate", False)
        )

        drop_failed_samples = bool(
            self.config.actor_rollout_ref.rollout.multi_turn.get("drop_failed_samples", False)
        )

        async def _run_with_index(idx: int, trajectory: dict[str, Any], trace: bool, **task_kwargs):
            try:
                result = await self._run_agent_loop(
                    sampling_params, trajectory, trace=trace, **task_kwargs
                )
            except Exception as exc:
                if drop_failed_samples:
                    return idx, exc
                raise
            return idx, result

        tasks = []
        for i in range(len(batch)):
            trace_this_sample = i in traced_indices
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
            tasks.append(
                asyncio.create_task(
                    _run_with_index(i, trajectory_info[i], trace_this_sample, **kwargs)
                )
            )

        outputs: list[Any] = [None] * len(batch)
        rollout_custom = self.config.actor_rollout_ref.rollout.get("custom", {}) or {}
        progress_enabled = bool(rollout_custom.get("progress_bar", False))
        progress_desc = rollout_custom.get("progress_desc", "rollouts")
        pbar = None
        if progress_enabled and not batch.meta_info.get("validate", False):
            step = batch.meta_info.get("global_steps", 0)
            worker_tag = self._get_worker_tag()
            pbar = tqdm(
                total=len(batch),
                desc=f"{progress_desc} step {step} worker {worker_tag[:8]}",
                leave=False,
            )

        dropped = []
        for task in asyncio.as_completed(tasks):
            idx, result = await task
            if isinstance(result, BaseException):
                outputs[idx] = None
                dropped.append((idx, result))
            else:
                outputs[idx] = result
            if pbar is not None:
                pbar.update(1)
                # print(f"[worker] task_done idx={idx} type={type(result).__name__}", flush=True)
        print(f"[router_rollout] worker {self._get_worker_tag()[:8]} rollout done", flush=True)
        if pbar is not None:
            pbar.close()
        if dropped:
            for idx, exc in dropped:
                sample_index = index[idx] if hasattr(index, "__getitem__") else idx
                print(
                    "[router_rollout] dropped sample "
                    f"index={sample_index} error={type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )

        if batch.meta_info.get("validate", False):
            selected = []
            for output in outputs:
                if output is None:
                    continue
                if isinstance(output, list):
                    if output:
                        selected.append(output[0])
                else:
                    selected.append(output)
            if not selected:
                raise RuntimeError("All rollouts failed in validation; no samples to process.")
            return self._postprocess(selected)

        # print(f"[worker] BEFORE flatten outputs_len={len(outputs)}", flush=True)
        flattened = []
        for output in outputs:
            if output is None:
                continue
            if isinstance(output, list):
                flattened.extend(output)
            else:
                flattened.append(output)

        if not flattened:
            raise RuntimeError("All rollouts failed; no samples to process.")
            
        # print(f"[worker] AFTER flatten flattened_len={len(flattened)}", flush=True)

        # print(f"[worker] BEFORE _postprocess flattened_len={len(flattened)}", flush=True)
        dp = self._postprocess(flattened)
        # print("[worker] AFTER _postprocess", flush=True)
        return dp

    async def _run_agent_loop(
        self,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
        *,
        agent_name: str,
        trace: bool = True,
        **kwargs,
    ):
        with rollout_trace_attr(
            step=trajectory["step"],
            sample_index=trajectory["sample_index"],
            rollout_n=trajectory["rollout_n"],
            validate=trajectory["validate"],
            name="agent_loop",
            trace=trace,
        ):
            self._ensure_agent_loop_registry(agent_name)
            agent_loop_config = _agent_loop_registry[agent_name]
            agent_loop = hydra.utils.instantiate(
                config=agent_loop_config,
                trainer_config=DictConfigWrap(config=self.config),
                server_manager=self.server_manager,
                tokenizer=self.tokenizer,
                processor=self.processor,
                dataset_cls=self.dataset_cls,
                dataset_config=self.config.data,
            )
            base_index = kwargs.get("index", 0)
            # print(f"[worker] idx={base_index} BEFORE agent_loop.run", flush=True)
            output = await agent_loop.run(sampling_params, **kwargs)
            # print(f"[worker] idx={base_index} AFTER agent_loop.run", flush=True)
            outputs = output if isinstance(output, list) else [output]

            # base_index = kwargs.get("index", 0)
            try:
                base_index_int = int(base_index)
            except (TypeError, ValueError):
                base_index_int = 0

            for item in outputs:
                turn_index = item.extra_fields.get("turn_index", 0)
                try:
                    turn_index_int = int(turn_index)
                except (TypeError, ValueError):
                    turn_index_int = 0
                item.extra_fields["index"] = base_index_int * 1000 + turn_index_int
                item.extra_fields["episode_index"] = base_index_int
                uid = kwargs.get("uid")
                if uid is not None:
                    item.extra_fields["uid"] = uid
                if self.config.algorithm.get("adv_estimator") == "grpo_rollout":
                    item.extra_fields["rollout_id"] = trajectory.get("rollout_n", 0)
            processed = []
            for j, item in enumerate(outputs):
                # print(f"[worker] idx={base_index} BEFORE _agent_loop_postprocess turn={j}", flush=True)
                pi = await self._agent_loop_postprocess(item, **kwargs)
                # print(f"[worker] idx={base_index} AFTER _agent_loop_postprocess turn={j}", flush=True)
                processed.append(pi)

            print(f"[worker] idx={base_index} RETURNING from _run_agent_loop", flush=True)
            return processed
            # return [await self._agent_loop_postprocess(item, **kwargs) for item in outputs]

    async def _agent_loop_postprocess(self, output, **kwargs):
        result = await super()._agent_loop_postprocess(output, **kwargs)
        for key in ("raw_prompt", "extra_info", "interaction_kwargs", "turn_index"):
            result.extra_fields.pop(key, None)
        return result


class RouterStepAgentLoopManager(AgentLoopManager):
    def __init__(self, config: OmegaConf, worker_group=None, rm_resource_pool=None):
        self.agent_loop_workers_class = RouterStepAgentLoopWorker
        super().__init__(config, worker_group, rm_resource_pool)
