from __future__ import annotations

from typing import Any, Optional
from uuid import uuid4

from verl.interactions.base import BaseInteraction

from maze_env import MazeEnv, parse_action


def _last_assistant_content(messages: list[dict[str, Any]]) -> str:
    for item in reversed(messages):
        if item.get("role") == "assistant":
            content = item.get("content", "")
            return content if isinstance(content, str) else str(content)
    return ""


class MazeInteraction(BaseInteraction):
    def __init__(self, config: dict):
        super().__init__(config)
        self._instances: dict[str, MazeEnv] = {}

    async def start_interaction(
        self,
        instance_id: Optional[str] = None,
        *,
        maze: list[list[int]],
        start: list[int] | tuple[int, int],
        goal: list[int] | tuple[int, int],
        max_steps: Optional[int] = None,
        **kwargs,
    ) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        env = MazeEnv(
            maze=maze,
            start=start,
            goal=goal,
            step_reward=self.config.get("step_reward", -1.0),
            success_reward=self.config.get("success_reward", 10.0),
            max_steps=max_steps or self.config.get("max_steps", 20),
        )
        self._instances[instance_id] = env
        return instance_id

    async def generate_response(
        self, instance_id: str, messages: list[dict[str, Any]], **kwargs
    ) -> tuple[bool, str, float, dict[str, Any]]:
        env = self._instances[instance_id]
        action_text = _last_assistant_content(messages)
        action = parse_action(action_text)
        observation, reward, done, info = env.step(action)
        if action is None:
            info["parse_error"] = True
        return done, observation, reward, info

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        self._instances.pop(instance_id, None)
