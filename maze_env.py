from __future__ import annotations

import re
from typing import Iterable, Optional

ACTION_DELTAS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
VALID_ACTIONS = tuple(ACTION_DELTAS.keys())


def parse_action(text: str | None) -> Optional[str]:
    if not text:
        return None
    match = re.search(r"\b(up|down|left|right)\b", text, flags=re.IGNORECASE)
    return match.group(1).lower() if match else None


def _coerce_pos(pos: Iterable[int]) -> tuple[int, int]:
    pos_list = list(pos)
    if len(pos_list) != 2:
        raise ValueError(f"Position must be length-2, got: {pos_list}")
    return int(pos_list[0]), int(pos_list[1])


def render_maze(
    maze: list[list[int]],
    agent_pos: tuple[int, int],
    start: tuple[int, int],
    goal: tuple[int, int],
) -> str:
    lines = []
    for r, row in enumerate(maze):
        cells = []
        for c, cell in enumerate(row):
            if (r, c) == agent_pos:
                ch = "A"
            elif (r, c) == start:
                ch = "S"
            elif (r, c) == goal:
                ch = "G"
            elif cell == 1:
                ch = "#"
            else:
                ch = "."
            cells.append(ch)
        lines.append(" ".join(cells))
    return "\n".join(lines)


def format_observation(
    maze: list[list[int]],
    agent_pos: tuple[int, int],
    start: tuple[int, int],
    goal: tuple[int, int],
    steps: int,
    max_steps: int,
    last_action: Optional[str] = None,
    valid_action: Optional[bool] = None,
    status: Optional[str] = None,
) -> str:
    lines = [
        "Maze task:",
        "maze = " + repr(maze),
        f"position = {agent_pos}, goal = {goal}",
        f"steps = {steps}/{max_steps}",
    ]
    if last_action is not None:
        if valid_action is None:
            validity = "unknown"
        else:
            validity = "valid" if valid_action else "invalid"
        lines.append(f"last_action = {last_action} ({validity})")
    if status:
        lines.append(f"status = {status}")
    lines.append("legend: A=agent, S=start, G=goal, #=wall, .=empty")
    lines.append(render_maze(maze, agent_pos, start, goal))
    lines.append("reply with one of: up, down, left, right")
    return "\n".join(lines)


class MazeEnv:
    def __init__(
        self,
        maze: list[list[int]],
        start: Iterable[int],
        goal: Iterable[int],
        *,
        step_reward: float = -1.0,
        success_reward: float = 10.0,
        max_steps: int = 20,
    ) -> None:
        if not maze or not maze[0]:
            raise ValueError("Maze cannot be empty.")
        row_len = len(maze[0])
        if any(len(row) != row_len for row in maze):
            raise ValueError("Maze rows must have the same length.")
        self.maze = [list(row) for row in maze]
        self.start = _coerce_pos(start)
        self.goal = _coerce_pos(goal)
        self.step_reward = float(step_reward)
        self.success_reward = float(success_reward)
        self.max_steps = int(max_steps)
        self._validate_positions()
        self.reset()

    def _validate_positions(self) -> None:
        for name, pos in (("start", self.start), ("goal", self.goal)):
            r, c = pos
            if not self._in_bounds(r, c):
                raise ValueError(f"{name} out of bounds: {pos}")
            if self.maze[r][c] == 1:
                raise ValueError(f"{name} cannot be on a wall: {pos}")

    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < len(self.maze) and 0 <= c < len(self.maze[0])

    def reset(self) -> str:
        self.pos = self.start
        self.steps = 0
        return self.initial_observation()

    def initial_observation(self) -> str:
        return format_observation(
            self.maze,
            self.pos,
            self.start,
            self.goal,
            self.steps,
            self.max_steps,
            last_action="none",
            valid_action=None,
            status="start",
        )

    def step(self, action: Optional[str]) -> tuple[str, float, bool, dict]:
        self.steps += 1
        reward = self.step_reward
        valid_action = action in ACTION_DELTAS
        new_pos = self.pos
        if valid_action:
            dr, dc = ACTION_DELTAS[action]
            candidate = (self.pos[0] + dr, self.pos[1] + dc)
            if self._in_bounds(*candidate) and self.maze[candidate[0]][candidate[1]] != 1:
                new_pos = candidate
            else:
                valid_action = False
        self.pos = new_pos

        done = False
        if self.pos == self.goal:
            reward += self.success_reward
            done = True
            status = "reached_goal"
        elif self.steps >= self.max_steps:
            done = True
            status = "max_steps"
        else:
            status = "continue"

        observation = format_observation(
            self.maze,
            self.pos,
            self.start,
            self.goal,
            self.steps,
            self.max_steps,
            last_action=action or "none",
            valid_action=valid_action,
            status=status,
        )
        info = {
            "action": action,
            "valid_action": valid_action,
            "position": self.pos,
            "status": status,
        }
        return observation, float(reward), done, info
