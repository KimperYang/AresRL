from __future__ import annotations

from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop


class MazeAgentLoop(ToolAgentLoop):
    async def run(self, sampling_params, **kwargs):
        output = await super().run(sampling_params, **kwargs)
        turn_scores = output.extra_fields.get("turn_scores", [])
        total_reward = float(sum(turn_scores)) if turn_scores else 0.0
        output.reward_score = total_reward
        output.extra_fields["total_reward"] = total_reward
        return output
