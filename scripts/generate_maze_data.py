import argparse
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import datasets

from data.mazes import MAZES
from maze_env import MazeEnv


SYSTEM_PROMPT = (
    "You are a maze navigation agent. Reply with exactly one word: up, down, left, or right. "
    "Coordinates are (row, col), row increases downward, col increases to the right."
)


def build_sample(entry, index, split):
    env = MazeEnv(
        maze=entry["maze"],
        start=entry["start"],
        goal=entry["goal"],
        max_steps=entry["max_steps"],
    )
    user_prompt = env.initial_observation()
    return {
        "data_source": "maze",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "maze": entry["maze"],
        "start": entry["start"],
        "goal": entry["goal"],
        "max_steps": entry["max_steps"],
        "extra_info": {
            "split": split,
            "index": index,
            "maze_id": entry["id"],
            "interaction_kwargs": {
                "name": "maze",
                "maze": entry["maze"],
                "start": entry["start"],
                "goal": entry["goal"],
                "max_steps": entry["max_steps"],
            },
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="data/maze", help="Output directory for parquet files.")
    parser.add_argument("--val_count", type=int, default=1, help="How many mazes to use for validation.")
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    val_count = max(1, min(len(MAZES) - 1, args.val_count))
    train_entries = MAZES[:-val_count]
    val_entries = MAZES[-val_count:]

    train_rows = [build_sample(entry, idx, "train") for idx, entry in enumerate(train_entries)]
    val_rows = [build_sample(entry, idx, "val") for idx, entry in enumerate(val_entries)]

    datasets.Dataset.from_list(train_rows).to_parquet(os.path.join(out_dir, "train.parquet"))
    datasets.Dataset.from_list(val_rows).to_parquet(os.path.join(out_dir, "val.parquet"))


if __name__ == "__main__":
    main()
