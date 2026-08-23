"""Evaluate Taxi-v4 Q-Learning and DQN models on the same fixed seeds."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN

ENV_ID = "Taxi-v4"
EXPECTED_QTABLE_SHAPE = (500, 6)


def evaluation_seeds(seed_start: int, num_seeds: int) -> list[int]:
    if seed_start < 0:
        raise ValueError("seed_start must be non-negative")
    if num_seeds < 1:
        raise ValueError("num_seeds must be at least 1")
    return list(range(seed_start, seed_start + num_seeds))


def _summarize(
    returns: Sequence[float],
    lengths: Sequence[int],
    successes: Sequence[bool],
) -> dict[str, float | int]:
    return {
        "episodes": len(returns),
        "successes": int(np.sum(successes)),
        "success_rate": float(np.mean(successes)),
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_length": float(np.mean(lengths)),
        "std_length": float(np.std(lengths)),
    }


def _load_qtable(model_path: Path) -> np.ndarray:
    if not model_path.is_file():
        raise FileNotFoundError(f"Q-Learning model not found: {model_path}")
    q_table = np.load(model_path, allow_pickle=False)
    if q_table.shape != EXPECTED_QTABLE_SHAPE:
        raise ValueError(
            f"Q-Learning table must have shape {EXPECTED_QTABLE_SHAPE}, "
            f"got {q_table.shape}"
        )
    if not np.isfinite(q_table).all():
        raise ValueError("Q-Learning table contains non-finite values")
    return q_table


def evaluate_qlearning(model_path: Path, seeds: Sequence[int]) -> dict[str, float | int]:
    q_table = _load_qtable(model_path)
    env = gym.make(ENV_ID)
    returns: list[float] = []
    lengths: list[int] = []
    successes: list[bool] = []

    try:
        for seed in seeds:
            policy_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
            observation, _ = env.reset(seed=seed)
            total_return = 0.0
            length = 0

            while True:
                q_values = q_table[int(observation)]
                greedy_actions = np.flatnonzero(q_values == np.max(q_values))
                action = int(policy_rng.choice(greedy_actions))
                observation, reward, terminated, truncated, _ = env.step(action)
                total_return += float(reward)
                length += 1
                if terminated or truncated:
                    returns.append(total_return)
                    lengths.append(length)
                    successes.append(bool(terminated))
                    break
    finally:
        env.close()

    return _summarize(returns, lengths, successes)


def evaluate_dqn(
    model_path: Path,
    seeds: Sequence[int],
) -> dict[str, float | int]:
    if not model_path.is_file():
        raise FileNotFoundError(f"DQN model not found: {model_path}")

    model = DQN.load(model_path, device="cpu")
    env = gym.make(ENV_ID)

    returns: list[float] = []
    lengths: list[int] = []
    successes: list[bool] = []

    try:
        if (
            model.observation_space != env.observation_space
            or model.action_space != env.action_space
        ):
            raise ValueError(
                "DQN model spaces do not match Taxi-v4: "
                f"model=({model.observation_space}, {model.action_space}), "
                f"environment=({env.observation_space}, {env.action_space})"
            )

        for seed in seeds:
            observation, _ = env.reset(seed=seed)
            total_return = 0.0

            for length in range(1, 201):
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, _ = env.step(int(action))
                total_return += float(reward)
                if terminated or truncated:
                    returns.append(total_return)
                    lengths.append(length)
                    successes.append(bool(terminated))
                    break
            else:
                raise RuntimeError("Taxi-v4 did not finish within its 200-step limit")
    finally:
        env.close()

    return _summarize(returns, lengths, successes)


def evaluate_models(
    *,
    qlearning_model: Path | None,
    dqn_model: Path | None,
    seed_start: int,
    num_seeds: int,
) -> dict[str, object]:
    if qlearning_model is None and dqn_model is None:
        raise ValueError("At least one model path must be provided")

    seeds = evaluation_seeds(seed_start, num_seeds)
    results: dict[str, dict[str, float | int]] = {}
    if qlearning_model is not None:
        results["qlearning"] = evaluate_qlearning(qlearning_model, seeds)
    if dqn_model is not None:
        results["dqn"] = evaluate_dqn(dqn_model, seeds)

    return {
        "environment": ENV_ID,
        "seed_start": seed_start,
        "num_seeds": num_seeds,
        "results": results,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare Taxi-v4 models on identical fixed seeds",
        allow_abbrev=False,
    )
    parser.add_argument("--qlearning-model", type=Path)
    parser.add_argument("--dqn-model", type=Path)
    parser.add_argument("--seed-start", type=int, default=10_000)
    parser.add_argument("--num-seeds", type=int, default=100)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        results = evaluate_models(
            qlearning_model=args.qlearning_model,
            dqn_model=args.dqn_model,
            seed_start=args.seed_start,
            num_seeds=args.num_seeds,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
