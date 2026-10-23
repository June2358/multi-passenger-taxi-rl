"""
AdvancedTaxi-v2 학습 모델 평가 스크립트 (이동 구간 개략도 포함)

- random / cluster / complex 시나리오에 대해
  - Random 정책
  - Greedy 정책
  - 전체 이동거리 Greedy 정책
  - RL(MaskablePPO) 정책
  의 총 완료시간 합(sum of completion times)을 요약해서 출력합니다.

- 시각화: 각 정책의 이동 구간을 그리드월드에 표시
  - 승객 출발지/목적지
  - 빈차 이동 경로
  - 승객 운행 경로
  - 택시 시작/종료 위치
"""

from __future__ import annotations

import argparse
import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.envs import AdvancedTimeAttackTaxiEnv

ROOT = Path(__file__).resolve().parent

ALGO = "maskableppo"
ENV_ID = "AdvancedTaxi-v2"
RL_EVALUATION_BATCH_SIZE = 100
SCENARIOS = ("random", "cluster", "complex")
POLICIES = ("random", "greedy", "shortest_trip", "rl")
BASELINE_POLICIES = POLICIES[:-1]


@dataclass(frozen=True)
class RunArtifacts:
    run_dir: Path
    checkpoint: str
    model: Path
    vecnormalize: Path | None
    config: Path
    args: Path


def _load_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"설정 파일의 최상위 값이 mapping이 아닙니다: {path}")
    return data


def _uses_vecnormalize(value: Any) -> bool:
    """저장된 학습 설정에서 VecNormalize 사용 여부를 반환한다."""
    if isinstance(value, dict):
        return True
    if isinstance(value, bool):
        return value
    raise ValueError("config.yml의 normalize는 boolean 또는 mapping이어야 합니다.")


def resolve_run_artifacts(
    logs_dir: Path,
    algo: str,
    env_id: str,
    exp_id: int,
    checkpoint: str,
) -> RunArtifacts:
    """명시한 checkpoint와 짝을 이루는 정규화 상태를 찾는다."""
    if exp_id < 1:
        raise ValueError("exp_id는 1 이상이어야 합니다.")
    if checkpoint not in {"final", "best"}:
        raise ValueError("checkpoint는 'final' 또는 'best'여야 합니다.")

    run_dir = logs_dir / algo / f"{env_id}_{exp_id}"
    params_dir = run_dir / env_id
    model_path = run_dir / (f"{env_id}.zip" if checkpoint == "final" else "best_model.zip")
    vecnormalize_path = (
        params_dir / "vecnormalize.pkl"
        if checkpoint == "final"
        else run_dir / "best_model_vecnormalize.pkl"
    )
    artifacts = RunArtifacts(
        run_dir=run_dir,
        checkpoint=checkpoint,
        model=model_path,
        vecnormalize=vecnormalize_path,
        config=params_dir / "config.yml",
        args=params_dir / "args.yml",
    )

    if not artifacts.run_dir.is_dir():
        raise FileNotFoundError(f"run을 찾을 수 없습니다: {artifacts.run_dir}")

    for label, path in {
        "선택한 모델": artifacts.model,
        "config": artifacts.config,
        "args": artifacts.args,
    }.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label}을(를) 찾을 수 없습니다: {path}")

    config = _load_mapping(artifacts.config)
    if "normalize" not in config:
        raise ValueError(
            f"config.yml에 normalize 계약이 없습니다: {artifacts.config}"
        )
    normalize = config["normalize"]
    uses_normalization = _uses_vecnormalize(normalize)
    if uses_normalization and not artifacts.vecnormalize.is_file():
        raise FileNotFoundError(
            "이 실험은 VecNormalize를 사용하지만 통계 파일이 없습니다: "
            f"{artifacts.vecnormalize}"
        )

    if not uses_normalization:
        return RunArtifacts(
            run_dir=artifacts.run_dir,
            checkpoint=artifacts.checkpoint,
            model=artifacts.model,
            vecnormalize=None,
            config=artifacts.config,
            args=artifacts.args,
        )
    return artifacts


def load_evaluation_env_kwargs(artifacts: RunArtifacts) -> dict[str, Any]:
    """학습에서 저장한 평가 환경 계약 전체를 읽는다."""
    saved_args = _load_mapping(artifacts.args)
    if saved_args.get("algo") != ALGO or saved_args.get("env") != ENV_ID:
        raise ValueError(
            f"저장된 실행이 {ENV_ID}/{ALGO}가 아닙니다: "
            f"algo={saved_args.get('algo')}, env={saved_args.get('env')}"
        )

    signature = inspect.signature(AdvancedTimeAttackTaxiEnv.__init__)
    expected_keys = {
        name
        for name, parameter in signature.parameters.items()
        if name != "self" and parameter.default is not inspect.Parameter.empty
    }
    saved_train_kwargs = saved_args.get("env_kwargs")
    saved_eval_kwargs = saved_args.get("eval_env_kwargs")
    for label, values in (
        ("env_kwargs", saved_train_kwargs),
        ("eval_env_kwargs", saved_eval_kwargs),
    ):
        if not isinstance(values, dict):
            raise ValueError(f"args.yml의 {label}가 mapping이 아닙니다: {artifacts.args}")
        missing = expected_keys - values.keys()
        unknown = values.keys() - expected_keys
        if missing or unknown:
            details = []
            if missing:
                details.append(f"누락={sorted(missing)}")
            if unknown:
                details.append(f"알 수 없음={sorted(unknown)}")
            raise ValueError(
                f"저장된 {label}가 현재 환경 계약과 일치하지 않습니다 "
                f"({', '.join(details)}). 새 환경으로 재학습해야 합니다."
            )

    return dict(saved_eval_kwargs)


def _new_path_data(taxi_start: np.ndarray, track_path: bool) -> dict[str, Any] | None:
    if not track_path:
        return None
    return {"taxi_start": taxi_start.tolist(), "trips": []}


def _record_step(path_data: dict[str, Any] | None, info: Mapping[str, Any]) -> None:
    if path_data is None:
        return
    origin = np.asarray(info["passenger_origin"], dtype=int).tolist()
    destination = np.asarray(info["passenger_destination"], dtype=int).tolist()
    taxi_before = np.asarray(info["taxi_before"], dtype=int).tolist()
    path_data["trips"].append(
        {
            "taxi_before": taxi_before,
            "origin": origin,
            "destination": destination,
        }
    )


def _run_direct_policy(
    env_kwargs: Mapping[str, Any],
    seed: int,
    select_action: Callable[[AdvancedTimeAttackTaxiEnv], int],
    track_path: bool,
) -> tuple[float, dict[str, Any] | None]:
    env = AdvancedTimeAttackTaxiEnv(**dict(env_kwargs))
    try:
        env.reset(seed=seed)
        path_data = _new_path_data(env.taxi_loc, track_path)
        total_cost = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            action = select_action(env)
            _, _, terminated, truncated, info = env.step(action)
            total_cost += float(info["completion_cost"])
            _record_step(path_data, info)
        if truncated:
            raise RuntimeError("AdvancedTaxi 평가 episode가 예기치 않게 truncated 되었습니다.")
        return total_cost, path_data
    finally:
        env.close()


def _run_random(
    env_kwargs: Mapping[str, Any],
    seed: int,
    track_path: bool = False,
) -> tuple[float, dict[str, Any] | None]:
    num_passengers = int(env_kwargs["num_passengers"])
    policy_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    order = iter(policy_rng.permutation(num_passengers).tolist())
    return _run_direct_policy(env_kwargs, seed, lambda _env: int(next(order)), track_path)


def _run_greedy(
    env_kwargs: Mapping[str, Any],
    seed: int,
    track_path: bool = False,
) -> tuple[float, dict[str, Any] | None]:
    def select_action(env: AdvancedTimeAttackTaxiEnv) -> int:
        available = np.flatnonzero(np.asarray(env.action_masks(), dtype=bool))
        if available.size == 0:
            raise RuntimeError("선택 가능한 승객이 없습니다.")
        taxi = np.asarray(env.taxi_loc)
        return int(
            min(
                available,
                key=lambda index: (
                    int(np.abs(taxi - np.asarray(env.passengers[int(index)][:2])).sum()),
                    int(index),
                ),
            )
        )

    return _run_direct_policy(env_kwargs, seed, select_action, track_path)


def _run_shortest_trip(
    env_kwargs: Mapping[str, Any],
    seed: int,
    track_path: bool = False,
) -> tuple[float, dict[str, Any] | None]:
    """픽업과 운행 거리의 합이 가장 짧은 승객을 선택한다."""

    def select_action(env: AdvancedTimeAttackTaxiEnv) -> int:
        available = np.flatnonzero(np.asarray(env.action_masks(), dtype=bool))
        if available.size == 0:
            raise RuntimeError("선택 가능한 승객이 없습니다.")
        taxi = np.asarray(env.taxi_loc)

        def total_trip(index: int) -> tuple[int, int]:
            passenger = np.asarray(env.passengers[index])
            pickup_distance = int(np.abs(taxi - passenger[:2]).sum())
            service_distance = int(np.abs(passenger[:2] - passenger[2:4]).sum())
            return pickup_distance + service_distance, index

        return int(min((int(index) for index in available), key=total_trip))

    return _run_direct_policy(env_kwargs, seed, select_action, track_path)


class _RLEpisodeRunner:
    """시각화 데이터를 수집하며 RL 에피소드 하나를 실행한다."""

    def __init__(
        self,
        model: MaskablePPO,
        env_kwargs: Mapping[str, Any],
        vecnormalize_path: Path | None,
    ) -> None:
        self.model = model
        self.base_env = AdvancedTimeAttackTaxiEnv(**dict(env_kwargs))
        base_env = self.base_env
        vec_env: DummyVecEnv | VecNormalize = DummyVecEnv([lambda: base_env])
        if vecnormalize_path is not None:
            vec_env = VecNormalize.load(str(vecnormalize_path), vec_env)
            vec_env.training = False
            vec_env.norm_reward = False
        self.vec_env = vec_env

    def close(self) -> None:
        self.vec_env.close()

    def run(
        self,
        seed: int,
        track_path: bool = False,
    ) -> tuple[float, dict[str, Any] | None]:
        self.vec_env.seed(seed)
        obs = self.vec_env.reset()
        path_data = _new_path_data(self.base_env.taxi_loc, track_path)
        total_cost = 0.0
        done = False
        while not done:
            try:
                masks = get_action_masks(self.vec_env)
            except Exception as exc:
                raise RuntimeError("Action mask를 가져오지 못해 RL 평가를 중단합니다.") from exc
            action, _ = self.model.predict(obs, action_masks=masks, deterministic=True)
            obs, _, dones, infos = self.vec_env.step(action)
            done = bool(np.asarray(dones)[0])
            info = infos[0]
            total_cost += float(info["completion_cost"])
            _record_step(path_data, info)
        return total_cost, path_data


def _run_rl_batch(
    model: MaskablePPO,
    env_kwargs: Mapping[str, Any],
    vecnormalize_path: Path | None,
    seeds: Sequence[int],
) -> list[float]:
    """에피소드 의미를 유지하면서 연속 seed를 묶어 평가한다."""
    seed_values = [int(seed) for seed in seeds]
    if not seed_values:
        return []
    expected = list(range(seed_values[0], seed_values[0] + len(seed_values)))
    if seed_values != expected:
        raise ValueError("배치 평가 seed는 연속된 오름차순이어야 합니다.")

    def make_env() -> AdvancedTimeAttackTaxiEnv:
        return AdvancedTimeAttackTaxiEnv(**dict(env_kwargs))

    vec_env: DummyVecEnv | VecNormalize = DummyVecEnv(
        [make_env for _ in seed_values]
    )
    try:
        if vecnormalize_path is not None:
            vec_env = VecNormalize.load(str(vecnormalize_path), vec_env)
            vec_env.training = False
            vec_env.norm_reward = False

        vec_env.seed(seed_values[0])
        obs = vec_env.reset()
        total_costs = np.zeros(len(seed_values), dtype=np.float64)
        finished = np.zeros(len(seed_values), dtype=bool)
        max_steps = int(env_kwargs["num_passengers"])

        for _ in range(max_steps):
            try:
                masks = get_action_masks(vec_env)
            except Exception as exc:
                raise RuntimeError(
                    "Action mask를 가져오지 못해 RL 배치 평가를 중단합니다."
                ) from exc
            actions, _ = model.predict(
                obs,
                action_masks=masks,
                deterministic=True,
            )
            obs, _, dones, infos = vec_env.step(actions)
            active = ~finished
            for index in np.flatnonzero(active):
                total_costs[index] += float(infos[int(index)]["completion_cost"])
            finished |= np.asarray(dones, dtype=bool)
            if np.all(finished):
                return total_costs.tolist()

        raise RuntimeError("RL 배치 평가가 예상된 승객 수 안에 종료되지 않았습니다.")
    finally:
        vec_env.close()


def _run_rl_seeds_batched(
    model: MaskablePPO,
    env_kwargs: Mapping[str, Any],
    vecnormalize_path: Path | None,
    seeds: Sequence[int],
) -> list[float]:
    seed_values = list(seeds)
    costs: list[float] = []
    for offset in range(0, len(seed_values), RL_EVALUATION_BATCH_SIZE):
        costs.extend(
            _run_rl_batch(
                model,
                env_kwargs,
                vecnormalize_path,
                seed_values[offset : offset + RL_EVALUATION_BATCH_SIZE],
            )
        )
    return costs


def visualize_path(
    path_data: dict[str, Any],
    passengers: list[list[int]],
    grid_size: int,
    scenario: str,
    policy: str,
    seed: int,
    output_dir: Path | None = None,
) -> None:
    """기록된 택시 이동 구간의 개략도를 저장한다."""
    _, ax = plt.subplots(figsize=(12, 12))
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax.set_xlabel("X Coordinate", fontsize=12)
    ax.set_ylabel("Y Coordinate", fontsize=12)
    ax.set_title(f"{scenario.upper()} Scenario - {policy.upper()} Policy (seed={seed})", fontsize=14, fontweight="bold")

    for i, p in enumerate(passengers):
        origin = [p[0], p[1]]
        dest = [p[2], p[3]]
        ax.scatter(origin[0], origin[1], c="green", s=150, marker="o", alpha=0.7, zorder=5, edgecolors="black", linewidths=1)
        ax.text(origin[0], origin[1], str(i), fontsize=9, ha="center", va="center",
                fontweight="bold", color="white", zorder=6)
        ax.scatter(dest[0], dest[1], c="red", s=150, marker="s", alpha=0.7, zorder=5, edgecolors="black", linewidths=1)
        ax.text(dest[0], dest[1], str(i), fontsize=9, ha="center", va="center",
                fontweight="bold", color="white", zorder=6)
        ax.plot([origin[0], dest[0]], [origin[1], dest[1]], "k--", alpha=0.2, linewidth=1, zorder=1)

    # 택시 시작 위치 (파란색 다이아몬드)
    if path_data["taxi_start"]:
        ax.scatter(
            path_data["taxi_start"][0],
            path_data["taxi_start"][1],
            c="blue",
            s=200,
            marker="D",
            edgecolors="black",
            linewidths=2,
            zorder=6,
        )

    # 택시 종료 위치 (보라색 다이아몬드)
    if path_data["trips"]:
        taxi_end = path_data["trips"][-1]["destination"]
        ax.scatter(
            taxi_end[0],
            taxi_end[1],
            c="purple",
            s=200,
            marker="D",
            edgecolors="black",
            linewidths=2,
            zorder=6,
        )

    # 각 승객별 이동 경로 표시
    colors = plt.cm.tab20(np.linspace(0, 1, len(path_data["trips"])))
    for trip_idx, trip in enumerate(path_data["trips"]):
        color = colors[trip_idx]
        # 빈차 이동 경로 (실선, 두껍게, 강조) - 최적화의 핵심
        ax.plot(
            [trip["taxi_before"][0], trip["origin"][0]],
            [trip["taxi_before"][1], trip["origin"][1]],
            linestyle="-",
            linewidth=4,
            color=color,
            alpha=0.9,
            zorder=3,
        )
        # 승객 운행 경로 (점선, 얇게) - 어쩔 수 없는 부분
        ax.plot(
            [trip["origin"][0], trip["destination"][0]],
            [trip["origin"][1], trip["destination"][1]],
            linestyle=":",
            linewidth=2,
            color=color,
            alpha=0.5,
            zorder=2,
        )

    empty_patch = mpatches.Patch(color="gray", alpha=0.9, label="Empty Taxi (solid, emphasized)")
    service_patch = mpatches.Patch(color="gray", alpha=0.5, label="With Passenger (dotted)")
    legend_elements = [
        mpatches.Patch(color="green", alpha=0.7, label="Passenger Origin"),
        mpatches.Patch(color="red", alpha=0.7, label="Passenger Destination"),
        empty_patch,
        service_patch,
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

    plt.tight_layout()

    target_dir = output_dir or ROOT / "visualizations"
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = target_dir / f"{scenario}_{policy}_seed{seed}.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"[정보] 시각화 저장: {filename}")

    plt.close()


def _paired_statistics(costs: np.ndarray) -> dict[str, np.ndarray]:
    baseline_costs = costs[:, : len(BASELINE_POLICIES), :]
    rl_costs = costs[:, -1:, :]
    savings = baseline_costs - rl_costs
    ddof = 1 if costs.shape[-1] > 1 else 0
    standard_deviation = savings.std(axis=2, ddof=ddof)
    return {
        "paired_mean_savings": savings.mean(axis=2),
        "paired_std_savings": standard_deviation,
        "paired_standard_errors": standard_deviation / np.sqrt(costs.shape[-1]),
        "rl_win_rates": (rl_costs < baseline_costs).mean(axis=2),
    }


def _save_evaluation_artifact(
    path: Path,
    checkpoint: str,
    env_kwargs: Mapping[str, Any],
    seeds: Sequence[int],
    costs: np.ndarray,
    paired: Mapping[str, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ddof = 1 if costs.shape[-1] > 1 else 0
    np.savez_compressed(
        path,
        checkpoint=np.asarray(checkpoint),
        env_kwargs_yaml=np.asarray(
            yaml.safe_dump(dict(env_kwargs), sort_keys=True, allow_unicode=True)
        ),
        scenarios=np.asarray(SCENARIOS),
        policies=np.asarray(POLICIES),
        baseline_policies=np.asarray(BASELINE_POLICIES),
        seeds=np.asarray(seeds, dtype=np.int64),
        costs=costs,
        mean_costs=costs.mean(axis=2),
        std_costs=costs.std(axis=2, ddof=ddof),
        **paired,
    )


def evaluate_scenarios(
    model: MaskablePPO,
    vecnormalize_path: Path | None,
    env_kwargs: Mapping[str, Any],
    results_path: Path,
    checkpoint: str,
    num_seeds: int = 100,
    seed_start: int = 10_000,
    visualize: bool = False,
    visualize_seed: int = 10_000,
    output_dir: Path | None = None,
) -> dict[str, dict[str, float]]:
    if seed_start < 0:
        raise ValueError("seed_start는 0 이상이어야 합니다.")
    if num_seeds < 1:
        raise ValueError("num_seeds는 1 이상이어야 합니다.")
    evaluation_seeds = list(range(seed_start, seed_start + num_seeds))
    scenario_costs: list[np.ndarray] = []

    for scenario in SCENARIOS:
        scenario_env_kwargs = {**env_kwargs, "scenario_mode": scenario}
        random_vals = [
            _run_random(scenario_env_kwargs, seed, track_path=False)[0]
            for seed in evaluation_seeds
        ]
        greedy_vals = [
            _run_greedy(scenario_env_kwargs, seed, track_path=False)[0]
            for seed in evaluation_seeds
        ]
        shortest_trip_vals = [
            _run_shortest_trip(scenario_env_kwargs, seed, track_path=False)[0]
            for seed in evaluation_seeds
        ]
        rl_vals = _run_rl_seeds_batched(
            model,
            scenario_env_kwargs,
            vecnormalize_path,
            evaluation_seeds,
        )
        scenario_costs.append(
            np.asarray(
                [random_vals, greedy_vals, shortest_trip_vals, rl_vals],
                dtype=np.float64,
            )
        )

        if visualize:
            rl_runner = _RLEpisodeRunner(
                model,
                scenario_env_kwargs,
                vecnormalize_path,
            )
            try:
                snapshot_env = AdvancedTimeAttackTaxiEnv(**scenario_env_kwargs)
                try:
                    snapshot_env.reset(seed=visualize_seed)
                    initial_passengers = [
                        list(passenger) for passenger in snapshot_env.passengers
                    ]
                finally:
                    snapshot_env.close()

                visual_policies = (
                    ("random", _run_random),
                    ("greedy", _run_greedy),
                    ("shortest_trip", _run_shortest_trip),
                )
                for policy_name, run_policy in visual_policies:
                    _, path_data = run_policy(
                        scenario_env_kwargs,
                        visualize_seed,
                        track_path=True,
                    )
                    if path_data:
                        visualize_path(
                            path_data,
                            initial_passengers,
                            int(scenario_env_kwargs["grid_size"]),
                            scenario,
                            policy_name,
                            visualize_seed,
                            output_dir,
                        )

                _, path_data_rl = rl_runner.run(visualize_seed, track_path=True)
                if path_data_rl:
                    visualize_path(
                        path_data_rl,
                        initial_passengers,
                        int(scenario_env_kwargs["grid_size"]),
                        scenario,
                        "rl",
                        visualize_seed,
                        output_dir,
                    )
            finally:
                rl_runner.close()

    costs = np.stack(scenario_costs)
    paired = _paired_statistics(costs)
    _save_evaluation_artifact(
        results_path,
        checkpoint,
        env_kwargs,
        evaluation_seeds,
        costs,
        paired,
    )

    results: dict[str, dict[str, float]] = {}
    ddof = 1 if costs.shape[-1] > 1 else 0
    for scenario_index, scenario in enumerate(SCENARIOS):
        scenario_results: dict[str, float] = {}
        for policy_index, policy in enumerate(POLICIES):
            values = costs[scenario_index, policy_index]
            scenario_results[f"{policy}_mean"] = float(values.mean())
            scenario_results[f"{policy}_std"] = float(values.std(ddof=ddof))
        for baseline_index, baseline in enumerate(BASELINE_POLICIES):
            prefix = f"rl_vs_{baseline}"
            scenario_results[f"{prefix}_mean_savings"] = float(
                paired["paired_mean_savings"][scenario_index, baseline_index]
            )
            scenario_results[f"{prefix}_standard_error"] = float(
                paired["paired_standard_errors"][scenario_index, baseline_index]
            )
            scenario_results[f"{prefix}_win_rate"] = float(
                paired["rl_win_rates"][scenario_index, baseline_index]
            )
        results[scenario] = scenario_results
    return results


def _validate_model_spaces(model: MaskablePPO, env_kwargs: Mapping[str, Any]) -> None:
    env = AdvancedTimeAttackTaxiEnv(**dict(env_kwargs))
    try:
        if model.observation_space != env.observation_space:
            raise ValueError(
                "모델과 평가 환경의 observation space가 다릅니다: "
                f"model={model.observation_space}, env={env.observation_space}"
            )
        if model.action_space != env.action_space:
            raise ValueError(
                "모델과 평가 환경의 action space가 다릅니다: "
                f"model={model.action_space}, env={env.action_space}"
            )
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=f"{ENV_ID} 학습 모델 평가 스크립트 (시각화 포함)")
    parser.add_argument("--logs-dir", type=str, default="logs", help="로그 디렉터리 (기본: logs)")
    parser.add_argument("--exp-id", type=int, required=True, help="평가할 학습 실험 ID")
    parser.add_argument(
        "--checkpoint",
        choices=("final", "best"),
        required=True,
        help="평가할 checkpoint와 대응 VecNormalize 통계 선택",
    )
    parser.add_argument(
        "--center-start",
        action="store_true",
        help="random/cluster를 중앙 출발로 바꾼 OOD 평가를 수행",
    )
    parser.add_argument("--num-seeds", type=int, default=100, help="시나리오별 평가 seed 수 (기본: 100)")
    parser.add_argument("--seed-start", type=int, default=10_000, help="평가 seed 시작값 (기본: 10000)")
    parser.add_argument("--num-threads", type=int, default=1, help="PyTorch CPU intra-op thread 수")
    parser.add_argument("--visualize", action="store_true", help="시각화 생성 여부")
    parser.add_argument("--visualize-seed", type=int, default=10_000, help="시각화에 사용할 시드 (기본: 10000)")
    parser.add_argument("--output-dir", type=str, default="visualizations", help="시각화 저장 디렉터리 (기본: visualizations)")
    parser.add_argument(
        "--results-file",
        type=Path,
        help="per-seed 비용과 paired 통계를 저장할 NPZ 경로",
    )
    args = parser.parse_args()
    if args.num_threads < 1:
        parser.error("--num-threads는 1 이상이어야 합니다.")
    torch.set_num_threads(args.num_threads)

    logs_dir = (ROOT / args.logs_dir).resolve()
    artifacts = resolve_run_artifacts(
        logs_dir,
        ALGO,
        ENV_ID,
        args.exp_id,
        args.checkpoint,
    )
    env_kwargs = load_evaluation_env_kwargs(artifacts)
    evaluation_mode = "저장된 평가 조건"
    if args.center_start:
        env_kwargs["start_at_center"] = True
        evaluation_mode = "random/cluster 중앙 출발 OOD"

    print(f"[정보] 모델 로드: {artifacts.model}")
    if artifacts.vecnormalize:
        print(f"[정보] VecNormalize 로드: {artifacts.vecnormalize}")
    print(f"[정보] 평가 모드: {evaluation_mode}")
    print(f"[정보] 저장된 평가 환경 설정 복원: {env_kwargs}")

    model = MaskablePPO.load(str(artifacts.model), device="cpu")
    _validate_model_spaces(model, env_kwargs)

    evaluation_tag = "center" if args.center_start else "saved"
    output_dir = (
        (ROOT / args.output_dir).resolve()
        / f"{ENV_ID}_{args.exp_id}"
        / artifacts.checkpoint
        / evaluation_tag
        if args.visualize
        else None
    )
    results_path = (
        args.results_file.expanduser().resolve()
        if args.results_file is not None
        else artifacts.run_dir
        / (
            f"evaluation_{artifacts.checkpoint}_"
            f"{evaluation_tag}_"
            f"{args.seed_start}_{args.num_seeds}.npz"
        )
    )
    if results_path.suffix != ".npz":
        results_path = Path(f"{results_path}.npz")

    results = evaluate_scenarios(
        model=model,
        vecnormalize_path=artifacts.vecnormalize,
        env_kwargs=env_kwargs,
        results_path=results_path,
        checkpoint=artifacts.checkpoint,
        num_seeds=args.num_seeds,
        seed_start=args.seed_start,
        visualize=args.visualize,
        visualize_seed=args.visualize_seed,
        output_dir=output_dir,
    )
    print(f"[정보] 평가 산출물 저장: {results_path}")

    print(
        "\n=== 성과 지표 "
        "(총 완료시간 합: 낮을수록 좋음, "
        f"seed={args.seed_start}~{args.seed_start + args.num_seeds - 1}, "
        f"{evaluation_mode}) ===\n"
    )
    header = f"{'Scenario':<10} | {'Policy':<13} | {'Mean':>12} | {'Std':>12}"
    sep = "-" * len(header)
    print(header)
    print(sep)
    for scenario, stats in results.items():
        rows = [
            (policy, stats[f"{policy}_mean"], stats[f"{policy}_std"])
            for policy in POLICIES
        ]
        for policy, mean, std in rows:
            print(f"{scenario:<10} | {policy:<13} | {mean:12.2f} | {std:12.2f}")
        print(sep)

    print("\n=== 짝지은 RL 절감량 (baseline - RL; 양수면 RL 우수) ===\n")
    paired_header = (
        f"{'Scenario':<10} | {'Baseline':<13} | {'Mean saving':>12} | "
        f"{'SE':>10} | {'Win rate':>10}"
    )
    paired_sep = "-" * len(paired_header)
    print(paired_header)
    print(paired_sep)
    for scenario, stats in results.items():
        for baseline in BASELINE_POLICIES:
            prefix = f"rl_vs_{baseline}"
            print(
                f"{scenario:<10} | {baseline:<13} | "
                f"{stats[f'{prefix}_mean_savings']:12.2f} | "
                f"{stats[f'{prefix}_standard_error']:10.2f} | "
                f"{stats[f'{prefix}_win_rate']:9.1%}"
            )
        print(paired_sep)


if __name__ == "__main__":
    main()
