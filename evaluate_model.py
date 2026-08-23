"""
AdvancedTaxi-v1 학습 모델 평가 스크립트 (시각화 포함)

- random / cluster / complex 시나리오에 대해
  - Random 정책
  - Greedy 정책
  - RL(MaskablePPO) 정책
  의 총 완료시간 합(sum of completion times)을 요약해서 출력합니다.

- 시각화: 각 정책의 이동 경로를 그리드월드에 표시
  - 승객 출발지/목적지
  - 빈차 이동 경로
  - 승객 운행 경로
  - 택시 시작/종료 위치
"""

from __future__ import annotations

import argparse
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import yaml
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# 프로젝트 루트를 Python 경로에 추가
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 환경 등록
import envs  # noqa: F401
from envs.envs import AdvancedTimeAttackTaxiEnv

ALGO = "maskableppo"
ENV_ID = "AdvancedTaxi-v1"


@dataclass(frozen=True)
class RunArtifacts:
    run_dir: Path
    model: Path
    vecnormalize: Path | None
    config: Path
    args: Path


class _RLZooSafeLoader(yaml.SafeLoader):
    """RL-Zoo가 저장한 OrderedDict 태그만 안전하게 읽는 로더."""


def _construct_ordered_dict(loader: yaml.SafeLoader, node: yaml.Node) -> dict[str, Any]:
    values = loader.construct_sequence(node, deep=True)
    if len(values) != 1:
        raise yaml.constructor.ConstructorError(
            None,
            None,
            "Invalid RL-Zoo OrderedDict payload",
            node.start_mark,
        )
    return dict(values[0])


_RLZooSafeLoader.add_constructor(
    "tag:yaml.org,2002:python/object/apply:collections.OrderedDict",
    _construct_ordered_dict,
)


def _load_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.load(file, Loader=_RLZooSafeLoader)
    if not isinstance(data, dict):
        raise ValueError(f"설정 파일의 최상위 값이 mapping이 아닙니다: {path}")
    return data


def _uses_vecnormalize(value: Any) -> bool:
    """Mirror RL-Zoo's normalization activation semantics."""
    if isinstance(value, (dict, str)):
        return True
    return bool(value)


def resolve_run_artifacts(
    logs_dir: Path,
    algo: str,
    env_id: str,
    exp_id: int,
) -> RunArtifacts:
    """명시한 RL-Zoo 실행의 final model과 설정 파일을 해석한다."""
    if exp_id < 1:
        raise ValueError("exp_id는 1 이상이어야 합니다.")

    run_dir = logs_dir / algo / f"{env_id}_{exp_id}"
    params_dir = run_dir / env_id
    artifacts = RunArtifacts(
        run_dir=run_dir,
        model=run_dir / f"{env_id}.zip",
        vecnormalize=(params_dir / "vecnormalize.pkl"),
        config=params_dir / "config.yml",
        args=params_dir / "args.yml",
    )

    if not artifacts.run_dir.is_dir():
        raise FileNotFoundError(f"run을 찾을 수 없습니다: {artifacts.run_dir}")

    for label, path in {
        "final model": artifacts.model,
        "config": artifacts.config,
        "args": artifacts.args,
    }.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label}을(를) 찾을 수 없습니다: {path}")

    config = _load_mapping(artifacts.config)
    normalize = config.get("normalize", False)
    uses_normalization = _uses_vecnormalize(normalize)
    if uses_normalization and not artifacts.vecnormalize.is_file():
        raise FileNotFoundError(
            "이 실험은 VecNormalize를 사용하지만 통계 파일이 없습니다: "
            f"{artifacts.vecnormalize}"
        )

    if not uses_normalization:
        return RunArtifacts(
            run_dir=artifacts.run_dir,
            model=artifacts.model,
            vecnormalize=None,
            config=artifacts.config,
            args=artifacts.args,
        )
    return artifacts


def load_evaluation_env_kwargs(artifacts: RunArtifacts) -> dict[str, Any]:
    """Load the complete evaluation environment contract saved by training."""
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


def _scenario_kwargs(base_kwargs: Mapping[str, Any], scenario: str) -> dict[str, Any]:
    kwargs = dict(base_kwargs)
    kwargs["scenario_mode"] = scenario
    kwargs["mix_noise_std"] = 0.0
    return kwargs


def _evaluation_seeds(seed_start: int, num_seeds: int) -> range:
    if seed_start < 0:
        raise ValueError("seed_start는 0 이상이어야 합니다.")
    if num_seeds < 1:
        raise ValueError("num_seeds는 1 이상이어야 합니다.")
    return range(seed_start, seed_start + num_seeds)


def _new_path_data(taxi_start: np.ndarray, track_path: bool) -> Dict[str, Any] | None:
    if not track_path:
        return None
    return {"taxi_start": taxi_start.tolist(), "taxi_end": None, "trips": []}


def _record_step(path_data: Dict[str, Any] | None, info: Mapping[str, Any]) -> None:
    if path_data is None:
        return
    origin = np.asarray(info["passenger_origin"], dtype=int).tolist()
    destination = np.asarray(info["passenger_destination"], dtype=int).tolist()
    taxi_before = np.asarray(info["taxi_before"], dtype=int).tolist()
    taxi_after = np.asarray(info["taxi_after"], dtype=int).tolist()
    path_data["trips"].append(
        {
            "passenger_idx": int(info["selected_passenger"]),
            "passenger_origin": origin,
            "passenger_dest": destination,
            "empty_path_start": taxi_before,
            "empty_path_end": origin,
            "service_path_start": origin,
            "service_path_end": destination,
        }
    )
    path_data["taxi_end"] = taxi_after


def _run_direct_policy(
    env_kwargs: Mapping[str, Any],
    seed: int,
    select_action: Callable[[AdvancedTimeAttackTaxiEnv], int],
    track_path: bool,
) -> Tuple[float, Dict[str, Any] | None]:
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
) -> Tuple[float, Dict[str, Any] | None]:
    num_passengers = int(env_kwargs["num_passengers"])
    policy_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    order = iter(policy_rng.permutation(num_passengers).tolist())
    return _run_direct_policy(env_kwargs, seed, lambda _env: int(next(order)), track_path)


def _run_greedy(
    env_kwargs: Mapping[str, Any],
    seed: int,
    track_path: bool = False,
) -> Tuple[float, Dict[str, Any] | None]:
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


class _RLEpisodeRunner:
    """한 시나리오에서 VecNormalize를 한 번만 로드해 여러 seed를 평가한다."""

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
    ) -> Tuple[float, Dict[str, Any] | None]:
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


def visualize_path(
    path_data: Dict[str, Any],
    passengers: List[List[int]],
    grid_size: int,
    scenario: str,
    policy: str,
    seed: int,
    output_dir: Path | None = None,
) -> None:
    """경로 데이터를 시각화하여 저장"""
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax.set_xlabel("X Coordinate", fontsize=12)
    ax.set_ylabel("Y Coordinate", fontsize=12)
    ax.set_title(f"{scenario.upper()} Scenario - {policy.upper()} Policy (seed={seed})", fontsize=14, fontweight="bold")

    # 승객 출발지/목적지 표시 (번호 포함)
    for i, p in enumerate(passengers):
        origin = [p[0], p[1]]
        dest = [p[2], p[3]]
        # 출발지 (녹색 원)
        ax.scatter(origin[0], origin[1], c="green", s=150, marker="o", alpha=0.7, zorder=5, edgecolors="black", linewidths=1)
        # 출발지에 승객 번호 표시
        ax.text(origin[0], origin[1], str(i), fontsize=9, ha="center", va="center", 
                fontweight="bold", color="white", zorder=6)
        # 목적지 (빨간색 사각형)
        ax.scatter(dest[0], dest[1], c="red", s=150, marker="s", alpha=0.7, zorder=5, edgecolors="black", linewidths=1)
        # 목적지에 승객 번호 표시
        ax.text(dest[0], dest[1], str(i), fontsize=9, ha="center", va="center", 
                fontweight="bold", color="white", zorder=6)
        # 출발지-목적지 연결선 (회색 점선)
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
            label="Taxi Start",
        )

    # 택시 종료 위치 (보라색 다이아몬드)
    if path_data["taxi_end"]:
        ax.scatter(
            path_data["taxi_end"][0],
            path_data["taxi_end"][1],
            c="purple",
            s=200,
            marker="D",
            edgecolors="black",
            linewidths=2,
            zorder=6,
            label="Taxi End",
        )

    # 각 승객별 이동 경로 표시
    colors = plt.cm.tab20(np.linspace(0, 1, len(path_data["trips"])))
    for trip_idx, trip in enumerate(path_data["trips"]):
        color = colors[trip_idx]
        # 빈차 이동 경로 (실선, 두껍게, 강조) - 최적화의 핵심
        ax.plot(
            [trip["empty_path_start"][0], trip["empty_path_end"][0]],
            [trip["empty_path_start"][1], trip["empty_path_end"][1]],
            linestyle="-",
            linewidth=4,
            color=color,
            alpha=0.9,
            zorder=3,
        )
        # 승객 운행 경로 (점선, 얇게) - 어쩔 수 없는 부분
        ax.plot(
            [trip["service_path_start"][0], trip["service_path_end"][0]],
            [trip["service_path_start"][1], trip["service_path_end"][1]],
            linestyle=":",
            linewidth=2,
            color=color,
            alpha=0.5,
            zorder=2,
        )

    # 범례
    empty_patch = mpatches.Patch(color="gray", alpha=0.9, label="Empty Taxi (solid, emphasized)")
    service_patch = mpatches.Patch(color="gray", alpha=0.5, label="With Passenger (dashed)")
    legend_elements = [
        mpatches.Patch(color="green", alpha=0.7, label="Passenger Origin"),
        mpatches.Patch(color="red", alpha=0.7, label="Passenger Destination"),
        empty_patch,
        service_patch,
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = output_dir / f"{scenario}_{policy}_seed{seed}.png"
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"[INFO] 시각화 저장: {filename}")
    else:
        plt.show()

    plt.close()


def evaluate_scenarios(
    model: MaskablePPO,
    vecnormalize_path: Path | None,
    env_kwargs: Mapping[str, Any],
    num_seeds: int = 100,
    seed_start: int = 10_000,
    visualize: bool = False,
    visualize_seed: int = 10_000,
    output_dir: Path | None = None,
) -> Dict[str, Dict[str, float]]:
    evaluation_seeds = _evaluation_seeds(seed_start, num_seeds)

    results: Dict[str, Dict[str, float]] = {}
    scenario_modes = ["random", "cluster", "complex"]

    for scenario in scenario_modes:
        scenario_env_kwargs = _scenario_kwargs(env_kwargs, scenario)
        random_vals = []
        greedy_vals = []
        rl_vals = []
        rl_runner = _RLEpisodeRunner(model, scenario_env_kwargs, vecnormalize_path)
        try:
            for seed in evaluation_seeds:
                total_cost, _ = _run_random(scenario_env_kwargs, seed, track_path=False)
                random_vals.append(total_cost)
                total_cost, _ = _run_greedy(scenario_env_kwargs, seed, track_path=False)
                greedy_vals.append(total_cost)
                total_cost, _ = rl_runner.run(seed, track_path=False)
                rl_vals.append(total_cost)

            if visualize:
                snapshot_env = AdvancedTimeAttackTaxiEnv(**scenario_env_kwargs)
                try:
                    snapshot_env.reset(seed=visualize_seed)
                    initial_passengers = [list(passenger) for passenger in snapshot_env.passengers]
                finally:
                    snapshot_env.close()

                _, path_data_random = _run_random(
                    scenario_env_kwargs,
                    visualize_seed,
                    track_path=True,
                )
                if path_data_random:
                    visualize_path(
                        path_data_random,
                        initial_passengers,
                        int(scenario_env_kwargs["grid_size"]),
                        scenario,
                        "random",
                        visualize_seed,
                        output_dir,
                    )

                _, path_data_greedy = _run_greedy(
                    scenario_env_kwargs,
                    visualize_seed,
                    track_path=True,
                )
                if path_data_greedy:
                    visualize_path(
                        path_data_greedy,
                        initial_passengers,
                        int(scenario_env_kwargs["grid_size"]),
                        scenario,
                        "greedy",
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

        results[scenario] = {
            "random_mean": float(np.mean(random_vals)),
            "random_std": float(np.std(random_vals)),
            "greedy_mean": float(np.mean(greedy_vals)),
            "greedy_std": float(np.std(greedy_vals)),
            "rl_mean": float(np.mean(rl_vals)),
            "rl_std": float(np.std(rl_vals)),
        }

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
    parser.add_argument("--exp-id", type=int, required=True, help="평가할 RL-Zoo 실험 ID")
    parser.add_argument(
        "--center-start",
        action="store_true",
        help="학습 출발 설정 대신 중앙 출발 OOD 평가를 수행",
    )
    parser.add_argument("--num-seeds", type=int, default=100, help="시나리오별 평가 seed 수 (기본: 100)")
    parser.add_argument("--seed-start", type=int, default=10_000, help="평가 seed 시작값 (기본: 10000)")
    parser.add_argument("--visualize", action="store_true", help="시각화 생성 여부")
    parser.add_argument("--visualize-seed", type=int, default=10_000, help="시각화에 사용할 시드 (기본: 10000)")
    parser.add_argument("--output-dir", type=str, default="visualizations", help="시각화 저장 디렉터리 (기본: visualizations)")
    args = parser.parse_args()

    logs_dir = (ROOT / args.logs_dir).resolve()
    artifacts = resolve_run_artifacts(logs_dir, ALGO, ENV_ID, args.exp_id)
    env_kwargs = load_evaluation_env_kwargs(artifacts)
    evaluation_mode = "저장된 평가 조건"
    if args.center_start:
        env_kwargs["start_at_center"] = True
        evaluation_mode = "중앙 출발 OOD"

    print(f"[INFO] 모델 로드: {artifacts.model}")
    if artifacts.vecnormalize:
        print(f"[INFO] VecNormalize 로드: {artifacts.vecnormalize}")
    print(f"[INFO] 평가 모드: {evaluation_mode}")
    print(f"[INFO] 저장된 평가 환경 설정 복원: {env_kwargs}")

    model = MaskablePPO.load(str(artifacts.model))
    _validate_model_spaces(model, env_kwargs)

    output_dir = (ROOT / args.output_dir).resolve() if args.visualize else None

    results = evaluate_scenarios(
        model=model,
        vecnormalize_path=artifacts.vecnormalize,
        env_kwargs=env_kwargs,
        num_seeds=args.num_seeds,
        seed_start=args.seed_start,
        visualize=args.visualize,
        visualize_seed=args.visualize_seed,
        output_dir=output_dir,
    )

    print(
        "\n=== 성과 지표 "
        "(총 완료시간 합: 낮을수록 좋음, "
        f"seed={args.seed_start}~{args.seed_start + args.num_seeds - 1}, "
        f"{evaluation_mode}) ===\n"
    )
    header = f"{'Scenario':<10} | {'Policy':<8} | {'Mean':>12} | {'Std':>12}"
    sep = "-" * len(header)
    print(header)
    print(sep)
    for scenario, stats in results.items():
        rows = [
            ("random", stats["random_mean"], stats["random_std"]),
            ("greedy", stats["greedy_mean"], stats["greedy_std"]),
            ("RL", stats["rl_mean"], stats["rl_std"]),
        ]
        for policy, mean, std in rows:
            print(f"{scenario:<10} | {policy:<8} | {mean:12.2f} | {std:12.2f}")
        print(sep)


if __name__ == "__main__":
    main()
