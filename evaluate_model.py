"""
AdvancedTaxi-v0 학습 모델 평가 스크립트 (시각화 포함)

- random / cluster / complex 시나리오에 대해
  - Random 정책
  - Greedy 정책
  - RL(MaskablePPO) 정책
  의 총 대기시간 지표를 요약해서 출력합니다.

- 시각화: 각 정책의 이동 경로를 그리드월드에 표시
  - 승객 출발지/목적지
  - 빈차 이동 경로
  - 승객 운행 경로
  - 택시 시작/종료 위치
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# 프로젝트 루트를 Python 경로에 추가
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 환경 등록
import envs  # noqa: F401
from envs.envs import AdvancedTimeAttackTaxiEnv


def _find_latest_model(logs_dir: Path, algo: str, env: str) -> Dict[str, Path]:
    """
    logs/{algo}/{env}_*/ 하위에서 가장 최신 실험을 찾아
    모델(.zip)과 VecNormalize(.pkl, 선택)를 반환.
    """
    base = logs_dir / algo
    if not base.exists():
        raise FileNotFoundError(f"로그 디렉터리가 없습니다: {base}")

    # 예: AdvancedTaxi-v0_1, AdvancedTaxi-v0_2 ...
    candidates = sorted(
        [p for p in base.iterdir() if p.is_dir() and p.name.startswith(f"{env}_")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"{base}에서 {env}_* 디렉터리를 찾을 수 없습니다.")

    exp_dir = candidates[0]
    # RL-Zoo3 스타일: 실험 루트(예: AdvancedTaxi-v0_4) 바로 아래에 모델 zip들이 위치
    model_files = sorted(exp_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not model_files:
        raise FileNotFoundError(f"모델(.zip)을 찾을 수 없습니다: {exp_dir}")

    model_path = model_files[0]

    # VecNormalize 통계 (있으면 사용, 없어도 진행)
    vecnormalize_path = None
    vec_files = sorted(exp_dir.glob("*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if vec_files:
        vecnormalize_path = vec_files[0]

    return {"model": model_path, "vecnormalize": vecnormalize_path}


def _run_random(
    num_passengers: int,
    grid_size: int,
    seed: int,
    scenario_mode: str,
    start_at_center: bool = True,
    track_path: bool = False,
) -> Tuple[float, Dict[str, Any] | None]:
    env = AdvancedTimeAttackTaxiEnv(
        num_passengers=num_passengers,
        grid_size=grid_size,
        scenario_mode=scenario_mode,
        start_at_center=start_at_center,
    )
    env.reset(seed=seed)

    passengers = [list(p) for p in env.passengers]
    taxi_pos = list(env.taxi_loc)
    taxi_start = list(taxi_pos)

    rng = np.random.RandomState(seed)
    indices = list(range(num_passengers))
    rng.shuffle(indices)

    total_wait_time = 0.0
    path_data = {
        "taxi_start": taxi_start,
        "taxi_end": None,
        "trips": [],
    } if track_path else None

    for idx in indices:
        p = passengers[idx]
        empty_dist = abs(taxi_pos[0] - p[0]) + abs(taxi_pos[1] - p[1])
        service_dist = abs(p[0] - p[2]) + abs(p[1] - p[3])
        current_trip_time = empty_dist + service_dist
        remaining = sum(1 for q in passengers if q[4] == 0)
        total_wait_time += current_trip_time * remaining

        if track_path:
            path_data["trips"].append({
                "passenger_idx": idx,
                "passenger_origin": [p[0], p[1]],
                "passenger_dest": [p[2], p[3]],
                "empty_path_start": list(taxi_pos),
                "empty_path_end": [p[0], p[1]],
                "service_path_start": [p[0], p[1]],
                "service_path_end": [p[2], p[3]],
            })

        taxi_pos = [p[2], p[3]]
        passengers[idx][4] = 1

    if track_path:
        path_data["taxi_end"] = list(taxi_pos)

    return float(total_wait_time), path_data


def _run_greedy(
    num_passengers: int,
    grid_size: int,
    seed: int,
    scenario_mode: str,
    start_at_center: bool = True,
    track_path: bool = False,
) -> Tuple[float, Dict[str, Any] | None]:
    env = AdvancedTimeAttackTaxiEnv(
        num_passengers=num_passengers,
        grid_size=grid_size,
        scenario_mode=scenario_mode,
        start_at_center=start_at_center,
    )
    env.reset(seed=seed)

    passengers = [list(p) for p in env.passengers]
    taxi_pos = list(env.taxi_loc)
    taxi_start = list(taxi_pos)

    total_wait_time = 0.0
    path_data = {
        "taxi_start": taxi_start,
        "taxi_end": None,
        "trips": [],
    } if track_path else None

    for _ in range(num_passengers):
        best_idx = -1
        min_dist = 10**9
        for i, p in enumerate(passengers):
            if p[4] == 0:
                dist = abs(taxi_pos[0] - p[0]) + abs(taxi_pos[1] - p[1])
                if dist < min_dist:
                    min_dist = dist
                    best_idx = i
        if best_idx == -1:
            break

        p = passengers[best_idx]
        empty_dist = abs(taxi_pos[0] - p[0]) + abs(taxi_pos[1] - p[1])
        service_dist = abs(p[0] - p[2]) + abs(p[1] - p[3])
        current_trip_time = empty_dist + service_dist
        remaining = sum(1 for q in passengers if q[4] == 0)
        total_wait_time += current_trip_time * remaining

        if track_path:
            path_data["trips"].append({
                "passenger_idx": best_idx,
                "passenger_origin": [p[0], p[1]],
                "passenger_dest": [p[2], p[3]],
                "empty_path_start": list(taxi_pos),
                "empty_path_end": [p[0], p[1]],
                "service_path_start": [p[0], p[1]],
                "service_path_end": [p[2], p[3]],
            })

        taxi_pos = [p[2], p[3]]
        passengers[best_idx][4] = 1

    if track_path:
        path_data["taxi_end"] = list(taxi_pos)

    return float(total_wait_time), path_data


def _run_rl(
    model: MaskablePPO,
    num_passengers: int,
    grid_size: int,
    seed: int,
    scenario_mode: str,
    vecnormalize_path: Path | None,
    start_at_center: bool = True,
    track_path: bool = False,
) -> Tuple[float, Dict[str, Any] | None]:
    if vecnormalize_path:
        base_env = AdvancedTimeAttackTaxiEnv(
            num_passengers=num_passengers,
            grid_size=grid_size,
            scenario_mode=scenario_mode,
            start_at_center=start_at_center,
        )
        masked_env = ActionMasker(base_env, lambda e: e.action_masks())
        vec_env = DummyVecEnv([lambda: masked_env])
        vec_env = VecNormalize.load(str(vecnormalize_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        vec_env.seed(seed)
        obs = vec_env.reset()
        inner_env = vec_env.venv.envs[0]
        base_ref = getattr(inner_env, "env", inner_env)
    else:
        vec_env = None
        env = AdvancedTimeAttackTaxiEnv(
            num_passengers=num_passengers,
            grid_size=grid_size,
            scenario_mode=scenario_mode,
            start_at_center=start_at_center,
        )
        env = ActionMasker(env, lambda e: e.action_masks())
        obs, _ = env.reset(seed=seed)
        base_ref = getattr(env, "env", env)

    passengers = [list(p) for p in base_ref.passengers]
    taxi_pos = list(base_ref.taxi_loc)
    taxi_start = list(taxi_pos)

    total_wait_time = 0.0
    done = False
    path_data = {
        "taxi_start": taxi_start,
        "taxi_end": None,
        "trips": [],
    } if track_path else None

    while not done:
        if vec_env:
            # VecEnv에서 action mask 추출
            try:
                masks = vec_env.env_method("action_masks")
                if isinstance(masks, list) and masks:
                    mask = np.asarray(masks[0], dtype=bool)
                else:
                    mask = None
            except (AttributeError, TypeError, IndexError) as e:
                # action_masks() 메서드가 없거나 호출 실패 시 None으로 폴백
                mask = None
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, _, dones, _ = vec_env.step(action)
            done = bool(np.array(dones)[0])
            a = int(np.array(action)[0])
        else:
            mask = base_ref.action_masks()
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, _, done, _, _ = env.step(action)
            a = int(action)

        p = passengers[a]
        empty_dist = abs(taxi_pos[0] - p[0]) + abs(taxi_pos[1] - p[1])
        service_dist = abs(p[0] - p[2]) + abs(p[1] - p[3])
        current_trip_time = empty_dist + service_dist
        remaining = sum(1 for q in passengers if q[4] == 0)
        total_wait_time += current_trip_time * remaining

        if track_path:
            path_data["trips"].append({
                "passenger_idx": a,
                "passenger_origin": [p[0], p[1]],
                "passenger_dest": [p[2], p[3]],
                "empty_path_start": list(taxi_pos),
                "empty_path_end": [p[0], p[1]],
                "service_path_start": [p[0], p[1]],
                "service_path_end": [p[2], p[3]],
            })

        taxi_pos = [p[2], p[3]]
        passengers[a][4] = 1

    if track_path:
        path_data["taxi_end"] = list(taxi_pos)

    return float(total_wait_time), path_data


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
    num_passengers: int,
    grid_size: int,
    start_at_center: bool = True,
    visualize: bool = False,
    visualize_seed: int = 1,
    output_dir: Path | None = None,
) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    scenario_modes = ["random", "cluster", "complex"]
    num_seeds = 100  # 시드 1~100 고정 평균

    for scenario in scenario_modes:
        random_vals = []
        greedy_vals = []
        rl_vals = []
        for seed in range(1, num_seeds + 1):
            wait_time, _ = _run_random(num_passengers, grid_size, seed, scenario, start_at_center, track_path=False)
            random_vals.append(wait_time)
            wait_time, _ = _run_greedy(num_passengers, grid_size, seed, scenario, start_at_center, track_path=False)
            greedy_vals.append(wait_time)
            wait_time, _ = _run_rl(
                model,
                num_passengers,
                grid_size,
                seed,
                scenario,
                vecnormalize_path,
                start_at_center,
                track_path=False,
            )
            rl_vals.append(wait_time)

        results[scenario] = {
            "random_mean": float(np.mean(random_vals)),
            "random_std": float(np.std(random_vals)),
            "greedy_mean": float(np.mean(greedy_vals)),
            "greedy_std": float(np.std(greedy_vals)),
            "rl_mean": float(np.mean(rl_vals)),
            "rl_std": float(np.std(rl_vals)),
        }

        # 시각화 (지정된 시드에 대해서만)
        if visualize:
            # 환경을 다시 생성하여 초기 승객 정보 가져오기
            env = AdvancedTimeAttackTaxiEnv(
                num_passengers=num_passengers,
                grid_size=grid_size,
                scenario_mode=scenario,
                start_at_center=start_at_center,
            )
            env.reset(seed=visualize_seed)
            initial_passengers = [list(p) for p in env.passengers]  # 초기 상태 저장

            # 각 정책별 시각화
            # Random 정책
            _, path_data_random = _run_random(
                num_passengers, grid_size, visualize_seed, scenario, start_at_center, track_path=True
            )
            if path_data_random:
                visualize_path(
                    path_data_random, initial_passengers, grid_size, scenario, "random", visualize_seed, output_dir
                )

            # Greedy 정책
            _, path_data_greedy = _run_greedy(
                num_passengers, grid_size, visualize_seed, scenario, start_at_center, track_path=True
            )
            if path_data_greedy:
                visualize_path(
                    path_data_greedy, initial_passengers, grid_size, scenario, "greedy", visualize_seed, output_dir
                )

            # RL 정책
            _, path_data_rl = _run_rl(
                model,
                num_passengers,
                grid_size,
                visualize_seed,
                scenario,
                vecnormalize_path,
                start_at_center,
                track_path=True,
            )
            if path_data_rl:
                visualize_path(
                    path_data_rl, initial_passengers, grid_size, scenario, "rl", visualize_seed, output_dir
                )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="AdvancedTaxi-v0 학습 모델 평가 스크립트 (시각화 포함)")
    parser.add_argument("--logs-dir", type=str, default="logs", help="로그 디렉터리 (기본: logs)")
    parser.add_argument("--num-passengers", type=int, default=20, help="승객 수 (기본: 20)")
    parser.add_argument("--grid-size", type=int, default=100, help="그리드 크기 (기본: 100)")
    parser.add_argument("--visualize", action="store_true", help="시각화 생성 여부")
    parser.add_argument("--visualize-seed", type=int, default=1, help="시각화에 사용할 시드 (기본: 1)")
    parser.add_argument("--output-dir", type=str, default="visualizations", help="시각화 저장 디렉터리 (기본: visualizations)")
    args = parser.parse_args()

    # 고정 설정: AdvancedTaxi-v0 + maskableppo
    algo = "maskableppo"
    env_id = "AdvancedTaxi-v0"

    logs_dir = (ROOT / args.logs_dir).resolve()
    paths = _find_latest_model(logs_dir, algo, env_id)

    print(f"[INFO] 모델 로드: {paths['model']}")
    if paths["vecnormalize"]:
        print(f"[INFO] VecNormalize 로드: {paths['vecnormalize']}")

    model = MaskablePPO.load(str(paths["model"]))
    vec_path = paths["vecnormalize"]

    output_dir = (ROOT / args.output_dir).resolve() if args.visualize else None

    results = evaluate_scenarios(
        model=model,
        vecnormalize_path=vec_path,
        num_passengers=args.num_passengers,
        grid_size=args.grid_size,
        start_at_center=True,
        visualize=args.visualize,
        visualize_seed=args.visualize_seed,
        output_dir=output_dir,
    )

    print("\n=== 성과 지표 (총 대기시간: 낮을수록 좋음, seed=1~100 평균) ===\n")
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


