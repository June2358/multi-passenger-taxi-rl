"""
Q-Learning 학습 모듈 (RL-Zoo 스타일)

Taxi-v4 원본 환경(Discrete 500)을 사용하는 Tabular Q-Learning 구현.
학습 설정과 실행 metadata를 분리해 저장하고 Monitor/TensorBoard를 기록합니다.
"""

import argparse
import time
from collections import deque
from contextlib import ExitStack
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import yaml
from stable_baselines3.common.monitor import Monitor
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parent

QLEARNING_HYPERPARAM_KEYS = frozenset(
    {
        "learning_rate",
        "gamma",
        "epsilon_start",
        "epsilon_end",
        "epsilon_decay_episodes",
    }
)

ROLLING_WINDOW_SIZE = 100
TENSORBOARD_LOG_INTERVAL_EPISODES = 200


def display_path(path: Path) -> str:
    """프로젝트 외부 경로도 허용하면서 읽기 쉬운 경로를 반환한다."""
    resolved_path = path.resolve()
    try:
        return str(resolved_path.relative_to(ROOT))
    except ValueError:
        return str(resolved_path)


def derive_rng_seeds(seed: int | None) -> dict[str, int | None]:
    """seed 하나에서 독립적인 환경·정책 RNG 스트림을 파생한다."""
    names = ("train_env", "train_policy", "eval_env", "eval_policy")
    if seed is None:
        return dict.fromkeys(names)

    children = np.random.SeedSequence(seed).spawn(len(names))
    return {
        name: int(child.generate_state(1, dtype=np.uint32)[0])
        for name, child in zip(names, children)
    }


def select_epsilon_greedy_action(
    q_values: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
) -> int:
    """전체 이산 행동 집합에서 행동을 선택한다.

    기존 Taxi-v4 Q-Learning 기준선은 mask를 쓰지 않으므로 탐험과 탐욕 선택 모두
    전체 행동을 고려한다. 탐욕 동점은 무작위로 해소해 낮은 인덱스 편향을 막는다."""
    if rng.random() < epsilon:
        return int(rng.integers(q_values.size))

    max_value = np.max(q_values)
    greedy_actions = np.flatnonzero(q_values == max_value)
    return int(rng.choice(greedy_actions))


def compute_td_target(
    reward: float,
    gamma: float,
    next_q_values: np.ndarray,
    *,
    terminated: bool,
) -> float:
    """시간 제한 truncation에서도 bootstrap하는 Q-Learning 목표값을 계산한다."""
    if terminated:
        return float(reward)
    return float(reward + gamma * np.max(next_q_values))


def create_experiment_dir(
    log_dir: Path,
    env_id: str,
    exp_id: int | None,
) -> tuple[int, Path]:
    """다른 실행을 덮어쓰지 않고 실행 디렉터리 하나를 원자적으로 만든다."""
    log_dir.mkdir(parents=True, exist_ok=True)

    if exp_id is not None:
        exp_dir = log_dir / f"{env_id}_{exp_id}"
        try:
            exp_dir.mkdir()
        except FileExistsError as exc:
            raise FileExistsError(
                f"실험 디렉터리가 이미 있습니다 (already exists): {exp_dir}"
            ) from exc
        return exp_id, exp_dir

    candidate = 1
    while True:
        exp_dir = log_dir / f"{env_id}_{candidate}"
        try:
            exp_dir.mkdir()
            return candidate, exp_dir
        except FileExistsError:
            candidate += 1


def load_hyperparams(conf_file: Path, env_id: str) -> dict[str, Any]:
    """YAML 파일에서 하이퍼파라미터를 로드합니다"""
    with open(conf_file, encoding="utf-8") as f:
        hyperparams = yaml.safe_load(f)

    if not isinstance(hyperparams, dict) or env_id not in hyperparams:
        raise ValueError(f"{conf_file}에서 환경 {env_id}를 찾을 수 없습니다.")

    unknown_top_level = set(hyperparams) - {env_id, "metadata"}
    if unknown_top_level:
        raise ValueError(
            f"알 수 없는 최상위 설정 키: {sorted(unknown_top_level)}"
        )
    if "metadata" in hyperparams and not isinstance(hyperparams["metadata"], dict):
        raise ValueError("metadata는 mapping이어야 합니다.")

    env_hyperparams = hyperparams[env_id]
    if not isinstance(env_hyperparams, dict):
        raise ValueError(f"{env_id}의 하이퍼파라미터는 mapping이어야 합니다.")
    unknown = set(env_hyperparams) - QLEARNING_HYPERPARAM_KEYS
    missing = QLEARNING_HYPERPARAM_KEYS - set(env_hyperparams)
    if unknown or missing:
        details = []
        if unknown:
            details.append(f"unknown={sorted(unknown)}")
        if missing:
            details.append(f"missing={sorted(missing)}")
        raise ValueError(f"잘못된 Q-Learning 하이퍼파라미터: {', '.join(details)}")
    return env_hyperparams


def _real_hyperparameter(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label}는 실수여야 합니다.")
    return float(value)


def _integer_hyperparameter(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{label}는 정수여야 합니다.")
    return int(value)


def save_config(
    config_path: Path,
    hyperparams: dict[str, Any],
    env_id: str,
    metadata: dict[str, Any],
) -> None:
    """재사용할 하이퍼파라미터를 실행 메타데이터와 분리해 저장한다."""
    config = {env_id: hyperparams, "metadata": metadata}
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(
            config,
            f,
            default_flow_style=False,
            allow_unicode=False,
            sort_keys=False,
        )


def train_qlearning(
    env_id: str = "Taxi-v4",
    conf_file: Path | None = None,
    n_timesteps: int = 500_000,
    seed: int | None = None,
    exp_id: int | None = None,
    log_dir: Path | None = None,
    verbose: int = 1,
):
    """
    Q-Learning 학습 함수 (RL-Zoo 스타일)

    인자:
        env_id: 환경 ID (예: "Taxi-v4")
        conf_file: 하이퍼파라미터 YAML 파일 경로
        n_timesteps: 총 학습 타임스텝 수
        seed: 랜덤 시드
        exp_id: 실험 ID (None이면 자동 생성)
        log_dir: 로그 디렉토리 (기본값: logs/qlearning)
        verbose: 출력 상세도 (0, 1, 2)
    """
    if env_id != "Taxi-v4":
        raise ValueError(
            f"Tabular Q-Learning에서 지원하지 않는 환경: {env_id}. "
            "Taxi-v4만 지원합니다 (Only Taxi-v4 is supported)."
        )
    if isinstance(n_timesteps, bool) or not isinstance(n_timesteps, Integral):
        raise ValueError("n_timesteps는 정수여야 합니다.")
    if n_timesteps < 1:
        raise ValueError("n_timesteps는 1 이상이어야 합니다.")
    if exp_id is not None:
        if isinstance(exp_id, bool) or not isinstance(exp_id, Integral):
            raise ValueError("exp_id는 정수여야 합니다.")
        if exp_id < 1:
            raise ValueError("exp_id는 1 이상이어야 합니다.")
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, Integral):
            raise ValueError("seed는 정수여야 합니다.")
        if seed < 0:
            raise ValueError("seed는 0 이상이어야 합니다.")
        seed = int(seed)
    n_timesteps = int(n_timesteps)
    if exp_id is not None:
        exp_id = int(exp_id)

    if log_dir is None:
        log_dir = ROOT / "logs" / "qlearning"

    if conf_file is None:
        env_dir = ROOT / "hyperparameters" / "taxi_v4"
        conf_file = env_dir / "qlearning.yml"

    if not conf_file.exists():
        raise FileNotFoundError(f"하이퍼파라미터 파일을 찾을 수 없습니다: {conf_file}")

    hyperparams = load_hyperparams(conf_file, env_id)

    # 하이퍼파라미터 및 실행 인자를 실제 사용값으로 해석합니다.
    learning_rate = _real_hyperparameter(
        hyperparams["learning_rate"], "learning_rate"
    )
    gamma = _real_hyperparameter(hyperparams["gamma"], "gamma")
    epsilon_start = _real_hyperparameter(
        hyperparams["epsilon_start"], "epsilon_start"
    )
    epsilon_end = _real_hyperparameter(
        hyperparams["epsilon_end"], "epsilon_end"
    )
    epsilon_decay_episodes = _integer_hyperparameter(
        hyperparams["epsilon_decay_episodes"],
        "epsilon_decay_episodes",
    )
    if not np.isfinite(learning_rate) or not 0.0 < learning_rate <= 1.0:
        raise ValueError("learning_rate는 유한한 (0, 1] 범위여야 합니다.")
    if not np.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma는 유한한 [0, 1] 범위여야 합니다.")
    if (
        not np.isfinite(epsilon_start)
        or not np.isfinite(epsilon_end)
        or not 0.0 <= epsilon_end <= epsilon_start <= 1.0
    ):
        raise ValueError(
            "epsilon은 0 <= epsilon_end <= epsilon_start <= 1을 만족해야 합니다."
        )
    if epsilon_decay_episodes < 1:
        raise ValueError("epsilon_decay_episodes는 1 이상이어야 합니다.")
    rng_seeds = derive_rng_seeds(seed)

    resolved_hyperparams = {
        "learning_rate": learning_rate,
        "gamma": gamma,
        "epsilon_start": epsilon_start,
        "epsilon_end": epsilon_end,
        "epsilon_decay_episodes": epsilon_decay_episodes,
    }
    run_metadata = {
        "n_timesteps": n_timesteps,
        "seed": seed,
        "rng_seeds": rng_seeds,
    }

    # 실행 ID를 원자적으로 선점해 병렬 실행과 명시 ID의 덮어쓰기를 막는다.
    exp_id, exp_dir = create_experiment_dir(log_dir, env_id, exp_id)

    config_path = exp_dir / "config.yml"
    save_config(config_path, resolved_hyperparams, env_id, run_metadata)

    if verbose >= 1:
        print("=" * 60)
        print("Q-Learning 학습 시작")
        print(f"  환경: {env_id}")
        print(f"  하이퍼파라미터: {display_path(conf_file)}")
        print(f"  타임스텝: {n_timesteps:,}")
        print(f"  실험 ID: {exp_id}")
        print(f"  로그 디렉토리: {display_path(exp_dir)}")
        print("=" * 60)

    resources = ExitStack()
    try:
        raw_env = gym.make(env_id)
        try:
            monitor_file = exp_dir / "monitor.csv"
            env = Monitor(raw_env, str(monitor_file))
        except BaseException:
            raw_env.close()
            raise
        resources.callback(env.close)

        rng = np.random.default_rng(rng_seeds["train_policy"])

        state_size = env.observation_space.n
        action_size = env.action_space.n
        q_table = np.zeros((state_size, action_size), dtype=np.float32)

        if verbose >= 1:
            print("\n[1] 환경 생성 완료")
            print(f"  상태 공간: Discrete({state_size})")
            print(f"  행동 공간: Discrete({action_size})")
            print(f"  Q-table 크기: {q_table.shape}")

        epsilon = epsilon_start
        eps_decay = (epsilon_start - epsilon_end) / epsilon_decay_episodes

        writer = SummaryWriter(log_dir=str(exp_dir))
        resources.callback(writer.close)

        total_steps = 0
        start_time = time.time()

        ep_reward_window = deque(maxlen=ROLLING_WINDOW_SIZE)
        ep_len_window = deque(maxlen=ROLLING_WINDOW_SIZE)
        td_error_window = deque(maxlen=ROLLING_WINDOW_SIZE)
        last_logged_episode = 0

        def log_training_metrics(step: int) -> tuple[float, float, float]:
            """TensorBoard에 현재 이동 지표를 기록하고 그 값을 반환합니다."""
            ep_rew_mean = float(np.mean(ep_reward_window))
            ep_len_mean = float(np.mean(ep_len_window))
            td_error_mean_window = float(np.mean(td_error_window))

            writer.add_scalar("rollout/ep_rew_mean", ep_rew_mean, step)
            writer.add_scalar("rollout/ep_len_mean", ep_len_mean, step)
            writer.add_scalar("train/epsilon", epsilon, step)
            writer.add_scalar("train/td_error_mean", td_error_mean_window, step)

            elapsed = time.time() - start_time
            fps = step / max(1e-9, elapsed)
            writer.add_scalar("time/fps", fps, step)
            writer.add_scalar("time/total_timesteps", step, step)
            return ep_rew_mean, ep_len_mean, fps

        ep = 0
        while total_steps < n_timesteps:
            # 첫 reset에만 시드를 전달하면 이후 에피소드는 환경의 동일한 RNG
            # 스트림을 이어 사용하므로 실행 전체가 재현 가능합니다.
            obs, _ = env.reset(seed=rng_seeds["train_env"] if ep == 0 else None)
            state = int(obs)
            terminated = False
            truncated = False
            ep_reward = 0.0
            ep_len = 0
            ep_td_errors = []

            while not (terminated or truncated) and total_steps < n_timesteps:
                action = select_epsilon_greedy_action(q_table[state], epsilon, rng)

                next_obs, reward, terminated, truncated, _ = env.step(action)
                next_state = int(next_obs)

                # Q-Learning 업데이트
                # truncation은 시간 제한일 뿐 MDP의 terminal state가 아니므로
                # bootstrap을 중단하지 않습니다.
                td_target = compute_td_target(
                    reward,
                    gamma,
                    q_table[next_state],
                    terminated=terminated,
                )

                td_error = td_target - q_table[state, action]
                q_table[state, action] += learning_rate * td_error
                ep_td_errors.append(abs(td_error))

                state = next_state
                ep_reward += reward
                ep_len += 1
                total_steps += 1

            # n_timesteps가 에피소드 도중 소진되면 해당 partial episode는
            # 완료 episode 지표, epsilon decay 및 episode 수에 포함하지 않습니다.
            if not (terminated or truncated):
                break

            if epsilon > epsilon_end:
                epsilon = max(epsilon_end, epsilon - eps_decay)

            ep_reward_window.append(ep_reward)
            ep_len_window.append(ep_len)
            td_error_mean = float(np.mean(ep_td_errors))
            td_error_window.append(td_error_mean)

            completed_episodes = ep + 1
            last_completed_step = total_steps

            # TensorBoard 기록은 I/O 부담을 줄이기 위해 일정 episode 간격으로 제한합니다.
            if completed_episodes % TENSORBOARD_LOG_INTERVAL_EPISODES == 0:
                ep_rew_mean, ep_len_mean, fps = log_training_metrics(total_steps)
                last_logged_episode = completed_episodes

                if verbose >= 1:
                    print(
                        f"\nEpisode {completed_episodes} "
                        f"(Steps: {total_steps:,}/{n_timesteps:,})"
                    )
                    print(
                        f"  보상 평균: {ep_rew_mean:.2f} | "
                        f"길이 평균: {ep_len_mean:.1f} | "
                        f"epsilon: {epsilon:.4f} | fps: {fps:.1f}"
                    )

            ep += 1

        # 마지막 완료 episode가 기록 주기와 일치하지 않아도 최종 이동 지표는 남깁니다.
        if ep > 0 and last_logged_episode != ep:
            log_training_metrics(last_completed_step)

        if verbose >= 1:
            print("\n[2] 최종 평가 (탐욕 정책, 20 에피소드)...")

        eval_env = gym.make(env_id)
        resources.callback(eval_env.close)
        eval_rng = np.random.default_rng(rng_seeds["eval_policy"])

        eval_rewards = []
        for eval_ep in range(20):
            obs, _ = eval_env.reset(
                seed=rng_seeds["eval_env"] if eval_ep == 0 else None
            )
            state = int(obs)
            terminated = truncated = False
            total_reward = 0.0
            while not (terminated or truncated):
                action = select_epsilon_greedy_action(q_table[state], 0.0, eval_rng)
                obs, reward, terminated, truncated, _ = eval_env.step(action)
                state = int(obs)
                total_reward += reward
            eval_rewards.append(total_reward)

        mean_reward = float(np.mean(eval_rewards))
        std_reward = float(np.std(eval_rewards))
        writer.add_scalar("eval/mean_reward", mean_reward, total_steps)
        writer.add_scalar("eval/std_reward", std_reward, total_steps)

        if verbose >= 1:
            print(f"평균 보상: {mean_reward:.2f} +/- {std_reward:.2f}")

        model_filename = f"qlearning_{env_id}_{exp_id}_steps.npy"
        model_path = exp_dir / model_filename
        np.save(model_path, q_table)

        if verbose >= 1:
            print(f"\n[3] 모델 저장 완료: {display_path(model_path)}")
            print("\n[완료] 학습 종료")
            print(f"  총 에피소드: {ep}")
            print(f"  총 타임스텝: {total_steps:,}")
            print(f"  최종 평균 보상: {mean_reward:.2f} +/- {std_reward:.2f}")

        return exp_dir, model_path
    finally:
        resources.close()


def main():
    parser = argparse.ArgumentParser(
        description="Tabular Q-Learning (Taxi-v4 전용)",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--n-timesteps",
        "-n",
        type=int,
        default=500_000,
        help="총 학습 스텝 수 (기본: 500000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="환경 및 행동 선택 랜덤 시드 (기본: 미지정)",
    )
    args = parser.parse_args()
    if args.n_timesteps < 1:
        parser.error("--n-timesteps는 1 이상이어야 합니다.")
    if args.seed is not None and args.seed < 0:
        parser.error("--seed는 0 이상이어야 합니다.")

    train_qlearning(
        env_id="Taxi-v4",
        conf_file=None,
        n_timesteps=args.n_timesteps,
        seed=args.seed,
        exp_id=None,
        log_dir=None,
        verbose=1,
    )


if __name__ == "__main__":
    main()
