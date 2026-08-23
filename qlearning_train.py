"""
Q-Learning 학습 모듈 (RL-Zoo 스타일)

Taxi-v3 원본 환경(Discrete 500)을 사용하는 Tabular Q-Learning 구현.
RL-Zoo와 동일한 저장 구조 및 로깅 형식을 사용합니다.
"""

import sys
import time
import argparse
import yaml
import numpy as np
import gymnasium as gym
from pathlib import Path
from typing import Optional, Dict, Any
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.monitor import Monitor

# 프로젝트 루트를 Python 경로에 추가
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 통합 환경 패키지 import
import envs  # noqa: F401

QLEARNING_HYPERPARAM_KEYS = frozenset(
    {
        "learning_rate",
        "gamma",
        "epsilon_start",
        "epsilon_end",
        "epsilon_decay_episodes",
    }
)


def display_path(path: Path) -> str:
    """Return a readable path without requiring it to be inside the project."""
    resolved_path = path.resolve()
    try:
        return str(resolved_path.relative_to(ROOT))
    except ValueError:
        return str(resolved_path)


def derive_eval_seed(seed: Optional[int]) -> Optional[int]:
    """Derive a deterministic evaluation seed independent of the training seed."""
    if seed is None:
        return None
    child = np.random.SeedSequence(seed).spawn(1)[0]
    return int(child.generate_state(1, dtype=np.uint32)[0])


def select_epsilon_greedy_action(
    q_values: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
) -> int:
    """Select an action from the full discrete action set.

    Taxi-v3's existing Q-Learning benchmark is unmasked, so exploration and
    greedy selection both consider every action. Greedy ties are randomized to
    avoid a permanent preference for the lowest action index.
    """
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
    """Compute a Q-Learning target, bootstrapping across time-limit truncation."""
    if terminated:
        return float(reward)
    return float(reward + gamma * np.max(next_q_values))


def get_next_exp_id(log_dir: Path, env_id: str) -> int:
    """다음 실험 ID를 찾습니다 (RL-Zoo 스타일)"""
    exp_id = 1
    while (log_dir / f"{env_id}_{exp_id}").exists():
        exp_id += 1
    return exp_id


def load_hyperparams(conf_file: Path, env_id: str) -> Dict[str, Any]:
    """YAML 파일에서 하이퍼파라미터를 로드합니다"""
    with open(conf_file, 'r', encoding='utf-8') as f:
        hyperparams = yaml.safe_load(f)
    
    if not isinstance(hyperparams, dict) or env_id not in hyperparams:
        raise ValueError(f"Environment {env_id} not found in {conf_file}")

    env_hyperparams = hyperparams[env_id]
    if not isinstance(env_hyperparams, dict):
        raise ValueError(f"Hyperparameters for {env_id} must be a mapping")
    unknown = set(env_hyperparams) - QLEARNING_HYPERPARAM_KEYS
    if unknown:
        raise ValueError(f"Unknown Q-Learning hyperparameters: {sorted(unknown)}")
    return env_hyperparams


def save_config(config_path: Path, hyperparams: Dict[str, Any], env_id: str):
    """config.yml 파일을 저장합니다 (RL-Zoo 스타일)"""
    config = {env_id: hyperparams}
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=False)


def train_qlearning(
    env_id: str = "Taxi-v3",
    conf_file: Optional[Path] = None,
    n_timesteps: int = 500_000,
    seed: Optional[int] = None,
    exp_id: Optional[int] = None,
    log_dir: Path | None = None,
    verbose: int = 1,
):
    """
    Q-Learning 학습 함수 (RL-Zoo 스타일)
    
    Args:
        env_id: 환경 ID (예: "Taxi-v3")
        conf_file: 하이퍼파라미터 YAML 파일 경로
        n_timesteps: 총 학습 타임스텝 수
        seed: 랜덤 시드
        exp_id: 실험 ID (None이면 자동 생성)
        log_dir: 로그 디렉토리 (기본값: logs/qlearning)
        verbose: 출력 상세도 (0, 1, 2)
    """
    if env_id != "Taxi-v3":
        raise ValueError(
            f"Unsupported environment for tabular Q-Learning: {env_id}. "
            "Only Taxi-v3 is supported."
        )
    if n_timesteps < 1:
        raise ValueError("n_timesteps must be at least 1")
    if exp_id is not None and exp_id < 1:
        raise ValueError("exp_id must be at least 1")

    if log_dir is None:
        log_dir = ROOT / "logs" / "qlearning"
    
    if conf_file is None:
        # 기본 하이퍼파라미터 파일 찾기
        env_dir = ROOT / "hyperparameters" / "taxi_v3"
        conf_file = env_dir / "qlearning.yml"
    
    if not conf_file.exists():
        raise FileNotFoundError(f"Hyperparameter file not found: {conf_file}")
    
    # 하이퍼파라미터 로드
    hyperparams = load_hyperparams(conf_file, env_id)

    # 하이퍼파라미터 및 실행 인자를 실제 사용값으로 해석합니다.
    learning_rate = float(hyperparams.get("learning_rate", 0.1))
    gamma = float(hyperparams.get("gamma", 0.99))
    epsilon_start = float(hyperparams.get("epsilon_start", 1.0))
    epsilon_end = float(hyperparams.get("epsilon_end", 0.05))
    epsilon_decay_episodes = int(hyperparams.get("epsilon_decay_episodes", 500))
    eval_seed = derive_eval_seed(seed)

    resolved_hyperparams = {
        "learning_rate": learning_rate,
        "gamma": gamma,
        "epsilon_start": epsilon_start,
        "epsilon_end": epsilon_end,
        "epsilon_decay_episodes": epsilon_decay_episodes,
        "n_timesteps": int(n_timesteps),
        "seed": seed,
        "eval_seed": eval_seed,
    }
    
    # 실험 디렉토리 생성
    if exp_id is None:
        exp_id = get_next_exp_id(log_dir, env_id)
    
    exp_dir = log_dir / f"{env_id}_{exp_id}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # config.yml 저장
    config_path = exp_dir / "config.yml"
    save_config(config_path, resolved_hyperparams, env_id)
    
    if verbose >= 1:
        print("=" * 60)
        print(f"Q-Learning 학습 시작")
        print(f"  환경: {env_id}")
        print(f"  하이퍼파라미터: {display_path(conf_file)}")
        print(f"  타임스텝: {n_timesteps:,}")
        print(f"  실험 ID: {exp_id}")
        print(f"  로그 디렉토리: {display_path(exp_dir)}")
        print("=" * 60)
    
    # 환경 생성 (원본 환경 사용, wrapper 없음)
    env = gym.make(env_id)
    
    # Monitor wrapper 적용 (RL-Zoo 스타일)
    monitor_file = exp_dir / "monitor.csv"
    env = Monitor(env, str(monitor_file))

    # 학습의 모든 행동 선택은 하나의 로컬 Generator를 사용합니다.
    rng = np.random.default_rng(seed)

    # action_space.sample()은 사용하지 않지만 외부 진단/확장을 위해 함께 시드합니다.
    if seed is not None:
        env.action_space.seed(seed)
    
    # Q-table 초기화 (Taxi-v3는 Discrete(500) 상태 공간)
    state_size = env.observation_space.n  # 500
    action_size = env.action_space.n  # 6
    q_table = np.zeros((state_size, action_size), dtype=np.float32)
    
    if verbose >= 1:
        print(f"\n[1] 환경 생성 완료")
        print(f"  상태 공간: Discrete({state_size})")
        print(f"  행동 공간: Discrete({action_size})")
        print(f"  Q-table 크기: {q_table.shape}")
    
    # Epsilon 스케줄
    epsilon = epsilon_start
    eps_decay = (epsilon_start - epsilon_end) / max(1, epsilon_decay_episodes)
    
    # TensorBoard 로거
    tb_log_dir = exp_dir
    writer = SummaryWriter(log_dir=str(tb_log_dir))
    
    # 학습 루프 변수
    total_steps = 0
    start_time = time.time()
    
    # SB3 형식: 이동 평균을 위한 윈도우
    ep_reward_window = []
    ep_len_window = []
    td_error_window = []
    window_size = 100  # SB3 기본값과 동일
    
    ep = 0
    while total_steps < n_timesteps:
        # 첫 reset에만 시드를 전달하면 이후 에피소드는 환경의 동일한 RNG
        # 스트림을 이어 사용하므로 실행 전체가 재현 가능합니다.
        obs, info = env.reset(seed=seed if ep == 0 else None)
        state = int(obs)  # Discrete 상태는 정수 그대로 사용
        terminated = False
        truncated = False
        ep_reward = 0.0
        ep_len = 0
        ep_td_errors = []
        
        while not (terminated or truncated) and total_steps < n_timesteps:
            # Epsilon-greedy 행동 선택
            action = select_epsilon_greedy_action(q_table[state], epsilon, rng)
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = int(next_obs)  # Discrete 상태는 정수 그대로 사용
            
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
        
        # Epsilon decay
        if epsilon > epsilon_end:
            epsilon = max(epsilon_end, epsilon - eps_decay)
        
        # 이동 평균 윈도우 업데이트
        ep_reward_window.append(ep_reward)
        ep_len_window.append(ep_len)
        td_error_mean = float(np.mean(ep_td_errors)) if ep_td_errors else 0.0
        td_error_window.append(td_error_mean)
        
        # 윈도우 크기 제한
        if len(ep_reward_window) > window_size:
            ep_reward_window.pop(0)
        if len(ep_len_window) > window_size:
            ep_len_window.pop(0)
        if len(td_error_window) > window_size:
            td_error_window.pop(0)
        
        # TensorBoard 로깅 (RL-Zoo 스타일)
        ep_rew_mean = float(np.mean(ep_reward_window)) if ep_reward_window else 0.0
        ep_len_mean = float(np.mean(ep_len_window)) if ep_len_window else 0.0
        td_error_mean_window = float(np.mean(td_error_window)) if td_error_window else 0.0
        
        writer.add_scalar("rollout/ep_rew_mean", ep_rew_mean, total_steps)
        writer.add_scalar("rollout/ep_len_mean", ep_len_mean, total_steps)
        writer.add_scalar("train/epsilon", epsilon, total_steps)
        writer.add_scalar("train/td_error_mean", td_error_mean_window, total_steps)
        
        # 주기적 로깅
        if verbose >= 1 and ((ep + 1) % 200 == 0 or ep == 0):
            elapsed = time.time() - start_time
            fps = total_steps / max(1e-9, elapsed)
            writer.add_scalar("time/fps", fps, total_steps)
            writer.add_scalar("time/total_timesteps", total_steps, total_steps)
            print(f"\nEpisode {ep + 1} (Steps: {total_steps:,}/{n_timesteps:,})")
            print(f"  보상 평균: {ep_rew_mean:.2f} | 길이 평균: {ep_len_mean:.1f} | epsilon: {epsilon:.4f} | fps: {fps:.1f}")
        
        ep += 1
        
        # 타임스텝 제한 확인
        if total_steps >= n_timesteps:
            break
    
    # 최종 평가
    if verbose >= 1:
        print("\n[2] 최종 평가 (탐욕 정책, 20 에피소드)...")
    
    eval_env = gym.make(env_id)
    eval_rng = np.random.default_rng(eval_seed)
    if eval_seed is not None:
        eval_env.action_space.seed(eval_seed)
    
    eval_rewards = []
    for eval_ep in range(20):
        obs, info = eval_env.reset(seed=eval_seed if eval_ep == 0 else None)
        state = int(obs)
        terminated = truncated = False
        total_reward = 0.0
        while not (terminated or truncated):
            action = select_epsilon_greedy_action(q_table[state], 0.0, eval_rng)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            state = int(obs)
            total_reward += reward
        eval_rewards.append(total_reward)
    eval_env.close()
    
    mean_reward = float(np.mean(eval_rewards))
    std_reward = float(np.std(eval_rewards))
    writer.add_scalar("eval/mean_reward", mean_reward, total_steps)
    writer.add_scalar("eval/std_reward", std_reward, total_steps)
    
    if verbose >= 1:
        print(f"평균 보상: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    # 모델 저장 (RL-Zoo 스타일)
    model_filename = f"qlearning_{env_id}_{exp_id}_steps.npy"
    model_path = exp_dir / model_filename
    np.save(model_path, q_table)
    
    if verbose >= 1:
        print(f"\n[3] 모델 저장 완료: {display_path(model_path)}")
    
    # 정리
    env.close()
    writer.flush()
    writer.close()
    
    if verbose >= 1:
        print("\n[완료] 학습 종료")
        print(f"  총 에피소드: {ep}")
        print(f"  총 타임스텝: {total_steps:,}")
        print(f"  최종 평균 보상: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    return exp_dir, model_path


def main():
    parser = argparse.ArgumentParser(description="Tabular Q-Learning (Taxi-v3 전용)")
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

    train_qlearning(
        env_id="Taxi-v3",
        conf_file=None,
        n_timesteps=args.n_timesteps,
        seed=args.seed,
        exp_id=None,
        log_dir=None,
        verbose=1,
    )


if __name__ == "__main__":
    main()
