"""
RL-Zoo 기반 훈련 스크립트 (Taxi-v3 등 표준 환경 전용)

- 커스텀 AdvancedTaxi-v0/MaskablePPO는 별도 스크립트 사용
- 심플한 인자 파싱 + YAML 자동 매핑 후 RL-Zoo에 위임
"""

import argparse
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 환경 등록 (Taxi wrappers/AdvancedTaxi-v0)
import envs  # noqa: F401


def resolve_config_file(env_id: str, algo: str, param_mode: str, hyperparams_dir: Path) -> Path:
    """환경/알고리즘별 하이퍼파라미터 파일 경로 반환 (공용)."""
    env_folder_map = {"Taxi-v3": "taxi_v3"}
    env_dir = hyperparams_dir / env_folder_map.get(env_id, "")
    base_name = "qlearning" if algo == "qlearning" else algo
    filename = f"{base_name}_base.yml" if param_mode == "base" else f"{base_name}.yml"
    candidates = [env_dir / filename, hyperparams_dir / filename]
    for cand in candidates:
        if cand.exists():
            return cand
    return candidates[0]


if __name__ == "__main__":
    import rl_zoo3
    from rl_zoo3.train import train

    HYPERPARAMS_DIR = ROOT / "hyperparameters"

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--algo", type=str, required=False)
    parser.add_argument("--env", type=str, required=False)
    parser.add_argument("--param-mode", type=str, choices=["tuned", "base"], default="tuned")
    parser.add_argument("--conf-file", type=str, required=False)
    parser.add_argument("--n-timesteps", "-n", type=int, required=False)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("-optimize", "--optimize-hyperparameters", action="store_true")
    parser.add_argument("--n-jobs", type=int, required=False)
    parser.add_argument("--device", type=str, default="cpu")  # Taxi-v3는 CPU가 기본적으로 효율적
    parser.add_argument("--tensorboard-log", "-tb", type=str, default=str(ROOT / "logs"))
    parser.add_argument("--log-folder", type=str, required=False)

    original_args = sys.argv[1:]
    known_args, unknown_args = parser.parse_known_args(original_args)
    # 주입이 필요한 추가 인자만 모아 중복 없이 전달
    injected_args = []

    # --device 기본값을 명시적으로 주입 (original_args는 parser 기본값을 포함하지 않음)
    if known_args.device and not any(
        arg == "--device" or arg.startswith("--device=") for arg in original_args
    ):
        injected_args.extend(["--device", known_args.device])

    # TensorBoard 로그 경로를 명시적으로 주입해 RL-Zoo가 항상 TF 로그를 남기도록 함
    has_tb_arg = any(
        arg in {"--tensorboard-log", "-tb"} or arg.startswith("--tensorboard-log=")
        for arg in original_args + unknown_args
    )
    if not has_tb_arg:
        tb_root = ROOT / "logs"
        injected_args.extend(["--tensorboard-log", str(tb_root)])
        print(f"[INFO] TensorBoard 로그 기본 경로 설정: {tb_root}")

    # MaskablePPO는 전용 스크립트 사용
    if known_args.algo == "maskableppo":
        print("[ERROR] maskableppo는 전용 스크립트 사용: train_advanced_taxi_maskableppo.py")
        sys.exit(1)

    # Optimize 모드 기본 n_jobs 추천 (이미 지정된 경우 중복 추가 방지)
    if (
        known_args.optimize_hyperparameters
        and known_args.algo
        and known_args.n_jobs is None
        and "--n-jobs" not in unknown_args
    ):
        on_policy_algos = {"ppo", "a2c", "trpo", "ppo_lstm"}
        off_policy_algos = {"dqn", "qrdqn", "sac", "td3", "ddpg", "tqc", "crossq"}
        if known_args.algo.lower() in on_policy_algos:
            recommended_n_jobs = 1
        elif known_args.algo.lower() in off_policy_algos:
            recommended_n_jobs = 8
        else:
            recommended_n_jobs = 1
        injected_args.extend(["--n-jobs", str(recommended_n_jobs)])

    # 하이퍼파라미터 파일 자동 선택
    auto_conf = None
    if not known_args.conf_file and "--conf-file" not in unknown_args and known_args.algo and known_args.env:
        auto_conf = resolve_config_file(known_args.env, known_args.algo, known_args.param_mode, HYPERPARAMS_DIR)
        if auto_conf.exists():
            injected_args.extend(["--conf-file", str(auto_conf)])
            print(f"[INFO] 하이퍼파라미터 파일: {auto_conf.relative_to(ROOT)} (mode: {known_args.param_mode})")
        else:
            print(f"[ERROR] 하이퍼파라미터 파일을 찾을 수 없습니다: {auto_conf}")
            print("지원 환경: Taxi-v3")
            print("입력 예: --algo ppo --env Taxi-v3")
            sys.exit(1)

    # Q-Learning 분기
    if known_args.algo == "qlearning":
        from qlearning_train import train_qlearning

        conf_file = Path(known_args.conf_file) if known_args.conf_file else auto_conf
        n_timesteps = known_args.n_timesteps or 500_000
        train_qlearning(
            env_id=known_args.env or "Taxi-v3",
            conf_file=conf_file if conf_file and conf_file.exists() else None,
            n_timesteps=n_timesteps,
            seed=None,
            exp_id=None,
            verbose=known_args.verbose,
        )
    else:
        # RL-Zoo로 전달: 사용자가 준 인자 + 자동 주입 인자만 추가 (unknown_args 중복 방지)
        sys.argv = [sys.argv[0]] + original_args + injected_args
        train()