"""
RL-Zoo 기반 훈련 스크립트 (Taxi-v3 전용)

- 커스텀 AdvancedTaxi-v1/MaskablePPO는 별도 스크립트 사용
- 심플한 인자 파싱 + YAML 자동 매핑 후 RL-Zoo에 위임
"""

import argparse
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 환경 등록 (Taxi wrappers/AdvancedTaxi-v1)
import envs  # noqa: F401


def parse_wrapper_args(argv: list[str]) -> tuple[argparse.Namespace, argparse.Namespace, list[str]]:
    """Parse local options separately and inspect the remaining RL-Zoo options.

    ``--param-mode`` belongs to this wrapper and must not be forwarded to
    RL-Zoo's training parser.  All other arguments remain in their original
    form and order.
    """
    local_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    local_parser.add_argument("--param-mode", choices=["tuned", "base"], default="tuned")
    local_args, rl_zoo_args = local_parser.parse_known_args(argv)

    inspection_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    inspection_parser.add_argument("--algo")
    inspection_parser.add_argument("--env")
    inspection_parser.add_argument("-conf", "--conf-file")
    inspection_parser.add_argument("--n-timesteps", "-n", type=int)
    inspection_parser.add_argument("--verbose", type=int, default=1)
    inspection_parser.add_argument("-optimize", "--optimize-hyperparameters", action="store_true")
    inspection_parser.add_argument("--n-jobs", type=int)
    inspection_parser.add_argument("--seed", type=int)
    inspection_parser.add_argument("--device")
    inspection_parser.add_argument("--tensorboard-log", "-tb")
    inspection_parser.add_argument("--log-folder", "-f")
    inspected_args, _ = inspection_parser.parse_known_args(rl_zoo_args)
    return local_args, inspected_args, rl_zoo_args


def resolve_config_file(env_id: str, algo: str, param_mode: str, hyperparams_dir: Path) -> Path:
    """환경/알고리즘별 하이퍼파라미터 파일 경로 반환 (공용)."""
    env_folder_map = {"Taxi-v3": "taxi_v3"}
    try:
        env_folder = env_folder_map[env_id]
    except KeyError as exc:
        raise ValueError(f"지원하지 않는 환경입니다: {env_id}") from exc
    env_dir = hyperparams_dir / env_folder
    base_name = "qlearning" if algo == "qlearning" else algo
    filename = f"{base_name}_base.yml" if param_mode == "base" else f"{base_name}.yml"
    return env_dir / filename


def require_config_file(path: Path | None) -> Path | None:
    """Reject an explicitly resolved config path instead of silently falling back."""
    if path is not None and not path.is_file():
        raise FileNotFoundError(f"하이퍼파라미터 파일을 찾을 수 없습니다: {path}")
    return path


def resolve_qlearning_log_dir(log_folder: str | None) -> Path | None:
    if log_folder is None:
        return None
    return Path(log_folder).resolve() / "qlearning"


if __name__ == "__main__":
    from rl_zoo3.train import train

    HYPERPARAMS_DIR = ROOT / "hyperparameters"

    original_args = sys.argv[1:]
    local_args, known_args, forwarded_args = parse_wrapper_args(original_args)
    is_qlearning = known_args.algo == "qlearning"
    # 주입이 필요한 추가 인자만 모아 중복 없이 전달
    injected_args = []

    # Taxi-v3는 CPU가 기본적으로 효율적
    if not is_qlearning and known_args.device is None:
        injected_args.extend(["--device", "cpu"])

    # TensorBoard 로그 경로를 명시적으로 주입해 RL-Zoo가 항상 TF 로그를 남기도록 함
    if not is_qlearning and known_args.tensorboard_log is None:
        tb_root = ROOT / "logs"
        injected_args.extend(["--tensorboard-log", str(tb_root)])
        print(f"[INFO] TensorBoard 로그 기본 경로 설정: {tb_root}")

    # MaskablePPO는 전용 스크립트 사용
    if known_args.algo == "maskableppo":
        print("[ERROR] maskableppo는 전용 스크립트 사용: train_advanced_taxi_maskableppo.py")
        sys.exit(1)
    if known_args.env is not None and known_args.env != "Taxi-v3":
        print("[ERROR] 이 스크립트는 Taxi-v3 전용입니다.")
        sys.exit(1)
    if is_qlearning and known_args.optimize_hyperparameters:
        print("[ERROR] Tabular Q-Learning은 이 스크립트에서 Optuna 최적화를 지원하지 않습니다.")
        sys.exit(1)
    if is_qlearning and known_args.tensorboard_log is not None:
        print("[ERROR] Q-Learning 로그 위치는 --log-folder로 지정하세요.")
        sys.exit(1)

    # Optimize 모드 기본 n_jobs 추천 (이미 지정된 경우 중복 추가 방지)
    if (
        known_args.optimize_hyperparameters
        and known_args.algo
        and known_args.n_jobs is None
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
    config_env = known_args.env
    if config_env is None and known_args.algo == "qlearning":
        config_env = "Taxi-v3"
    if known_args.conf_file is None and known_args.algo and config_env:
        try:
            auto_conf = resolve_config_file(
                config_env,
                known_args.algo,
                local_args.param_mode,
                HYPERPARAMS_DIR,
            )
        except ValueError as exc:
            print(f"[ERROR] {exc}")
            sys.exit(1)
        if auto_conf.exists():
            injected_args.extend(["--conf-file", str(auto_conf)])
            print(f"[INFO] 하이퍼파라미터 파일: {auto_conf.relative_to(ROOT)} (mode: {local_args.param_mode})")
        else:
            print(f"[ERROR] 하이퍼파라미터 파일을 찾을 수 없습니다: {auto_conf}")
            print("지원 환경: Taxi-v3")
            print("입력 예: --algo ppo --env Taxi-v3")
            sys.exit(1)

    # Q-Learning 분기
    if known_args.algo == "qlearning":
        from qlearning_train import train_qlearning

        conf_file = Path(known_args.conf_file) if known_args.conf_file else auto_conf
        try:
            conf_file = require_config_file(conf_file)
        except FileNotFoundError as exc:
            print(f"[ERROR] {exc}")
            sys.exit(1)
        n_timesteps = (
            known_args.n_timesteps
            if known_args.n_timesteps is not None
            else 500_000
        )
        train_qlearning(
            env_id=known_args.env or "Taxi-v3",
            conf_file=conf_file,
            n_timesteps=n_timesteps,
            seed=known_args.seed,
            exp_id=None,
            log_dir=resolve_qlearning_log_dir(known_args.log_folder),
            verbose=known_args.verbose,
        )
    else:
        # RL-Zoo로 전달: 로컬 전용 옵션을 제거하고 필요한 기본값만 추가
        sys.argv = [sys.argv[0]] + forwarded_args + injected_args
        train()
