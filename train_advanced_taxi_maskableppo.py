"""Train MaskablePPO on AdvancedTaxi-v1 using public SB3 APIs only."""

from __future__ import annotations

import argparse
import ast
import inspect
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from torch import nn

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import envs  # noqa: E402,F401  # Register AdvancedTaxi-v1 with Gymnasium.
from envs.envs import AdvancedTimeAttackTaxiEnv  # noqa: E402


ALGO = "maskableppo"
ENV_ID = "AdvancedTaxi-v1"
DEFAULT_CONFIG_PATH = ROOT / "hyperparameters" / "advanced_taxi" / "maskableppo_base.yml"

TRAIN_ENV_KWARGS: dict[str, Any] = {
    "num_passengers": 20,
    "grid_size": 100,
    "scenario_mode": "mixed",
    "start_at_center": False,
    "scenario_mix": [0.4, 0.3, 0.3],
    "mix_noise_std": 0.05,
    "mix_epsilon": 1e-3,
}
EVAL_ENV_OVERRIDES: dict[str, Any] = {"mix_noise_std": 0.0}

_TOP_LEVEL_CONFIG_KEYS = {
    "policy",
    "n_timesteps",
    "n_envs",
    "normalize",
    "model_kwargs",
}
_MODEL_KWARG_KEYS = {
    name
    for name in inspect.signature(MaskablePPO.__init__).parameters
    if name
    not in {
        "self",
        "policy",
        "env",
        "tensorboard_log",
        "verbose",
        "seed",
        "device",
        "_init_setup_model",
    }
}
_NORMALIZE_KWARG_KEYS = {
    name
    for name in inspect.signature(VecNormalize.__init__).parameters
    if name not in {"self", "venv", "training"}
}
_POLICY_KWARG_KEYS = {"net_arch", "activation_fn"}
_ACTIVATION_FUNCTIONS = {
    "elu": nn.ELU,
    "leaky_relu": nn.LeakyReLU,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
}


class SaveBestVecNormalizeCallback(BaseCallback):
    """Persist normalization statistics paired with ``best_model.zip``."""

    def __init__(self, save_path: Path) -> None:
        super().__init__()
        self.save_path = save_path

    def _on_step(self) -> bool:
        vec_normalize = self.model.get_vec_normalize_env()
        if vec_normalize is None:
            raise RuntimeError("best model의 VecNormalize 환경을 찾을 수 없습니다.")
        vec_normalize.save(str(self.save_path))
        return True


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label}는 1 이상의 정수여야 합니다.")
    return value


def _non_negative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label}는 0 이상의 정수여야 합니다.")
    return value


def _reject_unknown_keys(values: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"{label}에 알 수 없는 키가 있습니다: {sorted(unknown)}")


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as file:
            loaded = yaml.safe_load(file)
    except OSError as exc:
        raise FileNotFoundError(f"설정 파일을 읽을 수 없습니다: {path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"올바르지 않은 YAML 설정입니다: {path}") from exc

    if not isinstance(loaded, dict):
        raise ValueError(f"설정 파일의 최상위 값은 mapping이어야 합니다: {path}")
    return loaded


def _validate_net_arch(value: Any) -> dict[str, list[int]]:
    if not isinstance(value, dict) or set(value) != {"pi", "vf"}:
        raise ValueError("policy_kwargs.net_arch는 pi와 vf 목록을 가져야 합니다.")

    result: dict[str, list[int]] = {}
    for branch in ("pi", "vf"):
        layers = value[branch]
        if not isinstance(layers, list) or not layers:
            raise ValueError(f"policy_kwargs.net_arch.{branch}는 비어 있지 않은 목록이어야 합니다.")
        result[branch] = [
            _positive_int(width, f"policy_kwargs.net_arch.{branch} 항목")
            for width in layers
        ]
    return result


def _resolve_policy_kwargs(value: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if value is None:
        return {}, {}
    if not isinstance(value, dict):
        raise ValueError("model_kwargs.policy_kwargs는 mapping이어야 합니다.")
    _reject_unknown_keys(value, _POLICY_KWARG_KEYS, "model_kwargs.policy_kwargs")

    saved: dict[str, Any] = {}
    resolved: dict[str, Any] = {}
    if "net_arch" in value:
        net_arch = _validate_net_arch(value["net_arch"])
        saved["net_arch"] = net_arch
        resolved["net_arch"] = net_arch
    if "activation_fn" in value:
        activation_name = value["activation_fn"]
        if not isinstance(activation_name, str):
            raise ValueError("policy_kwargs.activation_fn은 문자열이어야 합니다.")
        activation_key = activation_name.lower()
        if activation_key not in _ACTIVATION_FUNCTIONS:
            raise ValueError(
                "지원하지 않는 activation_fn입니다. "
                f"지원 값: {sorted(_ACTIVATION_FUNCTIONS)}"
            )
        saved["activation_fn"] = activation_key
        resolved["activation_fn"] = _ACTIVATION_FUNCTIONS[activation_key]
    return saved, resolved


def load_training_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and validate safe YAML, returning saved and runtime forms."""
    config = _load_mapping(path)
    _reject_unknown_keys(config, _TOP_LEVEL_CONFIG_KEYS, "설정")

    policy = config.get("policy", "MlpPolicy")
    if policy != "MlpPolicy":
        raise ValueError("AdvancedTaxi 학습기는 policy: MlpPolicy만 지원합니다.")

    n_timesteps = _positive_int(config.get("n_timesteps", 1_000_000), "n_timesteps")
    n_envs = _positive_int(config.get("n_envs", 1), "n_envs")

    normalize = config.get("normalize", False)
    if not isinstance(normalize, (bool, dict)):
        raise ValueError("normalize는 boolean 또는 mapping이어야 합니다.")
    if isinstance(normalize, dict):
        _reject_unknown_keys(normalize, _NORMALIZE_KWARG_KEYS, "normalize")
        normalize_config: bool | dict[str, Any] = dict(normalize)
    else:
        normalize_config = normalize

    model_kwargs = config.get("model_kwargs", {})
    if not isinstance(model_kwargs, dict):
        raise ValueError("model_kwargs는 mapping이어야 합니다.")
    _reject_unknown_keys(model_kwargs, _MODEL_KWARG_KEYS, "model_kwargs")
    saved_model_kwargs = dict(model_kwargs)
    runtime_model_kwargs = dict(model_kwargs)
    saved_policy_kwargs, runtime_policy_kwargs = _resolve_policy_kwargs(
        model_kwargs.get("policy_kwargs")
    )
    if "policy_kwargs" in model_kwargs:
        saved_model_kwargs["policy_kwargs"] = saved_policy_kwargs
        runtime_model_kwargs["policy_kwargs"] = runtime_policy_kwargs

    n_steps = runtime_model_kwargs.get("n_steps", 2_048)
    batch_size = runtime_model_kwargs.get("batch_size", 64)
    _positive_int(n_steps, "model_kwargs.n_steps")
    if batch_size is not None:
        _positive_int(batch_size, "model_kwargs.batch_size")
        if not is_valid_rollout_batch_size(n_steps, batch_size, n_envs):
            raise ValueError(
                "model_kwargs.batch_size는 n_steps * n_envs 이하이며 rollout 크기를 "
                "나누어떨어지게 해야 합니다."
            )

    saved_config = {
        "policy": policy,
        "n_timesteps": n_timesteps,
        "n_envs": n_envs,
        "normalize": normalize_config,
        "model_kwargs": saved_model_kwargs,
    }
    runtime_config = {
        **saved_config,
        "model_kwargs": runtime_model_kwargs,
    }
    return saved_config, runtime_config


def _parse_env_value(raw_value: str) -> Any:
    try:
        value = ast.literal_eval(raw_value)
    except (SyntaxError, ValueError):
        value = raw_value
    if isinstance(value, tuple):
        return list(value)
    return value


def parse_env_kwargs(
    defaults: Mapping[str, Any],
    groups: Sequence[Sequence[str]] | None,
) -> dict[str, Any]:
    """Merge ``key:value`` CLI groups and reject unknown environment keys."""
    merged = dict(defaults)
    for group in groups or ():
        for token in group:
            key, separator, raw_value = token.partition(":")
            if not separator or not key or not raw_value:
                raise ValueError(f"환경 인자는 key:value 형식이어야 합니다: {token!r}")
            merged[key] = _parse_env_value(raw_value)

    allowed = set(inspect.signature(AdvancedTimeAttackTaxiEnv.__init__).parameters) - {
        "self"
    }
    _reject_unknown_keys(merged, allowed, "환경 인자")

    validation_env = AdvancedTimeAttackTaxiEnv(**merged)
    validation_env.close()
    return merged


def resolve_env_kwargs(
    train_groups: Sequence[Sequence[str]] | None,
    eval_groups: Sequence[Sequence[str]] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    train_kwargs = parse_env_kwargs(TRAIN_ENV_KWARGS, train_groups)
    eval_defaults = {**train_kwargs, **EVAL_ENV_OVERRIDES}
    eval_kwargs = parse_env_kwargs(eval_defaults, eval_groups)
    return train_kwargs, eval_kwargs


def is_valid_rollout_batch_size(n_steps: int, batch_size: int, n_envs: int) -> bool:
    if min(n_steps, batch_size, n_envs) < 1:
        return False
    rollout_size = n_steps * n_envs
    return batch_size <= rollout_size and rollout_size % batch_size == 0


def _create_run_dir(log_folder: Path) -> Path:
    algo_dir = log_folder / ALGO
    algo_dir.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(rf"^{re.escape(ENV_ID)}_(\d+)$")
    ids = [
        int(match.group(1))
        for child in algo_dir.iterdir()
        if child.is_dir() and (match := pattern.fullmatch(child.name))
    ]
    next_id = max(ids, default=0) + 1
    while True:
        run_dir = algo_dir / f"{ENV_ID}_{next_id}"
        try:
            run_dir.mkdir()
            return run_dir
        except FileExistsError:
            next_id += 1


def _dump_yaml(path: Path, values: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        yaml.safe_dump(dict(values), file, sort_keys=False, allow_unicode=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"{ENV_ID} 전용 MaskablePPO 학습기",
        allow_abbrev=False,
    )
    parser.add_argument("--n-timesteps", type=int, help="설정 파일의 총 timestep을 덮어씀")
    parser.add_argument("--eval-freq", type=int, default=10_000, help="평가 간격; 0이면 평가 안 함")
    parser.add_argument("--eval-episodes", type=int, default=5, help="평가 시 episode 수")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="PyTorch CPU intra-op thread 수",
    )
    parser.add_argument("--verbose", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("-P", "--progress", action="store_true", help="학습 진행률 표시")
    parser.add_argument("--log-folder", type=Path, default=Path("logs"))
    parser.add_argument(
        "-conf",
        "--conf-file",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
    )
    parser.add_argument(
        "--tensorboard-log",
        type=Path,
        help="TensorBoard 루트; 생략하면 --log-folder 사용",
    )
    parser.add_argument("--env-kwargs", nargs="+", action="append", metavar="KEY:VALUE")
    parser.add_argument(
        "--eval-env-kwargs",
        nargs="+",
        action="append",
        metavar="KEY:VALUE",
    )
    return parser


def train(args: argparse.Namespace) -> Path:
    config_path = args.conf_file.expanduser().resolve()
    saved_config, runtime_config = load_training_config(config_path)

    n_timesteps = (
        runtime_config["n_timesteps"]
        if args.n_timesteps is None
        else _positive_int(args.n_timesteps, "--n-timesteps")
    )
    eval_freq = _non_negative_int(args.eval_freq, "--eval-freq")
    eval_episodes = _positive_int(args.eval_episodes, "--eval-episodes")
    num_threads = _positive_int(args.num_threads, "--num-threads")
    if args.seed < 0:
        raise ValueError("--seed는 0 이상이어야 합니다.")

    # 작은 CPU MLP에서는 thread 관리 비용이 연산보다 커지므로 명시적으로 제한한다.
    torch.set_num_threads(num_threads)

    train_env_kwargs, eval_env_kwargs = resolve_env_kwargs(
        args.env_kwargs,
        args.eval_env_kwargs,
    )
    n_envs = runtime_config["n_envs"]
    eval_seed = args.seed + n_envs

    log_folder = args.log_folder.expanduser().resolve()
    run_dir = _create_run_dir(log_folder)
    params_dir = run_dir / ENV_ID
    params_dir.mkdir()
    tensorboard_log = (
        args.tensorboard_log.expanduser().resolve()
        if args.tensorboard_log is not None
        else log_folder
    )
    tensorboard_log.mkdir(parents=True, exist_ok=True)

    saved_config = {**saved_config, "n_timesteps": n_timesteps}
    saved_args = {
        "algo": ALGO,
        "env": ENV_ID,
        "conf_file": str(config_path),
        "log_folder": str(log_folder),
        "tensorboard_log": str(tensorboard_log),
        "n_timesteps": n_timesteps,
        "eval_freq": eval_freq,
        "eval_episodes": eval_episodes,
        "seed": args.seed,
        "eval_seed": eval_seed,
        "device": args.device,
        "num_threads": num_threads,
        "verbose": args.verbose,
        "progress": args.progress,
        "env_kwargs": train_env_kwargs,
        "eval_env_kwargs": eval_env_kwargs,
    }
    _dump_yaml(params_dir / "config.yml", saved_config)
    _dump_yaml(params_dir / "args.yml", saved_args)

    train_env = None
    eval_env = None
    try:
        train_env = make_vec_env(
            AdvancedTimeAttackTaxiEnv,
            n_envs=n_envs,
            seed=args.seed,
            env_kwargs=train_env_kwargs,
        )

        normalize_config = runtime_config["normalize"]
        if normalize_config is not False:
            normalize_kwargs = {} if normalize_config is True else dict(normalize_config)
            train_env = VecNormalize(train_env, training=True, **normalize_kwargs)

        callback = None
        if eval_freq > 0:
            eval_env = make_vec_env(
                AdvancedTimeAttackTaxiEnv,
                n_envs=1,
                seed=eval_seed,
                env_kwargs=eval_env_kwargs,
            )
            if normalize_config is not False:
                eval_normalize_kwargs = (
                    {} if normalize_config is True else dict(normalize_config)
                )
                eval_normalize_kwargs["norm_reward"] = False
                eval_env = VecNormalize(
                    eval_env,
                    training=False,
                    **eval_normalize_kwargs,
                )

            callback_freq = max(eval_freq // n_envs, 1)
            callback_on_new_best = None
            if isinstance(train_env, VecNormalize):
                callback_on_new_best = SaveBestVecNormalizeCallback(
                    run_dir / "best_model_vecnormalize.pkl"
                )
            callback = MaskableEvalCallback(
                eval_env,
                callback_on_new_best=callback_on_new_best,
                best_model_save_path=str(run_dir),
                log_path=str(run_dir),
                eval_freq=callback_freq,
                n_eval_episodes=eval_episodes,
                deterministic=True,
                verbose=args.verbose,
                warn=True,
            )

        model = MaskablePPO(
            runtime_config["policy"],
            train_env,
            tensorboard_log=str(tensorboard_log),
            seed=args.seed,
            verbose=args.verbose,
            device=args.device,
            **runtime_config["model_kwargs"],
        )
        model.learn(
            total_timesteps=n_timesteps,
            callback=callback,
            progress_bar=args.progress,
            tb_log_name=f"{ENV_ID}_{run_dir.name.rsplit('_', 1)[-1]}",
        )
        model.save(str(run_dir / ENV_ID))
        if isinstance(train_env, VecNormalize):
            train_env.save(str(params_dir / "vecnormalize.pkl"))
    finally:
        if eval_env is not None:
            eval_env.close()
        if train_env is not None:
            train_env.close()

    print(f"[INFO] 학습 완료: {run_dir}")
    return run_dir


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        train(args)
    except (FileNotFoundError, TypeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
