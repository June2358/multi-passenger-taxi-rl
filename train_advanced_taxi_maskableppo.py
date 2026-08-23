"""
AdvancedTaxi-v1 전용 MaskablePPO 학습 스크립트

- MaskablePPO 등록 및 RL-Zoo 패치(Optuna 포함)
- 기본 하이퍼파라미터: hyperparameters/advanced_taxi/maskableppo_base.yml
- 기본 디바이스: CPU, 기본 시드: 42
"""

import argparse
import ast
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 환경 등록
import envs  # noqa: F401

ENV_ID = "AdvancedTaxi-v1"

# --------------------------
# 환경 분포 기본값 (CLI env kwargs로 명시적 오버라이드 가능)
# --------------------------
TRAIN_ENV_KWARGS = {
    "num_passengers": 20,
    "grid_size": 100,
    "start_at_center": False,
    "scenario_mode": "mixed",
    "scenario_mix": [0.4, 0.3, 0.3],  # random / cluster / complex
    "mix_noise_std": 0.05,  # 학습 시 약간의 노이즈로 과적합 완화
    "mix_epsilon": 1e-3,
}
EVAL_ENV_KWARGS = {
    **TRAIN_ENV_KWARGS,
    "mix_noise_std": 0.0,  # 평가 시 완전 고정
}
EVAL_ENV_OVERRIDES = {"mix_noise_std": 0.0}


def parse_wrapper_args(argv: list[str]) -> tuple[argparse.Namespace, list[str], argparse.ArgumentParser]:
    """Remove wrapper-owned options while preserving RL-Zoo arguments."""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--param-mode", choices=["base"], default="base")
    parser.add_argument("--exp-id", type=int)
    parser.add_argument("-uuid", "--uuid", action="store_true")
    parser.add_argument("-conf", "--conf-file", dest="conf_files", action="append")
    parser.add_argument("--env-kwargs", nargs="+", action="append")
    parser.add_argument("--eval-env-kwargs", nargs="+", action="append")
    wrapper_args, rl_zoo_args = parser.parse_known_args(argv)
    if wrapper_args.exp_id is not None:
        parser.error("--exp-id는 학습 명령에서 지원하지 않습니다.")
    if wrapper_args.uuid:
        parser.error("--uuid는 평가 실행 경로를 모호하게 하므로 지원하지 않습니다.")
    return wrapper_args, rl_zoo_args, parser


def inspect_rl_zoo_args(argv: list[str]) -> argparse.Namespace:
    """Read options used for defaults without removing them from ``argv``."""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--algo")
    parser.add_argument("--env")
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    parser.add_argument("-optimize", "--optimize-hyperparameters", action="store_true")
    parser.add_argument("--n-jobs", type=int)
    parser.add_argument("--tensorboard-log", "-tb")
    inspected_args, _ = parser.parse_known_args(argv)
    return inspected_args


def merge_env_kwargs(defaults: dict[str, object], groups: list[list[str]] | None) -> list[str]:
    """Merge RL-Zoo ``key:value`` tokens, with the last user value winning."""
    merged = {key: f"{key}:{value!r}" for key, value in defaults.items()}
    for group in groups or []:
        for token in group:
            key, separator, raw_value = token.partition(":")
            if not separator or not key or not raw_value:
                raise ValueError(f"환경 인자는 key:value 형식이어야 합니다: {token!r}")
            try:
                value = ast.literal_eval(raw_value)
            except (SyntaxError, ValueError):
                value = raw_value
            if isinstance(value, tuple):
                value = list(value)
            merged[key] = f"{key}:{value!r}"
    return list(merged.values())


def resolve_env_kwarg_args(
    train_groups: list[list[str]] | None,
    eval_groups: list[list[str]] | None,
) -> tuple[list[str], list[str]]:
    """Resolve complete train/eval kwargs while keeping their spaces compatible."""
    train_args = merge_env_kwargs(TRAIN_ENV_KWARGS, train_groups)
    eval_override_args = [
        f"{key}:{value!r}" for key, value in EVAL_ENV_OVERRIDES.items()
    ]
    combined_eval_groups = [train_args, eval_override_args, *(eval_groups or [])]
    eval_args = merge_env_kwargs({}, combined_eval_groups)
    return train_args, eval_args


def _eval_log_root(log_path: str | None) -> str | None:
    """Undo EvalCallback's automatic ``evaluations`` filename suffix."""
    if log_path is None:
        return None
    return str(Path(log_path).parent)


def is_valid_rollout_batch_size(n_steps: int, batch_size: int, n_envs: int) -> bool:
    """Return whether PPO can split a rollout into full mini-batches."""
    if min(n_steps, batch_size, n_envs) < 1:
        return False
    rollout_size = n_steps * n_envs
    return batch_size <= rollout_size and rollout_size % batch_size == 0


def _register_maskableppo():
    """MaskablePPO를 RL-Zoo에 등록하고 평가/Optuna 콜백을 패치."""
    import rl_zoo3
    import rl_zoo3.train as rl_zoo3_train
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
    from rl_zoo3 import hyperparams_opt
    from rl_zoo3.exp_manager import ExperimentManager
    from stable_baselines3.common.callbacks import EvalCallback
    import numpy as np
    import optuna
    import torch

    # ALGOS 등록
    rl_zoo3.ALGOS["maskableppo"] = MaskablePPO
    rl_zoo3_train.ALGOS = rl_zoo3.ALGOS
    if hasattr(rl_zoo3, "exp_manager"):
        rl_zoo3.exp_manager.ALGOS = rl_zoo3.ALGOS
    print("[INFO] MaskablePPO 알고리즘 등록 완료")

    # Optuna 샘플러: gamma=1.0 고정
    def sample_maskableppo_params(trial, n_actions, n_envs, additional_args):
        batch_size_pow = trial.suggest_int("batch_size_pow", 2, 10)
        n_steps_pow = trial.suggest_int("n_steps_pow", 5, 12)
        batch_size = 2**batch_size_pow
        n_steps = 2**n_steps_pow
        if not is_valid_rollout_batch_size(n_steps, batch_size, n_envs):
            raise optuna.TrialPruned(
                "PPO batch_size must divide n_steps * n_envs and not exceed it"
            )
        one_minus_gamma = 0.0
        one_minus_gae_lambda = trial.suggest_float("one_minus_gae_lambda", 0.0001, 0.1, log=True)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 0.002, log=True)
        ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
        clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
        n_epochs = trial.suggest_categorical("n_epochs", [1, 5, 10, 20])
        max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 2)
        net_arch = trial.suggest_categorical("net_arch", ["tiny", "small", "medium"])
        activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

        trial.set_user_attr("gamma", 1.0)
        trial.set_user_attr("n_steps", n_steps)
        trial.set_user_attr("batch_size", batch_size)

        sampled_params = {
            "n_steps_pow": n_steps_pow,
            "batch_size_pow": batch_size_pow,
            "one_minus_gamma": one_minus_gamma,
            "one_minus_gae_lambda": one_minus_gae_lambda,
            "learning_rate": learning_rate,
            "ent_coef": ent_coef,
            "clip_range": clip_range,
            "n_epochs": n_epochs,
            "max_grad_norm": max_grad_norm,
            "net_arch": net_arch,
            "activation_fn": activation_fn,
        }

        hyperparams = hyperparams_opt.convert_onpolicy_params(sampled_params)
        assert hyperparams["gamma"] == 1.0
        return hyperparams

    hyperparams_opt.HYPERPARAMS_SAMPLER["maskableppo"] = sample_maskableppo_params
    hyperparams_opt.HYPERPARAMS_CONVERTER["maskableppo"] = hyperparams_opt.HYPERPARAMS_CONVERTER["ppo"]
    print("[INFO] MaskablePPO Optuna sampler: gamma is fixed to 1.0")

    # 평가 콜백 패치: 기본 콜백 체인은 유지하고, MaskablePPO일 때만 EvalCallback을 교체
    original_create_callbacks = ExperimentManager.create_callbacks

    def patched_create_callbacks(self):
        # 1) RL-Zoo 기본 콜백 구성을 그대로 생성
        original_create_callbacks(self)

        # 2) MaskablePPO가 아닌 경우에는 기본 동작 유지
        if getattr(self, "algo", None) != "maskableppo":
            return

        # 3) 기존 EvalCallback들을 MaskableEvalCallback으로 교체
        new_callbacks = []
        eval_callbacks = []
        for cb in self.callbacks:
            if isinstance(cb, EvalCallback):
                eval_callbacks.append(cb)
            else:
                new_callbacks.append(cb)

        # EvalCallback이 없으면 아무 것도 하지 않음
        if not eval_callbacks:
            self.callbacks = new_callbacks
            return

        # EvalCallback을 제거하고, 같은 설정으로 MaskableEvalCallback을 추가
        self.callbacks = new_callbacks
        for old_cb in eval_callbacks:
            eval_callback = MaskableEvalCallback(
                old_cb.eval_env,
                callback_on_new_best=old_cb.callback_on_new_best,
                callback_after_eval=old_cb.callback,
                best_model_save_path=old_cb.best_model_save_path,
                n_eval_episodes=old_cb.n_eval_episodes,
                log_path=_eval_log_root(old_cb.log_path),
                eval_freq=old_cb.eval_freq,
                deterministic=old_cb.deterministic,
                render=old_cb.render,
                verbose=old_cb.verbose,
                warn=old_cb.warn,
            )
            self.callbacks.append(eval_callback)

        if getattr(self, "verbose", 0) > 0:
            print("[INFO] Replaced EvalCallback with MaskableEvalCallback for MaskablePPO (action masking support)")

    ExperimentManager.create_callbacks = patched_create_callbacks
    print("[INFO] ExperimentManager.create_callbacks patched for MaskableEvalCallback support")

    # Optuna TrialEvalCallback 패치
    OPTUNA_FIXED_SEED = 42
    original_objective = ExperimentManager.objective

    class MaskableTrialEvalCallback(MaskableEvalCallback):
        def __init__(
            self,
            eval_env,
            trial,
            n_eval_episodes=5,
            eval_freq=10000,
            deterministic=True,
            verbose=0,
            best_model_save_path=None,
            log_path=None,
        ):
            super().__init__(
                eval_env=eval_env,
                n_eval_episodes=n_eval_episodes,
                eval_freq=eval_freq,
                deterministic=deterministic,
                verbose=verbose,
                best_model_save_path=best_model_save_path,
                log_path=log_path,
            )
            self.trial = trial
            self.eval_idx = 0
            self.is_pruned = False

        def _on_step(self) -> bool:
            if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
                continue_training = super()._on_step()
                self.eval_idx += 1
                self.trial.report(self.last_mean_reward, self.eval_idx)
                if self.trial.should_prune():
                    self.is_pruned = True
                    return False
                return continue_training
            return True

    def patched_objective(self, trial: optuna.Trial) -> float:
        # MaskablePPO가 아닌 경우에는 RL-Zoo 기본 objective를 그대로 사용
        if getattr(self, "algo", None) != "maskableppo":
            return original_objective(self, trial)

        from rl_zoo3.utils import ALGOS, get_callback_list
        from rl_zoo3.hyperparams_opt import HYPERPARAMS_SAMPLER
        import os
        import random

        random.seed(OPTUNA_FIXED_SEED)
        np.random.seed(OPTUNA_FIXED_SEED)
        torch.manual_seed(OPTUNA_FIXED_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(OPTUNA_FIXED_SEED)

        kwargs = self._hyperparams.copy()
        n_envs = self.n_envs

        additional_args = {
            "using_her_replay_buffer": kwargs.get("replay_buffer_class") == "HerReplayBuffer",
            "her_kwargs": kwargs.get("replay_buffer_kwargs", {}),
        }
        sampled_hyperparams = HYPERPARAMS_SAMPLER[self.algo](trial, self.n_actions, n_envs, additional_args)
        kwargs.update(sampled_hyperparams)

        trial_verbosity = 0
        if self.verbose >= 2:
            trial_verbosity = self.verbose

        env = None
        model = None
        eval_env = None
        try:
            env = self.create_envs(n_envs, no_log=True)
            model = ALGOS[self.algo](
                env=env,
                tensorboard_log=None,
                seed=OPTUNA_FIXED_SEED,
                verbose=trial_verbosity,
                device=self.device,
                **kwargs,
            )

            eval_env = self.create_envs(n_envs=self.n_eval_envs, eval_env=True)
            optuna_eval_freq = int(self.n_timesteps / self.n_evaluations)
            optuna_eval_freq = max(optuna_eval_freq // self.n_envs, 1)

            path = None
            if self.optimization_log_path is not None:
                path = os.path.join(self.optimization_log_path, f"trial_{trial.number!s}")

            callbacks = get_callback_list({"callback": self.specified_callbacks})
            eval_callback = MaskableTrialEvalCallback(
                eval_env,
                trial,
                best_model_save_path=path,
                log_path=path,
                n_eval_episodes=self.n_eval_episodes,
                eval_freq=optuna_eval_freq,
                deterministic=self.deterministic_eval,
            )
            if self.verbose > 0:
                print("[INFO] Using MaskableTrialEvalCallback for MaskablePPO (action masking support in Optuna)")
            callbacks.append(eval_callback)

            model.learn(self.n_timesteps, callback=callbacks)
            is_pruned = eval_callback.is_pruned
            reward = eval_callback.last_mean_reward
        except (AssertionError, ValueError) as e:
            print(e)
            print("============")
            print("Sampled hyperparams:")
            from pprint import pprint as _pprint

            _pprint(sampled_hyperparams)
            raise optuna.exceptions.TrialPruned() from e
        finally:
            if model is not None and model.env is not None:
                model.env.close()
            elif env is not None:
                env.close()
            if eval_env is not None:
                eval_env.close()

        if is_pruned:
            raise optuna.exceptions.TrialPruned()

        return reward

    ExperimentManager.objective = patched_objective
    print("[INFO] ExperimentManager.objective patched for MaskableTrialEvalCallback support (Optuna optimization)")


if __name__ == "__main__":
    try:
        _register_maskableppo()
    except ImportError as e:
        print("[ERROR] sb3-contrib가 설치되지 않아 MaskablePPO를 사용할 수 없습니다.")
        print(f"ImportError: {e}")
        print("\n해결 방법:")
        print("1. pixi 환경 사용: pixi run python train_advanced_taxi_maskableppo.py ...")
        print("2. 또는 직접 설치: pip install sb3-contrib")
        sys.exit(1)

    from rl_zoo3.train import train

    HYPERPARAMS_DIR = ROOT / "hyperparameters"
    wrapper_args, args, wrapper_parser = parse_wrapper_args(sys.argv[1:])
    if wrapper_args.conf_files and len(wrapper_args.conf_files) > 1:
        wrapper_parser.error("--conf-file은 한 번만 지정할 수 있습니다.")
    known_args = inspect_rl_zoo_args(args)
    algo = known_args.algo
    env = known_args.env

    # 강제 설정: maskableppo + AdvancedTaxi-v1
    if algo is None:
        algo = "maskableppo"
        args.extend(["--algo", algo])
        print("[INFO] --algo가 지정되지 않아 maskableppo로 설정합니다.")
    if algo != "maskableppo":
        print("[ERROR] 이 스크립트는 maskableppo 전용입니다. --algo maskableppo 를 사용하세요.")
        sys.exit(1)

    if env is None:
        env = ENV_ID
        args.extend(["--env", env])
        print(f"[INFO] --env가 지정되지 않아 {ENV_ID}로 설정합니다.")
    if env != ENV_ID:
        print(f"[ERROR] 이 스크립트는 {ENV_ID} 전용입니다.")
        sys.exit(1)

    # 디바이스 기본: CPU
    if known_args.device is None:
        device = "cpu"
        print("[INFO] Device not specified, using CPU (default)")
        args.extend(["--device", device])

    # 시드 기본: 42
    if known_args.seed is None:
        seed = 42
        print("[INFO] Seed not specified, using 42 (default for reproducibility)")
        args.extend(["--seed", str(seed)])

    # Optimize 모드 기본 n_jobs 제안 (on-policy이므로 1)
    if known_args.optimize_hyperparameters and known_args.n_jobs is None:
        print(f"[INFO] Optimize 모드 감지: 알고리즘 '{algo}'")
        print("[INFO] 권장 n_jobs: 1 (on-policy, 병렬 환경을 내부에서 사용)")
        args.extend(["--n-jobs", "1"])

    # TensorBoard 로그 경로를 명시적으로 주입해 RL-Zoo가 항상 TF 로그를 남기도록 함
    if known_args.tensorboard_log is None:
        tb_root = ROOT / "logs"
        args.extend(["--tensorboard-log", str(tb_root)])
        print(f"[INFO] TensorBoard 로그 기본 경로 설정: {tb_root}")

    # 하이퍼파라미터 파일 자동 선택 (AdvancedTaxi + maskableppo 전용)
    if wrapper_args.conf_files:
        args.extend(["--conf-file", wrapper_args.conf_files[0]])
    else:
        env_dir = HYPERPARAMS_DIR / "advanced_taxi"
        filename = "maskableppo_base.yml"
        conf_file = env_dir / filename

        if conf_file.exists():
            args.extend(["--conf-file", str(conf_file)])
            print(
                f"[INFO] 하이퍼파라미터 파일: {conf_file.relative_to(ROOT)} "
                f"(mode: {wrapper_args.param_mode})"
            )
        else:
            print(f"[ERROR] 하이퍼파라미터 파일이 존재하지 않습니다: {conf_file}")
            print("[INFO] --conf-file 옵션으로 파일을 명시적으로 지정해주세요.")
            sys.exit(1)

    # 완전히 해석된 환경 설정을 한 번만 전달하여 args.yml에도 그대로 남긴다.
    try:
        train_env_kwargs_args, eval_env_kwargs_args = resolve_env_kwarg_args(
            wrapper_args.env_kwargs,
            wrapper_args.eval_env_kwargs,
        )
    except ValueError as exc:
        wrapper_parser.error(str(exc))

    args.extend(["--env-kwargs"] + train_env_kwargs_args)
    print(f"[INFO] 학습 env kwargs 적용: {train_env_kwargs_args}")

    args.extend(["--eval-env-kwargs"] + eval_env_kwargs_args)
    print(f"[INFO] 평가 env kwargs 적용: {eval_env_kwargs_args}")

    # RL-Zoo에 전달
    sys.argv = [sys.argv[0]] + args
    train()
