import tempfile
import unittest
from pathlib import Path

import yaml

from train_advanced_taxi_maskableppo import (
    DEFAULT_CONFIG_PATH,
    TRAIN_ENV_KWARGS,
    build_parser,
    is_valid_rollout_batch_size,
    load_training_config,
    parse_env_kwargs,
    resolve_env_kwargs,
)
from train_rlzoo import (
    parse_wrapper_args as parse_rlzoo_wrapper_args,
    require_config_file,
    resolve_config_file,
    resolve_qlearning_log_dir,
)


class TrainRlZooCliTests(unittest.TestCase):
    def test_local_options_are_not_forwarded(self):
        local, inspected, forwarded = parse_rlzoo_wrapper_args(
            [
                "--algo",
                "qlearning",
                "--env",
                "Taxi-v4",
                "--param-mode",
                "base",
                "--seed",
                "11",
            ]
        )

        self.assertEqual(local.param_mode, "base")
        self.assertEqual(inspected.seed, 11)
        self.assertEqual(
            forwarded,
            ["--algo", "qlearning", "--env", "Taxi-v4", "--seed", "11"],
        )

    def test_all_rl_zoo_conf_file_forms_are_detected(self):
        cases = (
            ["--conf-file", "custom.yml"],
            ["--conf-file=custom.yml"],
            ["-conf", "custom.yml"],
            ["-conf=custom.yml"],
        )
        for cli_args in cases:
            with self.subTest(cli_args=cli_args):
                _, inspected, forwarded = parse_rlzoo_wrapper_args(cli_args)
                self.assertEqual(inspected.conf_file, "custom.yml")
                self.assertEqual(forwarded, cli_args)

    def test_rl_zoo_thread_override_is_detected_and_forwarded(self):
        _, inspected, forwarded = parse_rlzoo_wrapper_args(
            ["--algo", "dqn", "--env", "Taxi-v4", "--num-threads", "3"]
        )

        self.assertEqual(inspected.num_threads, 3)
        self.assertEqual(
            forwarded,
            ["--algo", "dqn", "--env", "Taxi-v4", "--num-threads", "3"],
        )

    def test_config_resolution_has_no_legacy_root_fallback(self):
        hyperparams_dir = Path("hyperparameters")

        resolved = resolve_config_file(
            "Taxi-v4",
            "dqn",
            "base",
            hyperparams_dir,
        )

        self.assertEqual(resolved, hyperparams_dir / "taxi_v4" / "dqn_base.yml")
        with self.assertRaisesRegex(ValueError, "지원하지 않는 환경"):
            resolve_config_file("Unknown-v0", "dqn", "base", hyperparams_dir)

    def test_missing_explicit_config_is_not_silently_replaced(self):
        with self.assertRaisesRegex(FileNotFoundError, "찾을 수 없습니다"):
            require_config_file(Path("does-not-exist.yml"))

    def test_qlearning_log_folder_is_forwarded_under_algorithm_directory(self):
        resolved = resolve_qlearning_log_dir("custom-logs")

        self.assertEqual(resolved, Path("custom-logs").resolve() / "qlearning")
        self.assertIsNone(resolve_qlearning_log_dir(None))


class AdvancedTaxiCliTests(unittest.TestCase):
    def test_parser_uses_reproducible_direct_trainer_defaults(self):
        args = build_parser().parse_args([])

        self.assertEqual(args.seed, 42)
        self.assertEqual(args.device, "cpu")
        self.assertEqual(args.num_threads, 1)
        self.assertEqual(args.conf_file, DEFAULT_CONFIG_PATH)
        self.assertIsNone(args.n_timesteps)

    def test_parser_reads_direct_trainer_options(self):
        args = build_parser().parse_args(
            [
                "--conf-file=custom.yml",
                "--env-kwargs",
                "grid_size:120",
                "scenario_mode:'complex'",
                "--eval-env-kwargs",
                "start_at_center:True",
                "--seed",
                "5",
                "--num-threads",
                "2",
            ]
        )

        self.assertEqual(args.conf_file, Path("custom.yml"))
        self.assertEqual(args.seed, 5)
        self.assertEqual(args.num_threads, 2)
        self.assertEqual(args.env_kwargs, [["grid_size:120", "scenario_mode:'complex'"]])
        self.assertEqual(args.eval_env_kwargs, [["start_at_center:True"]])

    def test_training_shape_overrides_propagate_to_evaluation(self):
        train_args, eval_args = resolve_env_kwargs(
            [["num_passengers:30", "scenario_mode:'complex'", "mix_noise_std:0.2"]],
            [["start_at_center:True"]],
        )

        self.assertEqual(train_args["num_passengers"], 30)
        self.assertEqual(eval_args["num_passengers"], 30)
        self.assertEqual(eval_args["scenario_mode"], "complex")
        self.assertEqual(eval_args["mix_noise_std"], 0.0)
        self.assertTrue(eval_args["start_at_center"])

    def test_cli_environment_values_are_safe_yaml_compatible(self):
        merged = parse_env_kwargs(
            TRAIN_ENV_KWARGS,
            [["scenario_mode:mixed", "scenario_mix:(0.4, 0.3, 0.3)"]],
        )

        self.assertEqual(merged["scenario_mode"], "mixed")
        self.assertEqual(merged["scenario_mix"], [0.4, 0.3, 0.3])

    def test_ppo_rollout_requires_full_minibatches(self):
        self.assertTrue(is_valid_rollout_batch_size(512, 128, 8))
        self.assertFalse(is_valid_rollout_batch_size(32, 1_024, 8))
        self.assertFalse(is_valid_rollout_batch_size(10, 16, 2))

    def test_malformed_env_kwarg_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "key:value"):
            parse_env_kwargs(TRAIN_ENV_KWARGS, [["grid_size=120"]])

    def test_unknown_env_kwarg_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "알 수 없는 키"):
            parse_env_kwargs(TRAIN_ENV_KWARGS, [["custom_flag:True"]])

    def test_safe_training_config_is_resolved_for_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "policy": "MlpPolicy",
                        "n_timesteps": 8,
                        "n_envs": 1,
                        "normalize": {"norm_obs": True, "norm_reward": False},
                        "model_kwargs": {
                            "n_steps": 4,
                            "batch_size": 4,
                            "policy_kwargs": {
                                "net_arch": {"pi": [16], "vf": [16]},
                                "activation_fn": "relu",
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )

            saved, runtime = load_training_config(path)

            self.assertEqual(saved["model_kwargs"]["policy_kwargs"]["activation_fn"], "relu")
            self.assertEqual(
                runtime["model_kwargs"]["policy_kwargs"]["activation_fn"].__name__,
                "ReLU",
            )

    def test_training_config_rejects_unknown_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yml"
            path.write_text("unexpected: true\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "알 수 없는 키"):
                load_training_config(path)


if __name__ == "__main__":
    unittest.main()
