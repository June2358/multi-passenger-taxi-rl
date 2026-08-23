import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import yaml

from train_advanced_taxi_maskableppo import (
    DEFAULT_CONFIG_PATH,
    TRAIN_ENV_KWARGS,
    build_parser,
    load_training_config,
    parse_env_kwargs,
    resolve_env_kwargs,
)
from train_rlzoo import (
    parse_qlearning_args,
    parse_wrapper_args,
    resolve_config_file,
    resolve_qlearning_log_dir,
)


class TrainRlZooCliTests(unittest.TestCase):
    def test_local_options_are_not_forwarded(self):
        local, inspected, forwarded = parse_wrapper_args(
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
                _, inspected, forwarded = parse_wrapper_args(cli_args)
                self.assertEqual(inspected.conf_file, "custom.yml")
                self.assertEqual(forwarded, cli_args)

    def test_rl_zoo_thread_override_is_detected_and_forwarded(self):
        _, inspected, forwarded = parse_wrapper_args(
            ["--algo", "dqn", "--env", "Taxi-v4", "--num-threads", "3"]
        )

        self.assertEqual(inspected.num_threads, 3)
        self.assertEqual(
            forwarded,
            ["--algo", "dqn", "--env", "Taxi-v4", "--num-threads", "3"],
        )

    def test_config_resolution_uses_taxi_v4_directory(self):
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

    def test_qlearning_log_directory_is_nested_under_algorithm(self):
        resolved = resolve_qlearning_log_dir("custom-logs")

        self.assertEqual(resolved, Path("custom-logs").resolve() / "qlearning")
        self.assertIsNone(resolve_qlearning_log_dir(None))

    def test_qlearning_cli_accepts_only_its_explicit_contract(self):
        args = parse_qlearning_args(
            [
                "--algo",
                "qlearning",
                "--env",
                "Taxi-v4",
                "--n-timesteps",
                "123",
                "--seed",
                "7",
                "--log-folder",
                "custom-logs",
            ]
        )

        self.assertEqual(args.n_timesteps, 123)
        self.assertEqual(args.seed, 7)
        self.assertEqual(args.log_folder, "custom-logs")

        for unsupported in (
            ["--eval-freq", "10"],
            ["--device", "cpu"],
            ["--n-timestpes", "123"],
            ["--n-timesteps", "0"],
            ["--seed", "-1"],
        ):
            with (
                self.subTest(unsupported=unsupported),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                parse_qlearning_args(["--algo", "qlearning", *unsupported])


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
            [["num_passengers:30", "scenario_mode:'complex'"]],
            [["start_at_center:True"]],
        )

        self.assertEqual(train_args["num_passengers"], 30)
        self.assertEqual(eval_args["num_passengers"], 30)
        self.assertEqual(eval_args["scenario_mode"], "complex")
        self.assertTrue(eval_args["start_at_center"])

    def test_evaluation_overrides_cannot_change_model_spaces(self):
        for override in (["num_passengers:30"], ["grid_size:120"]):
            with (
                self.subTest(override=override),
                self.assertRaisesRegex(ValueError, "observation/action space"),
            ):
                resolve_env_kwargs(None, [override])

    def test_cli_environment_values_are_safe_yaml_compatible(self):
        merged = parse_env_kwargs(
            TRAIN_ENV_KWARGS,
            [["scenario_mode:mixed", "scenario_mix:(0.4, 0.3, 0.3)"]],
        )

        self.assertEqual(merged["scenario_mode"], "mixed")
        self.assertEqual(merged["scenario_mix"], [0.4, 0.3, 0.3])

    def test_training_config_rejects_partial_rollout_minibatches(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "n_envs": 2,
                        "model_kwargs": {"n_steps": 10, "batch_size": 16},
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "나누어떨어지게"):
                load_training_config(path)

    def test_malformed_env_kwarg_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "key:value"):
            parse_env_kwargs(TRAIN_ENV_KWARGS, [["grid_size=120"]])

    def test_unknown_env_kwarg_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "알 수 없는 키"):
            parse_env_kwargs(TRAIN_ENV_KWARGS, [["custom_flag:True"]])

    def test_training_config_rejects_unknown_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yml"
            path.write_text("unexpected: true\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "알 수 없는 키"):
                load_training_config(path)


if __name__ == "__main__":
    unittest.main()
