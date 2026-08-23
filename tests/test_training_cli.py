import contextlib
import io
import unittest
from pathlib import Path

from train_advanced_taxi_maskableppo import (
    EVAL_ENV_KWARGS,
    TRAIN_ENV_KWARGS,
    _eval_log_root,
    is_valid_rollout_batch_size,
    merge_env_kwargs,
    parse_wrapper_args as parse_advanced_wrapper_args,
    resolve_env_kwarg_args,
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
                "Taxi-v3",
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
            ["--algo", "qlearning", "--env", "Taxi-v3", "--seed", "11"],
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

    def test_config_resolution_has_no_legacy_root_fallback(self):
        hyperparams_dir = Path("hyperparameters")

        resolved = resolve_config_file(
            "Taxi-v3",
            "dqn",
            "base",
            hyperparams_dir,
        )

        self.assertEqual(resolved, hyperparams_dir / "taxi_v3" / "dqn_base.yml")
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
    def test_leak_free_environment_defaults_to_baseline_parameters(self):
        wrapper, _, _ = parse_advanced_wrapper_args([])

        self.assertEqual(wrapper.param_mode, "base")

    def test_stale_tuned_mode_is_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_advanced_wrapper_args(["--param-mode", "tuned"])

    def test_wrapper_options_are_consumed(self):
        wrapper, forwarded, _ = parse_advanced_wrapper_args(
            [
                "--algo",
                "maskableppo",
                "--param-mode=base",
                "--conf-file=custom.yml",
                "--env-kwargs",
                "grid_size:120",
                "scenario_mode:'complex'",
                "--eval-env-kwargs",
                "start_at_center:True",
                "--seed",
                "5",
            ]
        )

        self.assertEqual(wrapper.param_mode, "base")
        self.assertEqual(wrapper.conf_files, ["custom.yml"])
        self.assertEqual(wrapper.env_kwargs, [["grid_size:120", "scenario_mode:'complex'"]])
        self.assertEqual(wrapper.eval_env_kwargs, [["start_at_center:True"]])
        self.assertEqual(forwarded, ["--algo", "maskableppo", "--seed", "5"])

    def test_exp_id_is_rejected_before_rl_zoo(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_advanced_wrapper_args(["--exp-id", "4"])

    def test_uuid_is_rejected_before_it_can_make_run_paths_ambiguous(self):
        for option in ("--uuid", "-uuid"):
            with self.subTest(option=option):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    parse_advanced_wrapper_args([option])

    def test_all_rl_zoo_conf_file_forms_are_consumed(self):
        cases = (
            ["--conf-file", "custom.yml"],
            ["--conf-file=custom.yml"],
            ["-conf", "custom.yml"],
            ["-conf=custom.yml"],
        )
        for cli_args in cases:
            with self.subTest(cli_args=cli_args):
                wrapper, forwarded, _ = parse_advanced_wrapper_args(cli_args)
                self.assertEqual(wrapper.conf_files, ["custom.yml"])
                self.assertEqual(forwarded, [])

    def test_resolved_defaults_and_user_overrides_are_emitted_once(self):
        merged = merge_env_kwargs(
            TRAIN_ENV_KWARGS,
            [["grid_size:120", "custom_flag:True"], ["grid_size:140"]],
        )
        keys = [token.partition(":")[0] for token in merged]

        self.assertEqual(keys.count("grid_size"), 1)
        self.assertIn("grid_size:140", merged)
        self.assertIn("custom_flag:True", merged)
        self.assertEqual(set(keys), set(TRAIN_ENV_KWARGS) | {"custom_flag"})
        self.assertIn(f"scenario_mode:{TRAIN_ENV_KWARGS['scenario_mode']!r}", merged)

    def test_eval_defaults_include_explicit_start_policy(self):
        merged = merge_env_kwargs(EVAL_ENV_KWARGS, None)

        self.assertIn("start_at_center:False", merged)
        self.assertIn("mix_epsilon:0.001", merged)

    def test_training_shape_overrides_propagate_to_evaluation(self):
        train_args, eval_args = resolve_env_kwarg_args(
            [["num_passengers:30", "scenario_mode:'complex'", "mix_noise_std:0.2"]],
            [["start_at_center:True"]],
        )

        self.assertIn("num_passengers:30", train_args)
        self.assertIn("num_passengers:30", eval_args)
        self.assertIn("scenario_mode:'complex'", eval_args)
        self.assertIn("mix_noise_std:0.0", eval_args)
        self.assertIn("start_at_center:True", eval_args)

    def test_unquoted_string_env_kwarg_is_forwarded_as_a_literal(self):
        merged = merge_env_kwargs(
            TRAIN_ENV_KWARGS,
            [["scenario_mode:mixed"]],
        )

        self.assertIn("scenario_mode:'mixed'", merged)

    def test_tuple_scenario_mix_is_saved_as_safe_yaml_compatible_list(self):
        merged = merge_env_kwargs(
            TRAIN_ENV_KWARGS,
            [["scenario_mix:(0.4, 0.3, 0.3)"]],
        )

        self.assertIn("scenario_mix:[0.4, 0.3, 0.3]", merged)

    def test_eval_callback_log_root_does_not_duplicate_filename(self):
        original_root = str(Path("logs") / "maskableppo" / "run")
        callback_log_path = str(Path(original_root) / "evaluations")

        self.assertEqual(_eval_log_root(callback_log_path), original_root)

    def test_ppo_rollout_requires_full_minibatches(self):
        self.assertTrue(is_valid_rollout_batch_size(512, 128, 8))
        self.assertFalse(is_valid_rollout_batch_size(32, 1_024, 8))
        self.assertFalse(is_valid_rollout_batch_size(10, 16, 2))

    def test_malformed_env_kwarg_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "key:value"):
            merge_env_kwargs(TRAIN_ENV_KWARGS, [["grid_size=120"]])


if __name__ == "__main__":
    unittest.main()
