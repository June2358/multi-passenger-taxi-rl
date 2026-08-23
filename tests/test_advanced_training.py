"""End-to-end regression test for the direct AdvancedTaxi trainer."""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import yaml
from sb3_contrib import MaskablePPO

from evaluate_model import load_evaluation_env_kwargs, resolve_run_artifacts
from train_advanced_taxi_maskableppo import build_parser, train


class AdvancedTaxiTrainingTests(unittest.TestCase):
    def test_short_training_run_writes_evaluator_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.yml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "policy": "MlpPolicy",
                        "n_timesteps": 8,
                        "n_envs": 1,
                        "normalize": {
                            "norm_obs": True,
                            "norm_reward": False,
                            "gamma": 1.0,
                        },
                        "model_kwargs": {
                            "n_steps": 4,
                            "batch_size": 4,
                            "n_epochs": 1,
                            "gamma": 1.0,
                            "policy_kwargs": {
                                "net_arch": {"pi": [16], "vf": [16]},
                                "activation_fn": "relu",
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            logs_dir = root / "logs"
            tensorboard_dir = root / "tensorboard"
            args = build_parser().parse_args(
                [
                    "--conf-file",
                    str(config_path),
                    "--log-folder",
                    str(logs_dir),
                    "--tensorboard-log",
                    str(tensorboard_dir),
                    "--eval-freq",
                    "4",
                    "--eval-episodes",
                    "1",
                    "--seed",
                    "7",
                    "--verbose",
                    "0",
                    "--env-kwargs",
                    "num_passengers:2",
                    "grid_size:20",
                    "scenario_mode:random",
                ]
            )

            with contextlib.redirect_stdout(io.StringIO()):
                run_dir = train(args)

            artifacts = resolve_run_artifacts(
                logs_dir,
                "maskableppo",
                "AdvancedTaxi-v1",
                1,
            )
            self.assertEqual(artifacts.run_dir, run_dir)
            self.assertTrue((run_dir / "best_model.zip").is_file())
            self.assertTrue((run_dir / "best_model_vecnormalize.pkl").is_file())
            self.assertTrue((run_dir / "evaluations.npz").is_file())
            self.assertTrue(artifacts.vecnormalize.is_file())

            eval_kwargs = load_evaluation_env_kwargs(artifacts)
            self.assertEqual(eval_kwargs["num_passengers"], 2)
            self.assertEqual(eval_kwargs["grid_size"], 20)
            self.assertEqual(eval_kwargs["scenario_mode"], "random")
            self.assertEqual(eval_kwargs["mix_noise_std"], 0.0)

            model = MaskablePPO.load(str(artifacts.model), device="cpu")
            self.assertEqual(model.action_space.n, 2)


if __name__ == "__main__":
    unittest.main()
