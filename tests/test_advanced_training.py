"""End-to-end regression test for the direct AdvancedTaxi trainer."""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml
from sb3_contrib import MaskablePPO

from evaluate_model import (
    evaluate_scenarios,
    load_evaluation_env_kwargs,
    resolve_run_artifacts,
)
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
                    "--validation-seed-start",
                    "100",
                    "--validation-seeds-per-scenario",
                    "2",
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
                "AdvancedTaxi-v2",
                1,
                "final",
            )
            self.assertEqual(artifacts.run_dir, run_dir)
            self.assertTrue((run_dir / "best_model.zip").is_file())
            self.assertTrue((run_dir / "best_model_vecnormalize.pkl").is_file())
            self.assertTrue((run_dir / "evaluations.npz").is_file())
            self.assertTrue(artifacts.vecnormalize.is_file())

            best_artifacts = resolve_run_artifacts(
                logs_dir,
                "maskableppo",
                "AdvancedTaxi-v2",
                1,
                "best",
            )
            self.assertEqual(
                best_artifacts.vecnormalize,
                run_dir / "best_model_vecnormalize.pkl",
            )

            with np.load(run_dir / "evaluations.npz") as evaluations:
                self.assertEqual(
                    evaluations["scenarios"].tolist(),
                    ["random", "cluster", "complex"],
                )
                self.assertEqual(evaluations["validation_seeds"].tolist(), [100, 101])
                self.assertEqual(evaluations["results"].shape, (2, 3, 2))

            eval_kwargs = load_evaluation_env_kwargs(artifacts)
            self.assertEqual(eval_kwargs["num_passengers"], 2)
            self.assertEqual(eval_kwargs["grid_size"], 20)
            self.assertEqual(eval_kwargs["scenario_mode"], "random")
            saved_args = yaml.safe_load(artifacts.args.read_text(encoding="utf-8"))
            self.assertEqual(saved_args["actual_timesteps"], 8)

            model = MaskablePPO.load(str(artifacts.model), device="cpu")
            self.assertEqual(model.action_space.n, 2)

            results_path = run_dir / "smoke_final.npz"
            evaluate_scenarios(
                model=model,
                vecnormalize_path=artifacts.vecnormalize,
                env_kwargs=eval_kwargs,
                results_path=results_path,
                checkpoint=artifacts.checkpoint,
                num_seeds=1,
                seed_start=200,
            )
            self.assertTrue(results_path.is_file())


if __name__ == "__main__":
    unittest.main()
