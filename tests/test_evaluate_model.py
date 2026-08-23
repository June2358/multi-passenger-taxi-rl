"""Regression tests for experiment artifact and evaluation configuration loading."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from evaluate_model import (
    _RLEpisodeRunner,
    _evaluation_seeds,
    _load_mapping,
    _run_rl_seeds_batched,
    load_evaluation_env_kwargs,
    resolve_run_artifacts,
)


class _FirstAvailableModel:
    def predict(self, _obs, *, action_masks, deterministic):
        assert deterministic
        return np.argmax(np.asarray(action_masks, dtype=bool), axis=1), None


class EvaluationArtifactTests(unittest.TestCase):
    def test_batched_rl_evaluation_matches_scalar_seed_results(self):
        env_kwargs = {
            "num_passengers": 6,
            "grid_size": 30,
            "scenario_mode": "complex",
            "start_at_center": False,
            "scenario_mix": [0.4, 0.3, 0.3],
            "mix_noise_std": 0.0,
            "mix_epsilon": 0.001,
        }
        model = _FirstAvailableModel()
        seeds = list(range(100, 107))
        scalar_runner = _RLEpisodeRunner(model, env_kwargs, None)
        try:
            scalar = [scalar_runner.run(seed)[0] for seed in seeds]
        finally:
            scalar_runner.close()

        batched = _run_rl_seeds_batched(model, env_kwargs, None, seeds)

        self.assertEqual(batched, scalar)

    def test_evaluation_seed_range_is_explicit_and_disjoint_from_default_train_seed(self):
        seeds = _evaluation_seeds(10_000, 3)

        self.assertEqual(list(seeds), [10_000, 10_001, 10_002])
        self.assertNotIn(42, seeds)
        with self.assertRaises(ValueError):
            _evaluation_seeds(-1, 3)
        with self.assertRaises(ValueError):
            _evaluation_seeds(10_000, 0)

    def test_experiment_id_zero_is_not_treated_as_latest(self):
        with self.assertRaisesRegex(ValueError, "1 이상"):
            resolve_run_artifacts(
                Path("logs"),
                "maskableppo",
                "AdvancedTaxi-v1",
                0,
            )

    def _create_run(
        self,
        root: Path,
        *,
        exp_id: int = 1,
        normalize: bool = True,
        create_nested_stats: bool = True,
    ) -> Path:
        run_dir = root / "maskableppo" / f"AdvancedTaxi-v1_{exp_id}"
        params_dir = run_dir / "AdvancedTaxi-v1"
        params_dir.mkdir(parents=True)
        (run_dir / "AdvancedTaxi-v1.zip").write_bytes(b"model")
        (params_dir / "config.yml").write_text(
            yaml.safe_dump({"normalize": {"norm_obs": True}} if normalize else {"normalize": False}),
            encoding="utf-8",
        )
        (params_dir / "args.yml").write_text(
            yaml.safe_dump(
                {
                    "algo": "maskableppo",
                    "env": "AdvancedTaxi-v1",
                    "env_kwargs": {
                        "num_passengers": 12,
                        "grid_size": 80,
                        "start_at_center": False,
                        "scenario_mode": "mixed",
                        "scenario_mix": [0.4, 0.3, 0.3],
                        "mix_noise_std": 0.05,
                        "mix_epsilon": 0.001,
                    },
                    "eval_env_kwargs": {
                        "num_passengers": 12,
                        "grid_size": 80,
                        "start_at_center": False,
                        "scenario_mode": "mixed",
                        "scenario_mix": [0.4, 0.3, 0.3],
                        "mix_noise_std": 0.0,
                        "mix_epsilon": 0.001,
                    },
                }
            ),
            encoding="utf-8",
        )
        if create_nested_stats:
            (params_dir / "vecnormalize.pkl").write_bytes(b"stats")
        return run_dir

    def test_resolves_explicit_final_model_and_nested_vecnormalize(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected_run = self._create_run(root, exp_id=1)
            self._create_run(root, exp_id=2)

            artifacts = resolve_run_artifacts(
                root,
                "maskableppo",
                "AdvancedTaxi-v1",
                1,
            )

            self.assertEqual(artifacts.run_dir, expected_run)
            self.assertEqual(artifacts.model, expected_run / "AdvancedTaxi-v1.zip")
            self.assertEqual(
                artifacts.vecnormalize,
                expected_run / "AdvancedTaxi-v1" / "vecnormalize.pkl",
            )

    def test_normalized_run_fails_when_only_root_stats_exist(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self._create_run(root, create_nested_stats=False)
            (run_dir / "vecnormalize.pkl").write_bytes(b"wrong location")

            with self.assertRaisesRegex(FileNotFoundError, "VecNormalize"):
                resolve_run_artifacts(root, "maskableppo", "AdvancedTaxi-v1", 1)

    def test_non_normalized_run_does_not_require_stats(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._create_run(root, normalize=False, create_nested_stats=False)

            artifacts = resolve_run_artifacts(
                root,
                "maskableppo",
                "AdvancedTaxi-v1",
                1,
            )

            self.assertIsNone(artifacts.vecnormalize)

    def test_empty_normalization_mapping_still_requires_vecnormalize(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self._create_run(root)
            config_path = run_dir / "AdvancedTaxi-v1" / "config.yml"
            config_path.write_text(
                yaml.safe_dump({"normalize": {}}),
                encoding="utf-8",
            )

            artifacts = resolve_run_artifacts(
                root,
                "maskableppo",
                "AdvancedTaxi-v1",
                1,
            )

            self.assertIsNotNone(artifacts.vecnormalize)

    def test_training_environment_kwargs_are_resolved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._create_run(root)
            artifacts = resolve_run_artifacts(
                root,
                "maskableppo",
                "AdvancedTaxi-v1",
                1,
            )

            kwargs = load_evaluation_env_kwargs(artifacts)

            self.assertEqual(kwargs["num_passengers"], 12)
            self.assertEqual(kwargs["grid_size"], 80)
            self.assertFalse(kwargs["start_at_center"])
            self.assertEqual(kwargs["scenario_mode"], "mixed")
            self.assertIn("scenario_mix", kwargs)

    def test_incomplete_environment_contract_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self._create_run(root)
            args_path = run_dir / "AdvancedTaxi-v1" / "args.yml"
            args_path.write_text(
                yaml.safe_dump(
                    {
                        "algo": "maskableppo",
                        "env": "AdvancedTaxi-v1",
                        "env_kwargs": {"num_passengers": 12},
                        "eval_env_kwargs": {"num_passengers": 12},
                    }
                ),
                encoding="utf-8",
            )

            artifacts = resolve_run_artifacts(
                root,
                "maskableppo",
                "AdvancedTaxi-v1",
                1,
            )
            with self.assertRaisesRegex(ValueError, "재학습"):
                load_evaluation_env_kwargs(artifacts)

    def test_python_object_yaml_tag_is_rejected(self):
        content = """\
!!python/object/apply:collections.OrderedDict
- - - algo
    - maskableppo
  - - env
    - AdvancedTaxi-v1
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "args.yml"
            path.write_text(content, encoding="utf-8")

            with self.assertRaises(yaml.constructor.ConstructorError):
                _load_mapping(path)


if __name__ == "__main__":
    unittest.main()
