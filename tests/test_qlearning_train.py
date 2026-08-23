import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import gymnasium as gym
import numpy as np
import yaml

import qlearning_train
from qlearning_train import (
    compute_td_target,
    derive_eval_seed,
    select_epsilon_greedy_action,
)


class _ThreeStepEnv(gym.Env):
    metadata = {}

    def __init__(self):
        self.observation_space = gym.spaces.Discrete(2)
        self.action_space = gym.spaces.Discrete(2)
        self.reset_seeds = []
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.reset_seeds.append(seed)
        self.steps = 0
        return 0, {}

    def step(self, action):
        self.steps += 1
        terminated = self.steps == 3
        return self.steps % 2, 1.0, terminated, False, {}


class _OneStepEnv(gym.Env):
    metadata = {}

    def __init__(self, *, truncate):
        self.observation_space = gym.spaces.Discrete(2)
        self.action_space = gym.spaces.Discrete(2)
        self.truncate = truncate

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return 0, {}

    def step(self, action):
        return 1, 1.0, not self.truncate, self.truncate, {}


class _RecordingWriter:
    instances = []

    def __init__(self, *args, **kwargs):
        self.scalars = []
        self.__class__.instances.append(self)

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, value, step))

    def flush(self):
        pass

    def close(self):
        pass


class ComputeTdTargetTests(unittest.TestCase):
    def test_terminated_transition_does_not_bootstrap(self):
        target = compute_td_target(
            reward=2.0,
            gamma=0.9,
            next_q_values=np.array([10.0, 20.0]),
            terminated=True,
        )

        self.assertEqual(target, 2.0)

    def test_nonterminated_transition_bootstraps(self):
        # A truncated transition is represented by terminated=False: truncation
        # deliberately is not an input to the TD target.
        target = compute_td_target(
            reward=2.0,
            gamma=0.9,
            next_q_values=np.array([10.0, 20.0]),
            terminated=False,
        )

        self.assertEqual(target, 20.0)


class EpsilonGreedyActionTests(unittest.TestCase):
    def test_same_seed_reproduces_exploration(self):
        q_values = np.array([0.0, 1.0, 2.0, 3.0])
        first_rng = np.random.default_rng(123)
        second_rng = np.random.default_rng(123)

        first = [
            select_epsilon_greedy_action(q_values, 1.0, first_rng)
            for _ in range(30)
        ]
        second = [
            select_epsilon_greedy_action(q_values, 1.0, second_rng)
            for _ in range(30)
        ]

        self.assertEqual(first, second)
        self.assertTrue(all(0 <= action < q_values.size for action in first))

    def test_greedy_ties_are_random_and_reproducible(self):
        q_values = np.array([5.0, 1.0, 5.0, 0.0])
        first_rng = np.random.default_rng(456)
        second_rng = np.random.default_rng(456)

        first = [
            select_epsilon_greedy_action(q_values, 0.0, first_rng)
            for _ in range(30)
        ]
        second = [
            select_epsilon_greedy_action(q_values, 0.0, second_rng)
            for _ in range(30)
        ]

        self.assertEqual(first, second)
        self.assertEqual(set(first), {0, 2})


class TrainQlearningIntegrationTests(unittest.TestCase):
    def setUp(self):
        _RecordingWriter.instances.clear()

    def test_resolved_config_partial_episode_and_eval_rng_are_independent(self):
        created_envs = []

        def make_env(_env_id):
            env = _ThreeStepEnv()
            created_envs.append(env)
            return env

        original_select = select_epsilon_greedy_action
        selection_calls = []

        def record_selection(q_values, epsilon, rng):
            selection_calls.append((epsilon, rng))
            return original_select(q_values, epsilon, rng)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            conf_file = temp_root / "qlearning.yml"
            conf_file.write_text(
                """
Taxi-v3:
  learning_rate: 0.2
  gamma: 0.9
  epsilon_start: 1.0
  epsilon_end: 0.1
  epsilon_decay_episodes: 2
""".lstrip(),
                encoding="utf-8",
            )
            output = io.StringIO()

            with (
                mock.patch.object(qlearning_train.gym, "make", side_effect=make_env),
                mock.patch.object(qlearning_train, "SummaryWriter", _RecordingWriter),
                mock.patch.object(
                    qlearning_train,
                    "select_epsilon_greedy_action",
                    side_effect=record_selection,
                ),
                contextlib.redirect_stdout(output),
            ):
                exp_dir, model_path = qlearning_train.train_qlearning(
                    env_id="Taxi-v3",
                    conf_file=conf_file,
                    n_timesteps=5,
                    seed=17,
                    exp_id=1,
                    log_dir=temp_root / "external-logs",
                    verbose=1,
                )

            saved_config = yaml.safe_load(
                (exp_dir / "config.yml").read_text(encoding="utf-8")
            )["Taxi-v3"]
            self.assertEqual(saved_config["n_timesteps"], 5)
            self.assertEqual(saved_config["seed"], 17)
            self.assertEqual(saved_config["eval_seed"], derive_eval_seed(17))
            self.assertNotIn("episodes", saved_config)
            self.assertTrue(model_path.exists())

            writer = _RecordingWriter.instances[-1]
            rollout_rewards = [
                scalar for scalar in writer.scalars if scalar[0] == "rollout/ep_rew_mean"
            ]
            epsilon_values = [
                scalar for scalar in writer.scalars if scalar[0] == "train/epsilon"
            ]
            self.assertEqual(len(rollout_rewards), 1)
            self.assertEqual(len(epsilon_values), 1)
            self.assertAlmostEqual(epsilon_values[0][1], 0.55)
            self.assertIn("총 에피소드: 1", output.getvalue())

            train_env, eval_env = created_envs
            self.assertEqual(train_env.reset_seeds[:2], [17, None])
            self.assertEqual(eval_env.reset_seeds[0], derive_eval_seed(17))
            self.assertNotEqual(train_env.reset_seeds[0], eval_env.reset_seeds[0])

            train_rngs = {id(rng) for epsilon, rng in selection_calls if epsilon > 0.0}
            eval_rngs = {id(rng) for epsilon, rng in selection_calls if epsilon == 0.0}
            self.assertEqual(len(train_rngs), 1)
            self.assertEqual(len(eval_rngs), 1)
            self.assertTrue(train_rngs.isdisjoint(eval_rngs))

    def test_same_seed_reproduces_complete_training_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            conf_file = (
                qlearning_train.ROOT
                / "hyperparameters"
                / "taxi_v3"
                / "qlearning_base.yml"
            )

            with mock.patch.object(qlearning_train, "SummaryWriter", _RecordingWriter):
                _, first_model = qlearning_train.train_qlearning(
                    conf_file=conf_file,
                    n_timesteps=250,
                    seed=23,
                    exp_id=1,
                    log_dir=temp_root / "first",
                    verbose=0,
                )
                _, second_model = qlearning_train.train_qlearning(
                    conf_file=conf_file,
                    n_timesteps=250,
                    seed=23,
                    exp_id=1,
                    log_dir=temp_root / "second",
                    verbose=0,
                )

            self.assertTrue(
                np.array_equal(np.load(first_model), np.load(second_model))
            )

    def test_training_bootstraps_when_environment_truncates(self):
        make_count = 0

        def make_env(_env_id):
            nonlocal make_count
            make_count += 1
            return _OneStepEnv(truncate=make_count == 1)

        original_compute_target = compute_td_target
        terminated_arguments = []

        def record_target(reward, gamma, next_q_values, *, terminated):
            terminated_arguments.append(terminated)
            return original_compute_target(
                reward,
                gamma,
                next_q_values,
                terminated=terminated,
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            conf_file = temp_root / "qlearning.yml"
            conf_file.write_text(
                """
Taxi-v3:
  learning_rate: 0.2
  gamma: 0.9
  epsilon_start: 1.0
  epsilon_end: 0.1
  epsilon_decay_episodes: 2
""".lstrip(),
                encoding="utf-8",
            )

            with (
                mock.patch.object(qlearning_train.gym, "make", side_effect=make_env),
                mock.patch.object(qlearning_train, "SummaryWriter", _RecordingWriter),
                mock.patch.object(
                    qlearning_train,
                    "compute_td_target",
                    side_effect=record_target,
                ),
            ):
                qlearning_train.train_qlearning(
                    env_id="Taxi-v3",
                    conf_file=conf_file,
                    n_timesteps=1,
                    seed=31,
                    exp_id=1,
                    log_dir=temp_root / "logs",
                    verbose=0,
                )

        self.assertEqual(terminated_arguments, [False])

    def test_non_taxi_environment_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Only Taxi-v3 is supported"):
            qlearning_train.train_qlearning(env_id="AdvancedTaxi-v1")

    def test_obsolete_or_unknown_config_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            conf_file = Path(temp_dir) / "qlearning.yml"
            conf_file.write_text(
                "Taxi-v3:\n  episodes: 100\n  learning_rate: 0.1\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "episodes"):
                qlearning_train.load_hyperparams(conf_file, "Taxi-v3")

    def test_invalid_run_limits_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "n_timesteps"):
            qlearning_train.train_qlearning(n_timesteps=0)
        with self.assertRaisesRegex(ValueError, "exp_id"):
            qlearning_train.train_qlearning(exp_id=0)


if __name__ == "__main__":
    unittest.main()
