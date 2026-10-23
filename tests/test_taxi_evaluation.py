"""공유 Taxi-v4 평가기의 외부 계약 테스트."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import DQN

from evaluate_taxi import evaluate_dqn

ROOT = Path(__file__).resolve().parents[1]


def _optimal_qtable() -> np.ndarray:
    env = gym.make("Taxi-v4")
    transitions = env.unwrapped.P
    values = np.zeros(500, dtype=np.float64)

    try:
        for _ in range(1_000):
            q_values = np.zeros((500, 6), dtype=np.float64)
            for state in range(500):
                for action in range(6):
                    q_values[state, action] = sum(
                        probability
                        * (
                            reward
                            if terminated
                            else reward + 0.99 * values[next_state]
                        )
                        for probability, next_state, reward, terminated
                        in transitions[state][action]
                    )
            next_values = np.max(q_values, axis=1)
            if np.max(np.abs(next_values - values)) < 1e-10:
                break
            values = next_values
    finally:
        env.close()

    # 알고리즘 간 정확한 비교를 위해 최적 행동의 동점을 결정적으로 해소한다.
    q_values -= np.arange(6, dtype=np.float64) * 1e-4
    return q_values.astype(np.float32)


class TaxiEvaluationContractTests(unittest.TestCase):
    def test_cli_compares_qlearning_and_dqn_on_identical_seeds(self):
        q_table = _optimal_qtable()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            qlearning_path = temp_root / "qlearning.npy"
            dqn_path = temp_root / "dqn.zip"
            np.save(qlearning_path, q_table)

            env = gym.make("Taxi-v4")
            model = DQN(
                "MlpPolicy",
                env,
                policy_kwargs={"net_arch": []},
                buffer_size=10,
                learning_starts=1,
                seed=3,
                device="cpu",
                verbose=0,
            )
            with torch.no_grad():
                for network in (model.q_net, model.q_net_target):
                    linear = network.q_net[0]
                    linear.weight.copy_(torch.as_tensor(q_table.T))
                    linear.bias.zero_()
            model.save(dqn_path)
            model.get_env().close()

            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "evaluate_taxi.py"),
                    "--qlearning-model",
                    str(qlearning_path),
                    "--dqn-model",
                    str(dqn_path),
                    "--seed-start",
                    "100",
                    "--num-seeds",
                    "8",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            payload = json.loads(completed.stdout)
            qlearning = payload["results"]["qlearning"]
            dqn = payload["results"]["dqn"]
            self.assertEqual(payload["seed_start"], 100)
            self.assertEqual(payload["num_seeds"], 8)
            self.assertEqual(qlearning["successes"], 8)
            self.assertEqual(dqn["successes"], 8)
            self.assertEqual(qlearning["mean_return"], dqn["mean_return"])
            self.assertEqual(qlearning["mean_length"], dqn["mean_length"])

    def test_dqn_with_non_taxi_spaces_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "wrong-spaces.zip"
            env = gym.make("FrozenLake-v1")
            model = DQN(
                "MlpPolicy",
                env,
                buffer_size=10,
                learning_starts=1,
                seed=3,
                device="cpu",
                verbose=0,
            )
            model.save(model_path)
            model.get_env().close()

            with self.assertRaisesRegex(ValueError, "do not match Taxi-v4"):
                evaluate_dqn(model_path, [100])


if __name__ == "__main__":
    unittest.main()
