"""Short end-to-end smoke test for the native AdvancedTaxi action mask."""

import unittest

import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO

import envs  # noqa: F401


class MaskablePpoSmokeTests(unittest.TestCase):
    def test_short_rollout_uses_environment_action_masks_directly(self):
        env = gym.make(
            "AdvancedTaxi-v1",
            num_passengers=3,
            grid_size=20,
            scenario_mode="complex",
        )
        try:
            model = MaskablePPO(
                "MlpPolicy",
                env,
                n_steps=4,
                batch_size=4,
                n_epochs=1,
                seed=7,
                verbose=0,
            )
            model.learn(8)

            observation, _ = env.reset(seed=11)
            mask = env.unwrapped.action_masks()
            action, _ = model.predict(
                observation,
                action_masks=mask,
                deterministic=True,
            )

            self.assertTrue(mask[int(np.asarray(action).item())])
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
