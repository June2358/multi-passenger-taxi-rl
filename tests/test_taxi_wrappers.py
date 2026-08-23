import unittest

import gymnasium as gym
import numpy as np
from gymnasium.utils import RecordConstructorArgs

from envs import TaxiDecodedObservationWrapper, TaxiOneHotObservationWrapper


class TaxiObservationWrapperTests(unittest.TestCase):
    def test_wrappers_record_constructor_args_and_can_be_recreated(self):
        for wrapper_class in (
            TaxiOneHotObservationWrapper,
            TaxiDecodedObservationWrapper,
        ):
            with self.subTest(wrapper=wrapper_class.__name__):
                wrapped = wrapper_class(gym.make("Taxi-v4"))
                recreated = None
                try:
                    self.assertIsInstance(wrapped, RecordConstructorArgs)
                    self.assertIsNotNone(wrapped.spec)

                    recreated = gym.make(wrapped.spec)
                    self.assertIsInstance(recreated, wrapper_class)
                    observation, _ = recreated.reset(seed=42)
                    self.assertTrue(recreated.observation_space.contains(observation))
                finally:
                    wrapped.close()
                    if recreated is not None:
                        recreated.close()

    def test_one_hot_wrapper_returns_float32_unit_vector(self):
        env = TaxiOneHotObservationWrapper(gym.make("Taxi-v4"))
        try:
            observation, _ = env.reset(seed=42)
            self.assertEqual(observation.shape, (500,))
            self.assertEqual(observation.dtype, np.float32)
            self.assertEqual(float(observation.sum()), 1.0)
        finally:
            env.close()

    def test_decoded_wrapper_returns_expected_feature_shape(self):
        env = TaxiDecodedObservationWrapper(gym.make("Taxi-v4"))
        try:
            observation, _ = env.reset(seed=42)
            self.assertEqual(observation.shape, (11,))
            self.assertEqual(observation.dtype, np.float32)
            self.assertTrue(env.observation_space.contains(observation))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
