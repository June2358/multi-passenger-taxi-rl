"""Regression tests for the AdvancedTimeAttackTaxiEnv contract."""

from __future__ import annotations

import unittest

import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env

import envs  # noqa: F401
from envs.envs import AdvancedTimeAttackTaxiEnv
from envs.utils import compute_trip_time


def passenger_snapshot(env: AdvancedTimeAttackTaxiEnv) -> np.ndarray:
    return np.asarray(env.passengers, dtype=int)


class AdvancedTaxiConstructionTests(unittest.TestCase):
    def test_gymnasium_environment_contract(self):
        env = gym.make("AdvancedTaxi-v1")
        try:
            check_env(env.unwrapped)
        finally:
            env.close()

    def test_registered_environment_has_no_conflicting_time_limit(self):
        env = gym.make("AdvancedTaxi-v1", num_passengers=1_001)
        try:
            self.assertIsNone(env.spec.max_episode_steps)
        finally:
            env.close()

    def test_rejects_invalid_constructor_values(self):
        cases = (
            ({"num_passengers": 0}, ValueError),
            ({"num_passengers": 2.5}, TypeError),
            ({"grid_size": 1}, ValueError),
            ({"grid_size": True}, TypeError),
            ({"scenario_mode": "unknown"}, ValueError),
            ({"start_at_center": 1}, TypeError),
            ({"scenario_mix": [0.5, 0.5]}, ValueError),
            ({"scenario_mix": [0.5, np.nan, 0.5]}, ValueError),
            ({"scenario_mix": [0.0, 0.0, 0.0]}, ValueError),
            ({"mix_noise_std": -0.1}, ValueError),
            ({"mix_epsilon": 0.0}, ValueError),
        )

        for kwargs, error_type in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(error_type):
                    AdvancedTimeAttackTaxiEnv(**kwargs)


class AdvancedTaxiRandomnessTests(unittest.TestCase):
    def test_same_seed_reproduces_every_scenario(self):
        for scenario_mode in ("random", "cluster", "complex", "mixed"):
            with self.subTest(scenario_mode=scenario_mode):
                first = AdvancedTimeAttackTaxiEnv(
                    num_passengers=20,
                    scenario_mode=scenario_mode,
                    mix_noise_std=0.2,
                )
                second = AdvancedTimeAttackTaxiEnv(
                    num_passengers=20,
                    scenario_mode=scenario_mode,
                    mix_noise_std=0.2,
                )

                first_obs, _ = first.reset(seed=2026)
                second_obs, _ = second.reset(seed=2026)

                np.testing.assert_array_equal(first.taxi_loc, second.taxi_loc)
                np.testing.assert_array_equal(
                    passenger_snapshot(first), passenger_snapshot(second)
                )
                np.testing.assert_array_equal(first_obs, second_obs)

    def test_reset_reports_the_generated_scenario(self):
        env = AdvancedTimeAttackTaxiEnv(
            num_passengers=3,
            grid_size=12,
            scenario_mode="cluster",
        )

        _, info = env.reset(seed=7)

        self.assertEqual(
            info,
            {"scenario": "cluster", "num_passengers": 3, "grid_size": 12},
        )

    def test_environment_rng_is_independent_of_other_rngs(self):
        first = AdvancedTimeAttackTaxiEnv(scenario_mode="mixed", mix_noise_std=0.3)
        second = AdvancedTimeAttackTaxiEnv(scenario_mode="mixed", mix_noise_std=0.3)
        intruder = AdvancedTimeAttackTaxiEnv(scenario_mode="mixed", mix_noise_std=0.3)

        first.reset(seed=91)
        second.reset(seed=91)
        first_obs, _ = first.reset()
        first_passengers = passenger_snapshot(first)
        first_taxi = first.taxi_loc.copy()

        np.random.seed(777)
        np.random.random(1_000)
        intruder.reset(seed=12345)
        intruder.reset()

        second_obs, _ = second.reset()
        np.testing.assert_array_equal(first_taxi, second.taxi_loc)
        np.testing.assert_array_equal(first_passengers, passenger_snapshot(second))
        np.testing.assert_array_equal(first_obs, second_obs)

    def test_center_start_changes_only_the_taxi_start_for_a_seed(self):
        for scenario_mode in ("random", "cluster", "complex"):
            with self.subTest(scenario_mode=scenario_mode):
                random_start = AdvancedTimeAttackTaxiEnv(
                    scenario_mode=scenario_mode,
                    start_at_center=False,
                )
                center_start = AdvancedTimeAttackTaxiEnv(
                    scenario_mode=scenario_mode,
                    start_at_center=True,
                )

                random_start.reset(seed=17)
                center_start.reset(seed=17)

                np.testing.assert_array_equal(
                    passenger_snapshot(random_start),
                    passenger_snapshot(center_start),
                )
                np.testing.assert_array_equal(
                    center_start.taxi_loc,
                    [center_start.grid_size // 2] * 2,
                )

    def test_complex_passenger_types_are_not_fixed_to_action_slots(self):
        trap_slot_sets = set()

        for seed in range(10):
            env = AdvancedTimeAttackTaxiEnv(
                num_passengers=20,
                grid_size=100,
                scenario_mode="complex",
                start_at_center=True,
            )
            env.reset(seed=seed)
            trap_slots = tuple(
                index
                for index, passenger in enumerate(env.passengers)
                if compute_trip_time(passenger[:2], passenger)[1] > 2
            )
            self.assertEqual(len(trap_slots), 6)
            trap_slot_sets.add(trap_slots)

        self.assertGreater(len(trap_slot_sets), 1)
        self.assertNotIn(tuple(range(6)), trap_slot_sets)


class AdvancedTaxiScenarioTests(unittest.TestCase):
    def test_all_scenarios_stay_in_bounds_on_small_and_large_grids(self):
        for grid_size in (2, 3, 7, 37, 100, 200):
            for scenario_mode in ("random", "cluster", "complex"):
                with self.subTest(
                    grid_size=grid_size, scenario_mode=scenario_mode
                ):
                    env = AdvancedTimeAttackTaxiEnv(
                        num_passengers=20,
                        grid_size=grid_size,
                        scenario_mode=scenario_mode,
                    )
                    observation, _ = env.reset(seed=42)
                    passengers = passenger_snapshot(env)

                    self.assertTrue(np.all(passengers[:, :4] >= 0))
                    self.assertTrue(np.all(passengers[:, :4] < grid_size))
                    self.assertTrue(np.all(env.taxi_loc >= 0))
                    self.assertTrue(np.all(env.taxi_loc < grid_size))
                    self.assertTrue(env.observation_space.contains(observation))


class AdvancedTaxiStepTests(unittest.TestCase):
    def setUp(self):
        self.env = AdvancedTimeAttackTaxiEnv(
            num_passengers=2,
            grid_size=10,
            scenario_mode="random",
        )
        self.env.reset(seed=1)

    def test_rejects_negative_out_of_range_and_served_actions(self):
        original_passengers = passenger_snapshot(self.env).copy()
        original_taxi = self.env.taxi_loc.copy()

        for action in (-1, 2):
            with self.subTest(action=action):
                with self.assertRaises(ValueError):
                    self.env.step(action)

        np.testing.assert_array_equal(original_passengers, passenger_snapshot(self.env))
        np.testing.assert_array_equal(original_taxi, self.env.taxi_loc)

        self.env.step(0)
        served_state = passenger_snapshot(self.env).copy()
        served_taxi = self.env.taxi_loc.copy()
        self.assertIsInstance(self.env.action_masks(), np.ndarray)
        self.assertEqual(self.env.action_masks().dtype, np.bool_)
        with self.assertRaisesRegex(ValueError, "already been served"):
            self.env.step(0)
        np.testing.assert_array_equal(served_state, passenger_snapshot(self.env))
        np.testing.assert_array_equal(served_taxi, self.env.taxi_loc)

    def test_reward_and_info_are_the_authoritative_transition_cost(self):
        self.env.taxi_loc = np.array([0, 0], dtype=int)
        self.env.passengers = [
            [1, 1, 3, 1, 0],
            [4, 4, 4, 5, 0],
        ]

        _, reward, terminated, truncated, info = self.env.step(0)

        self.assertAlmostEqual(reward, -0.08)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(
            {
                "selected_passenger": info["selected_passenger"],
                "empty_distance": info["empty_distance"],
                "service_distance": info["service_distance"],
                "trip_time": info["trip_time"],
                "remaining_passengers": info["remaining_passengers"],
                "completion_cost": info["completion_cost"],
            },
            {
                "selected_passenger": 0,
                "empty_distance": 2,
                "service_distance": 2,
                "trip_time": 4,
                "remaining_passengers": 2,
                "completion_cost": 8,
            },
        )
        np.testing.assert_array_equal(info["taxi_before"], [0, 0])
        np.testing.assert_array_equal(info["taxi_after"], [3, 1])
        np.testing.assert_array_equal(info["passenger_origin"], [1, 1])
        np.testing.assert_array_equal(info["passenger_destination"], [3, 1])
        self.assertAlmostEqual(reward, -info["completion_cost"] / 100.0)
        np.testing.assert_array_equal(self.env.passengers[0][:4], [1, 1, 3, 1])

        info["taxi_after"][0] = 99
        info["passenger_destination"][0] = 99
        np.testing.assert_array_equal(self.env.taxi_loc, [3, 1])

    def test_permuting_slots_and_remapping_actions_preserves_cost(self):
        passengers = [
            [1, 1, 3, 1, 0],
            [4, 4, 4, 5, 0],
            [2, 3, 0, 3, 0],
        ]
        physical_order = [2, 0, 1]
        permutation = [1, 2, 0]
        remapped_order = [permutation.index(index) for index in physical_order]

        def rollout(passenger_rows, actions):
            env = AdvancedTimeAttackTaxiEnv(num_passengers=3, grid_size=10)
            env.reset(seed=1)
            env.taxi_loc = np.array([0, 0], dtype=int)
            env.passengers = [list(row) for row in passenger_rows]
            return sum(env.step(action)[4]["completion_cost"] for action in actions)

        original_cost = rollout(passengers, physical_order)
        permuted_cost = rollout(
            [passengers[index] for index in permutation],
            remapped_order,
        )

        self.assertEqual(original_cost, permuted_cost)


if __name__ == "__main__":
    unittest.main()
