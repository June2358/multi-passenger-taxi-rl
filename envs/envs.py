"""커스텀 환경 구현."""

from __future__ import annotations

from numbers import Integral
from typing import ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.utils import compute_trip_time


class AdvancedTimeAttackTaxiEnv(gym.Env):
    """
    고급 택시 디스패치 환경 (시간 공격형)
    - action: 승객 인덱스 선택 (Discrete num_passengers)
    - observation: 승객별 상대좌표/목적지/상태/거리 벡터 (Box)
    - reward: 남은 승객 수 * 이동시간 (총 완료시간 합의 음수)
    """

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}
    _SCENARIOS = ("random", "cluster", "complex")
    _SCENARIO_MODES = ("mixed", *_SCENARIOS)

    def __init__(
        self,
        num_passengers: int = 20,
        grid_size: int = 100,
        scenario_mode: str = "mixed",
        start_at_center: bool = False,
        scenario_mix: tuple | list | None = None,
    ):
        super().__init__()

        if isinstance(num_passengers, bool) or not isinstance(num_passengers, Integral):
            raise TypeError("num_passengers must be an integer")
        if num_passengers <= 0:
            raise ValueError("num_passengers must be greater than 0")
        if isinstance(grid_size, bool) or not isinstance(grid_size, Integral):
            raise TypeError("grid_size must be an integer")
        if grid_size < 2:
            raise ValueError("grid_size must be at least 2")
        if scenario_mode not in self._SCENARIO_MODES:
            modes = ", ".join(repr(mode) for mode in self._SCENARIO_MODES)
            raise ValueError(f"scenario_mode must be one of: {modes}")
        if not isinstance(start_at_center, (bool, np.bool_)):
            raise TypeError("start_at_center must be a boolean")

        try:
            validated_mix = np.asarray(
                scenario_mix if scenario_mix is not None else [0.3, 0.4, 0.3],
                dtype=float,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("scenario_mix must contain three numeric weights") from exc
        if validated_mix.shape != (3,):
            raise ValueError(
                "scenario_mix must be length-3 array-like for "
                "['random', 'cluster', 'complex']"
            )
        if not np.all(np.isfinite(validated_mix)):
            raise ValueError("scenario_mix weights must be finite")
        mix_sum = float(validated_mix.sum())
        if np.any(validated_mix < 0) or not np.isfinite(mix_sum) or mix_sum <= 0:
            raise ValueError(
                "scenario_mix weights must be non-negative and contain a positive value"
            )

        self.num_passengers = int(num_passengers)
        self.grid_size = int(grid_size)
        self.scenario_mode = scenario_mode
        self.start_at_center = bool(start_at_center)
        self.scenario_mix = validated_mix / mix_sum
        self.action_space = spaces.Discrete(self.num_passengers)

        # 승객 1명당 7개 정보 (정규화된 상대 좌표/목적지/상태/거리)
        # [p_dx, p_dy, dest_dx, dest_dy, status, pickup_dist, service_dist]
        # 좌표와 거리는 고정 스케일 100으로 나눈다.
        max_abs = self.grid_size / 100.0  # 예: grid_size=100 -> 1.0, 200 -> 2.0
        dist_max = 2 * max_abs  # 맨해튼 거리 최대치 정규화 (0~2*grid_size/100)
        passenger_low = np.array(
            [-max_abs, -max_abs, -max_abs, -max_abs, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )
        passenger_high = np.array(
            [max_abs, max_abs, max_abs, max_abs, 1.0, dist_max, dist_max],
            dtype=np.float32,
        )
        low = np.tile(passenger_low, self.num_passengers)
        high = np.tile(passenger_high, self.num_passengers)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._init_scenario()
        return self._get_state(), {
            "scenario": self.current_scenario,
            "num_passengers": self.num_passengers,
            "grid_size": self.grid_size,
        }

    def step(self, action):
        if isinstance(action, bool) or not isinstance(action, (Integral, np.integer)):
            raise TypeError("action must be an integer passenger index")
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(
                f"action must be between 0 and {self.num_passengers - 1}; got {action}"
            )

        target_p = self.passengers[action]
        if target_p[4] != 0:
            raise ValueError(f"passenger {action} has already been served")

        taxi_before = np.asarray(self.taxi_loc, dtype=int).copy()
        passenger_origin = np.asarray(target_p[:2], dtype=int).copy()
        passenger_destination = np.asarray(target_p[2:4], dtype=int).copy()

        dist_empty, dist_service = compute_trip_time(self.taxi_loc, target_p)
        time_elapsed = int(dist_empty + dist_service)

        # 완료시간 비용에는 이번에 운송할 승객까지 포함한다.
        remaining_passengers = self._remaining_passengers
        completion_cost = int(time_elapsed * remaining_passengers)
        reward = -completion_cost / 100.0  # 스케일 다운

        self.taxi_loc = passenger_destination.copy()
        self.passengers[action, 4] = 1
        self._remaining_passengers -= 1
        terminated = self._remaining_passengers == 0
        truncated = False

        info = {
            "selected_passenger": action,
            "passenger_origin": passenger_origin,
            "passenger_destination": passenger_destination,
            "taxi_before": taxi_before,
            "taxi_after": self.taxi_loc.copy(),
            "empty_distance": dist_empty,
            "service_distance": dist_service,
            "trip_time": time_elapsed,
            "remaining_passengers": remaining_passengers,
            "completion_cost": completion_cost,
        }
        return self._get_state(), reward, terminated, truncated, info

    def action_masks(self):
        return self.passengers[:, 4] == 0

    def _init_scenario(self):
        raw_passengers = []
        scenario = self._pick_scenario()
        self.current_scenario = scenario
        sampled_taxi_loc = self.np_random.integers(0, self.grid_size, size=2)
        if scenario == "complex" or self.start_at_center:
            center = self.grid_size // 2
            self.taxi_loc = np.array([center, center], dtype=int)
        else:
            self.taxi_loc = sampled_taxi_loc

        if scenario == "random":
            for _ in range(self.num_passengers):
                loc = self.np_random.integers(0, self.grid_size, size=2)
                dest = self.np_random.integers(0, self.grid_size, size=2)
                raw_passengers.append(list(loc) + list(dest) + [0])

        elif scenario == "cluster":
            centers = [
                np.array([int(self.grid_size * 0.2), int(self.grid_size * 0.2)]),  # Source
                np.array([int(self.grid_size * 0.8), int(self.grid_size * 0.8)]),  # Sink
                np.array([int(self.grid_size * 0.2), int(self.grid_size * 0.8)]),  # Hub
            ]
            noise_range = max(5, int(self.grid_size * 0.05))  # 그리드 크기의 5% 또는 최소 5
            
            for _ in range(self.num_passengers):
                start_node_idx = int(self.np_random.choice([0, 2]))
                start_center = centers[start_node_idx]
                noise = self.np_random.integers(
                    -noise_range, noise_range + 1, size=2
                )
                loc = np.clip(start_center + noise, 0, self.grid_size - 1)

                if start_node_idx == 0:
                    dest_center = (
                        centers[1] if self.np_random.random() < 0.5 else centers[2]
                    )
                else:
                    dest_center = centers[int(self.np_random.choice([0, 1]))]
                dest_noise = self.np_random.integers(
                    -noise_range, noise_range + 1, size=2
                )
                dest = np.clip(dest_center + dest_noise, 0, self.grid_size - 1)
                raw_passengers.append(list(loc) + list(dest) + [0])
        elif scenario == "complex":
            center = self.grid_size // 2

            # 100x100에서의 기존 거리 구성을 비율로 유지하고, 작은
            # 그리드에서도 모든 범위가 항상 유효하도록 클립한다.
            max_coord = self.grid_size - 1
            trap_far_offset = max(1, round(self.grid_size * 0.03))
            trap_near_offset = max(1, round(self.grid_size * 0.01))
            trap_start_min = max(0, center - trap_far_offset)
            trap_start_max = max(0, center - trap_near_offset)
            trap_col_radius = max(1, round(self.grid_size * 0.05))
            trap_col_min = max(0, center - trap_col_radius)
            trap_col_max = min(max_coord, center + trap_col_radius)

            honey_near_offset = max(1, round(self.grid_size * 0.12))
            honey_far_offset = max(1, round(self.grid_size * 0.22))
            honey_zone_start = min(max_coord, center + honey_near_offset)
            honey_zone_end = min(max_coord, center + honey_far_offset)
            north_band_size = min(
                self.grid_size, max(1, round(self.grid_size * 0.05))
            )

            # 승객의 30%는 중앙 근처의 장거리 trap이다.
            num_traps = int(self.num_passengers * 0.3)

            for i in range(self.num_passengers):
                if i < num_traps:
                    # 중앙 근처 pickup, 북쪽 끝의 장거리 destination.
                    r_loc = self.np_random.integers(
                        trap_start_min, trap_start_max + 1
                    )
                    c_loc = self.np_random.integers(
                        trap_col_min, trap_col_max + 1
                    )
                    loc = np.array([r_loc, c_loc])

                    r_dest = self.np_random.integers(0, north_band_size)
                    c_dest = self.np_random.integers(0, self.grid_size)
                    dest = np.array([r_dest, c_dest])

                    raw_passengers.append(list(loc) + list(dest) + [0])

                else:
                    # 남동쪽 pickup, 인접한 단거리 destination.
                    r_loc = self.np_random.integers(
                        honey_zone_start, honey_zone_end + 1
                    )
                    c_loc = self.np_random.integers(
                        honey_zone_start, honey_zone_end + 1
                    )
                    loc = np.array([r_loc, c_loc])

                    move_noise = self.np_random.integers(-1, 2, size=2)
                    dest = np.clip(loc + move_noise, 0, self.grid_size - 1)
                    
                    if np.array_equal(loc, dest):
                        dest[0] = dest[0] - 1 if dest[0] == max_coord else dest[0] + 1

                    raw_passengers.append(list(loc) + list(dest) + [0])

        # 승객 슬롯에 의미가 없도록 매 에피소드에 한 번 순서를 섞는다.
        # 이로써 complex의 trap/honey 유형을 행동 인덱스로 암기할 수 없다.
        order = self.np_random.permutation(self.num_passengers)
        passenger_array = np.asarray(raw_passengers, dtype=int)
        self.passengers = np.ascontiguousarray(passenger_array[order])
        self._remaining_passengers = self.num_passengers

    def _pick_scenario(self) -> str:
        if self.scenario_mode == "mixed":
            return str(self.np_random.choice(self._SCENARIOS, p=self.scenario_mix))
        return self.scenario_mode

    def _get_state(self):
        norm = np.float32(100.0)  # 고정 정규화 스케일
        active = self.passengers[:, 4] == 0
        active_passengers = self.passengers[active]

        # 완료된 승객은 status만 1로 남기고 좌표/거리는 0으로 정리한다.
        state = np.zeros((self.num_passengers, 7), dtype=np.float32)
        state[:, 4] = self.passengers[:, 4]

        taxi_loc = self.taxi_loc.astype(np.float32)
        pickup_delta = active_passengers[:, :2] - taxi_loc
        destination_delta = active_passengers[:, 2:4] - active_passengers[:, :2]

        state[active, :2] = pickup_delta / norm
        state[active, 2:4] = destination_delta / norm
        state[active, 5] = np.abs(pickup_delta).sum(axis=1) / norm
        state[active, 6] = np.abs(destination_delta).sum(axis=1) / norm
        return state.reshape(-1)

