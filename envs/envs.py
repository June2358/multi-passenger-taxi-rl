"""커스텀 환경 구현."""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from envs.utils import (
    compute_trip_time,
    count_remaining_passengers,
    compute_manhattan_distance,
)


class AdvancedTimeAttackTaxiEnv(gym.Env):
    """
    고급 택시 디스패치 환경 (시간 공격형)
    - action: 승객 인덱스 선택 (Discrete num_passengers)
    - observation: 승객별 상대좌표/목적지/상태/거리 벡터 (Box)
    - reward: 대기 인원수 * 이동시간 (음수, 작을수록 좋음)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        num_passengers: int = 20,
        grid_size: int = 100,
        scenario_mode: str = "mixed",  # mixed | random | cluster | complex
        start_at_center: bool = False,  # 검증 시 중앙 고정 용도
        scenario_mix: tuple | list | None = None,  # mixed 모드에서 사용할 분포
        mix_noise_std: float = 0.0,  # mixed 분포에 가우시안 노이즈(학습용 완화)
        mix_epsilon: float = 1e-3,  # 분포 하한 (0 방지)
    ):
        super().__init__()
        self.num_passengers = num_passengers
        self.grid_size = grid_size
        self.scenario_mode = scenario_mode
        self.start_at_center = start_at_center
        self.scenario_mix = (
            np.array(scenario_mix, dtype=float)
            if scenario_mix is not None
            else np.array([0.3, 0.4, 0.3], dtype=float)
        )
        self.mix_noise_std = float(mix_noise_std)
        self.mix_epsilon = float(mix_epsilon)
        self.action_space = spaces.Discrete(num_passengers)

        # 승객 1명당 6개 정보 (정규화된 상대 좌표/목적지/상태/거리)
        # [p_dx, p_dy, dest_dx, dest_dy, status, dist]
        # 좌표는 고정 스케일 100으로 나눔 (세부 표준화는 VecNormalize가 담당)
        max_abs = grid_size / 100.0  # 예: grid_size=100 -> 1.0, 200 -> 2.0
        dist_max = 2 * max_abs  # 맨해튼 거리 최대치 정규화 (0~2*grid_size/100)
        passenger_low = np.array(
            [-max_abs, -max_abs, -max_abs, -max_abs, 0.0, 0.0], dtype=np.float32
        )
        passenger_high = np.array(
            [max_abs, max_abs, max_abs, max_abs, 2.0, dist_max], dtype=np.float32
        )
        low = np.tile(passenger_low, num_passengers)
        high = np.tile(passenger_high, num_passengers)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    # -------------------------
    # Gym API
    # -------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        self._init_scenario()
        return self._get_state(), {}

    def step(self, action):
        target_p = self.passengers[action]

        # 거리(시간) 계산 - 유틸리티 함수 사용
        dist_empty, dist_service = compute_trip_time(self.taxi_loc, target_p)
        time_elapsed = dist_empty + dist_service

        # remaining_passengers 계산 - 유틸리티 함수 사용
        remaining_passengers = count_remaining_passengers(self.passengers)
        wait_cost = time_elapsed * remaining_passengers
        reward = -wait_cost / 100.0  # 스케일 다운

        # 상태 업데이트
        self.taxi_loc = np.array([target_p[2], target_p[3]])
        self.passengers[action][0] = 0
        self.passengers[action][1] = 0
        self.passengers[action][2] = 0
        self.passengers[action][3] = 0
        self.passengers[action][4] = 1
        terminated = all(p[4] == 1 for p in self.passengers)
        truncated = False

        return self._get_state(), reward, terminated, truncated, {}

    # -------------------------
    # Helpers
    # -------------------------
    def action_masks(self):
        return [p[4] == 0 for p in self.passengers]

    def _init_scenario(self):
        raw_passengers = []
        scenario = self._pick_scenario()
        if self.start_at_center:
            center = self.grid_size // 2
            self.taxi_loc = np.array([center, center], dtype=int)
        else:
            self.taxi_loc = np.random.randint(0, self.grid_size, size=2)

        if scenario == "random":
            for _ in range(self.num_passengers):
                loc = np.random.randint(0, self.grid_size, size=2)
                dest = np.random.randint(0, self.grid_size, size=2)
                raw_passengers.append(list(loc) + list(dest) + [0])

        elif scenario == "cluster":
            # 비율 기반 클러스터 중심 정의
            centers = [
                np.array([int(self.grid_size * 0.2), int(self.grid_size * 0.2)]),  # Source
                np.array([int(self.grid_size * 0.8), int(self.grid_size * 0.8)]),  # Sink
                np.array([int(self.grid_size * 0.2), int(self.grid_size * 0.8)]),  # Hub
            ]
            noise_range = max(5, int(self.grid_size * 0.05))  # 그리드 크기의 5% 또는 최소 5
            
            for _ in range(self.num_passengers):
                start_node_idx = np.random.choice([0, 2])
                start_center = centers[start_node_idx]
                noise = np.random.randint(-noise_range, noise_range + 1, size=2)
                loc = np.clip(start_center + noise, 0, self.grid_size - 1)

                if start_node_idx == 0:
                    dest_center = centers[1] if np.random.rand() < 0.5 else centers[2]
                else:
                    dest_center = centers[np.random.choice([0, 1])]
                dest_noise = np.random.randint(-noise_range, noise_range + 1, size=2)
                dest = np.clip(dest_center + dest_noise, 0, self.grid_size - 1)
                raw_passengers.append(list(loc) + list(dest) + [0])
        elif scenario == "complex":
            center = self.grid_size // 2
            
            # [설정값 조정]
            # 1. 함정 구역: 중앙에서 아주 살짝 북쪽 (유혹 거리: 2~3칸)
            trap_start_min = center - 3
            trap_start_max = center - 1
            
            # 2. 꿀통 구역: 중앙에서 남쪽으로 약간 거리 둠 (무시 거리: 10~15칸)
            # 그리디가 보기엔 당장 멀어 보임
            honey_zone_start = center + 12 
            honey_zone_end = center + 22

            # 3. 함정의 개수 (적당히 섞어서 반복적으로 괴롭힘)
            num_traps = int(self.num_passengers * 0.3)  # 30%는 함정

            for i in range(self.num_passengers):
                if i < num_traps:
                    # [함정 승객: 악마의 셔틀]
                    # 위치: 중앙 바로 위 (Greedy가 1순위 포착)
                    r_loc = np.random.randint(trap_start_min, trap_start_max + 1)
                    c_loc = np.random.randint(center - 5, center + 6)  # 가로로는 중앙 부근
                    loc = np.array([r_loc, c_loc])

                    # 목적지: 맵의 최북단 끝 (지옥행)
                    # 0행 근처로 강제 이동
                    r_dest = np.random.randint(0, 5) 
                    c_dest = np.random.randint(0, self.grid_size)  # 가로로는 아무데나
                    dest = np.array([r_dest, c_dest])

                    raw_passengers.append(list(loc) + list(dest) + [0])

                else:
                    # [꿀통 승객: 남쪽의 천국]
                    # 위치: 함정보다는 멀지만 일단 도착하면 노다지
                    r_loc = np.random.randint(honey_zone_start, honey_zone_end + 1)
                    c_loc = np.random.randint(honey_zone_start, honey_zone_end + 1)  # 좁은 구역에 밀집
                    loc = np.array([r_loc, c_loc])

                    # 목적지: 아주 짧은 거리 (거의 안 움직임)
                    # 바로 옆칸 수준 (Walking distance)
                    move_noise = np.random.randint(-1, 2, size=2)  # -1, 0, 1 칸 이동
                    dest = np.clip(loc + move_noise, 0, self.grid_size - 1)
                    
                    # 출발지와 도착지가 같으면 강제로 한 칸 옆으로
                    if np.array_equal(loc, dest):
                        dest[0] = min(dest[0] + 1, self.grid_size - 1)

                    raw_passengers.append(list(loc) + list(dest) + [0])

        else:
            raise ValueError(f"Unknown scenario_mode: {scenario}")

        self.passengers = raw_passengers

    def _pick_scenario(self) -> str:
        if self.scenario_mode == "mixed":
            base_p = self.scenario_mix
            if base_p.shape != (3,):
                raise ValueError("scenario_mix must be length-3 array-like for ['random','cluster','complex']")

            if self.mix_noise_std > 0:
                noise = np.random.normal(0.0, self.mix_noise_std, size=base_p.shape)
                p = base_p + noise
            else:
                p = base_p

            p = np.clip(p, self.mix_epsilon, None)
            p = p / p.sum()

            return np.random.choice(["random", "cluster", "complex"], p=p)
        return self.scenario_mode

    def _get_state(self):
        taxi_loc_arr = np.array(self.taxi_loc, dtype=np.float32)
        norm = 100.0  # 고정 정규화 스케일

        state = []
        for p in self.passengers:
            status = float(p[4])

            if status == 1.0:
                # 완료된 승객은 좌표/거리 정보를 0으로 정리해 관찰 노이즈를 제거
                state.extend([0.0, 0.0, 0.0, 0.0, status, 0.0])
                continue

            # 택시 -> 승객 (정규화)
            rel_x = float(p[0] - taxi_loc_arr[0]) / norm
            rel_y = float(p[1] - taxi_loc_arr[1]) / norm

            # 승객 -> 목적지 (정규화)
            dest_dx = float(p[2] - p[0]) / norm
            dest_dy = float(p[3] - p[1]) / norm

            # 택시-승객 맨해튼 거리 (정규화)
            dist = float(compute_manhattan_distance(taxi_loc_arr, p[:2])) / norm

            state.extend([rel_x, rel_y, dest_dx, dest_dy, status, dist])

        return np.array(state, dtype=np.float32)

