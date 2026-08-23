"""Taxi-v3 관찰 래퍼."""

import numpy as np
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box


class TaxiOneHotObservationWrapper(ObservationWrapper):
    """
    Taxi-v3의 정수 상태(0~499)를 길이 500의 원핫 벡터(Box)로 변환.
    - 출력: Box(low=0, high=1, shape=(500,), dtype=float32)
    """

    def __init__(self, env):
        super().__init__(env)
        self.n = env.observation_space.n  # 500
        self.observation_space = Box(
            low=np.zeros(self.n, dtype=np.float32),
            high=np.ones(self.n, dtype=np.float32),
            dtype=np.float32,
        )

    def observation(self, obs):
        onehot = np.zeros(self.n, dtype=np.float32)
        onehot[int(obs)] = 1.0
        return onehot


class TaxiDecodedObservationWrapper(ObservationWrapper):
    """
    Taxi-v3의 정수 상태를 decode()하여 11차원의 의미 있는 벡터로 변환하는 Wrapper.
    
    Taxi-v3의 원형(Discrete(500))을 받아서 decode()만 수행 (정규화는 RL-Zoo의 VecNormalize에 위임):
    - 택시 위치: 원본 좌표 그대로 사용 (0~4 범위)
    - 승객 위치: 원핫 벡터 (범주형 데이터)
    - 목적지: 원핫 벡터 (범주형 데이터)
    
    입력: Discrete(500) → decode() → 11차원 벡터
    - 입력: 정수 상태 (Discrete(500), Taxi-v3 원형)
    - 출력: Box(low=[0,0,0,0,0,0,0,0,0,0,0], high=[4,4,1,1,1,1,1,1,1,1,1], shape=(11,), dtype=float32)
    
    출력 벡터 구성:
    [taxi_row, taxi_col,                    # 택시 위치 (2차원, 원본 좌표 0~4)
     passenger_location_onehot[0~4],         # 승객 위치 원핫 (5차원: 0=R, 1=G, 2=Y, 3=B, 4=탑승)
     destination_onehot[0~3]]                # 목적지 원핫 (4차원: 0=R, 1=G, 2=Y, 3=B)
    
    총 11차원: 2 (택시 위치) + 5 (승객 위치) + 4 (목적지)
    
    정규화는 RL-Zoo의 VecNormalize를 사용하여 관리합니다.
    """

    def __init__(self, env):
        super().__init__(env)
        # Taxi-v3의 원형(Discrete(500))을 입력으로 받음
        # 출력은 11차원 벡터 (택시 위치 2 + 승객 위치 원핫 5 + 목적지 원핫 4)
        # low/high: 택시 위치는 0~4, 원핫 벡터는 0~1
        self.observation_space = Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([4.0, 4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            shape=(11,), 
            dtype=np.float32
        )

    def observation(self, obs):
        """
        Taxi-v3의 정수 상태를 11차원 벡터로 디코딩 (정규화 없이 원본 값 반환)
        
        Args:
            obs: 정수 상태 (int, np.integer, Taxi-v3의 원형)
        
        Returns:
            np.array: 11차원 벡터 [taxi_row, taxi_col,
                                  passenger_location_onehot (5차원),
                                  destination_onehot (4차원)]
        """
        # 정수 상태로 변환
        state_int = int(obs)
        
        # 환경의 내장 decode 함수 사용 (원형 그대로 활용)
        # unwrapped를 통해 실제 Taxi-v3 환경에 접근
        taxi_row, taxi_col, passenger_location, destination = self.env.unwrapped.decode(
            state_int
        )

        # 1. 택시 위치: 원본 좌표 그대로 사용 (정규화 없음, 0~4 범위)
        taxi_row_val = float(taxi_row)  # 0~4
        taxi_col_val = float(taxi_col)  # 0~4

        # 2. 승객 위치: 원핫 벡터 (5차원: 0=R, 1=G, 2=Y, 3=B, 4=탑승)
        passenger_location_onehot = np.zeros(5, dtype=np.float32)
        passenger_location_onehot[int(passenger_location)] = 1.0

        # 3. 목적지: 원핫 벡터 (4차원: 0=R, 1=G, 2=Y, 3=B)
        destination_onehot = np.zeros(4, dtype=np.float32)
        destination_onehot[int(destination)] = 1.0

        # 11차원 벡터로 조합
        return np.concatenate([
            [taxi_row_val, taxi_col_val],  # 택시 위치 (2차원, 원본 좌표)
            passenger_location_onehot,      # 승객 위치 원핫 (5차원)
            destination_onehot,            # 목적지 원핫 (4차원)
        ], dtype=np.float32)

