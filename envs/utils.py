"""공통 유틸리티 함수 모듈"""
import numpy as np
from typing import Tuple, List, Union


def compute_manhattan_distance(pos1: Union[List, np.ndarray], pos2: Union[List, np.ndarray]) -> int:
    """두 좌표 간 맨해튼 거리 계산"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def compute_trip_time(taxi_pos: Union[List, np.ndarray], passenger: Union[List, np.ndarray]) -> Tuple[int, int]:
    """
    승객 픽업을 위한 이동 시간 계산
    
    Args:
        taxi_pos: 택시 위치 [x, y]
        passenger: 승객 정보 [start_x, start_y, dest_x, dest_y, status]
    
    Returns:
        (empty_dist, service_dist): 빈 차 이동 거리, 승객 운송 거리
    """
    empty_dist = compute_manhattan_distance(taxi_pos, passenger[:2])
    service_dist = compute_manhattan_distance(passenger[:2], passenger[2:4])
    return empty_dist, service_dist


def count_remaining_passengers(passengers: List) -> int:
    """대기 중인 승객 수 계산 (status == 0)"""
    return sum(1 for p in passengers if p[4] == 0)


def update_passenger_state(passengers: List, taxi_pos: Union[List, np.ndarray], passenger_idx: int) -> None:
    """
    승객 상태 업데이트 및 택시 위치 이동
    
    Args:
        passengers: 승객 리스트 (in-place 수정)
        taxi_pos: 택시 위치 리스트/배열 (in-place 수정)
        passenger_idx: 완료 처리할 승객 인덱스
    """
    p = passengers[passenger_idx]
    taxi_pos[0] = p[2]
    taxi_pos[1] = p[3]
    passengers[passenger_idx][4] = 1
