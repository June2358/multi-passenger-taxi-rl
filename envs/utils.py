"""공통 유틸리티 함수 모듈"""

from typing import List, Tuple, Union

import numpy as np


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
