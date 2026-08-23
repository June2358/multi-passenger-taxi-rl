"""AdvancedTaxi distance helpers."""

from collections.abc import Sequence

import numpy as np


def compute_manhattan_distance(
    pos1: Sequence[int] | np.ndarray,
    pos2: Sequence[int] | np.ndarray,
) -> int:
    """Return the Manhattan distance between two coordinates."""
    return int(abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]))


def compute_trip_time(
    taxi_pos: Sequence[int] | np.ndarray,
    passenger: Sequence[int] | np.ndarray,
) -> tuple[int, int]:
    """Return pickup and service distances for one passenger."""
    empty_dist = compute_manhattan_distance(taxi_pos, passenger[:2])
    service_dist = compute_manhattan_distance(passenger[:2], passenger[2:4])
    return empty_dist, service_dist
