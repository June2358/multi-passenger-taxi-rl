"""
통합 환경 패키지

- Taxi-v3 wrapper (원핫/decoded 관찰)
- AdvancedTimeAttackTaxiEnv (커스텀 환경)
"""

from gymnasium.envs.registration import register

# Taxi-v3는 표준 Gymnasium 환경이므로 별도 등록 불필요
# Wrapper만 제공: envs.wrappers.TaxiOneHotObservationWrapper

# AdvancedTimeAttackTaxiEnv 등록
register(
    id="AdvancedTaxi-v1",
    entry_point="envs.envs:AdvancedTimeAttackTaxiEnv",
)

# Wrapper와 환경 클래스 export
from envs.wrappers import (
    TaxiOneHotObservationWrapper,
    TaxiDecodedObservationWrapper,
)
from envs.envs import AdvancedTimeAttackTaxiEnv

__all__ = [
    "TaxiOneHotObservationWrapper",
    "TaxiDecodedObservationWrapper",
    "AdvancedTimeAttackTaxiEnv",
]

