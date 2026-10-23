"""AdvancedTaxi Gymnasium 환경 패키지."""

from gymnasium.envs.registration import register

# AdvancedTimeAttackTaxiEnv 등록
register(
    id="AdvancedTaxi-v2",
    entry_point="envs.envs:AdvancedTimeAttackTaxiEnv",
)

from envs.envs import AdvancedTimeAttackTaxiEnv

__all__ = ["AdvancedTimeAttackTaxiEnv"]

