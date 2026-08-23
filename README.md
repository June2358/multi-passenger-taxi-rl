# Taxi RL 실험 프로젝트

Gymnasium 커스텀 환경 `AdvancedTaxi-v1`과 표준 `Taxi-v3`에서 MaskablePPO, DQN, Tabular Q-Learning을 비교하는 프로젝트입니다.

> `AdvancedTaxi-v1`은 환경 난수·승객 슬롯 누출·평가 설정 문제를 수정한 새 계약입니다. 기존 `AdvancedTaxi-v0` 모델과 로그는 새 환경의 성능 근거로 재사용할 수 없으며, 반드시 다시 학습해야 합니다.

## 목적함수

`AdvancedTaxi-v1`은 승객을 한 명 선택할 때 다음 비용을 부과합니다.

```text
step_cost = 현재 남은 승객 수 × (빈차 이동거리 + 승객 운행거리)
reward = -step_cost / 100
```

에피소드 비용의 합은 픽업 대기시간이 아니라 **모든 승객의 완료시간 합**(sum of passenger completion times, total flow time)입니다. 서비스 거리의 합은 순서와 무관하므로 픽업 대기시간을 최소화하는 순서와 최적 순서는 같지만, 절대 지표의 정확한 명칭은 “총 완료시간 합”입니다.

## 환경 계약

- 행동: `Discrete(num_passengers)`, 아직 처리하지 않은 승객 인덱스 선택
- 관찰: 승객당 `[상대 x, 상대 y, 목적지 dx, 목적지 dy, 처리 상태, 빈차 거리]` 6개 특징
- 시나리오: `random`, `cluster`, `complex`, 또는 이들을 섞는 `mixed`
- 기본값: 승객 20명, 100×100 그리드, 랜덤 택시 시작점
- 모든 난수는 Gymnasium의 환경별 `self.np_random`을 사용
- 모든 시나리오에서 승객 슬롯을 에피소드마다 섞어 `complex`의 trap/honey 유형이 고정 인덱스로 노출되지 않음
- 환경 자체의 `action_masks()`를 사용하므로 별도 ActionMasker 호환 래퍼가 필요하지 않음
- 음수·범위 밖·이미 처리한 행동은 환경이 즉시 거부
- 처리된 승객의 원본 좌표는 보존하고 관찰에서만 완료 특징을 0으로 정리
- 같은 seed의 중앙 출발 OOD 평가는 승객 집합을 그대로 유지하고 택시 시작점만 변경
- 에피소드 길이는 승객 수와 같으므로 별도의 고정 `TimeLimit`을 두지 않음

`step()`의 `info["completion_cost"]`에는 단계별 완료시간 비용이 제공됩니다. 평가기는 비용을 다시 계산하지 않고 이 값을 단일 기준으로 사용합니다.

## 설치와 테스트

```bash
pixi install
pixi run test
```

`test` task는 전체 `unittest` 회귀 테스트를 실행합니다.

## AdvancedTaxi-v1 학습

누출이 제거된 환경에서는 과거 tuned 설정을 폐기했습니다. 기본 모드는 현재의 보수적인 baseline 설정인 `hyperparameters/advanced_taxi/maskableppo_base.yml`입니다.

```bash
pixi run python train_advanced_taxi_maskableppo.py --n-timesteps 1000000 --seed 42 -P
```

Optuna 탐색:

```bash
pixi run python train_advanced_taxi_maskableppo.py -optimize --n-trials 100 --n-timesteps 200000 --study-name adv_taxi_v1_maskableppo --storage sqlite:///logs/adv_taxi_v1_optuna.db --n-jobs 1
```

환경 오버라이드:

```bash
pixi run python train_advanced_taxi_maskableppo.py --n-timesteps 1000000 --env-kwargs num_passengers:30 grid_size:120 scenario_mode:mixed
```

학습 환경의 승객 수·그리드·시나리오 설정은 평가 환경에도 전달됩니다. 평가 전용 값은 `--eval-env-kwargs`로 다시 지정할 수 있으며, 기본적으로 평가 시 분포 노이즈만 `0.0`으로 고정됩니다.

```bash
pixi run python train_advanced_taxi_maskableppo.py --env-kwargs num_passengers:30 --eval-env-kwargs start_at_center:True
```

- 기본 seed는 `42`입니다.
- `--param-mode base`가 기본입니다.
- 누출 환경에서 얻은 과거 tuned 모드는 제거했으며 현재는 `base`만 허용합니다.
- 실행 경로와 평가 대상을 모호하게 만드는 `--uuid` 및 학습용 `--exp-id`는 지원하지 않습니다.
- 사용자 `--conf-file`을 지정하면 자동 설정 선택 대신 그 파일만 사용합니다.

## Taxi-v3 기준선

DQN baseline:

```bash
pixi run python train_rlzoo.py --algo dqn --env Taxi-v3 --param-mode base --n-timesteps 500000 --seed 42 -P
```

`dqn_base.yml`은 현재 고정된 SB3 2.7.1의 기본 `learning_starts=100`, `[64, 64]` 네트워크를 사용합니다.

Tabular Q-Learning baseline:

```bash
pixi run python train_rlzoo.py --algo qlearning --env Taxi-v3 --param-mode base --n-timesteps 500000 --seed 42
```

직접 실행하면 `qlearning.yml`을 사용합니다.

```bash
pixi run python qlearning_train.py --n-timesteps 500000 --seed 42
```

Q-Learning은 termination에서만 bootstrap을 중단하고 truncation에서는 계속 bootstrap합니다. 탐욕 행동의 Q값이 같을 때는 무작위로 동률을 해소하며, 최종 평가는 학습과 독립적으로 파생한 seed와 RNG를 사용합니다. `seed`를 생략하면 비결정적으로 실행됩니다.

## AdvancedTaxi-v1 평가

평가할 실행 ID를 반드시 지정합니다.

```bash
pixi run python evaluate_model.py --exp-id 1
```

평가기의 기본 계약은 다음과 같습니다.

- 정확히 `logs/maskableppo/AdvancedTaxi-v1_<exp-id>/AdvancedTaxi-v1.zip` 최종 모델을 사용
- `config.yml`, `args.yml`, `vecnormalize.pkl`은 실행 폴더 아래의 `AdvancedTaxi-v1/` 디렉터리에서만 로드
- 정규화 실행인데 통계가 없으면 원시 관찰로 계속하지 않고 즉시 실패
- 저장된 전체 `eval_env_kwargs`를 복원하며, 누락된 과거 형식은 현재 기본값으로 추측하지 않고 재학습을 요구
- Random, nearest-pickup Greedy, RL에 동일한 seed의 동일 승객 인스턴스를 사용
- Random 정책 RNG는 문제 생성 RNG와 분리
- 기본 테스트 seed는 학습 기본 seed와 분리된 `10000~10099`
- 결과 비용은 환경의 `info["completion_cost"]`만 합산

중앙 출발 OOD 비교:

```bash
pixi run python evaluate_model.py --exp-id 1 --center-start
```

seed 범위와 평가 개수 변경:

```bash
pixi run python evaluate_model.py --exp-id 1 --seed-start 20000 --num-seeds 200
```

경로 시각화:

```bash
pixi run python evaluate_model.py --exp-id 1 --visualize --visualize-seed 10000
```

PNG는 기본적으로 `visualizations/`에 저장됩니다.

## 출력 구조

```text
logs/
└── maskableppo/
    └── AdvancedTaxi-v1_1/
        ├── AdvancedTaxi-v1.zip
        ├── best_model.zip
        ├── evaluations.npz
        └── AdvancedTaxi-v1/
            ├── args.yml
            ├── config.yml
            └── vecnormalize.pkl
```

평가기에서는 최종 모델만 지원합니다. `best_model.zip`은 저장 시점과 동일한 VecNormalize 통계를 별도로 보존하는 구조가 마련되기 전까지 선택하지 않습니다.

TensorBoard:

```bash
pixi run tensorboard --logdir logs/
```

## 현재 해석상의 한계

- `complex`의 trap은 실제 수요 데이터가 아니라 의도적으로 만든 합성 스트레스 시나리오입니다.
- 에피소드별 shuffle은 고정 슬롯 누출을 제거하지만, 현재 평면 MLP 자체가 순열 등가적인 set policy라는 뜻은 아닙니다.
- 현재 평가 기준선은 Random과 nearest-pickup Greedy입니다. 최적성 주장을 하려면 작은 문제의 exact DP, beam/local search 등 더 강한 조합최적화 기준선이 필요합니다.
- 정량 결과는 학습 seed 하나로 결론 내리지 말고 여러 독립 학습 실행의 평균과 분산을 함께 보고해야 합니다.
