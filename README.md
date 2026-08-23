# Taxi RL 실험 프로젝트

Gymnasium 커스텀 환경 `AdvancedTaxi-v2`와 표준 `Taxi-v4`에서 MaskablePPO, DQN, Tabular Q-Learning을 비교하는 프로젝트입니다.

> `AdvancedTaxi-v2`는 환경 난수·승객 슬롯 누출·평가 설정을 수정하고, `complex` 중앙 시작과 `service_distance` 관찰을 명시한 새 계약입니다. 이전 버전으로 학습한 모델과 로그는 성능 근거로 재사용하지 않으며, 현재 환경에서 다시 학습합니다. 오래된 모델과 `VecNormalize` 통계의 호환성도 지원하지 않습니다.

## 목적함수

`AdvancedTaxi-v2`는 승객을 한 명 선택할 때 다음 비용을 부과합니다.

```text
step_cost = 현재 남은 승객 수 × (빈차 이동거리 + 승객 운행거리)
reward = -step_cost / 100
```

에피소드 비용의 합은 픽업 대기시간이 아니라 **모든 승객의 완료시간 합**(sum of passenger completion times, total flow time)입니다. 서비스 거리의 합은 순서와 무관하므로 픽업 대기시간을 최소화하는 순서와 최적 순서는 같지만, 절대 지표의 정확한 명칭은 “총 완료시간 합”입니다.

## 환경 계약

- 행동: `Discrete(num_passengers)`, 아직 처리하지 않은 승객 인덱스 선택
- 관찰: 승객당 `[상대 x, 상대 y, 목적지 dx, 목적지 dy, 처리 상태, 빈차 거리, 서비스 거리]` 7개 특징
- 서비스 거리는 목적지 상대좌표에서 계산되는 맨해튼 거리이며 새로운 정보를 누출하지 않음
- 시나리오: `random`, `cluster`, `complex`, 또는 이들을 섞는 `mixed`
- 기본값: 승객 20명, 100×100 그리드
- `random`과 `cluster`는 기본적으로 랜덤 위치에서 시작하고, 의도적인 낚시 상황인 `complex`는 항상 중앙에서 시작
- `scenario_mix`의 0 가중치는 해당 시나리오를 정확히 제외하며, episode별 확률 노이즈나 숨은 확률 하한을 적용하지 않음
- 모든 난수는 Gymnasium의 환경별 `self.np_random`을 사용
- 모든 시나리오에서 승객 슬롯을 에피소드마다 섞어 `complex`의 trap/honey 유형이 고정 인덱스로 노출되지 않음
- 환경 자체의 `action_masks()`를 사용하므로 별도 ActionMasker 호환 래퍼가 필요하지 않음
- 음수·범위 밖·이미 처리한 행동은 환경이 즉시 거부
- 처리된 승객의 원본 좌표는 보존하고 관찰에서만 완료 특징을 0으로 정리
- 같은 seed의 중앙 출발 OOD 평가는 `random`과 `cluster`의 승객 집합을 유지한 채 택시 시작점만 변경하며, `complex`는 원래부터 중앙 시작
- 에피소드 길이는 승객 수와 같으므로 별도의 고정 `TimeLimit`을 두지 않음

`step()`의 `info["completion_cost"]`에는 단계별 완료시간 비용이 제공됩니다. 평가기는 비용을 다시 계산하지 않고 이 값을 단일 기준으로 사용합니다.

## 설치와 테스트

```bash
pixi install --locked
pixi run test
```

`test` task는 전체 `unittest` 회귀 테스트를 실행합니다.

잠금 환경은 Python 3.13, NumPy 2.5, Gymnasium 1.3, PyTorch 2.13, Stable-Baselines3·SB3-Contrib 2.9 계열을 사용합니다. 최신 Gymnasium 계약에 따라 표준 Taxi 기준선은 `Taxi-v4`입니다.

## AdvancedTaxi-v2 학습

누출이 제거된 환경에서는 과거 tuned 설정을 폐기했습니다. 기본값은 보수적인 baseline 설정인 `hyperparameters/advanced_taxi/maskableppo_base.yml`입니다.

```bash
pixi run python train_advanced_taxi_maskableppo.py --n-timesteps 1000000 --seed 42 -P
```

환경 오버라이드:

```bash
pixi run python train_advanced_taxi_maskableppo.py --n-timesteps 1000000 --env-kwargs num_passengers:30 grid_size:120 scenario_mode:mixed
```

학습 환경의 승객 수·그리드·시나리오 설정은 평가 환경에도 전달됩니다. 평가 전용 값은 `--eval-env-kwargs`로 다시 지정할 수 있습니다.

```bash
pixi run python train_advanced_taxi_maskableppo.py --env-kwargs num_passengers:30 --eval-env-kwargs start_at_center:True
```

- 기본 seed는 `42`입니다.
- 작은 MLP 환경에 적합하도록 기본 device는 CPU입니다.
- PyTorch CPU thread는 기본적으로 `1`개를 사용하며 `--num-threads`로 변경할 수 있습니다.
- 학습기는 SB3-Contrib의 공개 `MaskablePPO` API를 사용하며, 전용 callback은 고정 seed 평가와 best checkpoint 저장만 담당합니다.
- best checkpoint는 `random`, `cluster`, `complex`별 고정 validation seed의 동일 가중 평균으로 선택합니다.
- validation 범위는 `--validation-seed-start`와 `--validation-seeds-per-scenario`로 지정합니다.
- 환경과 알고리즘이 고정되어 있으므로 `--algo`, `--env`, `--param-mode`는 받지 않습니다.
- 성능 탐색이 과제 목표가 아니므로 AdvancedTaxi용 Optuna 경로는 제공하지 않습니다.
- 사용자 `--conf-file`을 지정하면 그 안전한 YAML 설정만 사용합니다.
- `--eval-freq 0`이면 중간 평가와 `best_model.zip` 생성을 생략합니다.

## Taxi-v4 기준선

DQN:

```bash
pixi run python train_rlzoo.py --algo dqn --env Taxi-v4 --n-timesteps 500000 --seed 42 -P
```

기본 `tuned` 모드는 Taxi-v4의 원본 `Discrete(500)` 상태를 사용합니다. 선형 Q head에 3-step return, 긴 탐색 구간, 빠른 target 갱신을 적용한 `dqn.yml`은 500k-step 독립 학습 seed 3개에서 각각 고정 평가 seed 100개를 모두 성공했습니다. `--param-mode base`는 같은 raw-discrete/linear 구조에서 학습 안정화 설정만 제외한 비교용 control입니다.

Tabular Q-Learning baseline:

```bash
pixi run python train_rlzoo.py --algo qlearning --env Taxi-v4 --n-timesteps 500000 --seed 42
```

위 명령과 직접 실행은 모두 tuned `qlearning.yml`을 사용합니다. 비교용 control은 `--param-mode base`로 실행합니다.

```bash
pixi run python qlearning_train.py --n-timesteps 500000 --seed 42
```

Q-Learning은 termination에서만 bootstrap을 중단하고 truncation에서는 계속 bootstrap합니다. 탐욕 행동의 Q값이 같을 때는 무작위로 동률을 해소하며, 최종 평가는 학습과 독립적으로 파생한 seed와 RNG를 사용합니다. `seed`를 생략하면 비결정적으로 실행됩니다.

두 Taxi 모델을 동일한 고정 seed로 비교:

```bash
pixi run python evaluate_taxi.py --qlearning-model logs/qlearning/Taxi-v4_1/qlearning_Taxi-v4_1_steps.npy --dqn-model logs/dqn/Taxi-v4_1/Taxi-v4.zip --seed-start 10000 --num-seeds 100
```

평가기는 성공률, episode return과 길이의 평균·표준편차를 JSON으로 출력합니다. DQN은 현재 설정과 같은 `Discrete(500)` 관찰·`Discrete(6)` 행동 모델만 허용하며, 다른 wrapper나 정규화 통계를 추측해 복원하지 않습니다.

## AdvancedTaxi-v2 평가

평가할 실행 ID를 반드시 지정합니다.

```bash
pixi run python evaluate_model.py --exp-id 1 --checkpoint final
```

평가기의 기본 계약은 다음과 같습니다.

- `--checkpoint final|best`로 평가 대상을 반드시 명시하고 해당 시점의 `VecNormalize`를 함께 사용
- `config.yml`, `args.yml`, `vecnormalize.pkl`은 실행 폴더 아래의 `AdvancedTaxi-v2/` 디렉터리에서만 로드
- 정규화 실행인데 통계가 없으면 원시 관찰로 계속하지 않고 즉시 실패
- 저장된 전체 `eval_env_kwargs`를 복원하며, 누락된 과거 형식은 현재 기본값으로 추측하지 않고 재학습을 요구
- Random, nearest-pickup Greedy, shortest-total-trip Greedy, RL에 동일한 seed의 동일 승객 인스턴스를 사용
- Random 정책 RNG는 문제 생성 RNG와 분리
- 기본 테스트 seed는 학습 기본 seed와 분리된 `10000~10099`
- RL 정책의 여러 seed는 최대 100개씩 묶어 배치 평가
- 결과 비용은 환경의 `info["completion_cost"]`만 합산
- raw seed별 비용과 paired saving·표준오차·승률을 NPZ로 저장

중앙 출발 OOD 비교:

```bash
pixi run python evaluate_model.py --exp-id 1 --checkpoint final --center-start
```

seed 범위와 평가 개수 변경:

```bash
pixi run python evaluate_model.py --exp-id 1 --checkpoint final --seed-start 20000 --num-seeds 200
```

경로 시각화:

```bash
pixi run python evaluate_model.py --exp-id 1 --checkpoint final --visualize --visualize-seed 10000
```

PNG는 기본적으로 `visualizations/AdvancedTaxi-v2_<실행 ID>/<checkpoint>/<saved|center>/`에 저장되어 실행·checkpoint·평가 모드 사이에 서로 덮어쓰지 않습니다.

## 출력 구조

```text
logs/
└── maskableppo/
    └── AdvancedTaxi-v2_1/
        ├── AdvancedTaxi-v2.zip
        ├── best_model.zip
        ├── best_model_vecnormalize.pkl
        ├── evaluations.npz
        └── AdvancedTaxi-v2/
            ├── args.yml
            ├── config.yml
            └── vecnormalize.pkl
```

정규화를 사용하는 실행에서는 final 모델과 nested `vecnormalize.pkl`, best 모델과 `best_model_vecnormalize.pkl`을 각각 하나의 쌍으로 취급합니다.

TensorBoard:

```bash
pixi run tensorboard --logdir logs/
```

## 현재 해석상의 한계

- `complex`의 trap은 실제 수요 데이터가 아니라 중앙에서 시작한 Greedy가 가까운 장거리 승객에 낚이도록 의도적으로 만든 합성 스트레스 시나리오입니다.
- 에피소드별 shuffle은 고정 슬롯 누출을 제거하지만, 현재 평면 MLP 자체가 순열 등가적인 set policy라는 뜻은 아닙니다.
- 현재 평가 기준선은 Random, nearest-pickup Greedy, shortest-total-trip Greedy입니다. 최적성 주장을 하려면 작은 문제의 exact DP, beam/local search 등 더 강한 조합최적화 기준선이 필요합니다.
- 정량 결과는 학습 seed 하나로 결론 내리지 말고 여러 독립 학습 실행의 평균과 분산을 함께 보고해야 합니다.
