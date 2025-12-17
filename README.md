# RL 프로젝트 (RL-Zoo 기반)

SB3 + SB3-Contrib MaskablePPO로 고수준 택시 디스패치 정책을 학습하는 프로젝트입니다.  
커스텀 환경 `AdvancedTaxi-v0`(단일 택시 디스패치, Box 관찰)와 표준 환경 `Taxi-v3`를 대상으로 Stable-Baselines3, SB3-Contrib(MaskablePPO), RL-Zoo3, Optuna를 사용합니다.

---

## 프로젝트 개요

- **문제 정의**: 단일 택시가 여러 승객을 순차적으로 서비스하면서 **총 누적 대기시간을 최소화**하는 디스패치 정책 학습
- **환경**:
  - `AdvancedTaxi-v0`: 커스텀 고수준 디스패치 환경 (Discrete 승객 선택 + Action Masking)
  - `Taxi-v3`: Gymnasium 표준 택시 환경 (DQN/Tabular Q-Learning)
- **주요 스크립트**:
  - `train_advanced_taxi_maskableppo.py`: AdvancedTaxi-v0 + MaskablePPO 전용 학습/Optuna
  - `evaluate_model.py`: AdvancedTaxi-v0 평가 (랜덤/그리디/RL 총 대기시간 콘솔 요약)
  - `train_rlzoo.py`: Taxi-v3용 RL-Zoo 래퍼(DQN/Q-Learning 등)
  - `qlearning_train.py`: Taxi-v3 Tabular Q-Learning


---

## 4단계 워크플로우 개요

1. **Step 0 – 파라미터 최적화 (Optuna + SQLite)**  
   - Optuna로 알고리즘/환경 조합별 적당한 하이퍼파라미터를 찾고 SQLite DB에 저장합니다.
2. **Step 1 – 학습 (Training)**  
   - 찾은 설정을 YAML에 업데이트한 후 AdvancedTaxi-v0 / Taxi-v3에 대해 본 학습을 수행합니다.
3. **Step 2 – 평가 (Evaluate)**  
   - AdvancedTaxi-v0에 대해 `evaluate_model.py`로 시나리오별 성능을 비교합니다.
   - `--visualize` 옵션으로 각 정책의 이동 경로를 그리드월드에 시각화할 수 있습니다.
4. **Step 3 – TensorBoard로 학습 차트 확인**  
   - RL-Zoo가 남긴 로그를 TensorBoard로 열어 학습 곡선, KL, loss 등을 확인합니다.

아래 섹션에서 각 Step에 해당하는 명령과 옵션을 자세히 설명합니다.

---

## Step 0 – 파라미터 최적화 (Optuna + SQLite)

이 프로젝트의 권장 플로우는 **먼저 Optuna로 하이퍼파라미터를 튜닝한 뒤**, 그 설정을 기반으로 본 학습을 돌리는 것입니다.


### 단순 최적화 패턴

다른 알고리즘/환경으로 변경하려면 다음 패턴을 사용하세요 (※ `train_rlzoo.py`는 Taxi-v3 전용):

```bash
python train_rlzoo.py --algo {ALGO} --env {ENV} -optimize --n-trials 100 --n-timesteps 200000 --study-name {STUDY_NAME} --storage sqlite:///logs/{STUDY_NAME}.db --n-jobs {N_JOBS}
```

- `{ALGO}`: 알고리즘 이름 (dqn 등 – **maskableppo는 전용 스크립트 사용**)
- `{ENV}`: 환경 ID (현재 스크립트에서는 **Taxi-v3**만 지원)
- `{STUDY_NAME}`: Optuna study 이름
- `{N_JOBS}`: On-policy 알고리즘은 `1`, Off-policy 알고리즘은 `4~8` 권장

### Optuna 예시 모음 (복붙 전용)

#### 1. AdvancedTaxi-v0 + MaskablePPO (On-policy)

```bash
python train_advanced_taxi_maskableppo.py -optimize --n-trials 100 --n-timesteps 200000 --study-name adv_taxi_maskableppo --storage sqlite:///logs/adv_taxi_optuna.db --n-jobs 1
```

#### 2. Taxi-v3 + DQN (Off-policy)

```bash
python train_rlzoo.py --algo dqn --env Taxi-v3 -optimize --n-trials 100 --n-timesteps 200000 --study-name taxi_v3_dqn --storage sqlite:///logs/taxi_dqn_optuna.db --n-jobs 4
```

### Optuna 옵션 정리

- `-optimize`: Optuna 최적화 모드 활성화
- `--n-trials`: 시도할 trial 수 (예: 20, 100)
- `--n-timesteps`: 각 trial의 학습 스텝 수 (예: 200000)
- `--n-jobs`: 병렬 trial 수
  - On-policy (MaskablePPO, A2C 등): `1` 권장 (각 trial 내에서 병렬 환경 사용)
  - Off-policy (DQN, SAC 등): `4~8` 권장 (병렬 trial로 속도 향상)
- `--study-name`: Optuna study 이름 (재현성을 위해 명시 권장)
- `--storage`: Optuna storage 경로 (SQLite 권장, 예: `sqlite:///logs/study.db`)

---

## Step 1 – 학습 (Training)

> `--seed` 옵션은 선택 사항입니다. 지정하지 않으면 기본값 `42`를 사용합니다.  
> 다른 seed를 사용하려면 `--seed <값>`을 명시적으로 지정하세요.

### 하이퍼파라미터 파일 지정 규칙

- `train_rlzoo.py`는 `--algo`, `--env`, `--param-mode`를 기반으로 **YAML을 자동 선택**합니다.
- `train_advanced_taxi_maskableppo.py`는 기본으로 `hyperparameters/advanced_taxi/maskableppo.yml`을 사용합니다.

자동 선택 규칙:

- 환경별 폴더: `hyperparameters/{env_folder}/{algo}.yml`
  - `AdvancedTaxi-v0` → `hyperparameters/advanced_taxi/`
  - `Taxi-v3` → `hyperparameters/taxi_v3/`
- `--param-mode` 옵션:
  - `tuned` (기본값): `{algo}.yml` 사용
  - `base`: `{algo}_base.yml` 사용

**수동 지정 (`--conf-file`)**

- 특정 YAML을 강제로 사용하고 싶을 때는 `--conf-file`을 사용합니다:

```bash
python train_rlzoo.py --algo dqn --env Taxi-v3 --n-timesteps 500000 \
  --conf-file hyperparameters/taxi_v3/dqn.yml
```

**진행률 바(`tqdm`) – `-P` / `--progress`**

- `-P`는 RL-Zoo3의 `--progress` 플래그로 전달되어 **tqdm 진행률 바를 켜는 역할**만 합니다.
- YAML 로딩과는 완전히 독립입니다.

```bash
python train_rlzoo.py --algo dqn --env Taxi-v3 --n-timesteps 500000 \
  --conf-file hyperparameters/taxi_v3/dqn.yml -P
```

### Step 1-1. AdvancedTaxi-v0 학습 (MaskablePPO 전용)

기본 학습 (자동 하이퍼파라미터 선택, seed 기본값 42):

```bash
python train_advanced_taxi_maskableppo.py --n-timesteps 1000000 -P
```

- RL-Zoo 기본 구조 사용: `logs/{algo}/{env_id}_{exp}/...`

환경 파라미터 오버라이드:

```bash
python train_advanced_taxi_maskableppo.py --n-timesteps 1000000 -P \
  --env-kwargs num_passengers:30 grid_size:120 scenario_mode:mixed
```

- `--device` 미지정 시 CPU 기본
- MaskablePPO는 `sb3-contrib` 필요 (이미 `pixi.toml`에 포함)

### Step 1-2. Taxi-v3 학습 (DQN)

DQN 학습:

```bash
python train_rlzoo.py --algo dqn --env Taxi-v3 --n-timesteps 500000 -P
```

- Taxi-v3 관찰: 기본은 `Discrete(500)` 원형  
  필요 시 YAML에서 wrapper를 설정해 one-hot/decoded 관찰을 사용할 수 있습니다.

### Step 1-3. Taxi-v3 Tabular Q-Learning

```bash
python qlearning_train.py --n-timesteps 500000
```

- 인자는 타임스텝만 필요합니다.
- 하이퍼파라미터는 `hyperparameters/taxi_v3/qlearning*.yml`에서 자동 로드됩니다.

---

## Step 2 – 평가 (Evaluate)

### AdvancedTaxi-v0 평가 (`evaluate_model.py`)

`evaluate_model.py`는 **`AdvancedTaxi-v0` + `maskableppo` 조합을 고정**으로 사용해  
`logs/`에서 최신 실험을 자동으로 찾아 시나리오별 성능을 평가합니다.

#### 기본 평가 (텍스트 요약)

아무 인자도 넘기지 않으면, random/cluster/complex 시나리오에 대해  
Random, Greedy, RL 정책의 총 대기시간 지표를 콘솔에 출력합니다:

```bash
python evaluate_model.py
```

출력 예시:
```
=== 성과 지표 (총 대기시간: 낮을수록 좋음, seed=1~100 평균) ===

Scenario   | Policy   |         Mean |          Std
----------------------------------------------------
random     | random   |     12345.67 |      1234.56
random     | greedy   |      9876.54 |       987.65
random     | RL       |      8765.43 |       876.54
----------------------------------------------------
...
```

#### 시각화 기능

`--visualize` 옵션을 사용하면 각 정책의 이동 경로를 그리드월드에 시각화합니다:

```bash
python evaluate_model.py --visualize
```

**시각화 내용:**
- **승객 출발지** (녹색 원, 번호 표시)
- **승객 목적지** (빨간색 사각형, 번호 표시)
- **빈차 이동 경로** (실선, 두껍게, 강조) - 최적화의 핵심
- **승객 운행 경로** (점선, 얇게) - 어쩔 수 없는 부분
- **택시 시작 위치** (파란색 다이아몬드)
- **택시 종료 위치** (보라색 다이아몬드)

**시각화 옵션:**
- `--visualize`: 시각화 생성 (기본: False)
- `--visualize-seed`: 시각화에 사용할 시드 (기본: 1)
- `--output-dir`: 시각화 저장 디렉터리 (기본: `visualizations/`)
- `--num-passengers`: 승객 수 (기본: 20)
- `--grid-size`: 그리드 크기 (기본: 100)

**예시:**
```bash
# 기본 시각화 (seed=1)
python evaluate_model.py --visualize
```

**출력 파일:**
각 시나리오(random, cluster, complex)와 정책(random, greedy, rl) 조합별로 PNG 파일이 생성됩니다:
- `visualizations/random_random_seed1.png`
- `visualizations/random_greedy_seed1.png`
- `visualizations/random_rl_seed1.png`
- `visualizations/cluster_random_seed1.png`
- ... (총 9개 파일)

시각화를 통해 각 정책이 얼마나 효율적으로 승객을 배차하는지 시각적으로 비교할 수 있습니다.

---

## Step 3 – TensorBoard로 학습 차트 확인

### TensorBoard 사용법

기본 실행 (RL-Zoo 기본 경로 기준):

```bash
tensorboard --logdir logs/
```

경로 안내 (RL-Zoo 기본 정책):

- 모델/모니터/평가 결과: `logs/{algo}/{env_id}_{exp_id}/...`  
  예: `logs/maskableppo/AdvancedTaxi-v0_1/`
- TensorBoard 이벤트: `logs/{env_id}/{algo}_{exp_id}/...`  
  예: `logs/AdvancedTaxi-v0/MaskablePPO_1/`
- 두 경로를 맞추고 싶다면 `--tensorboard-log`를 명시적으로 `logs/{algo}/{env_id}_{exp_id}` 형태로 지정하세요.

### 주요 메트릭

- `rollout/ep_rew_mean`: 에피소드 평균 누적 보상
- `rollout/ep_len_mean`: 에피소드 평균 길이
- `train/value_loss`: 가치 함수 손실
- `train/policy_gradient_loss`: 정책 경사 손실
- `train/approx_kl`: KL 발산 근사값
- `train/clip_fraction`: 클리핑 비율

---

## AdvancedTaxi-v0 환경·아키텍처 개요

### 사용 가능한 환경

- `AdvancedTaxi-v0` (커스텀, MaskablePPO 권장)
- `Taxi-v3` (표준 Gymnasium, DQN/Tabular Q-Learning 지원)

### 커스텀 환경 요약 (`AdvancedTaxi-v0`)

- 클래스: `envs.envs.AdvancedTimeAttackTaxiEnv`
- 행동 공간: `Discrete(num_passengers)` (승객 인덱스 선택), Action Masking 필수 (`env.action_masks()`)
- 관찰 공간: `Box(shape=(num_passengers*6,), dtype=float32)`  
  - 승객별 `[rel_x, rel_y, dest_dx, dest_dy, status, dist]` (정규화 스케일 100)  
  - 현재 구현은 승객 생성 순서를 유지합니다.
- 보상: `reward = -(time_elapsed * remaining_passengers) / 100.0`  
  - `time_elapsed = dist_empty + dist_service`
  - 시스템 수준 KPI인 총 누적 대기시간을 근사하도록 설계
- 시나리오: `random` / `cluster` / `complex`, `mixed`는 (0.3, 0.4, 0.3) 비율로 무작위 선택
- 기본 env_kwargs: `num_passengers=20`, `grid_size=100`, `scenario_mode="mixed"`, `start_at_center=False`

학습·시연 간 일관성을 위해, **동일한 환경 ID(`AdvancedTaxi-v0`)와 동일한 env_kwargs**를 사용하는 것을 권장합니다.

### 실제 프로젝트 구조

```text
RL/
├── train_rlzoo.py                    # Taxi-v3 등 표준 환경 학습 (dqn/qlearning)
├── train_advanced_taxi_maskableppo.py  # AdvancedTaxi-v0 전용 MaskablePPO 학습/Optuna
├── qlearning_train.py                # Taxi-v3용 Tabular Q-Learning
├── evaluate_model.py                 # AdvancedTaxi-v0 평가 스크립트 (텍스트 요약 + 시각화)
├── envs/                             # 환경 패키지
│   ├── __init__.py                  # AdvancedTaxi-v0 등록, wrappers export
│   ├── envs.py                      # AdvancedTimeAttackTaxiEnv 정의
│   └── wrappers.py                  # Taxi-v3 one-hot/decoded, AdvancedTaxi ActionMasker
├── hyperparameters/                  # RL-Zoo YAML (환경별 디렉터리)
│   ├── advanced_taxi/
│   │   └── maskableppo.yml
│   └── taxi_v3/                     # dqn/qlearning 등 base·tuned 모음
├── logs/                             # RL-Zoo가 생성하는 실험 결과 및 TensorBoard 로그
├── visualizations/                   # evaluate_model.py --visualize로 생성된 경로 시각화 이미지
├── pixi.toml / pixi.lock             # 의존성(사전 CUDA 12.8 설정)
└── ...
```

---

## Optuna 결과 고급 분석 (Appendix)

### Optuna Dashboard 사용 (권장)

```bash
optuna-dashboard sqlite:///logs/adv_taxi_optuna.db
```

또는 상대 경로 사용:

```bash
optuna-dashboard logs/adv_taxi_optuna.db
```