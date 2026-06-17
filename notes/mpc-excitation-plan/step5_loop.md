# Step 5: MPCLoop (plan-execute-identify-replan)

## 作成ファイル

- `src/mjwarp_ur5e/identification/mpc/loop.py`

## 依存関係

- Step 2: `MPCConfig` (identification/mpc/config.py)
- Step 4: `ExcitationPlanner`, `PlanResult` (identification/mpc/planner.py)

## 既存インタフェース (変更不可)

```python
# identification/execution.py
@dataclass
class PlaybackConfig:
    dt: float = 0.002
    use_pd_control: bool = False
    noise_std_wrench: float = 0.0
    body_name: str = "payload_box_mount"
    site_name: str = "attachment_site"
    settle_time: float = 1.0
    force_sensor_name: str = "ft_force"
    torque_sensor_name: str = "ft_torque"
    # ... (kp, kd, noise_std_q, noise_std_dq も存在)

class TrajectoryPlayback:
    def __init__(self, model, data, config: PlaybackConfig) -> None: ...
    def execute(self, trajectory: TrajectorySample, rng=None) -> DataBuffer: ...

# identification/data_buffer.py
class DataBuffer:
    def to_arrays(self) -> dict[str, np.ndarray]:
        # keys: "timestamp", "q", "dq", "ddq", "ee_position", "ee_rotation", "wrench"
        # wrench shape: (N, 6), ordering: [τx, τy, τz, fx, fy, fz]

# identification/estimators/rtls.py
@dataclass
class RTLSConfig:
    n_params: int = 10
    forgetting_factor: float = 1.0
    window_size: int | None = None

class RecursiveTotalLeastSquares:
    def __init__(self, config: RTLSConfig | None = None) -> None: ...
    def initialize(self, A_init: np.ndarray, y_init: np.ndarray) -> None: ...
    def update(self, A_row: np.ndarray, y_row: np.ndarray) -> EstimationResult: ...
    def get_current_estimate(self) -> EstimationResult: ...
    # _initialized: bool

# identification/estimators/types.py
@dataclass
class EstimationResult:
    phi: np.ndarray           # (10,) [m, hx, hy, hz, Ixx, Iyy, Izz, Ixy, Ixz, Iyz]
    condition_number: float
    residual_norm: float
    n_samples: int

# identification/regressor.py
def compute_stacked_body_regressor(
    model, data, q, dq, ddq, body_name,
    subsample_factor=1, with_ft_offset=False, site_name=None,
) -> np.ndarray:  # (6*N_sub, 10)
```

## 出力インタフェース

```python
# identification/mpc/loop.py

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import mujoco

from mjwarp_ur5e.identification.estimators.types import EstimationResult
from mjwarp_ur5e.identification.mpc.planner import PlanResult


@dataclass
class MPCStepLog:
    """1 MPC イテレーションの記録。"""
    step: int
    plan_result: PlanResult
    estimation: EstimationResult
    condition_number_accumulated: float  # κ₂(W_accumulated) at this step
    q_start: np.ndarray                 # (6,) this step's start position
    q_end: np.ndarray                   # (6,) this step's end position
    n_samples_total: int                # cumulative data points (rows of W)
    wall_time_plan: float               # planning time [s]
    wall_time_execute: float            # execution time [s]


@dataclass
class MPCResult:
    """MPC ループ全体の結果。"""
    steps: list[MPCStepLog]
    final_estimation: EstimationResult
    total_wall_time: float
    total_samples: int
    final_condition_number: float
    converged: bool


class MPCLoop:
    def __init__(
        self,
        config: MPCConfig,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> None: ...

    def run(self) -> MPCResult: ...
```

## 実装詳細

### 初期化

```python
def __init__(self, config, model, data):
    self._config = config
    self._model = model
    self._data = data
    self._planner = ExcitationPlanner(config, model, data)
    self._rtls = RecursiveTotalLeastSquares(RTLSConfig(n_params=10))
    self._playback = TrajectoryPlayback(model, data, PlaybackConfig(
        use_pd_control=config.use_pd_control,
        body_name=config.body_name,
        site_name=config.site_name,
        noise_std_wrench=config.noise_std_wrench,
        settle_time=0.0,  # MPC では settle 不要 (連続実行)
    ))
```

### run メソッドの擬似コード

```python
def run(self) -> MPCResult:
    cfg = self._config
    q_current = cfg.q0.copy()
    dq_current = np.zeros(cfg.num_joints)
    W_accumulated = None
    steps = []
    prev_cond = None
    converged = False
    rng = np.random.default_rng(cfg.planner.seed + 1000)  # noise 用

    t0 = time.perf_counter()

    for step_idx in range(cfg.max_mpc_steps):
        # === PLAN ===
        t_plan = time.perf_counter()
        plan_result = self._planner.plan(q_current, dq_current, W_accumulated)
        wall_plan = time.perf_counter() - t_plan

        # === SLICE: replan_period 分だけ切り出す ===
        execute_traj = _slice_trajectory(plan_result.trajectory, 0.0, cfg.replan_period)

        # === EXECUTE ===
        t_exec = time.perf_counter()
        buffer = self._playback.execute(execute_traj, rng=rng if cfg.noise_std_wrench > 0 else None)
        wall_exec = time.perf_counter() - t_exec

        # === COLLECT: regressor と wrench を取得 ===
        arrays = buffer.to_arrays()
        q_meas = arrays["q"]       # (N, 6)
        dq_meas = arrays["dq"]     # (N, 6)
        ddq_meas = arrays["ddq"]   # (N, 6)
        wrench = arrays["wrench"]  # (N, 6) [τx,τy,τz,fx,fy,fz]

        W_new = compute_stacked_body_regressor(
            self._model, self._data,
            q_meas, dq_meas, ddq_meas,
            cfg.body_name,
        )
        y_new = wrench.ravel()  # (6*N,)

        # === ACCUMULATE regressor (for planner's cost function) ===
        if W_accumulated is None:
            W_accumulated = W_new
        else:
            W_accumulated = np.vstack([W_accumulated, W_new])

        cond_accumulated = compute_condition_number(W_accumulated)

        # === IDENTIFY ===
        if not self._rtls._initialized:
            self._rtls.initialize(W_new, y_new)
        else:
            self._rtls.update(W_new, y_new)
        estimation = self._rtls.get_current_estimate()

        # === UPDATE STATE ===
        q_current = q_meas[-1].copy()
        dq_current = dq_meas[-1].copy()

        # === LOG ===
        step_log = MPCStepLog(
            step=step_idx,
            plan_result=plan_result,
            estimation=estimation,
            condition_number_accumulated=cond_accumulated,
            q_start=arrays["q"][0].copy(),
            q_end=q_current.copy(),
            n_samples_total=W_accumulated.shape[0] // 6,
            wall_time_plan=wall_plan,
            wall_time_execute=wall_exec,
        )
        steps.append(step_log)

        # === CONVERGENCE CHECK ===
        if prev_cond is not None and prev_cond > 0:
            rel_improvement = (prev_cond - cond_accumulated) / prev_cond
            if rel_improvement < cfg.convergence_threshold:
                converged = True
                break
        prev_cond = cond_accumulated

    total_time = time.perf_counter() - t0
    return MPCResult(
        steps=steps,
        final_estimation=self._rtls.get_current_estimate(),
        total_wall_time=total_time,
        total_samples=W_accumulated.shape[0] // 6 if W_accumulated is not None else 0,
        final_condition_number=cond_accumulated,
        converged=converged,
    )
```

### _slice_trajectory

```python
def _slice_trajectory(
    trajectory: TrajectorySample,
    t_start: float,
    t_end: float,
) -> TrajectorySample:
    """TrajectorySample から時間範囲 [t_start, t_end] を切り出す。"""
    mask = (trajectory.time >= t_start - 1e-9) & (trajectory.time <= t_end + 1e-9)
    return TrajectorySample(
        time=trajectory.time[mask] - t_start,
        position=trajectory.position[mask],
        velocity=trajectory.velocity[mask],
        acceleration=trajectory.acceleration[mask],
    )
```

### RTLS への入力

`compute_stacked_body_regressor` は `(6*N, 10)` の regressor を返す.
`wrench.ravel()` は `(6*N,)` のベクトルになる.
`RTLS.update(A_row, y_row)` は `(k, n_params)` と `(k,)` を受け付ける.
したがって `W_new` と `y_new` をそのまま渡す.

### settle_time = 0.0

MPC ループは連続実行なので, 各ホライズンの先頭で settle する必要はない.
初回のみ settle が必要な場合は `run()` の冒頭で別途対応するが,
open-loop モード (`use_pd_control=False`) では不要.

### playback の FT wrench 取得フロー

`TrajectoryPlayback.execute()` は FT センサが存在すれば `data.sensordata` から
force/torque を読む (execution.py:161-169). wrench 順序は `[torque; force]`.
これは regressor の wrench 規約と一致する.

## 検証

### テスト

1. **短いループ**: `max_mpc_steps=2` で `run()` を実行, `MPCResult` が返ること
2. **steps 長**: `len(result.steps) <= max_mpc_steps`
3. **estimation**: `result.final_estimation.phi.shape == (10,)`
4. **condition_number**: `result.final_condition_number` が有限値
5. **total_samples**: `result.total_samples > 0`
6. **状態の連続性**: `steps[1].q_start ≈ steps[0].q_end` (前ステップの終了位置が次の開始)
7. **_slice_trajectory**: 3.0 s の軌道から [0, 1.5] を切り出して `time[-1] ≈ 1.5` を確認

### 実行方法

```bash
python -m pytest tests/test_mpc_loop.py -v
```

### スモークテスト

```python
from mjwarp_ur5e.model import load_and_reset
from mjwarp_ur5e.identification.mpc.config import MPCConfig, PlannerConfig, HorizonConfig
from mjwarp_ur5e.identification.mpc.loop import MPCLoop

loaded = load_and_reset()
config = MPCConfig(
    max_mpc_steps=3,
    planner=PlannerConfig(n_restarts=2, max_iter_per_start=10),
)
mpc = MPCLoop(config, loaded.model, loaded.data)
result = mpc.run()
print(f"converged={result.converged}, steps={len(result.steps)}")
print(f"final mass={result.final_estimation.mass:.4f}")
for s in result.steps:
    print(f"  step {s.step}: cond={s.condition_number_accumulated:.2f}, "
          f"mass={s.estimation.mass:.4f}")
```
