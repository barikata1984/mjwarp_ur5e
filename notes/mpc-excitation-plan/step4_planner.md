# Step 4: ExcitationPlanner (単一ホライズン最適化)

## 作成ファイル

- `src/mjwarp_ur5e/identification/mpc/planner.py`

## 依存関係

- Step 1: `build_quintic_from_decision_vars` (trajectories/quintic_spline.py)
- Step 2: `MPCConfig`, `HorizonConfig`, `PlannerConfig` (identification/mpc/config.py)
- Step 3: `gravity_sweep_angle`, `acceleration_peak` (identification/mpc/metrics.py)

## 既存インタフェース (変更不可)

```python
# identification/regressor.py
def compute_stacked_body_regressor(
    model, data, q, dq, ddq, body_name,
    subsample_factor=1, with_ft_offset=False, site_name=None,
) -> np.ndarray:  # (6*N_sub, 10) or (6*N_sub, 16) with ft_offset

def compute_condition_number(
    regressor, singular_value_floor=1e-12, column_scale=False,
) -> float:

# identification/constraints.py
@dataclass(frozen=True)
class JointLimits:
    q_min: np.ndarray   # (6,)
    q_max: np.ndarray   # (6,)
    dq_max: np.ndarray  # (6,)
    ddq_max: np.ndarray # (6,)

# identification/sampling.py
def set_model_state(model, data, qpos, qvel=None, qacc=None) -> None: ...

# trajectories/base.py
@dataclass(frozen=True)
class TrajectorySample:
    time: np.ndarray        # (N,)
    position: np.ndarray    # (N, 6)
    velocity: np.ndarray    # (N, 6)
    acceleration: np.ndarray  # (N, 6)
```

## 出力インタフェース

```python
# identification/mpc/planner.py

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import mujoco

from mjwarp_ur5e.trajectories.base import TrajectorySample


@dataclass
class PlanResult:
    """Result of a single horizon planning step."""
    trajectory: TrajectorySample
    waypoints: np.ndarray            # (N_seg, 6) optimized waypoints (excl. q0)
    condition_number: float          # κ₂(W) for this horizon (or accumulated)
    gravity_sweep: float             # gravity sweep angle [rad]
    acceleration_peak: float         # max |ddq| [rad/s²]
    cost: float                      # optimizer objective value
    feasible: bool                   # all constraints satisfied
    constraint_margins: dict[str, float]  # per-constraint margins
    wall_time: float                 # seconds
    n_evaluations: int               # total function evaluations across restarts


class ExcitationPlanner:
    def __init__(
        self,
        config: MPCConfig,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> None: ...

    def plan(
        self,
        q_current: np.ndarray,              # (6,) current joint position
        dq_current: np.ndarray,             # (6,) current joint velocity
        W_accumulated: np.ndarray | None = None,  # past regressor data
    ) -> PlanResult: ...

    def trajectory_from_x(
        self,
        x: np.ndarray,
        q0: np.ndarray,
        dq0: np.ndarray,
    ) -> TrajectorySample:
        """Decision variable -> TrajectorySample (for external use)."""
        ...
```

## 実装詳細

### 決定変数

```
x = waypoints.ravel()  # shape: (N_seg * num_joints,)
```

`waypoints` は `(N_seg, num_joints)` 行列. `waypoints[0]` = 最初の内部ウェイポイント,
`waypoints[-1]` = 終端位置. `q0` (= q_current) は固定で決定変数に含まない.

### 軌道キャッシュ

既存 `_TrajectoryCache` (constraints.py:35-71) のパターンに倣い,
`x.tobytes()` をキーとしたキャッシュを内部に持つ.
コスト関数と制約関数が同一の `x` に対して軌道を再計算しないようにする.

```python
class _QuinticCache:
    def __init__(self, q0, dq0, num_segments, num_joints, duration, fps):
        self._q0 = q0.copy()
        self._dq0 = dq0.copy()
        self._num_segments = num_segments
        self._num_joints = num_joints
        self._duration = duration
        self._fps = fps
        self._cache_key: bytes | None = None
        self._cache_value: TrajectorySample | None = None

    def get(self, x: np.ndarray) -> TrajectorySample:
        key = x.tobytes()
        if key == self._cache_key and self._cache_value is not None:
            return self._cache_value
        sample = build_quintic_from_decision_vars(
            x, self._q0, self._dq0,
            self._num_segments, self._num_joints,
            self._duration, self._fps,
        )
        self._cache_key = key
        self._cache_value = sample
        return sample
```

### コスト関数

```python
def _make_objective(cache, model, data, config, W_accumulated):
    body_name = config.body_name
    sub = config.horizon.subsample_factor

    def objective(x):
        sample = cache.get(x)
        W_new = compute_stacked_body_regressor(
            model, data,
            sample.position, sample.velocity, sample.acceleration,
            body_name, subsample_factor=sub,
        )
        if W_accumulated is not None:
            W_combined = np.vstack([W_accumulated, W_new])
        else:
            W_combined = W_new
        return compute_condition_number(W_combined)

    return objective
```

数値的に失敗した場合は `1e12` を返す (既存 `condition_number_objective` と同じパターン).

### 制約関数

既存の `make_joint_position_constraint` 等のパターンに倣う.
キャッシュが `_QuinticCache` に変わるだけ.

```python
def _make_position_constraint(cache, joint_limits):
    def constraint(x):
        sample = cache.get(x)
        margin_lo = sample.position - joint_limits.q_min
        margin_hi = joint_limits.q_max - sample.position
        return float(np.min([margin_lo, margin_hi]))
    return constraint

def _make_velocity_constraint(cache, joint_limits):
    def constraint(x):
        sample = cache.get(x)
        return float(np.min(joint_limits.dq_max - np.abs(sample.velocity)))
    return constraint

def _make_acceleration_constraint(cache, joint_limits):
    def constraint(x):
        sample = cache.get(x)
        return float(np.min(joint_limits.ddq_max - np.abs(sample.acceleration)))
    return constraint
```

### マルチスタート

`optimizer.py` の `_optimize_sequential` パターンに倣う.

```python
def plan(self, q_current, dq_current, W_accumulated=None):
    cfg = self.config
    pcfg = cfg.planner
    hcfg = cfg.horizon
    rng = np.random.default_rng(pcfg.seed)

    cache = _QuinticCache(q_current, dq_current,
                          hcfg.num_segments, cfg.num_joints,
                          hcfg.duration, hcfg.fps)

    objective = _make_objective(cache, self._model, self._data, cfg, W_accumulated)
    constraints = [
        {"type": "ineq", "fun": _make_position_constraint(cache, cfg.joint_limits)},
        {"type": "ineq", "fun": _make_velocity_constraint(cache, cfg.joint_limits)},
        {"type": "ineq", "fun": _make_acceleration_constraint(cache, cfg.joint_limits)},
    ]

    best_x = None
    best_cost = float("inf")
    total_evals = 0

    for i in range(pcfg.n_restarts):
        x0 = q_current + rng.uniform(
            -pcfg.waypoint_perturbation,
            pcfg.waypoint_perturbation,
            size=(hcfg.num_segments, cfg.num_joints),
        )
        x0 = np.clip(x0, cfg.joint_limits.q_min, cfg.joint_limits.q_max)
        x0 = x0.ravel()

        result = scipy.optimize.minimize(
            objective, x0,
            method=pcfg.method,
            constraints=constraints,
            options={"maxiter": pcfg.max_iter_per_start, "ftol": pcfg.ftol},
        )
        total_evals += result.nfev

        if result.fun < best_cost:
            best_cost = result.fun
            best_x = result.x.copy()

    # Build PlanResult from best_x
    ...
```

### PlanResult の構築

最良の `x` から `TrajectorySample` を生成し, メトリクスを計算して `PlanResult` を返す.
`gravity_sweep` は `gravity_sweep_angle()` (Step 3) で計算.
`feasible` は全制約の margin >= 0 で判定.

## 検証

### テスト

1. **実行可能性**: `load_and_reset` でモデルをロード, ホームポジションから `plan()` を呼び出して `PlanResult` が返ること
2. **終端速度**: `plan_result.trajectory.velocity[-1]` が全て ≈ 0
3. **条件数有限**: `plan_result.condition_number` が有限値
4. **制約充足**: `plan_result.feasible == True`
5. **W_accumulated 効果**: `W_accumulated=None` と `W_accumulated=np.random.randn(60, 10)` で条件数が異なること
6. **shape**: `plan_result.waypoints.shape == (N_seg, 6)`

### 実行方法

```bash
python -m pytest tests/test_mpc_planner.py -v
```

### スモークテスト (手動)

```python
from mjwarp_ur5e.model import load_and_reset
from mjwarp_ur5e.identification.mpc import MPCConfig
from mjwarp_ur5e.identification.mpc.planner import ExcitationPlanner

loaded = load_and_reset()
config = MPCConfig(planner=PlannerConfig(n_restarts=2, max_iter_per_start=20))
planner = ExcitationPlanner(config, loaded.model, loaded.data)
result = planner.plan(config.q0, np.zeros(6))
print(f"cond={result.condition_number:.2f}, sweep={np.degrees(result.gravity_sweep):.1f}°")
```
