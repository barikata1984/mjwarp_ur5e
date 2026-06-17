# Step 1: QuinticSpline 軌道パラメータ化

## 作成ファイル

- `src/mjwarp_ur5e/trajectories/quintic_spline.py`

## 依存関係

なし (既存コードの読み取りのみ)

## 既存インタフェース (変更不可)

```python
# trajectories/base.py
@dataclass(frozen=True, kw_only=True)
class BaseTrajectoryConfig:
    duration: float
    fps: float

@dataclass(frozen=True)
class TrajectorySample:
    time: np.ndarray      # (num_steps,)
    position: np.ndarray   # (num_steps, num_joints)
    velocity: np.ndarray   # (num_steps, num_joints)
    acceleration: np.ndarray  # (num_steps, num_joints)

class BaseTrajectory:
    def __init__(self, config: BaseTrajectoryConfig) -> None: ...
    def sample(self) -> TrajectorySample: ...
    @staticmethod
    def _build_time_array(duration: float, fps: float) -> np.ndarray: ...
```

## 出力インタフェース

```python
@dataclass(frozen=True, kw_only=True)
class QuinticSplineConfig(BaseTrajectoryConfig):
    num_joints: int           # 6 for UR5e
    num_segments: int         # N_seg (typically 3-6)
    q0: np.ndarray            # (num_joints,) initial position
    dq0: np.ndarray           # (num_joints,) initial velocity
    waypoints: np.ndarray     # (num_segments, num_joints) interior + terminal waypoints
    dq_terminal: np.ndarray   # (num_joints,) terminal velocity (default: zeros)

class QuinticSplineTrajectory(BaseTrajectory):
    def __init__(self, config: QuinticSplineConfig) -> None: ...
    def sample(self) -> TrajectorySample: ...

def build_quintic_from_decision_vars(
    x: np.ndarray,              # (num_segments * num_joints,) flat decision variables
    q0: np.ndarray,             # (num_joints,)
    dq0: np.ndarray,            # (num_joints,)
    num_segments: int,
    num_joints: int,
    duration: float,
    fps: float,
    dq_terminal: np.ndarray | None = None,  # None -> zeros
) -> TrajectorySample:
    """最適化変数から TrajectorySample を直接生成するコンビニエンス関数。

    x を (num_segments, num_joints) に reshape し、waypoints として扱う。
    q0 は固定 (決定変数に含まない)。
    """
```

## 実装詳細

### ウェイポイント構成

全ウェイポイント列 = `[q0, waypoints[0], waypoints[1], ..., waypoints[N_seg-1]]`

- `q0`: 固定 (MPC の現在状態). 決定変数に含まない
- `waypoints`: shape `(N_seg, num_joints)`. 決定変数
- ウェイポイント総数: N_seg + 1 (始端 q0 含む)
- セグメント数: N_seg

### セグメント境界の速度・加速度

中間ウェイポイント (index 1 ... N_seg-1) の速度・加速度は有限差分で近似:

```
dq_k = (wp[k+1] - wp[k-1]) / (2 * dt_seg)      # k = 1, ..., N_seg-1
ddq_k = (wp[k+1] - 2*wp[k] + wp[k-1]) / dt_seg^2
```

ここで `dt_seg = duration / num_segments`, `wp[0] = q0`, `wp[k] = waypoints[k-1]`.

端点:
- 始端 (k=0): `dq_0 = dq0` (MPC から与える), `ddq_0` は natural spline 近似:
  `ddq_0 = 2*(wp[1] - wp[0]) / dt_seg^2 - 2*dq0 / dt_seg`
- 終端 (k=N_seg): `dq_N = dq_terminal` (default: zeros), `ddq_N = 0`

### 5 次 Hermite 補間

各セグメント k は始端 (p0, v0, a0) と終端 (p1, v1, a1) の 6 つの境界条件から一意に決まる.
正規化時間 τ = (t - t_k) / dt_seg ∈ [0, 1] として:

```
q(τ) = h00(τ)*p0 + h10(τ)*dt_seg*v0 + h20(τ)*dt_seg^2*a0
      + h01(τ)*p1 + h11(τ)*dt_seg*v1 + h21(τ)*dt_seg^2*a1
```

基底関数:

```
h00(τ) = 1 - 10τ³ + 15τ⁴ - 6τ⁵
h10(τ) = τ - 6τ³ + 8τ⁴ - 3τ⁵
h20(τ) = 0.5τ² - 1.5τ³ + 1.5τ⁴ - 0.5τ⁵
h01(τ) = 10τ³ - 15τ⁴ + 6τ⁵
h11(τ) = -4τ³ + 7τ⁴ - 3τ⁵
h21(τ) = 0.5τ³ - τ⁴ + 0.5τ⁵
```

速度 (dq/dt = (1/dt_seg) * dq/dτ):

```
h00'(τ) = -30τ² + 60τ³ - 30τ⁴
h10'(τ) = 1 - 18τ² + 32τ³ - 15τ⁴
h20'(τ) = τ - 4.5τ² + 6τ³ - 2.5τ⁴
h01'(τ) = 30τ² - 60τ³ + 30τ⁴
h11'(τ) = -12τ² + 28τ³ - 15τ⁴
h21'(τ) = 1.5τ² - 4τ³ + 2.5τ⁴
```

加速度 (ddq/dt² = (1/dt_seg²) * d²q/dτ²):

```
h00''(τ) = -60τ + 180τ² - 120τ³
h10''(τ) = -36τ + 96τ² - 60τ³
h20''(τ) = 1 - 9τ + 18τ² - 10τ³
h01''(τ) = 60τ - 180τ² + 120τ³
h11''(τ) = -24τ + 84τ² - 60τ³
h21''(τ) = 3τ - 12τ² + 10τ³
```

### サンプリング

`BaseTrajectory._build_time_array(duration, fps)` で時間配列を生成.
各時刻 t に対し, 所属セグメント k と正規化時間 τ を求め, 上の Hermite 式で q, dq, ddq を計算.

### build_quintic_from_decision_vars

最適化ループで使うコンビニエンス関数. `x` を `(N_seg, num_joints)` に reshape して `waypoints` とし, `QuinticSplineConfig` + `QuinticSplineTrajectory.sample()` で `TrajectorySample` を返す.

## 検証

### 単体テスト (`tests/test_quintic_spline.py`)

1. **静止軌道**: 全ウェイポイント = q0, dq0 = 0 → velocity, acceleration が全て 0
2. **境界条件**: ランダム q0, dq0 で `position[0] == q0`, `velocity[0] ≈ dq0` (atol=1e-10)
3. **終端条件**: `velocity[-1] ≈ dq_terminal` (atol=1e-10)
4. **C2 連続性**: セグメント境界での position, velocity, acceleration の連続性を検証.
   境界の両側から評価して `np.allclose(left, right, atol=1e-10)` を確認
5. **shape**: `position.shape == (num_steps, num_joints)`
6. **build_quintic_from_decision_vars**: x の長さが `num_segments * num_joints` であること.
   返り値が `TrajectorySample` であること

### 実行方法

```bash
python -m pytest tests/test_quintic_spline.py -v
```
