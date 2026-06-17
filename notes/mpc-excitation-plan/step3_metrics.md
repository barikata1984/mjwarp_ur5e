# Step 3: Metrics モジュール

## 作成ファイル

- `src/mjwarp_ur5e/identification/mpc/metrics.py`

## 依存関係

なし (既存インタフェースのみ使用)

## 既存インタフェース (変更不可)

```python
# identification/sampling.py
def set_model_state(
    model: mujoco.MjModel, data: mujoco.MjData,
    qpos: np.ndarray, qvel: np.ndarray | None = None, qacc: np.ndarray | None = None,
) -> None:
    # qpos/qvel/qacc を設定し mj_kinematics + mj_comPos + mj_fwdVelocity を呼ぶ
    ...

# trajectories/base.py
@dataclass(frozen=True)
class TrajectorySample:
    time: np.ndarray       # (N,)
    position: np.ndarray   # (N, 6)
    velocity: np.ndarray   # (N, 6)
    acceleration: np.ndarray  # (N, 6)

# model.py
def get_named_object_id(model, object_type, name) -> int | None: ...
```

## 出力インタフェース

```python
# identification/mpc/metrics.py

def gravity_sweep_angle(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    q_trajectory: np.ndarray,
    body_name: str = "payload_box_mount",
    subsample: int = 1,
) -> float:
    """ボディフレームにおける重力方向ベクトルの累積掃引角 [rad]。

    各タイムステップで重力ベクトルをボディローカル座標に変換し、
    連続する方向ベクトル間の角度を累積する。
    値が大きいほど重力パラメータ (質量, 一次モーメント) の励起が良い。

    Args:
        model: MuJoCo モデル。
        data: MuJoCo データ。
        q_trajectory: (N, nq) 関節位置の時系列。
        body_name: 対象ボディ名。
        subsample: サブサンプリング間隔 (1 = 全フレーム)。

    Returns:
        累積角度 [rad]。静止軌道なら 0.0。
    """


def gravity_direction_spread(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    q_trajectory: np.ndarray,
    body_name: str = "payload_box_mount",
    subsample: int = 1,
) -> float:
    """ボディフレームにおける重力方向ベクトルの最大開き角 [rad]。

    全フレームの重力方向ベクトルの平均方向を求め、
    各フレームの方向と平均方向の角度の最大値を返す。
    HANDOFF ドキュメントの「重力方向の振り角」に対応する指標。

    Returns:
        最大開き角 [rad]。静止軌道なら 0.0。
    """


def acceleration_peak(
    trajectory: TrajectorySample,
) -> float:
    """全関節・全タイムステップの加速度絶対値の最大値 [rad/s²]。"""


def trajectory_excitation_summary(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    trajectory: TrajectorySample,
    body_name: str = "payload_box_mount",
) -> dict[str, float]:
    """軌道の励起品質をまとめて返す。

    Returns:
        {
            "gravity_sweep_angle": ...,    # [rad]
            "gravity_direction_spread": ...,  # [rad]
            "acceleration_peak": ...,      # [rad/s²]
            "velocity_peak": ...,          # [rad/s]
            "position_range": ...,         # [rad] max - min across all joints
        }
    """
```

## 実装詳細

### gravity_sweep_angle

```python
def gravity_sweep_angle(model, data, q_trajectory, body_name, subsample=1):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    g_world = np.array([0.0, 0.0, -9.81])

    g_hat_prev = None
    total_angle = 0.0

    for i in range(0, len(q_trajectory), subsample):
        data.qpos[:] = q_trajectory[i]
        mujoco.mj_kinematics(model, data)
        R = data.xmat[body_id].reshape(3, 3)
        g_body = R.T @ g_world
        g_hat = g_body / np.linalg.norm(g_body)

        if g_hat_prev is not None:
            cos_angle = np.clip(g_hat @ g_hat_prev, -1.0, 1.0)
            total_angle += np.arccos(cos_angle)
        g_hat_prev = g_hat

    return total_angle
```

注意: `set_model_state` は `mj_kinematics + mj_comPos + mj_fwdVelocity` を呼ぶが,
ここでは回転行列だけが必要なので `mj_kinematics` のみで十分. ただし `set_model_state` を使っても
副作用は問題ない. パフォーマンスが問題になるなら `mj_kinematics` 直接呼び出しに切り替える.

### gravity_direction_spread

```python
def gravity_direction_spread(model, data, q_trajectory, body_name, subsample=1):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    g_world = np.array([0.0, 0.0, -9.81])

    directions = []
    for i in range(0, len(q_trajectory), subsample):
        data.qpos[:] = q_trajectory[i]
        mujoco.mj_kinematics(model, data)
        R = data.xmat[body_id].reshape(3, 3)
        g_body = R.T @ g_world
        directions.append(g_body / np.linalg.norm(g_body))

    if len(directions) < 2:
        return 0.0

    directions = np.array(directions)  # (N, 3)
    mean_dir = directions.mean(axis=0)
    mean_dir /= np.linalg.norm(mean_dir)

    cos_angles = np.clip(directions @ mean_dir, -1.0, 1.0)
    return float(np.max(np.arccos(cos_angles)))
```

### acceleration_peak

```python
def acceleration_peak(trajectory):
    return float(np.max(np.abs(trajectory.acceleration)))
```

### trajectory_excitation_summary

各メトリクスを呼び出してまとめるだけ.

## 検証

### 単体テスト

1. **静止軌道**: 全フレーム同一 q → `gravity_sweep_angle == 0.0`, `gravity_direction_spread == 0.0`
2. **加速度ゼロ**: 位置が等速直線的 → `acceleration_peak == 0.0`
3. **非ゼロケース**: MuJoCo モデルをロードし, q を 5 ステップ分ランダムに振って `gravity_sweep_angle > 0` を確認
4. **subsample**: `subsample=2` で `subsample=1` より小さい値 (粗い近似) になること
5. **summary**: 返り値が期待するキーをすべて含むこと

### 実行方法

```bash
python -m pytest tests/test_mpc_metrics.py -v
```

テストには MuJoCo モデルのロードが必要. テスト内で `load_and_reset("assets/ur5e/mjcf/scene_with_box.xml")` を呼ぶ.
