# Step 6: パッケージエクスポート

## 変更ファイル

- `src/mjwarp_ur5e/identification/mpc/__init__.py` (Step 2 で作成済み → 更新)
- `src/mjwarp_ur5e/trajectories/__init__.py` (既存 → QuinticSpline 追加)

## 依存関係

Step 1, 2, 3, 4, 5 すべて

## 実装

### `identification/mpc/__init__.py`

```python
from .config import HorizonConfig, MPCConfig, PlannerConfig
from .loop import MPCLoop, MPCResult, MPCStepLog
from .metrics import (
    acceleration_peak,
    gravity_direction_spread,
    gravity_sweep_angle,
    trajectory_excitation_summary,
)
from .planner import ExcitationPlanner, PlanResult

__all__ = [
    "ExcitationPlanner",
    "HorizonConfig",
    "MPCConfig",
    "MPCLoop",
    "MPCResult",
    "MPCStepLog",
    "PlanResult",
    "PlannerConfig",
    "acceleration_peak",
    "gravity_direction_spread",
    "gravity_sweep_angle",
    "trajectory_excitation_summary",
]
```

### `trajectories/__init__.py`

既存の末尾に追加:

```python
from .quintic_spline import QuinticSplineConfig, QuinticSplineTrajectory

# __all__ に追加:
"QuinticSplineConfig",
"QuinticSplineTrajectory",
```

既存の `__all__` リストはアルファベット順. `QuinticSplineConfig` は `HAS_WARP` と `TrajectorySample` の間, `QuinticSplineTrajectory` は `QuinticSplineConfig` の次に挿入する.

## 検証

```bash
python -c "from mjwarp_ur5e.identification.mpc import MPCLoop, MPCConfig, ExcitationPlanner"
python -c "from mjwarp_ur5e.trajectories import QuinticSplineTrajectory, QuinticSplineConfig"
python -c "import mjwarp_ur5e"  # エラーなし
```
