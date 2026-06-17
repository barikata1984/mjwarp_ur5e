# Step 7: CLI デモ

## 作成/変更ファイル

- `src/mjwarp_ur5e/demos/run_mpc_identification.py` (新規)
- `src/mjwarp_ur5e/cli/configs.py` (MPCIdentificationConfig 追加)

## 依存関係

Step 5 (MPCLoop), Step 6 (exports)

## 既存インタフェース (変更不可)

```python
# cli/yaml_config.py
def load_config(config_class: type[T]) -> T:
    """tyro で CLI 引数をパースし、YAML defaults があれば読む。"""

# cli/configs.py
@dataclass(slots=True)
class ModelConfig:
    model: str = "assets/ur5e/mjcf/scene_with_box.xml"

# model.py
def load_and_reset(model_path: str | None = None) -> LoadedModel: ...

# identification/regressor.py
def body_inertial_parameters_from_model(model, body_name) -> InertialParameters: ...

# identification/types.py
@dataclass(frozen=True)
class InertialParameters:
    mass: float
    first_moments: np.ndarray     # (3,)
    inertia_matrix: np.ndarray    # (3, 3)
    def to_vector(self) -> np.ndarray:  # (10,) [m, hx, hy, hz, Ixx, Iyy, Izz, Ixy, Ixz, Iyz]

# identification/estimators/types.py
@dataclass
class EstimationResult:
    phi: np.ndarray     # (10,) same order as InertialParameters.to_vector()
    def to_dict(self) -> dict: ...
```

## 出力インタフェース

### cli/configs.py に追加

```python
@dataclass(slots=True)
class MPCIdentificationConfig(ModelConfig):
    """Config for MPC identification demo."""
    # Horizon
    horizon_duration: float = 3.0
    num_segments: int = 4
    fps: float = 100.0
    subsample_factor: int = 10
    # Planner
    n_restarts: int = 8
    max_iter: int = 100
    seed: int = 42
    waypoint_perturbation: float = 0.3
    # MPC loop
    replan_period: float = 1.5
    max_mpc_steps: int = 10
    convergence_threshold: float = 0.01
    # Robot
    body_name: str = "payload_box_mount"
    site_name: str = "attachment_site"
    # Execution
    use_pd_control: bool = False
    noise_std_wrench: float = 0.0
    # Output
    output: str = "results/mpc_identification_result.json"
```

### demos/run_mpc_identification.py

```python
def main() -> None: ...

if __name__ == "__main__":
    main()
```

## 実装詳細

### main() の流れ

既存の `run_inertial_identification.py` と `optimize_excitation_trajectory.py` のパターンに倣う.

```python
def main() -> None:
    config = load_config(MPCIdentificationConfig)
    loaded = load_and_reset(config.model)

    # Ground truth
    true_params = body_inertial_parameters_from_model(loaded.model, config.body_name)
    true_vec = true_params.to_vector()
    print(f"Ground truth: mass={true_params.mass:.4f} kg")

    # Build MPCConfig from CLI config
    mpc_config = MPCConfig(
        horizon=HorizonConfig(
            duration=config.horizon_duration,
            num_segments=config.num_segments,
            fps=config.fps,
            subsample_factor=config.subsample_factor,
        ),
        planner=PlannerConfig(
            n_restarts=config.n_restarts,
            max_iter_per_start=config.max_iter,
            seed=config.seed,
            waypoint_perturbation=config.waypoint_perturbation,
        ),
        replan_period=config.replan_period,
        max_mpc_steps=config.max_mpc_steps,
        convergence_threshold=config.convergence_threshold,
        body_name=config.body_name,
        site_name=config.site_name,
        model_path=config.model,
        use_pd_control=config.use_pd_control,
        noise_std_wrench=config.noise_std_wrench,
    )

    # Run MPC
    mpc = MPCLoop(mpc_config, loaded.model, loaded.data)
    result = mpc.run()

    # Print per-step summary
    print(f"\n{'Step':>4} {'Cond':>10} {'Mass':>8} {'Samples':>8} {'Plan[s]':>8} {'Exec[s]':>8}")
    print("-" * 56)
    for s in result.steps:
        print(f"{s.step:>4d} {s.condition_number_accumulated:>10.2f} "
              f"{s.estimation.mass:>8.4f} {s.n_samples_total:>8d} "
              f"{s.wall_time_plan:>8.1f} {s.wall_time_execute:>8.1f}")

    # Print final comparison
    est = result.final_estimation
    print(f"\nConverged: {result.converged}")
    print(f"Total time: {result.total_wall_time:.1f}s")
    print(f"\n{'Param':>8} {'True':>12} {'Estimated':>12} {'Error':>12} {'Rel%':>8}")
    print("-" * 56)
    param_names = ["m", "hx", "hy", "hz", "Ixx", "Iyy", "Izz", "Ixy", "Ixz", "Iyz"]
    for i, name in enumerate(param_names):
        t, e = true_vec[i], est.phi[i]
        err = e - t
        rel = err / t * 100 if abs(t) > 1e-10 else float("nan")
        print(f"{name:>8} {t:>12.6f} {e:>12.6f} {err:>+12.6f} {rel:>+7.1f}%")

    # Save JSON
    _save_result(result, true_vec, config.output)


def _save_result(result: MPCResult, true_vec: np.ndarray, path: str) -> None:
    import json
    from pathlib import Path

    out = {
        "converged": result.converged,
        "total_wall_time": result.total_wall_time,
        "total_samples": result.total_samples,
        "final_condition_number": result.final_condition_number,
        "true_params": true_vec.tolist(),
        "estimated_params": result.final_estimation.phi.tolist(),
        "steps": [
            {
                "step": s.step,
                "condition_number": s.condition_number_accumulated,
                "mass": s.estimation.mass,
                "n_samples": s.n_samples_total,
                "wall_time_plan": s.wall_time_plan,
                "wall_time_execute": s.wall_time_execute,
            }
            for s in result.steps
        ],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {path}")
```

### pyproject.toml への追加は不要

エントリポイントは `python -m mjwarp_ur5e.demos.run_mpc_identification` で実行.
既存デモ (`run_inertial_identification.py` 等) も同じパターン.

## 検証

### 実行テスト

```bash
# 最小パラメータで実行 (速度優先)
python -m mjwarp_ur5e.demos.run_mpc_identification \
    --max-mpc-steps 2 \
    --n-restarts 2 \
    --max-iter 10

# 確認事項:
# 1. エラーなく完了する
# 2. per-step テーブルが表示される
# 3. final comparison テーブルが表示される
# 4. results/mpc_identification_result.json が生成される
# 5. --help でオプション一覧が表示される
```

```bash
python -m mjwarp_ur5e.demos.run_mpc_identification --help
```
