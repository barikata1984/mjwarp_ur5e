# 励起軌道 パイプライン実装・インフラ記録

## 2026-03-12: フェーズ 3-9 一括実装

### 実施内容

フェーズ 3 (制約) から フェーズ 9 (品質保証) までを一括実装した。

### 新規モジュール

| フェーズ | モジュール | 概要 |
|---|---|---|
| 3 | `constraints.py`, `workspace.py`, `collision.py` | joint/workspace/collision 制約、scipy constraints builder |
| 4 | `objective.py`, `optimizer.py`, `io.py` | 条件数目的関数、multi-start SLSQP、JSON I/O |
| 5 | `demos/optimize_excitation_trajectory.py`, `demos/validate_excitation_trajectory.py` | tyro CLI |
| 6 | `data_buffer.py`, `execution.py`, `demos/run_identification_playback.py` | 軌道再生、SensorSample 収集 |
| 7 | `estimators/{batch_ls,batch_tls,rtls,types}.py`, `demos/run_inertial_identification.py` | LS/TLS/RTLS 推定器 |
| 8 | `fourier_warp.py`, `regressor_warp.py` | Warp kernel による Fourier 軌道並列計算、batch 行列演算 |
| 9 | `test_integration.py`, `test_smoke_cli.py` | end-to-end + CLI smoke テスト |

### テスト結果

- 87 テスト全通過
- ruff lint / format クリーン

### コードレビュー (/simplify) による修正

- 重複 `_condition_number` 関数を `compute_condition_number` への委譲に統合
- `condition_number_objective` の未使用パラメータ 6 個を除去
- `except Exception` → `except (LinAlgError, ValueError)` で例外範囲を限定
- `workspace.py` の FK ループ重複を排除
- `collision.py` の冗長 `mj_kinematics` 呼び出し統合 + radii 事前計算
- `-> callable` → `-> Callable[[np.ndarray], float]` 型修正
- `model: object` → `model: mujoco.MjModel` 型修正
- キャッシュの `hash(bytes)` → バイト列直接比較で衝突リスク排除

## 2026-03-12: コードベースのモジュラー化・品質リファクタリング

### 目的

3エージェント並列レビュー（再利用・品質・効率）を実施し、スタッフエンジニア品質への再編を行った。

### 主な変更

#### YAML コンフィグシステム
- `configs/default.yaml` を新設: パイプライン全体のデフォルト設定を一元管理
- `cli/yaml_config.py` を新設: YAML ローダー + tyro CLI オーバーライドの統合

#### コンフィグ集約
- `cli/configs.py` に全デモコンフィグを集約: `ModelConfig` / `ResultInputConfig` ベースクラスによる継承階層
- 5つのデモスクリプトからインラインコンフィグを削除

#### 共通ユーティリティ抽出
- `model.py` に `load_and_reset()` ヘルパーを追加 (5デモの3行ボイラープレートを解消)
- `optimizer.py` に `_build_cache_and_constraints()` を抽出 (`optimize()` / `validate_trajectory()` の重複排除)
- `workspace.py` に `_compute_box_margin()` を抽出 (box/payload 制約の重複排除)

#### 型安全性の改善
- `constraints.py`: `object | None` → `WorkspaceConstraintConfig | None` / `CollisionConfig | None`
- `estimators/batch_ls.py`, `batch_tls.py`: 冗長な `_condition_number()` ラッパー関数を削除

#### ホットパス最適化
- `trajectories/base.py`: `time` プロパティを copy → read-only view に変更 (最適化中の1000+ 配列コピーを解消)
- `workspace.py`: 制約クロージャ内の `np.asarray()` を factory スコープにホイスト

### 検証結果

- 92/92 テストパス
- ruff lint / format クリーン

---

## 2026-03-12: 最適軌道のサンプリング済みJSON出力

### 目的

最適化結果のフーリエ係数JSONでは、軌道再生にフーリエ軌道生成コードが必要となる。
ROSノード等の外部システムから直接再生可能な、各タイムステップの関節状態を記述した
自己完結型JSONを出力する機能を追加した。

### 主な変更

#### `save_trajectory_json()` — `identification/io.py`
- `TrajectorySample` を受け取り、各ステップの `t`, `q`, `dq`, `ddq` をJSON出力
- メタデータに `fps`, `dt`, `joint_names`, `condition_number` を記録
- ROS側は `metadata.dt` の制御周期で `trajectory[i].q` を順次送信するだけで再生可能

#### `result_to_trajectory()` のfpsオーバーライド対応
- `fps` 引数を追加。フーリエ表現から任意のレートで解析的に再サンプリング可能

#### `export_trajectory` CLIコマンド — `demos/export_trajectory.py`
- 既存の最適化結果JSONから軌道JSONへの変換コマンド
- `--fps` フラグで出力サンプリングレートを指定可能

#### 最適化完了時の自動出力
- `optimize_excitation_trajectory.py` の完了時に軌道JSONを自動生成
- `--trajectory-fps`, `--trajectory-output` フラグで制御

#### 出力ディレクトリ
- `results/` ディレクトリを新設し、軌道JSON出力を格納

### 出力JSONの構造

```json
{
  "metadata": {"fps": 100.0, "dt": 0.01, "joint_names": [...], ...},
  "trajectory": [
    {"t": 0.0, "q": [...], "dq": [...], "ddq": [...]},
    ...
  ]
}
```

### 検証結果

- 92/92 テストパス
- ruff lint / format クリーン
- 100Hz (101 steps) および 50Hz (51 steps) での出力を確認

---

## 2026-03-12: デフォルト出力パスの統一と既存結果の調査

### 出力パス統一

全パイプラインのデフォルト出力先を `debug/` → `results/` に変更。

- `configs/default.yaml` の output セクション
- `cli/configs.py` の全 Config クラスのデフォルト値
- `tests/test_cli_excitation.py` のアサーション
- 92/92 テストパス

### 既存 excitation_result.json の調査

`debug/excitation_result.json` はテスト用短縮設定で生成されていたことを確認:

| パラメータ | 既存結果 | default.yaml |
|---|---|---|
| duration | 1.0 秒 | 10.0 秒 |
| num_harmonics | 2 | 5 |
| n_monte_carlo | 1 | 20 |
| max_iter_per_start | 5 | 200 |

レンダリング動画（5秒）は `playback_speed=0.2` で引き伸ばしたもの。
実時間では1秒の軌道であり、デフォルト設定での本番再最適化が必要。

---

## 2026-03-13: wandb 実験追跡・アーリーストップ実装

### 目的

長時間の最適化ランのモニタリングと不要な計算の早期打ち切りを実現する。

### 実装内容

#### `WandbConfig` / `EarlyStopConfig` — `optimizer.py`

```python
@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "ur5e-excitation"
    run_name: str | None = None
    tags: list[str] = field(default_factory=list)

@dataclass
class EarlyStopConfig:
    enabled: bool = False
    patience: int = 5
    min_improvement: float = 1e-3
```

#### wandb メトリクス

| レベル | メトリクス |
|---|---|
| per-iteration | `iter/condition_number`, `iter/restart_index`, `iter/iter_in_restart`, `iter/wall_time` |
| per-restart | `restart/condition_number`, `restart/global_best_cond`, `restart/constraint_margin_min`, `restart/feasible`, `restart/n_func_evals`, `restart/n_iters`, `restart/wall_time_s`, `restart/improved`, `restart/index` |
| final summary | `final/condition_number`, `final/best_restart_index`, `final/total_restarts`, `final/total_func_evals`, `final/wall_time_s` |

#### per-iteration ロギングの工夫

scipy `minimize` の `callback` は `xk` のみ渡すため、目的関数クロージャ内で `_latest_cond` に最新値を保存し、callback 内でそれを読み取る設計とした。
これにより目的関数の再評価を回避している。

#### CLI 統合 — `cli/configs.py`, `demos/optimize_excitation_trajectory.py`

`OptimizeExcitationConfig` に `--wandb`, `--wandb-project`, `--wandb-run-name`, `--early-stop`, `--early-stop-patience` フラグを追加。

### 変更ファイル

- `src/mjwarp_ur5e/identification/optimizer.py`: `WandbConfig`, `EarlyStopConfig`, `_config_to_wandb_dict()` 追加、`optimize()` の引数拡張
- `src/mjwarp_ur5e/cli/configs.py`: CLI フラグ追加
- `src/mjwarp_ur5e/demos/optimize_excitation_trajectory.py`: wandb/early_stop config のワイヤリング

---

## 2026-03-13: 並列 Monte Carlo 最適化の設計

### 動機

Config D でも ~3.8h の計算時間がかかる。マルチコア並列化により線形に短縮可能。

### アーキテクチャ設計

```
ProcessPoolExecutor (N workers)
├── Worker 0: load model → run restart 0, N, 2N, ...
├── Worker 1: load model → run restart 1, N+1, 2N+1, ...
├── ...
└── Worker N-1
```

#### MuJoCo スレッド安全性

- `MjModel`: 読み取り専用 → プロセス間共有可能（ただし ProcessPool では各 worker がロード）
- `MjData`: ミュータブル → **各 worker が独自に `mj_makeData` で生成**

#### 設計ポイント

1. 各 worker は独立に `MjModel`/`MjData` をロードし、割り当てられた restart をシーケンシャルに実行
2. `concurrent.futures.as_completed()` で完了順に結果を収集
3. メインプロセスが wandb ロギングと early stopping 判定を担当
4. `OptimizationResult` は pickle 可能（numpy 配列 + dataclass）
5. 推定スピードアップ: 4 workers で ~4x（CPU バウンド）

### 未実装

コード実装は未着手。設計のみ完了。

---

## 2026-03-23: 並列 Monte Carlo 最適化の実装

### 概要

2026-03-13 に設計した `ProcessPoolExecutor` ベースの並列 Monte Carlo 最適化を実装。

### 実装内容

| 項目 | 内容 |
|---|---|
| 並列化手法 | `concurrent.futures.ProcessPoolExecutor` |
| Worker 関数 | `_run_single_restart()` (module-level, pickle 可能) |
| MuJoCo 安全性 | 各 worker が `load_and_reset(model_path)` で独自 MjModel/MjData を生成 |
| 結果収集 | `as_completed()` で完了順に収集 |
| Early stopping | 到着順に判定、`future.cancel()` で残りを中止 (best-effort) |
| RNG 再現性 | 全 x0 をメインプロセスで事前生成 → n_workers に依存しない |
| wandb 対応 | 並列モードでは restart-level ログのみ (per-iteration は逐次のみ) |

### CLI パラメータ

```
--n-workers INT  (default: 1, 1=逐次)
```

`n_workers=1` では従来と完全に同一の逐次実行パスを通るため後方互換性を維持。

### ベンチマーク結果

設定: harmonics=3, duration=3s, max_iter=10, n_monte_carlo=4, collision/payload OFF

| モード | workers | wall time | speedup |
|--------|---------|-----------|---------|
| 逐次 | 1 | 51.5s | 1.0x |
| 並列 | 4 | 13.9s | **3.7x** |

ほぼ理想的な線形スケーリング。両モードで同一の最適解 (cond=12.5976, best_start_index=1) を確認。

### 変更ファイル

| ファイル | 変更内容 |
|---|---|
| `optimizer.py` | `_RestartResult`, `_run_single_restart()`, `_optimize_parallel()`, `_build_final_result()` 追加。`optimize()` を sequential/parallel に分岐 |
| `configs.py` | `OptimizeExcitationConfig` に `n_workers` 追加 |
| `optimize_excitation_trajectory.py` | `n_workers`, `model_path` の受け渡し |

### 追記: 並列モードの wandb per-iteration ログ対応

初期実装では並列モードで `iter/*` メトリクスと `restart/n_iters` が欠落していた。
一括ログ方式で対応: worker 内の callback が iteration メトリクスをリストに蓄積し、restart 完了時にメインプロセスがまとめて wandb に送信。リアルタイム性は失われるが、記録されるデータは逐次モードと同一。

### 追記: wandb ログを restart ごとの独立 run に変更

全 restart を 1 つの wandb run に混在させていたのを、各 restart を独立した run + `group` でグルーピングする構造に変更。逐次・並列の両モードに適用。

```
group: "<run_name>"
├── restart-0  (iter step 1..N + summary)
├── restart-1
├── ...
└── summary    (final/* metrics)
```

各 restart の `condition_number` 収束曲線を wandb ダッシュボード上で個別に比較可能になった。

---

## 2026-03-23: YAML-tyro 統合と設定リファクタリング

### YAML ↔ tyro CLI の統合

`yaml_config.py` に `load_config()` を実装し、全9デモスクリプトで `tyro.cli(Config)` → `load_config(Config)` に統一。
`configs/default.yaml` がベースデフォルト → tyro CLI フラグがオーバーライドする2段構成が実際に機能するようになった。

YAML の値と dataclass フィールドの型が不一致（例: リスト vs スカラー）の場合は型チェックでスキップする安全策を追加。

### 設定デフォルトの変更

| パラメータ | 旧値 | 新値 | 理由 |
|---|---|---|---|
| `num_harmonics` | 3 | 5 | より高い周波数成分で励起品質向上 |
| `base_freq` | 0.3333 | 0.1 | 基本周期を長くする |
| `duration` | 3.0 | 5.0 | 軌道長の拡大 |
| `subsample_factor` | 1 | 5 | 速度と精度のバランス |
| `objective` | `d_optimal` | `condition_number` | 条件数を直接最小化 |
| `max_displacement` | 0.5 | 0.0 | デフォルト無効化 |
| `model` | `""` | `assets/ur5e/mjcf/scene_with_box.xml` | 暗黙フォールバックを明示化 |

### 制約セクションのリファクタリング

- `ddq_max: float = 0.0` を新規追加（`0.0` で無効化、`> 0` で有効化）
- `enable_acc_constraint` を削除（`ddq_max > 0` で代替）
- `use_fourier_bounds` は `dq_max > 0` or `ddq_max > 0` の場合のみ有効に

### リネーム

- `include_ft_offset` → `with_ft_offset`（全ソースファイル一括置換）

### subsample_factor ベンチマーク

n_monte_carlo=3, max_iter=20, harmonics=5, duration=5s で実測:

| subsample_factor | wall time | 条件数 |
|---|---|---|
| 1 | 691.7s | — |
| 5 | 540.8s | — |
| 10 | 523.2s | — |

n_monte_carlo=3, max_iter=100 で追加計測:

| subsample_factor | wall time | 最終条件数 |
|---|---|---|
| 5 | 2151s | 10.21 |
| 10 | 2436s | 10.70 |

sf=10 は間引きすぎで勾配精度が低下し、収束が遅れ逆に遅くなる結果。sf=5 が精度と速度のバランスが最良。

### base_freq 変更

Kubus et al. (2008) が最高周波数を 2Hz に制限していたことを参考に、base_freq を 0.1 → 0.2 に変更。
最高周波数 = harmonics(5) × base_freq(0.2) = 1.0Hz で、2Hz 制限に対して十分な余裕。

### 本番最適化ラン（FT オフセットなし、20 restarts × 200 iter）

n_workers=6 並列、base_freq=0.2、duration=5s、harmonics=5 で実施:

| dq_max | 条件数 | 速度違反 | workspace違反 | feasible |
|---|---|---|---|---|
| 1.5 rad/s | **1.93** | 0.0014 | 0.0006 | No（僅差）|
| 1.0 rad/s | **2.20** | 0.0019 | 0.0002 | No（僅差）|

条件数は非常に良好。制約違反は極めて小さく（速度 0.1%、workspace 0.6mm 以下）、実用上ほぼ feasible。

### 列スケーリングの数理的背景ノート

FT オフセット拡張時の列スケーリングに関する数理的背景を `docs/notes/column_scaling_background.md` にまとめた。
Golub & Van Loan (2013) の列均衡化、Swevers et al. (1997)、Gautier & Khalil (1992) のロボット同定での適用、
Pukelsheim (1993) の D-optimal 設計における知見を整理。

### FT オフセットあり最適化ラン（実行中）

n_workers=4、dq_max=1.0、with_ft_offset + column_scale で ddq_max を 3 条件で比較実行中:
