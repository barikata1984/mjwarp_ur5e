# 慣性パラメータ同定・最適励起 実験記録

## 2026-03-12: regressor rank 9 問題の調査と修正

### 問題

`payload_box_mount` の stacked regressor が動的軌道でも rank 9 止まりで、10 パラメータすべてが独立に励起されなかった。

### 原因

`set_model_state` が `mj_forward` を呼んでおり、2 つの問題を引き起こしていた:

1. ユーザ指定の `qacc` (関節加速度) が MuJoCo の順動力学で上書きされていた
2. MuJoCo 3.6 のキネマティクスパイプラインでは `data.cacc` (body Cartesian 加速度) が未計算のため、`mj_objectAcceleration` が返す angular acceleration が常にゼロだった

null space 解析により、欠落方向は慣性テンソルのトレース `Ixx + Iyy + Izz` であることを特定。トレース方向の回帰列は `[alpha_body, 0]` に等しく、`alpha_body = 0` なので常にゼロ列となっていた。

### 修正

- `set_model_state`: `mj_forward` → `mj_kinematics` + `mj_comPos` + `mj_fwdVelocity`
- `sample_body_kinematics`: `mj_jacBody` で body Jacobian を取得し、`J @ qacc` で body frame の加速度を補正

### 結果

| 指標 | 修正前 | 修正後 |
|------|--------|--------|
| rank | 9 | **10** |
| 条件数 | inf | **~5** |

base parameter 化は不要であることを確認。

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

## 2026-03-12: 最適励起軌道の生成と MuJoCo 再生動画

### 実施内容

実装したパイプライン全体を通して最適励起軌道を生成し、MuJoCo 上で再生した動画を出力した。

### 最適化設定

- harmonics=3, duration=5.0s, fps=50
- 3 Monte Carlo restarts, 50 iter/start
- workspace constraint: max_displacement=0.5m
- collision constraint: enabled

### 結果

| 指標 | 値 |
|------|------|
| 条件数 | **2.64** |
| 計算時間 | 110.8s |
| 評価回数 | 5791 |
| best start | 1 |

### 出力

- `debug/excitation_result.json` — 最適化結果 (係数、config)
- `debug/excitation_playback.mp4` — MuJoCo 再生動画 (5s, 30fps, 150 frames)

### 追加スクリプト

- `demos/render_excitation_playback.py` — 最適化結果 JSON → MuJoCo レンダリング → mp4 動画生成 CLI

## 2026-03-12: ペイロード ワークスペース制約と多視点カメラ

### 実施内容

1. ペイロード箱の8頂点がワークスペース立方体内に収まる制約を追加
2. 3軸カメラ (`view_x`, `view_y`, `view_z`) を追加し 2x2 グリッド動画を生成
3. ワークスペース X 範囲をベース座標系で EE ホーム基準 -25cm ~ +45cm に設定

### バグ修正

- `model.body_pos` (親ローカル座標) を使っていたため制約が効かなかった → `data.xpos` (ワールド座標) に修正
- body 原点のみの点チェック → geom 8頂点のワールド座標変換によるチェックに修正

### 最適化結果 (頂点ベース制約)

| 指標 | 値 |
|------|------|
| 条件数 | **3.03** |
| 計算時間 | 121.4s |
| 評価回数 | 5859 |
| 最小マージン | 5.8mm |

### 新規・変更モジュール

- `workspace.py`: `_box_vertices`, `_evaluate_payload_vertices`, `make_payload_workspace_constraint`
- `constraints.py`: `build_scipy_constraints` に `payload_workspace_config` パラメータ追加
- `optimizer.py`: `OptimizerConfig.payload_workspace_config` 追加
- `collision.py`: `CollisionConfig.payload_half_extents` / `payload_offset` (既存、次セッションで頂点ベースに改修予定)
- `scene_with_box.xml`: `view_x`, `view_y`, `view_z` カメラ追加
- `render_excitation_playback.py`: `--multi-camera`, `--save-frames` オプション追加

### 残課題

- ~~ペイロード外形に基づくロボット-ペイロード衝突判定 (現在は body 原点の点判定)~~ → 解決済み

## 2026-03-12: サーフェスベースのペイロード衝突判定

### 問題

`_check_payload_collision` がペイロードを body 原点の1点として扱っており、箱の面・辺がリンク球に食い込むケースを見逃していた。

### 修正

OBB (Oriented Bounding Box) と球の解析的最短距離を実装:

1. `_box_sphere_clearance()`: 球の中心をボックスのローカル座標系に変換し、各軸をクランプして最近接点を求め、サーフェス間距離を返す
2. `_check_payload_collision()`: 各非隣接リンク球に対して box-sphere 距離を計算
3. `_check_payload_ground_clearance()`: 8頂点の最小 z で地面クリアランスを判定（凸体に対して厳密）

### アルゴリズム

```python
local = box_rot.T @ (sphere_center - box_center)
closest = np.clip(local, -half_extents, half_extents)
clearance = ||local - closest|| - sphere_radius
```

面・辺・頂点すべてのケースを正しく処理する。

### テスト追加 (初版)

- `test_collision_box_sphere_clearance_unit`: 外部・内部・角の3ケースを検証
- `test_collision_large_payload_detected_by_surface`: 巨大ペイロードで面衝突を検出
- `test_collision_payload_ground_clearance`: ペイロードサイズによる地面クリアランス差を検証

### 初版の問題

- CollisionConfig のデフォルト値がハードコードされており MJCF の実ジオメトリと不一致
  - `payload_half_extents`: [0.05, 0.075, 0.10] → 実値 **[0.125, 0.125, 0.125]**
  - `payload_offset`: [0, 0, 0.10] → 実値 **[0, -0.1, 0.125]**
- リンクを body 原点中心の球で近似 → 細長いカプセルとの乖離が大きく衝突を見逃す

## 2026-03-12: Box-Capsule 衝突判定への改修

### 修正

1. **ペイロードジオメトリの自動抽出**: MuJoCo モデルの `geom_size`/`geom_pos` から box geom の half_extents と offset を読み取り
2. **リンクカプセルの自動抽出**: 各リンクの capsule/cylinder geom の端点・半径を body ローカル座標で保持
3. **Box-Capsule 距離**: 交互投影法 (`_segment_aabb_distance`) でボックスローカル座標のセグメント-AABB 距離を計算し、カプセル半径を減算
4. **自己衝突**: sphere-sphere のまま (UR5e 用チューニング済みデフォルト radii を維持)

### CollisionConfig の変更

| フィールド | 旧 | 新 |
|---|---|---|
| `link_radii` | `list[float]` (ハードコード) | 削除 |
| `self_collision_radii` | — | `list[float]` (UR5e デフォルト) |
| `payload_half_extents` | `list[float]` (ハードコード) | `list[float] \| None` (None=モデル自動) |
| `payload_offset` | `list[float]` (ハードコード) | `list[float] \| None` (None=モデル自動) |
| `safety_margin` | 0.005 | **0.02** |

### テスト結果

- 92 テスト全通過 (2件追加: box-capsule unit, auto-extract geometry)
- ruff lint / format クリーン

### 最適化結果 (Box-Capsule 衝突制約, safety_margin=5cm)

| 指標 | 値 |
|------|------|
| 条件数 | **3.02** |
| 計算時間 | 754.3s |
| 評価回数 | 5855 |
| best start | 1 |
| 最小クリアランス（制約値） | 5.5mm (step 39) |
| 実際のサーフェス間距離 | **55.5mm** |

### 出力

- `debug/excitation_result.json` — 最適化結果
- `debug/excitation_playback.mp4` — 多視点 MuJoCo 再生動画 (5s, 30fps, 4カメラ 2x2)

---

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

## 2026-03-13: Config D による本番最適化ラン

### 計算コスト分析

軽量テストラン (harmonics=2, duration=4s, subsample=10, mc=2, mi=5) の実測値をもとに、各設定の推定計算時間を算出した。

| 設定 | harmonics | duration | subsample | mc | mi | 推定時間 |
|---|---|---|---|---|---|---|
| A (default) | 5 | 10s | 10 | 20 | 200 | **9.2h** |
| B | 5 | 10s | 10 | 10 | 100 | 4.6h |
| C | 5 | 10s | 20 | 20 | 200 | 4.6h |
| **D** | **5** | **10s** | **20** | **10** | **100** | **2.3h** |
| E | 3 | 10s | 10 | 10 | 100 | 2.4h |
| F | 5 | 5s | 10 | 10 | 100 | 2.3h |

Config A は最初に実行を開始したが 6 時間超経過しても終了せず停止。
Config D を早期停止付きで実行した。

### 最適化結果 (Config D + early stop)

| 指標 | 値 |
|---|---|
| 条件数 | **2.5690** |
| 計算時間 | 13789.5s (~3.8h) |
| 評価回数 | 58,178 |
| 完了 restart | 9/10 (patience=5 で早期停止) |
| best start index | 3 |
| **feasible** | **全 restart で False** |

wandb ラン: `balanced-d` (project: `ur5e-excitation`)

### 制約違反の分析

全 restart が infeasible (最小制約マージン < 0) であった。
restart 1 の margin=-0.0125 が最も 0 に近いが、それでも制約を満たしていない。

原因: `subsample_factor=20` により、制約は 1001 timestep 中 51 点でしか評価されない。
サンプル間の制約違反を optimizer が検知できず、infeasible な解に収束する。

---

## 2026-03-13: Kubus et al. (2008) 論文調査

### 論文概要

"On-Line Estimation of Inertial Parameters Using a Recursive Total Least-Squares Approach" (IROS 2008)

### 軌道パラメータ

| パラメータ | 値 |
|---|---|
| 軌道持続時間 | **1.5s** |
| 最大周波数 | **2 Hz** |
| Fourier 高調波数 | 3 |
| サンプリング周波数 | 250 Hz |
| 条件数 | **7–8** |

### 本プロジェクトとの比較

| | Kubus et al. | 本プロジェクト |
|---|---|---|
| 持続時間 | 1.5s | 10.0s |
| 条件数 | 7–8 | 2.57 |
| 高調波 | 3 | 5 |
| 実行可能性 | 制約充足 | **全 infeasible** |

条件数は κ=7–8 で十分とされる。本プロジェクトの κ=2.57 は数値的に優れているが、
制約を満たさない軌道は実機で実行不可能であり、feasibility の改善が最優先事項。

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

## 2026-03-13: infeasibility 根本原因の再調査と最適化設定の改善

### 根本原因の訂正

前回セッションで「`subsample_factor=20` により制約が 51 点でしか評価されない」と記載したが、
コード精査により **制約関数は `subsample_factor` の影響を受けず、常に全 timestep で評価されている** ことを確認。
`subsample_factor` は目的関数（条件数計算）にのみ適用される。

真の原因は以下の複合要因:
1. **duration=10s が過剰**: timestep 数が線形に増え制約評価コストが膨大
2. **60 変数 (6 joints × 5 harmonics × 2)** の広い探索空間で SLSQP が feasible 領域を見つけられない
3. FK ベース制約 (workspace/payload/collision) × 全 timestep × 37 回/iteration (有限差分) が計算コストの ~97%

### 実装した改善

#### 1. duration 短縮 + harmonics 削減
- duration: 10.0s → **3.0s** (Kubus et al. は 1.5s で κ=7-8 を達成)
- harmonics: 5 → **3** (決定変数: 60 → 36)
- base_freq: 0.1 → **1/3 Hz**
- subsample_factor: 10 → **1** (目的関数も全点評価)

#### 2. EE 線速度制約の追加
- `EeVelocityConfig(max_linear_velocity=0.25)` — 0.25 m/s 上限
- site Jacobian × joint velocity で各 timestep の EE 線速度を計算
- `workspace.py` に `_evaluate_ee_linear_velocity()`, `make_ee_velocity_constraint()` を追加

#### 3. 関節速度上限の CLI 設定
- `--dq-max 0.0873` (≈5 deg/s) で全関節の速度上限を一律設定
- 人が近傍にいる環境での安全性を考慮した値

#### 4. 目標条件数 early stop
- `EarlyStopConfig.target_cond` を追加
- **現在の restart が feasible かつ κ ≤ target_cond の場合のみ** 停止
- infeasible な解での早期停止を防止

### 変更ファイル

| ファイル | 変更内容 |
|---|---|
| `workspace.py` | `EeVelocityConfig`, EE 速度評価・制約関数 |
| `constraints.py` | `build_scipy_constraints` に `ee_velocity_config`, `site_name` 引数追加 |
| `optimizer.py` | `target_cond`, `ee_velocity_config` 対応、early stop の feasibility チェック |
| `configs.py` | 新デフォルト値、`ee_max_linear_velocity`, `dq_max`, `early_stop_target_cond` |
| `optimize_excitation_trajectory.py` | 新制約のワイヤリング、`dq_max` による JointLimits 上書き |
| `default.yaml` | 新デフォルト設定 |
| `test_cli_excitation.py` | デフォルト値アサーションの更新 |

### 最適化実行

新設定 (duration=3s, harmonics=3, dq_max=5deg/s, EE vel≤0.25m/s, target κ≤5) で
最適化を開始したが、30 分以上経過しても最初の restart が完了せず、セッション時間内に結果を得られなかった。

### 残存課題

1. **1 iteration のプロファイリング**: 実測なしに推測で最適化するのは非効率
2. **制約の段階的評価**: 安い制約（joint limits, 純 NumPy）を先に評価し、違反なら FK 制約をスキップ
3. **解析的 velocity/acceleration 上界**: Fourier 係数の三角不等式で FK ループを排除
4. **並列 Monte Carlo**: restart 数を増やして feasible 解の発見確率を向上

---

## 2026-03-13: 1 iteration プロファイリングと診断ラン

### プロファイリング結果

`scripts/profile_optimizer.py` を作成し、各コンポーネントの所要時間を実測。

| コンポーネント | 1 回 | 1 iter (×37 FD) | 割合 |
|---|---|---|---|
| collision constraint | 67.2 ms | 2.49 s | 68.5% |
| objective (cond number) | 28.2 ms | 1.04 s | 28.8% |
| payload workspace | 1.05 ms | 39 ms | 1.1% |
| EE velocity | 1.01 ms | 37 ms | 1.0% |
| workspace | 0.56 ms | 21 ms | 0.6% |
| NumPy constraints (3つ) | 0.04 ms | 1.5 ms | ~0% |

- **1 SLSQP iteration = 3.6s** (プロファイルと実測が一致)
- collision constraint が全体の ~69% — Box-Capsule 距離計算が支配的
- EE velocity / dq_max 制約を除外しても 3.56s/iter — これらはボトルネックではない

### 診断ラン (constrained vs unconstrained 並列比較)

2 つの最適化を tmux で並列実行:
- **constrained**: dq_max=5°/s, EE vel≤0.25m/s, 全制約あり
- **unconstrained**: dq_max/EE vel 制約なし、workspace/collision/payload 制約のみ

共通設定: n_monte_carlo=5, max_iter=100, early_stop=true, patience=3, target_cond=10.0

#### 結果

| | unconstrained | constrained |
|---|---|---|
| 条件数 | **2.11** | **2.74** |
| feasible | No (margin=-0.004) | No (全 restart infeasible) |
| 最大関節速度 | 125°/s | 51.6°/s |
| 最大 EE 線速度 | 86.3 cm/s | 54.2 cm/s |
| wall time | 1596s (27min, 4 restart) | 2015s (34min, 5 restart) |
| best restart | #0 | #4 |

wandb ラン: `constrained-diag`, `unconstrained-diag` (project: `ur5e-excitation`)

#### constrained 各 restart

| restart | cond | margin | feasible | time |
|---|---|---|---|---|
| 1 | 7.28 | -0.243 | No | 401s |
| 2 | 3.73 | -0.530 | No | 405s |
| 3 | 6.78 | -0.589 | No | 403s |
| 4 | — | — | — | — |
| 5 | **2.74** | — | No | — |

#### unconstrained 各 restart

| restart | cond | margin | feasible | time |
|---|---|---|---|---|
| 1 | **2.11** | -0.004 | No | 397s |
| 2 | 2.20 | -0.024 | No | 398s |
| 3 | 2.12 | -0.078 | No | 396s |
| 4 (patience stop) | 2.26 | — | No | — |

### 分析

1. **SLSQP は制約を無視しているのではない**: constrained で dq_max=5°/s を要求 → 125°/s (無制約) から 51.6°/s まで低下。制約を尊重しようとしているが 100 iteration では到達しない
2. **unconstrained でも全 restart infeasible**: collision/workspace 制約だけでも feasible 解を見つけられない (ただし margin=-0.004 と極めて僅差)
3. **SLSQP は本問題に不適切**: 有限差分ベースの局所勾配法では、36 変数の非凸空間で狭い feasible 領域を探索できない

### 結論: 最適化アルゴリズムの変更が最優先

SLSQP の問題:
- 有限差分で 36+1=37 回/iter の関数評価 (うち collision が 69%)
- 局所探索のみ、restart 間の情報共有なし
- 制約付き非凸問題で feasible 領域に到達困難

候補アルゴリズム:
1. **COBYLA**: 有限差分不要 → 1 iter あたり ~37 倍速い iteration
2. **Differential Evolution**: scipy 組込み、大域探索、制約対応
3. **CMA-ES + augmented Lagrangian**: 文献でも excitation optimization に使用実績

---

## 2026-03-13: 最適化アルゴリズムの文献調査と適切性評価

### 調査動機

SLSQP が feasible 解を返さない問題に対し、アルゴリズム選択の妥当性を文献に照らして評価した。

### 問題の性質の整理

| 項目 | 内容 |
|---|---|
| 決定変数 | フーリエ係数 a_{j,k}, b_{j,k} → 2 × 6 × N_h 個 (N_h=5 で 60, N_h=3 で 36) |
| 軌道表現 | q(t) = q0 + w(t) × Σ[a sin(kωt) + b cos(kωt)]、窓関数で境界条件自動満足 |
| 目的関数 | cond(W) = σ_max / σ_min（リグレッサ行列の条件数） |
| 制約 | 関節位置/速度/加速度、ワークスペース、衝突回避、EE 速度 — 非線形不等式制約 |
| 現行手法 | Multi-start SLSQP (有限差分勾配) |

### SLSQP の問題点（文献に基づく分析）

現行手法は [[Swevers1997_excitation]](../REFERENCES/MAIN.md#Swevers1997_excitation) の SQP アプローチを直接踏襲している。しかし以下の問題がある:

1. **条件数は非平滑**: σ_max/σ_min は特異値の「担い手」が入れ替わる点で微分不可能。有限差分勾配がこの kink 付近で不正確になる
2. **有限差分コスト**: 36 変数で 1 勾配あたり 37 回の関数評価。collision constraint が 69% を占めるため 1 iter = 3.6s
3. **局所最適**: 非凸問題に対しマルチスタートで対処しているが効率が悪い

### 推奨改善策

#### 1. 目的関数を D-optimal 基準に変更（最優先）

`minimize -log det(W^T W) = -2 Σ log(σ_i)`

- [[Calafiore2001_calibration]](../REFERENCES/MAIN.md#Calafiore2001_calibration) が実験的有効性を実証
- [[Lee2021_excitation_geometric]](../REFERENCES/MAIN.md#Lee2021_excitation_geometric) が理論的正当性を示す
- 条件数と異なり全特異値が正である限り C^∞ で滑らか → 有限差分勾配の精度が向上
- 条件数はバリデーション指標として報告すればよい

#### 2. ソルバの改善

- **短期**: 目的関数を D-optimal に変えるだけで SLSQP の収束改善が見込める
- **中期**: [[Tian2024_virtual_constraints]](../REFERENCES/MAIN.md#Tian2024_virtual_constraints) が示すように CasADi + IPOPT で解析的勾配を供給すれば大幅に高品質な解を短時間で得られる (同論文: cond=51/10min vs メメティック法 57890/60min超)
- **代替**: CMA-ES でグローバル探索 → SLSQP でローカル研磨のハイブリッド

#### 3. ドロップイン改善

- `log(cond(W))` への変更 — スケール安定化
- `method='COBYLA'` — 有限差分不要、非平滑目的に robust

### 参考文献

- [[Swevers1997_excitation]](../REFERENCES/MAIN.md#Swevers1997_excitation) — フーリエ級数 + SQP の原論文
- [[Lee2021_excitation_geometric]](../REFERENCES/MAIN.md#Lee2021_excitation_geometric) — 幾何学的基準・解析的勾配
- [[Tian2024_virtual_constraints]](../REFERENCES/MAIN.md#Tian2024_virtual_constraints) — グラミアン代理指標 + IPOPT
- [[Calafiore2001_calibration]](../REFERENCES/MAIN.md#Calafiore2001_calibration) — D-optimal 基準の実験的検証
- [[Kubus2008_rtls]](../REFERENCES/MAIN.md#Kubus2008_rtls) — 回帰行列構成の基礎文献

---

## 2026-03-13: D-optimal 目的関数の実装と診断ラン

### 実装内容

`objective.py` に D-optimal 目的関数を追加し、`optimizer.py` で `objective_type` による切替を実装した。

#### `d_optimal_objective()` — `objective.py`

- 目的関数値: `-2 * Σ log(σ_i)` （リグレッサの全特異値の対数和の符号反転）
- C^∞ で滑らか（条件数と異なり特異値交差点での kink がない）
- 特異値フロア `1e-30` で log(0) を回避

#### `d_optimal_with_cond()` — `objective.py`

- 1 回の SVD から D-optimal 値と条件数の両方を返すヘルパー
- `optimizer.py` が D-optimal で最適化しつつ条件数をバリデーション指標として報告するために使用

#### `optimizer.py` の変更

- `OptimizerConfig.objective_type`: `"d_optimal"` (デフォルト) / `"condition_number"`
- `optimize()` 内で `d_optimal_with_cond()` または `condition_number_objective()` を自動切替
- 条件数は `objective_type` に関わらず常にトラッキング

#### CLI — `configs.py`, `optimize_excitation_trajectory.py`

- `--objective` フラグ: `"d_optimal"` / `"condition_number"`
- デフォルトを `d_optimal` に変更

### 診断ラン結果: D-optimal vs condition_number

2 つの D-optimal 診断ランを tmux 並列実行（n_mc=5, max_iter=100, early_stop, patience=3）:

#### D-optimal 目的関数

| | unconstrained | constrained (dq≤5°/s, EE≤25cm/s) |
|---|---|---|
| 条件数 | **10.72** | **6.68** |
| D-opt 値 | -157.3 | -153.0 |
| feasible | No (margin=-27.0) | No (margin=-60.3) |
| wall time | 1992s (33min) | 1754s (29min) |
| restarts | 5/5 | 5/5 (patience stop) |

#### 比較: condition_number 目的関数（前回）

| | unconstrained | constrained |
|---|---|---|
| 条件数 | **2.11** | **2.74** |
| feasible | No (margin=-0.004) | No (全 infeasible) |

### 制約別違反分析

D-optimal 結果に対し、制約ごとのマージンを個別評価:

| 制約 | unconstrained margin | constrained margin |
|---|---|---|
| joint_position | +1.49 (OK) | +2.47 (OK) |
| joint_velocity | **-3.56** | **-5.72** |
| joint_acceleration | **-27.03** | **-60.29** |
| workspace_ee | **-0.04** | **-0.19** |
| collision | **-0.11** | **-0.11** |

### 根本原因の特定: 文献との差異分析

D-optimal 目的関数は文献（Calafiore 2001, Lee 2021）では成功しているが、本プロジェクトでは全 restart が infeasible。体系的に文献との差異を調査した結果、以下が判明:

#### 1. 制約の種類が根本的に異なる

| 制約 | 文献 (Swevers, Lee, Calafiore, Kubus, Tian, Rackl) | 本プロジェクト |
|---|---|---|
| 関節位置 | Yes | Yes |
| 関節速度 | Yes (一部) | Yes |
| 関節加速度 | Yes (一部) | Yes |
| **EE ワークスペース変位** | **No** | **Yes** (≤0.5m) |
| **衝突回避** | **No** | **Yes** (自己衝突+地面+ペイロード) |

調査した6本の論文のいずれも衝突回避やワークスペース変位制約を使用していない。

#### 2. D-optimal が係数を制限の33倍まで押し込む

D-optimal は情報量（全特異値の積）を最大化するため、大きな振幅を要求する。Fourier 第3高調波の加速度安全上限は係数 0.20 rad だが、最適化結果は 6.64 rad まで到達（33倍超過）。

#### 3. 根本原因の修正

前回セッションの結論「目的関数の非滑らかさが主因」は不正確であった。真の根本原因は:

> **探索空間が制約に対して広すぎる**。SLSQP はバウンド制約なしの広い空間で、文献にない衝突・ワークスペース制約の狭い feasible 領域を見つけられない。目的関数の滑らかさ改善は副次的効果に留まる。

### 対策の優先順位再編

| 順位 | 施策 | 理由 |
|---|---|---|
| **1** | 解析的 Fourier 係数バウンド（三角不等式） | 加速度違反が最大の違反源。バウンドで vel/acc 違反を構造的に排除 |
| **2** | 制約の段階的評価（安い制約で早期棄却） | collision/workspace の非凸制約評価回数を削減 |
| **3** | collision constraint の高速化 | 全体の 69% を占めるボトルネック |
| **4** | アルゴリズム変更 (COBYLA/IPOPT) | バウンド導入後に再評価。探索空間の縮小で SLSQP でも動く可能性 |
| **5** | 並列 Monte Carlo | 正解性ではなく速度の問題 |

D-optimal 目的関数は実装済みのため維持するが、feasibility 改善の主要因ではないことが判明した。

---

## 2026-03-13: 解析的 Fourier 係数バウンドの実装と最適化ラン

### 実装内容

三角不等式に基づく Fourier 係数のボックスバウンドを実装し、速度制約を構造的に保証する手法を導入した。

#### `compute_fourier_velocity_bounds()` — `constraints.py`

窓付き Fourier 軌道 `v_j(t) = w'(t)*osc_j(t) + w(t)*osc_j'(t)` に対し、三角不等式から十分条件:

```
Σ_k (|a_{j,k}| + |b_{j,k}|) * α_k ≤ dq_max_j
α_k = max|w'(t)/T| + 2π * base_freq * k
```

均等予算配分で per-coefficient バウンド: `|a_{j,k}|, |b_{j,k}| ≤ dq_max_j / (2 * N_h * α_k)`

- 窓関数 `w(s) = 64s³(1-s)³` の導関数最大値を 10,000 点グリッドで数値計算
- `scipy.optimize.Bounds` オブジェクトとして SLSQP に供給

#### `optimizer.py` の変更

- `OptimizerConfig` に `use_fourier_bounds`, `enable_velocity_constraint`, `enable_acceleration_constraint` フラグ追加
- `_compute_fourier_bounds()`: `Bounds(-upper, upper)` を構築
- `_compute_named_margins()`: 制約辞書から名前付きマージンを計算
- `_compute_trajectory_stats()`: q_max, dq_max, ddq_max のジョイント別統計
- `OptimizationResult` に `constraint_margins: dict[str, float]`, `feasible: bool`, `trajectory_stats: dict` 追加
- `_generate_random_x0()`: バウンド範囲内でクリップ
- wandb ロギングに名前付き制約マージンと軌道統計を追加

#### `constraints.py` の変更

- `build_scipy_constraints()` の各制約辞書に `"name"` キーを追加（デバッグ・診断用）
- `enable_velocity_constraint`, `enable_acceleration_constraint` フラグで制約の ON/OFF 制御

#### CLI — `configs.py`

- `--use-fourier-bounds` フラグ追加
- `--enable-acc-constraint` フラグ追加
- `--wandb` デフォルトを True に変更

#### `optimize_excitation_trajectory.py`

- Fourier バウンド有効時に per-timestep velocity constraint を自動無効化
- 名前付き制約マージンと軌道統計の診断出力

#### `io.py`

- `feasible`, `constraint_margins`, `trajectory_stats` を JSON シリアライズに追加

### 最適化結果

#### ラン 1: dq_max=1.0 rad/s, T=5s, harmonics=3

wandb ラン: `fourier-bounds-dq1-T5`

| 指標 | 値 |
|---|---|
| 条件数 | **11.56** |
| D-opt 値 | -70.8 |
| feasible | **Yes** (全制約マージン ≥ 0) |
| wall time | ~3700s |
| restarts | 20 (patience=10, target_cond=5.0) |

初めて全 restart で feasible な解を獲得。Fourier バウンドにより速度制約が構造的に保証され、SLSQP が衝突・ワークスペース制約のみに集中できるようになった。

#### ラン 2: dq_max=2.0 rad/s, T=5s, harmonics=3

wandb ラン: `fourier-bounds-dq2-T5` (run ID: q25kplzw)

| 指標 | 値 |
|---|---|
| 条件数 | **6.90** |
| D-opt 値 | -95.2 |
| feasible | **実質 Yes** (margin ≥ -5e-11) |
| wall time | ~185s (1 restart 完了時点) |
| best start | 1 |

| 制約 | マージン | 状態 |
|---|---|---|
| joint_position | +4.41 | OK |
| workspace | +0.22 | OK |
| payload_workspace | -4.76e-11 | OK (数値精度) |
| collision | -1.75e-11 | OK (数値精度) |

### 分析

1. **Fourier バウンドは feasibility 問題を完全に解決**: dq_max=1.0 では全解 feasible、dq_max=2.0 でも数値精度レベルの違反のみ
2. **dq_max の緩和は条件数を大幅に改善**: 1.0 rad/s → 11.56、2.0 rad/s → 6.90
3. **κ=6.90 は実用範囲内**: Kubus et al. (2008) の κ=7-8 と同等
4. **SLSQP はバウンド導入後に十分機能**: アルゴリズム変更は不要と判断

### さらなる改善の方向性（議論のみ、未実装）

1. **線形制約への拡張**: 均等配分ではなく `Σ_k (|a_{j,k}| + |b_{j,k}|) * α_k ≤ dq_max_j` を線形制約として直接実装 → バウンドの保守性を低減
2. **高調波数の増加**: harmonics=3→5 で表現力向上（ただし計算コスト増）
3. **feasibility 閾値の導入**: margin > -1e-6 を feasible とみなす実用的判定

---

## 2026-03-13: harmonics=3 vs harmonics=5 比較実験

### 目的

Fourier バウンド (dq_max=2.0 rad/s) 下で高調波数の影響を比較する。

### 共通設定

- duration=5.0s, base_freq=1/3 Hz, dq_max=2.0 rad/s
- D-optimal 目的関数, Fourier bounds 有効, 加速度制約なし, EE 速度制約なし
- n_monte_carlo=20, max_iter=100, early_stop (patience=10, target_cond=5.0)

### 結果

| | h3 (harmonics=3) | h5 (harmonics=5) |
|---|---|---|
| wandb ラン | `h3-100iter` (q33flt3k) | `h5-100iter` (xkvsvxlk) |
| 決定変数 | 36 | 60 |
| **条件数** | **6.87** | **5.44** |
| D-opt 値 | -95.0 | -98.3 |
| feasible | No (margin=-3e-7) | **Yes** |
| wall time | 2913s (~49min) | 2792s (~47min) |
| restarts 完了 | 11/20 (patience stop) | 20/20 |
| evaluations | 17,286 | 16,382 |
| dq_max 実測 | 1.09 rad/s | 1.03 rad/s |
| ddq_max 実測 | 5.22 rad/s² | 7.29 rad/s² |
| collision margin | +0.007 | +0.025 |
| payload_workspace margin | -3e-7 | 0.000 |

### h5 制約マージン詳細

| 制約 | マージン | 状態 |
|---|---|---|
| joint_position | +4.53 | OK |
| workspace | +0.27 | OK |
| payload_workspace | 0.000 | OK (境界上) |
| collision | +0.025 | OK (2.5cm 余裕) |

### h5 軌道統計

| 指標 | 値 |
|---|---|
| dq_per_joint_max | [0.98, 0.71, 0.71, 0.71, 1.03, 0.85] rad/s |
| ddq_per_joint_max | [7.16, 5.00, 5.00, 5.00, 7.29, 6.54] rad/s² |

### 分析

1. **harmonics=5 が明確に優越**: 条件数 5.44 vs 6.87（21% 改善）、collision margin 2.5cm vs 0.7cm
2. **計算時間は同等**: h5 の方がむしろ速い（47min vs 49min）。決定変数は多いが SLSQP が早く収束
3. **h3 は patience stop**: restart 11 で打ち切り。h5 は 20 restart 完走しより良い解を探索
4. **dq_max の利用率は約 50%**: バウンド 2.0 rad/s に対し実測 ~1.0 rad/s。衝突・ワークスペース制約が支配的

### ワークスペース下端の修正

ワークスペース下端が地面から 3cm であり、collision safety_margin=5cm と不整合であることを発見。
下端を **7cm** に修正（`ur5e_with_box.xml`: center_z=0.479, half_z=0.409）。

### 結論

- harmonics=5, dq_max=2.0, duration=5.0s を推奨設定として採用
- 現時点のベスト: **κ=5.44** (feasible, Kubus et al. の κ=7-8 を上回る)

## 2026-03-22: FT センサオフセット推定のための励起軌道最適化

### 背景

FT センサの観測に定数オフセット `[f_ox, f_oy, f_oz, τ_ox, τ_oy, τ_oz]` が存在する状況を想定。
[[Kubus2007_ft_offset]](../REFERENCES/MAIN.md#Kubus2007_ft_offset) の式 10-11 に基づき、回帰行列を `[I₆ | V]` (6×16) に拡張し、16 パラメータの同時推定が可能な励起軌道を設計する。

### 実装内容

1. **回帰行列の拡張**: `compute_stacked_body_regressor()` に `include_ft_offset` パラメータ追加。スタック後に `I₆` ブロックを左側に hstack → `(6T, 16)`
2. **列スケーリング**: `compute_condition_number()` に `column_scale` パラメータ追加。`I₆` 列の L2 ノルム `√T` 膨張を正規化してから SVD を計算
3. **CLI フラグ**: `--include-ft-offset` / `--no-ft-offset-column-scale` で制御可能
4. **制約デフォルト変更**: dq_max=1.5 rad/s, 加速度制約無効化, EE 速度制約無効化
5. **ペイロードワークスペース制約**: 8 頂点 → 26 サーフェスポイント（頂点+辺中点+面中心）に変更
6. **制約違反ログ改善**: 違反制約を分離表示し定量的な violation 量を出力

### 最適化結果 (D-optimal, harmonics=3, 20 restarts)

| 条件 | スケーリング | 条件数 | feasible | 主な違反 |
|------|-------------|--------|----------|----------|
| dq=2.0, 5s | あり | **1.85** | False (margin=-0.000074) | joint_velocity のみ (ほぼ境界上) |
| dq=1.5, 10s | あり | **1.58** | False | joint_velocity: -2.26, payload: -0.29 |
| dq=2.0, 5s | なし | **21.7** | False | joint_velocity: -0.76, payload: -0.09 |
| dq=1.5, 10s | なし | **37.2** | False | joint_velocity: -1.10, payload: -0.37 |

### 考察

- **列スケーリングあり**: 条件数は良好 (1.6-1.9) だが、オフセット列のノルム正規化後の値。dq=2.0/5s はほぼ feasible。10s 条件では 1001 タイムステップで制約空間が複雑化し、SLSQP が制約違反方向に収束
- **列スケーリングなし**: D-optimal 値が負 (オフセット列の √T 膨張で行列式が支配される) となり、最適化が制約違反方向に暴走。条件数 20-40 と 1 桁悪化
- 列スケーリングは Kubus et al. (2007) の論文には含まれない独自導入。その妥当性は要検討
- dq=1.5/5s の追加実験を実行中 (スケールあり/なし)
