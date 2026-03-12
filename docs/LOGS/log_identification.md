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
