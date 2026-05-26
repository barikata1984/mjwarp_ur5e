# 励起軌道 制約実装記録

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
