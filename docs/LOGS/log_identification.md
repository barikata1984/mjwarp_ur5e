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

- ペイロード外形に基づくロボット-ペイロード衝突判定 (現在は body 原点の点判定)
