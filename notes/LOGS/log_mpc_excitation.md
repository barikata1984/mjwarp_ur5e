# MPC 励起軌道最適化 実装記録

## 2026-06-17: MPC 励起軌道最適化システムの実装

### 背景

Zhang-RSS2025 に基づき, 慣性パラメータ同定のための MPC スタイルのオンライン励起軌道最適化を実装した.
既存の Fourier 係数ベースのオフライン最適化とは異なり, 再計画型 (receding horizon) の軌道生成を行う.
実装は `mjwarp_ur5e` 側 (MuJoCo) に置いた.

### 実装構成

7 段階に分けて実装し, 全 22 テストが通過している.

| モジュール | 役割 |
|---|---|
| `trajectories/quintic_spline.py` | 5 次 Hermite 多項式による軌道表現. Fourier と異なり任意初期状態に対応 |
| `identification/mpc/config.py` | `MPCConfig`, `HorizonConfig`, `PlannerConfig` |
| `identification/mpc/metrics.py` | 重力掃引角度, 重力方向分散, 加速度ピーク |
| `identification/mpc/planner.py` | `ExcitationPlanner` (マルチスタート SLSQP) |
| `identification/mpc/loop.py` | `MPCLoop` (plan→execute→identify→replan) |
| `identification/mpc/__init__.py` | パッケージエクスポート |
| `demos/run_mpc_identification.py` | CLI エントリポイント |
| `demos/render_mpc_playback.py` | 保存済み軌道から 4 視点グリッド動画を生成 |
| `cli/configs.py` | `MPCIdentificationConfig` 追加 |

### コスト関数と軌道表現の選択

- コスト関数: 条件数 κ₂(W) (Zhang に倣う). D-optimal (-log det(WᵀW)) は採用しなかった
- 軌道表現: 5 次スプライン (Fourier 不使用). 再計画時の任意初期状態 (q, dq, ddq) の受け渡しが容易なため
- 終端制約: 速度 dq(T_h)=0 を構造的制約として強制 (安全フォールバック)

### /simplify レビューによる修正

実装後に /simplify レビューを適用し, 以下を修正した.

- 重複していた制約ファクトリを既存の `make_joint_*_constraint` の再利用に統一
- `metrics.py` での kinematics 二重スイープを解消
- `planner.py` の regressor 冗長再計算を排除

### エンドツーエンド動作確認

MuJoCo シミュレーション上で MPC ループが正常に動作することを確認した.
2 ステップ (最小設定) 実行時に条件数の改善傾向を確認.

### 判明した制約・課題

**計画速度の問題:**
- 現状の計画時間: 1 ステップあたり約 10 秒 (scipy SLSQP + 数値勾配)
- リアルタイム実行には CasADi + Ipopt による解析的勾配が必要
- 現在の計画と実行は逐次処理 (Zhang 論文は並列を想定)

**実装上の設計選択:**
- オープンループモード (`use_pd_control=False`) は `mj_set_state` で状態を直接代入
- FT センサレンチは `mj_inverse` から取得 (解析的に正確だが PD サーボモードとは挙動が異なる)

**衝突・作業域制約の未実装 (→ ISSUES.md に追記):**
- `ExcitationPlanner` に衝突制約・作業域制約が未接続
- 実行中にテーブルを貫通・危険な姿勢に到達するケースが確認された
- 既存インフラ (`WorkspaceConstraintConfig`, `CollisionConfig`, `make_collision_constraint`, `make_workspace_constraint` in `optimizer.py`/`constraints.py`) は流用可能

**収束検証の未完了:**
- 2 ステップ・2 スタートの最小設定では推定パラメータが大きくズレる
- 本番的な収束確認には長いランが必要

### 実装計画ドキュメント

`notes/mpc-excitation-plan/` に README および 7 ステップ分のドキュメントを作成した.
