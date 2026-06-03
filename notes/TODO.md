# TODO

## 慣性パラメータ同定・最適励起軌道

詳細計画: [inertial-identification-optimal-excitation-plan.md](inertial-identification-optimal-excitation-plan.md)

- [x] フェーズ 0: 要件固定と数理対象の確定
- [x] フェーズ 1: 軌道表現レイヤの実装 (`trajectories/`)
- [x] フェーズ 2: MuJoCo ベースの運動学・回帰行列構築 (`identification/`)
- [x] フェーズ 3: 制約評価レイヤの実装 (`constraints.py`, `collision.py`, `workspace.py`)
- [x] フェーズ 4: 最適化エンジンの実装 (`optimizer.py`, `objective.py`, `io.py`)
- [x] フェーズ 5: CLI とユーザー導線の整備
- [x] フェーズ 5+: 実機再生用サンプリング済み軌道JSON出力 (`export_trajectory` CLI)
- [x] フェーズ 6: MuJoCo 上での軌道再生と計測基盤 (`execution.py`, `data_buffer.py`)
- [x] フェーズ 7: 慣性推定器との接続 (LS / TLS / RTLS)
- [x] フェーズ 8: Warp 導入と高速化

## 最適化品質の改善 — feasible 解の獲得 (→ `docs/ISSUES.md`)

- [x] wandb 実験追跡 (per-iteration / per-restart メトリクス)
- [x] early stopping (patience ベースの restart 打ち切り)
- [x] duration 短縮 + harmonics 削減 (duration=3.0s, harmonics=3, base_freq=1/3Hz)
- [x] EE 線速度制約の追加 (`EeVelocityConfig`, Jacobian ベース)
- [x] 関節速度上限の CLI 設定 (`--dq-max`, デフォルト 5 deg/s)
- [x] 目標条件数 early stop (`EarlyStopConfig.target_cond`, feasible 時のみ発動)
- [x] 最適化の 1 iteration 所要時間の実測プロファイリング
- [x] stdout flush 対応 (`print(..., flush=True)`)
- [x] 診断ラン実施 (constrained / unconstrained 並列比較)
- [x] 目的関数を D-optimal 基準に変更 (`-log det(W^T W)`, 条件数はバリデーション指標に)
- [x] **解析的 Fourier 係数バウンドの導入** (三角不等式で vel/acc 違反を構造的に排除)
## FT センサオフセット推定対応

- [x] 回帰行列の `[I₆ | V]` 拡張 (`with_ft_offset`)
- [x] 列スケーリングオプション (`ft_offset_column_scale`)
- [x] CLI フラグ (`--with-ft-offset`, `--no-ft-offset-column-scale`)
- [x] 制約デフォルト変更 (dq_max=1.5, 加速度制約OFF, EE速度制約OFF)
- [x] ペイロードワークスペース制約を 26 サーフェスポイントに変更
- [x] 制約違反の定量的ログ出力
- [ ] 列スケーリングの妥当性検証(スケールあり/なしの推定精度比較)
- [ ] feasible 解が得られる制約設定の探索(特に 10s 条件)
  - [x] 5s 条件で EE/dq/ddq/作業領域をスイープ. ベスト = EE≤0.25 + dq≤π/2 + ddq≤2π(cond 1.83, 境界違反のみ). 作業域縮小は cond 悪化 + feasibility 破綻を確認(2026-06-03, log 参照)

## FT センサベースの同定・wrench 計測

- [x] MuJoCo force/torque sensor を tool0 (`ft_sensor` site) に追加
- [x] PD playback の制御入力バグ修正(servo 目標角度・substep・実測 qacc)
- [x] settling phase 追加(home の重力下平衡への整定)
- [x] wrench 符号反転(実機 FT 規約 child→parent に整合)
- [x] flange wrench プロットスクリプト(共通スパン・重力 tare)
- [x] FT センサ wrench での LS 同定検証(質量・重心 誤差 0%, 重力二重計上バグを特定・修正)
- [x] Kubus 2007 論文サマリ作成
- [ ] FT センサ整合 regressor を `identification/` に正式実装
- [ ] 推定慣性の CoM まわりへの変換(平行軸定理)と真値一致確認
- [ ] PD 追従誤差が同定精度・sim-real ギャップに与える影響評価

## ペイロードを Robotiq 2F-85 グリッパへ切替

- [x] Menagerie から 2F-85 資産取得・配置 (`assets/robotiq_2f85/`)
- [x] `ur5e_with_gripper.xml` / `scene_with_gripper.xml` 作成 (クラス・material・body 名衝突回避)
- [x] FT センサを tool0 整合点に配置, home keyframe を 14 qpos / 7 ctrl に拡張
- [x] 指全開保持 (actuator ctrl=0) で軌道追従 + FT 記録 (`playback_gripper.npz`)
- [x] FT wrench プロット (`flange_wrench_gripper.png`) と再生動画 (`playback_gripper.mp4`)
- [x] `execution.py` の単一剛体 `params` 解決を遅延化 (FT 経路で不在 body 名を許容)
- [ ] グリッパサブツリー合成慣性の算出 (全開固定剛体としての同定真値)

## 最適化品質の改善 — feasible 解の獲得 (→ `docs/ISSUES.md`)

- [ ] 制約の段階的評価(安い制約で早期棄却し FK ループをスキップ)
- [ ] collision constraint の高速化 (FK ループ共有化)
- [ ] 最適化アルゴリズムの変更 (SLSQP → COBYLA or IPOPT) ← バウンド導入後に再評価
- [x] 並列 Monte Carlo 最適化の実装 (`ProcessPoolExecutor`, 設計済み)
- [ ] `EarlyStopConfig.min_improvement` の early stopping ロジックへの組み込み
- [ ] 目的関数へのワークスペースカバレッジ項導入の検討(条件数最小化単独では作業空間全体を使い切らないことが 3 仮説検証で確認された)

## インフラ・共通

- [x] devcontainer + Docker Compose 基盤
- [x] UR5e MJCF モデル配置
- [x] ペイロード付き派生シーン (`scene_with_box.xml`)
- [ ] Warp kinematics / dynamics 実験モジュール
