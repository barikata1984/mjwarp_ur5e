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
- [ ] **目的関数を D-optimal 基準に変更** (`-log det(W^T W)`, 条件数はバリデーション指標に)
- [ ] **最適化アルゴリズムの変更** (SLSQP → COBYLA or CMA-ES+SLSQP or IPOPT)
- [ ] 制約の段階的評価（安い制約で早期棄却し FK ループをスキップ）
- [ ] 解析的 velocity/acceleration 上界の導入 (Fourier 係数の三角不等式)
- [ ] 並列 Monte Carlo 最適化の実装 (`ProcessPoolExecutor`, 設計済み)
- [ ] `EarlyStopConfig.min_improvement` の early stopping ロジックへの組み込み

## インフラ・共通

- [x] devcontainer + Docker Compose 基盤
- [x] UR5e MJCF モデル配置
- [x] ペイロード付き派生シーン (`scene_with_box.xml`)
- [ ] Warp kinematics / dynamics 実験モジュール
