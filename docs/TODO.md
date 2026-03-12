# TODO

## 慣性パラメータ同定・最適励起軌道

詳細計画: [inertial-identification-optimal-excitation-plan.md](inertial-identification-optimal-excitation-plan.md)

- [x] フェーズ 0: 要件固定と数理対象の確定
- [x] フェーズ 1: 軌道表現レイヤの実装 (`trajectories/`)
- [x] フェーズ 2: MuJoCo ベースの運動学・回帰行列構築 (`identification/`)
- [x] フェーズ 3: 制約評価レイヤの実装 (`constraints.py`, `collision.py`, `workspace.py`)
- [x] フェーズ 4: 最適化エンジンの実装 (`optimizer.py`, `objective.py`, `io.py`)
- [x] フェーズ 5: CLI とユーザー導線の整備
- [x] フェーズ 6: MuJoCo 上での軌道再生と計測基盤 (`execution.py`, `data_buffer.py`)
- [x] フェーズ 7: 慣性推定器との接続 (LS / TLS / RTLS)
- [x] フェーズ 8: Warp 導入と高速化

## 次の開発アイテム

- [ ] ペイロード外形に基づくロボット-ペイロード衝突判定の実装（現在は body 原点の点判定）

## インフラ・共通

- [x] devcontainer + Docker Compose 基盤
- [x] UR5e MJCF モデル配置
- [x] ペイロード付き派生シーン (`scene_with_box.xml`)
- [ ] Warp kinematics / dynamics 実験モジュール
