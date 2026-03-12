# mjwarp_ur5e

MuJoCo と NVIDIA Warp 上で UR5e を扱い、ロボット工学アルゴリズムを反復検証するための開発環境です。
現在の主要ワークロードは **ペイロード慣性パラメータ同定のための最適励起軌道生成** です。

## ディレクトリ構成

```
.
├── .devcontainer/
│   └── devcontainer.json
├── assets/ur5e/
│   ├── README.md                  # UR5e 資産の配置ルール
│   └── mjcf/
│       ├── scene.xml              # 標準シーン
│       ├── scene_with_box.xml     # ペイロード付き派生シーン
│       └── ur5e_with_box.xml      # UR5e + 直方体ペイロード
├── docs/
│   ├── README.md                  # ドキュメントガイド
│   ├── TODO.md                    # 進捗チェックリスト
│   ├── ISSUES.md                  # 未解決の技術課題
│   ├── LOGS/                      # 実験記録 (追記のみ)
│   ├── REFERENCES/                # 参考文献マスタ
│   ├── SURVEYS/                   # 文献調査レポート
│   ├── REPORTS/                   # 日次報告
│   ├── container-agent-notes.md
│   └── inertial-identification-optimal-excitation-plan.md
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yaml
│   ├── entrypoint.sh
│   ├── requirements.txt
│   ├── setup-env.sh
│   └── .env.example
├── literature/                    # 参照論文 PDF (git ignore)
├── scripts/
│   └── fetch_ur5e_menagerie_assets.sh
├── src/mjwarp_ur5e/
│   ├── assets.py                  # 資産解決
│   ├── model.py                   # MuJoCo モデルロード
│   ├── rendering.py               # レンダリング補助
│   ├── inspection.py              # モデル検査
│   ├── frames.py                  # 座標系ユーティリティ
│   ├── cli/configs.py             # tyro CLI 設定
│   ├── demos/                     # デモスクリプト
│   ├── trajectories/              # 軌道表現 (Fourier, windowed)
│   └── identification/            # 慣性パラメータ同定
│       ├── types.py               # BodyKinematics, InertialParameters 等
│       ├── sampling.py            # MuJoCo 状態設定・キネマティクス取得
│       └── regressor.py           # 剛体 wrench 回帰行列
├── tests/
│   ├── test_trajectories.py
│   ├── test_identification_regressor.py
│   └── ...
├── pyproject.toml
└── README.md
```

## クイックスタート

### 1. `docker/.env` を生成

```bash
./docker/setup-env.sh
```

### 2. Docker 単体で起動

```bash
docker compose -f docker/docker-compose.yaml --env-file docker/.env build
docker compose -f docker/docker-compose.yaml --env-file docker/.env up -d --wait
docker compose -f docker/docker-compose.yaml --env-file docker/.env exec dev zsh
```

### 3. VS Code devcontainer で開く

VS Code でこのワークスペースを開き、`Dev Containers: Reopen in Container` を実行します。

## 検証コマンド

```bash
python -c "import mujoco, warp; print(mujoco.__version__)"
python -c "import mjwarp_ur5e; print(mjwarp_ur5e.__all__)"
pytest tests/ -v
ruff check .
```

## 主要機能

### UR5e モデル操作

```bash
# UR5e 資産の取得 (mujoco_menagerie から)
./scripts/fetch_ur5e_menagerie_assets.sh

# MuJoCo ロード確認
python -m mjwarp_ur5e.demos.load_ur5e_mujoco --headless --steps 10

# joint 名・home 姿勢・EE 位置の確認
python -m mjwarp_ur5e.demos.inspect_ur5e_mujoco

# ホーム姿勢レンダリング
python -m mjwarp_ur5e.demos.render_ur5e_home --show-base-frame --show-ee-frame

# ペイロード付きシーン
python -m mjwarp_ur5e.demos.render_ur5e_home \
    --model assets/ur5e/mjcf/scene_with_box.xml \
    --camera payload_overview
```

### 慣性パラメータ同定基盤

`src/mjwarp_ur5e/trajectories/`: windowed Fourier 系の軌道表現

- `WindowedFourierTrajectory` — 境界条件 q(0)=q(T)=q0, dq(0)=dq(T)=0 を満たす励起軌道
- 軌道位置・速度・加速度を一括サンプリング

`src/mjwarp_ur5e/identification/`: MuJoCo ベースの回帰行列構築

- `set_model_state` — qpos, qvel, qacc を設定してキネマティクスを計算
- `sample_body_kinematics` — body frame の速度・加速度を取得 (Jacobian ベース)
- `rigid_body_wrench_regressor` — 剛体 wrench 回帰行列 (6x10) を構築
- `compute_stacked_body_regressor` — 軌道全体の積み上げ回帰行列
- `compute_condition_number` — 条件数評価

実装計画の詳細: [docs/inertial-identification-optimal-excitation-plan.md](docs/inertial-identification-optimal-excitation-plan.md)

## ペイロード付きシーン

`assets/ur5e/mjcf/scene_with_box.xml` は EE フランジ面に直方体ペイロード (25cm^3, 1kg) を剛体取り付けした派生シーンです。

- ペイロード body 名: `payload_box_mount`
- 初期姿勢: `shoulder_pan_joint = +pi/2`
- 作業領域 (半透明直方体): base frame で x: [-35, +35] cm, y: [-20, +60] cm, z: [3cm, EE+40cm]

## コーディング規約

- **Formatter / Linter**: `ruff format` / `ruff check`
- **Type hints**: Python 3.10+ annotations
- **Line length**: 100 characters
- **Import order**: stdlib → third-party → local (ruff isort)
- **Docstrings**: Google style, non-obvious functions only
- **Config**: Python dataclasses + tyro CLI
- **RNG**: `np.random.Generator` (legacy `np.random` 禁止)
- **Commit format**: Conventional Commits (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`)

## コンテナ環境

- Python venv: `/opt/venv/bin/python`
- MuJoCo Menagerie: `/opt/mujoco_menagerie/`
- editable install 済み: `pip install --no-deps -e /workspace`
- CUDA 付き Ubuntu 24.04 ベース、EGL ヘッドレス描画対応

## 注意点

- NVIDIA Container Toolkit がホストに入っている前提
- GUI 表示を使う場合はホスト側の X11 設定が必要
- MJCF の `<inertial>` はデフォルト値。ランタイムで `model.body_mass` / `model.body_inertia` を書き換え → `mj_setConst()` で派生量再計算する設計
