# mjwarp_ur5e

MuJoCo と NVIDIA Warp 上で UR5e を扱い、ロボット工学アルゴリズムを反復検証するための開発環境です。
VS Code の devcontainer を標準導線にしつつ、同じ Docker Compose 定義をそのまま `docker compose` でも使える構成にしています。

## ディレクトリ構成

```
.
├── .devcontainer/
│   └── devcontainer.json       # VS Code 側の最小設定
├── assets/
│   └── ur5e/
│       ├── README.md           # UR5e 資産の配置ルール
│       └── mjcf/
│           └── .gitkeep
├── docs/
│   └── container-agent-notes.md
├── scripts/
│   └── fetch_ur5e_menagerie_assets.sh
├── .vscode/
│   └── tasks.json              # docker compose 操作用タスク
├── docker/
│   ├── Dockerfile              # CUDA + MuJoCo/Warp 向け依存を含むベースイメージ
│   ├── docker-compose.yaml     # devcontainer/CLI 共通の起動定義
│   ├── entrypoint.sh           # 初期化と非 root ユーザー切替
│   ├── requirements.txt        # Python 依存
│   ├── setup-env.sh            # docker/.env をホスト値から生成
│   └── .env.example            # 手動編集用テンプレート
├── src/
│   └── mjwarp_ur5e/
│       ├── assets.py
│       ├── inspection.py
│       ├── mujoco.py
│       └── demos/
│           ├── inspect_ur5e_mujoco.py
│           ├── load_ur5e_mujoco.py
│           └── render_ur5e_home.py
├── tests/
│   ├── test_assets.py
│   ├── test_inspection.py
│   └── test_import.py
├── pyproject.toml              # Python パッケージ設定と ruff/pytest 設定
├── .dockerignore               # ルートを build context にしたときの除外設定
├── .gitignore
└── README.md
```

## クイックスタート

### 1. `docker/.env` を生成

```bash
./docker/setup-env.sh
```

必要であれば [docker/.env.example](/home/atsushi.kuno/workspace/mjwarp_ur5e/docker/.env.example) を参考に、生成された `docker/.env` の値を調整します。

### 2. Docker 単体で起動

```bash
docker compose -f docker/docker-compose.yaml --env-file docker/.env build
docker compose -f docker/docker-compose.yaml --env-file docker/.env up -d --wait
docker compose -f docker/docker-compose.yaml --env-file docker/.env exec dev zsh
```

### 3. VS Code devcontainer で開く

VS Code でこのワークスペースを開き、`Dev Containers: Reopen in Container` を実行します。
devcontainer 側は [docker/docker-compose.yaml](/home/atsushi.kuno/workspace/mjwarp_ur5e/docker/docker-compose.yaml) をそのまま使います。

## 何が揃うか

### 共通コンテナ基盤

- CUDA 付き Ubuntu 24.04 ベース
- MuJoCo と Warp を想定した EGL ヘッドレス描画設定
- ホスト UID/GID に揃えた非 root ユーザー
- `docker compose` と devcontainer の単一ソース化

### Python 開発基盤

- `src/` レイアウトの Python パッケージ
- `pytest` と `ruff` の最低限設定
- entrypoint で editable install を実施

### VS Code 補助

- devcontainer 設定
- Docker Compose 操作用タスク
- Python interpreter / pytest / formatter の設定

## 推奨ワークフロー

### UR5e モデル配置

UR5e の URDF や mesh は `assets/ur5e/` 配下に置く想定です。
配置ルールは [assets/ur5e/README.md](/home/atsushi.kuno/workspace/mjwarp_ur5e/assets/ur5e/README.md) に記載しています。
実データは巨大ファイル混入を避けるため git ignore しています。

最初の導入は次のコマンドで行えます。

```bash
./scripts/fetch_ur5e_menagerie_assets.sh
```

取得元は `google-deepmind/mujoco_menagerie` の `universal_robots_ur5e` です。

### アルゴリズム実装

実験コードは `src/mjwarp_ur5e/` に、検証は `tests/` に置く前提です。
Jupyter を使う場合はコンテナ内から `jupyter lab --ip 0.0.0.0 --no-browser` を実行してください。

### 最初の UR5e ロード確認

`assets/ur5e/` にモデルを置いたら、次のコマンドで MuJoCo ロードを確認できます。

```bash
python -m mjwarp_ur5e.demos.load_ur5e_mujoco --headless --steps 10
```

viewer を開く場合は `--headless` を外してください。

### joint 名と home 姿勢の確認

次のコマンドで joint 名、site 名、home 姿勢、EE 位置を確認できます。

```bash
python -m mjwarp_ur5e.demos.inspect_ur5e_mujoco
```

JSON が欲しい場合は次を使います。

```bash
python -m mjwarp_ur5e.demos.inspect_ur5e_mujoco --json-output
```

### ホーム姿勢のレンダリング

次のコマンドでホーム姿勢を `debug/ur5e_home.png` に出力できます。

```bash
python -m mjwarp_ur5e.demos.render_ur5e_home
```

同名の `debug/ur5e_home.json` に、レンダリング時の model 情報も出力します。

ベース座標系の軸ベクトルを重ねる場合は次を使います。

```bash
python -m mjwarp_ur5e.demos.render_ur5e_home --show-base-frame
```

EE 座標系 `attachment_site` の軸も重ねる場合は次を使います。

```bash
python -m mjwarp_ur5e.demos.render_ur5e_home --show-base-frame --show-ee-frame
```

EE フランジ面に直方体ペイロードを接触配置した派生シーンは `assets/ur5e/mjcf/scene_with_box.xml` です。
現在の寸法は EE 座標系で `x=25cm`, `y=25cm`, `z=25cm` です。
ペイロード中心は、フランジ面に接する元位置を基準に EE 座標系の `y` 方向へ `-10cm` オフセットしています。
この場合は `--model assets/ur5e/mjcf/scene_with_box.xml --camera payload_overview` を指定して読み込みます。

このシーンでは、初期姿勢を `shoulder_pan_joint = +0.5\pi` とし、
その姿勢での EE 基準位置からベース座標系で `x: [-35, +35] cm`, `y: [-20, +60] cm`、
`z` は床面から `3cm` を下端とし、上端は従来どおり EE 基準の `+40cm` とした作業可能領域を半透明直方体として追加しています。
作業領域まで含めて見る場合は `workspace_overview` カメラを使います。

## 注意点

- NVIDIA Container Toolkit がホストに入っている前提です。
- GUI 表示を使う場合はホスト側の X11 設定が必要です。
- UR5e 資産の取得元や表現形式はプロジェクト方針に応じて別途決めてください。

## 初期確認コマンド

```bash
python -c "import mujoco, warp; print(mujoco.__version__)"
python -c "import mjwarp_ur5e; print(mjwarp_ur5e.__all__)"
pytest
ruff check .
```

