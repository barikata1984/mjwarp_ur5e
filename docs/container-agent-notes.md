# Container agent notes

このリポジトリでコンテナ内実装を担当するエージェント向けの覚書です。

## 前提

- 開発の正系は devcontainer だが、コンテナ定義の単一ソースは [docker/docker-compose.yaml](/home/atsushi.kuno/workspace/mjwarp_ur5e/docker/docker-compose.yaml)
- standalone では `./docker/setup-env.sh` で `docker/.env` を生成してから `docker compose -f docker/docker-compose.yaml --env-file docker/.env up -d --wait`
- Python 実装は `src/` レイアウトを守る

## UR5e 資産配置ルール

- ローカル資産のルートは `assets/ur5e/`
- 標準取得スクリプトは `./scripts/fetch_ur5e_menagerie_assets.sh`
- 優先読込順は `assets/ur5e/mjcf/scene.xml` → `assets/ur5e/mjcf/ur5e.xml` → `assets/ur5e/urdf/ur5e.urdf`
- 実データは git ignore される。共有すべきなのは配置規則、README、ローダーコード
- MuJoCo 用の初期資産は `google-deepmind/mujoco_menagerie` の `universal_robots_ur5e` を採用

## 実装済みの入口

- 資産解決: `mjwarp_ur5e.assets.resolve_ur5e_model_path()`
- MuJoCo ロード: `mjwarp_ur5e.mujoco.load_ur5e_model()`
- 動作確認: `python -m mjwarp_ur5e.demos.load_ur5e_mujoco --headless --steps 10`
- 状態検査: `python -m mjwarp_ur5e.demos.inspect_ur5e_mujoco --json`
- readiness は compose healthcheck で `import mjwarp_ur5e, mujoco, warp` が通ることを見ている

## 現在わかっているモデル情報

- joint 名は `shoulder_pan_joint`, `shoulder_lift_joint`, `elbow_joint`, `wrist_1_joint`, `wrist_2_joint`, `wrist_3_joint`
- `home` keyframe があり、6 自由度の初期姿勢を持つ
- EE site 名は `attachment_site`、対応 body は `wrist_3_link`

## 次に作業するとよいもの

- Warp 側の kinematics / dynamics 実験モジュール
- MJCF にセンサ、床、ターゲットマーカを足した実験用 scene.xml
- EE site を基準にした IK / 到達点追従の雛形

## 注意点

- URDF を読む場合、mesh 相対パスが `assets/ur5e/urdf/` から壊れていないことを必ず確認する
- MuJoCo の viewer を使うコードと headless の検証コードは分けて保つ
- 資産そのものの配布条件は別管理。取得元やライセンスを README に追記するまではリポジトリへ同梱しない