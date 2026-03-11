# UR5e assets

このディレクトリには、UR5e のモデル資産を以下のどちらかの形で配置します。
現時点の標準導線は MuJoCo Menagerie 由来の MJCF を `assets/ur5e/mjcf/` へ同期する方法です。

## 標準取得方法

```bash
./scripts/fetch_ur5e_menagerie_assets.sh
```

このスクリプトは `google-deepmind/mujoco_menagerie` の `universal_robots_ur5e` を取得し、
`assets/ur5e/mjcf/` に `scene.xml`, `ur5e.xml`, `assets/` と upstream のライセンス情報を配置します。

## 推奨レイアウト

### MuJoCo MJCF を置く場合

```text
assets/ur5e/
  mjcf/
    scene.xml
    ur5e.xml
    meshes/
```

`scene.xml` を優先して読み込みます。`scene.xml` がない場合は `ur5e.xml` を探します。
MuJoCo Menagerie から取得した場合もこのレイアウトになります。

### URDF を置く場合

```text
assets/ur5e/
  urdf/
    ur5e.urdf
    meshes/
```

MuJoCo で直接読み込める URDF を置く前提です。相対 mesh パスは `urdf/` から解決されるように揃えてください。

## 推奨方針

- まずは `mjcf/scene.xml` を用意し、可視化やセンサ追加は MJCF 側で管理する
- 生の URDF は比較用・変換元として `urdf/` に残す
- 大きい mesh やバイナリは git に含めず、この README の下の実データだけローカル配置する
- upstream 由来のライセンス文書は `mjcf/LICENSE.menagerie` を確認する

## 最初の確認

```bash
python -m mjwarp_ur5e.demos.load_ur5e_mujoco --headless --steps 10
```

または

```bash
mjwarp-ur5e-load --headless --steps 10
```