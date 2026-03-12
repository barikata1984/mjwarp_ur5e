# 未解決の技術課題

解決したらエントリごと削除する。

## `attachment_site` 基準の wrench 観測モデル定義

- `attachment_site` を基準とした wrench 観測モデルをどう定義するか未決定
- 現在は `payload_box_mount` body frame 基準で回帰行列を構築している

## 最適化結果の保存先

- `debug/` と `data/` のどちらに置くか未決定

## 作業領域制約の基準フレーム

- base frame で持つか、初期 EE 基準で持つか未決定

## 衝突判定の方式

- 幾何近似で固定するか、MuJoCo 接触情報へ広げるか未決定
