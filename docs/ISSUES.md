# 未解決の技術課題

解決したらエントリごと削除する。

## `attachment_site` 基準の wrench 観測モデル定義

- `attachment_site` を基準とした wrench 観測モデルをどう定義するか未決定
- 現在は `payload_box_mount` body frame 基準で回帰行列を構築している

## ペイロード外形に基づくロボット-ペイロード衝突判定

- 現在の `_check_payload_collision` はペイロード body 原点（点）とリンク球の距離で判定
- ペイロードは一辺 0.25m の箱なので、外形（8頂点）を考慮した厳密な判定が必要
- 実機で最適励起軌道を実行するための安全要件

