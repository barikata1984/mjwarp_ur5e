# 未解決の技術課題

解決したらエントリごと削除する。

## `attachment_site` 基準の wrench 観測モデル定義

- `attachment_site` を基準とした wrench 観測モデルをどう定義するか未決定
- 現在は `payload_box_mount` body frame 基準で回帰行列を構築している

## `EarlyStopConfig.min_improvement` 未使用

- `min_improvement: float = 1e-3` フィールドが定義されているが、early stopping ロジックでは厳密な改善 (`cond < best_cond`) のみを判定
- 微小改善を無視する閾値として `min_improvement` を組み込む必要がある
