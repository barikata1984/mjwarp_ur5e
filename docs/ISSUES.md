# 未解決の技術課題

解決したらエントリごと削除する。

## `attachment_site` 基準の wrench 観測モデル定義

- `attachment_site` を基準とした wrench 観測モデルをどう定義するか未決定
- 現在は `payload_box_mount` body frame 基準で回帰行列を構築している

## `subsample_factor` による制約違反の見逃し

- **影響**: 最適化結果が全て infeasible (feasible=False)
- **原因**: `subsample_factor=20` で 1001 timestep 中 51 点のみで制約を評価。サンプル間での velocity/acceleration/workspace/collision 違反を optimizer が検知できない
- **再現**: Config D (subsample=20, mc=10, mi=100) の全 9 restart が infeasible
- **制約マージン例**: velocity=-0.40, acceleration=-4.50, workspace=-0.03, payload_workspace=-0.35, collision=-0.10

### 対策案

1. **subsample_factor=10 + 並列化**: 制約精度を上げつつ計算コストを並列化で吸収
2. **2段階最適化**: 粗い subsample で粗最適化 → 細かい subsample で refinement
3. **保守的な制約値**: joint limits / workspace bounds に安全マージンを追加して subsample 間の違反を予防

## `EarlyStopConfig.min_improvement` 未使用

- `min_improvement: float = 1e-3` フィールドが定義されているが、early stopping ロジックでは厳密な改善 (`cond < best_cond`) のみを判定
- 微小改善を無視する閾値として `min_improvement` を組み込む必要がある
