# 未解決の技術課題

解決したらエントリごと削除する。

## `attachment_site` 基準の wrench 観測モデル定義

- `attachment_site` を基準とした wrench 観測モデルをどう定義するか未決定
- 現在は `payload_box_mount` body frame 基準で回帰行列を構築している

## FT オフセット拡張時の列スケーリング妥当性

- 回帰行列 `[I₆ | V]` でオフセット列の L2 ノルムが `√T` 倍に膨張し、スケーリングなしでは D-optimal 目的関数が制約違反方向に暴走する
- スケーリングありでは条件数は良好だが、正規化後の値であり実推定精度との対応は未検証
- スケーリングの代替案: 重み付き D-optimal、ブロック対角前処理、オフセット列のサブサンプリング

## 10s 条件での feasibility 問題

- duration=10s (1001 timesteps) で subsample_factor=1 のとき、全 restart で feasible=False
- SLSQP が高次元制約空間で局所最適に陥り、制約遵守と目的関数改善を両立できない
- 対策候補: subsample_factor 引き上げ、IPOPT 移行、restart 数/イテレーション数増加

## `EarlyStopConfig.min_improvement` 未使用

- `min_improvement: float = 1e-3` フィールドが定義されているが、early stopping ロジックでは厳密な改善 (`cond < best_cond`) のみを判定
- 微小改善を無視する閾値として `min_improvement` を組み込む必要がある
