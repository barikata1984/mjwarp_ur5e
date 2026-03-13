# 未解決の技術課題

解決したらエントリごと削除する。

## `attachment_site` 基準の wrench 観測モデル定義

- `attachment_site` を基準とした wrench 観測モデルをどう定義するか未決定
- 現在は `payload_box_mount` body frame 基準で回帰行列を構築している

## 最適化結果が全て infeasible

- **影響**: Config D の全 9 restart が infeasible (feasible=False)
- **制約マージン例**: velocity=-0.40, acceleration=-4.50, workspace=-0.03, payload_workspace=-0.35, collision=-0.10

### 調査結果 (2026-03-13)

- `subsample_factor` は**目的関数（条件数計算）にのみ適用**。制約関数は既に全 1001 timestep で評価されており、「制約の見逃し」ではない
- 制約評価（workspace/payload_workspace/collision の FK ループ各 1000 回）が 1 iteration あたりの計算コストの ~97% を占め、目的関数の subsample 変更は全体に僅少な影響
- **duration=10s が過剰**: Kubus et al. (2008) は duration=1.5s, harmonics=3, max_freq=2Hz で κ=7-8 を達成。短い duration で timestep 数が線形に減り計算コストも線形に減少
- 真の原因は SLSQP が広い探索空間（60 変数 = 6 joints × 5 harmonics × 2 coeffs）で feasible 領域内の良い解を見つけられないこと

### 対策の実施状況 (2026-03-13)

- **duration 短縮 + harmonics 削減**: 実施済み (duration=3.0s, harmonics=3, 変数 60→36)
- **EE 線速度制約 / 関節速度制約**: 実施済み (≤0.25 m/s, ≤5 deg/s)
- **目標条件数 early stop**: 実施済み (κ≤5 + feasible で停止)

### 残存問題: 最適化が 30 分以上で最初の restart が完了しない

- 原因候補: (1) dq_max=0.0873 rad/s の厳しさにより SLSQP の収束が遅い (2) FK ベース制約 (workspace/payload/collision/EE velocity) × 301 点 × 37 回/iteration (有限差分) ≈ 55,000 FK/iteration が重い (3) stdout バッファリングで出力が見えないだけで実は動いている可能性
- **次のアクション**: 1 iteration の所要時間を実測し、ボトルネックを特定する

### 対策案（優先順、更新版）

1. **1 iteration のプロファイリング**: 実測なしに推測で最適化するのは非効率
2. **制約の段階的評価**: 安い制約（joint limits, 純 NumPy）を先に評価し、違反なら高い FK 制約をスキップ
3. **解析的 velocity/acceleration 上界**: Fourier 係数の三角不等式で FK ループを排除
4. **並列 Monte Carlo**: restart 数を増やして feasible 解の発見確率を向上（設計済み）

## `EarlyStopConfig.min_improvement` 未使用

- `min_improvement: float = 1e-3` フィールドが定義されているが、early stopping ロジックでは厳密な改善 (`cond < best_cond`) のみを判定
- 微小改善を無視する閾値として `min_improvement` を組み込む必要がある
