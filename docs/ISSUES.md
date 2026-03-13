# 未解決の技術課題

解決したらエントリごと削除する。

## `attachment_site` 基準の wrench 観測モデル定義

- `attachment_site` を基準とした wrench 観測モデルをどう定義するか未決定
- 現在は `payload_box_mount` body frame 基準で回帰行列を構築している

## SLSQP では feasible 解が得られない

- **影響**: duration=3s, harmonics=3 の設定でも、constrained / unconstrained 両方で全 restart が infeasible
- **根本原因**: SLSQP (有限差分ベース局所勾配法) が制約付き非凸最適化問題に不適切

### プロファイリング結果 (2026-03-13)

| コンポーネント | 1 回 | 1 iter (×37 FD) | 割合 |
|---|---|---|---|
| collision constraint | 67 ms | 2.49 s | 69% |
| objective (cond number) | 28 ms | 1.04 s | 29% |
| workspace / payload / EE vel | 2.6 ms | 97 ms | 3% |
| NumPy constraints (pos/vel/acc) | 0.04 ms | 1.5 ms | ~0% |

- **1 SLSQP iteration = 3.6s**, 100 iter/restart = ~400s/restart
- collision constraint が圧倒的ボトルネック (全体の 69%)

### 診断ラン結果 (2026-03-13)

| | unconstrained | constrained (dq≤5°/s, EE≤25cm/s) |
|---|---|---|
| 条件数 | **2.11** | **2.74** |
| feasible | No (margin=-0.004) | No (全 restart infeasible) |
| 最大関節速度 | 125°/s | 51.6°/s |
| 最大 EE 線速度 | 86.3 cm/s | 54.2 cm/s |
| wall time | 27 min (4 restart) | 34 min (5 restart) |

- SLSQP は制約を無視しているのではなく、**満たそうとしているが 100 iteration では収束しない**
- constrained: dq_max=5°/s を要求 → 51.6°/s まで下がったが未到達
- unconstrained: collision/workspace 制約だけでもほぼ feasible (margin=-0.004) だが到達せず

### 対策 (次セッション)

1. **最適化アルゴリズムの変更** (最優先)
   - COBYLA: 有限差分不要 → 1 iter あたり 37x 高速、ただし局所的
   - Differential Evolution (scipy 組込み): 大域探索、制約対応あり
   - CMA-ES + augmented Lagrangian: 文献でも使用実績あり
2. collision constraint の高速化 (FK ループ共有化)
3. 制約の段階的評価

## `EarlyStopConfig.min_improvement` 未使用

- `min_improvement: float = 1e-3` フィールドが定義されているが、early stopping ロジックでは厳密な改善 (`cond < best_cond`) のみを判定
- 微小改善を無視する閾値として `min_improvement` を組み込む必要がある
