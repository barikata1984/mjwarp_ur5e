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

### 文献との差異分析 (2026-03-13)

D-optimal 目的関数を実装・診断した結果、目的関数の滑らかさは副次的であり、根本原因は以下と判明:

1. **探索空間が制約に対して広すぎる**: D-optimal が Fourier 係数を制限の33倍まで押し込む（加速度 margin=-27）
2. **文献にない制約**: 衝突回避・ワークスペース変位は調査した6論文のいずれも使用していない
3. **SLSQP がバウンドなしの広い空間で非凸 feasible 領域を探索できない**

### 対策（優先順位順）

1. **解析的 Fourier 係数バウンド** (最優先) — 三角不等式で vel/acc 上界を `bounds` として設定し、探索空間を構造的に制限
2. 制約の段階的評価（安い制約で早期棄却）
3. collision constraint の高速化
4. アルゴリズム変更 — バウンド導入後に SLSQP で feasible 解が得られるか再評価してから判断

## `EarlyStopConfig.min_improvement` 未使用

- `min_improvement: float = 1e-3` フィールドが定義されているが、early stopping ロジックでは厳密な改善 (`cond < best_cond`) のみを判定
- 微小改善を無視する閾値として `min_improvement` を組み込む必要がある
