# 励起軌道 最適化実験・アルゴリズム分析記録

## 2026-03-12: 最適励起軌道の生成と MuJoCo 再生動画

### 実施内容

実装したパイプライン全体を通して最適励起軌道を生成し, MuJoCo 上で再生した動画を出力した.

### 最適化設定

- harmonics=3, duration=5.0s, fps=50
- 3 Monte Carlo restarts, 50 iter/start
- workspace constraint: max_displacement=0.5m
- collision constraint: enabled

### 結果

| 指標 | 値 |
|------|------|
| 条件数 | **2.64** |
| 計算時間 | 110.8s |
| 評価回数 | 5791 |
| best start | 1 |

### 出力

- `debug/excitation_result.json` — 最適化結果 (係数, config)
- `debug/excitation_playback.mp4` — MuJoCo 再生動画 (5s, 30fps, 150 frames)

### 追加スクリプト

- `demos/render_excitation_playback.py` — 最適化結果 JSON → MuJoCo レンダリング → mp4 動画生成 CLI

## 2026-03-13: Config D による本番最適化ラン

### 計算コスト分析

軽量テストラン (harmonics=2, duration=4s, subsample=10, mc=2, mi=5) の実測値をもとに, 各設定の推定計算時間を算出した.

| 設定 | harmonics | duration | subsample | mc | mi | 推定時間 |
|---|---|---|---|---|---|---|
| A (default) | 5 | 10s | 10 | 20 | 200 | **9.2h** |
| B | 5 | 10s | 10 | 10 | 100 | 4.6h |
| C | 5 | 10s | 20 | 20 | 200 | 4.6h |
| **D** | **5** | **10s** | **20** | **10** | **100** | **2.3h** |
| E | 3 | 10s | 10 | 10 | 100 | 2.4h |
| F | 5 | 5s | 10 | 10 | 100 | 2.3h |

Config A は最初に実行を開始したが 6 時間超経過しても終了せず停止.
Config D を早期停止付きで実行した.

### 最適化結果 (Config D + early stop)

| 指標 | 値 |
|---|---|
| 条件数 | **2.5690** |
| 計算時間 | 13789.5s (~3.8h) |
| 評価回数 | 58,178 |
| 完了 restart | 9/10 (patience=5 で早期停止) |
| best start index | 3 |
| **feasible** | **全 restart で False** |

wandb ラン: `balanced-d` (project: `ur5e-excitation`)

### 制約違反の分析

全 restart が infeasible (最小制約マージン < 0) であった.
restart 1 の margin=-0.0125 が最も 0 に近いが, それでも制約を満たしていない.

原因: `subsample_factor=20` により, 制約は 1001 timestep 中 51 点でしか評価されない.
サンプル間の制約違反を optimizer が検知できず, infeasible な解に収束する.

---

## 2026-03-13: Kubus et al. (2008) 論文調査

### 論文概要

"On-Line Estimation of Inertial Parameters Using a Recursive Total Least-Squares Approach" (IROS 2008)

### 軌道パラメータ

| パラメータ | 値 |
|---|---|
| 軌道持続時間 | **1.5s** |
| 最大周波数 | **2 Hz** |
| Fourier 高調波数 | 3 |
| サンプリング周波数 | 250 Hz |
| 条件数 | **7–8** |

### 本プロジェクトとの比較

| | Kubus et al. | 本プロジェクト |
|---|---|---|
| 持続時間 | 1.5s | 10.0s |
| 条件数 | 7–8 | 2.57 |
| 高調波 | 3 | 5 |
| 実行可能性 | 制約充足 | **全 infeasible** |

条件数は κ=7–8 で十分とされる. 本プロジェクトの κ=2.57 は数値的に優れているが,
制約を満たさない軌道は実機で実行不可能であり, feasibility の改善が最優先事項.

---

## 2026-03-13: infeasibility 根本原因の再調査と最適化設定の改善

### 根本原因の訂正

前回セッションで"`subsample_factor=20` により制約が 51 点でしか評価されない"と記載したが,
コード精査により **制約関数は `subsample_factor` の影響を受けず, 常に全 timestep で評価されている** ことを確認.
`subsample_factor` は目的関数(条件数計算)にのみ適用される.

真の原因は以下の複合要因:
1. **duration=10s が過剰**: timestep 数が線形に増え制約評価コストが膨大
2. **60 変数 (6 joints × 5 harmonics × 2)** の広い探索空間で SLSQP が feasible 領域を見つけられない
3. FK ベース制約 (workspace/payload/collision) × 全 timestep × 37 回/iteration (有限差分) が計算コストの ~97%

### 実装した改善

#### 1. duration 短縮 + harmonics 削減
- duration: 10.0s → **3.0s** (Kubus et al. は 1.5s で κ=7-8 を達成)
- harmonics: 5 → **3** (決定変数: 60 → 36)
- base_freq: 0.1 → **1/3 Hz**
- subsample_factor: 10 → **1** (目的関数も全点評価)

#### 2. EE 線速度制約の追加
- `EeVelocityConfig(max_linear_velocity=0.25)` — 0.25 m/s 上限
- site Jacobian × joint velocity で各 timestep の EE 線速度を計算
- `workspace.py` に `_evaluate_ee_linear_velocity()`, `make_ee_velocity_constraint()` を追加

#### 3. 関節速度上限の CLI 設定
- `--dq-max 0.0873` (≈5 deg/s) で全関節の速度上限を一律設定
- 人が近傍にいる環境での安全性を考慮した値

#### 4. 目標条件数 early stop
- `EarlyStopConfig.target_cond` を追加
- **現在の restart が feasible かつ κ ≤ target_cond の場合のみ** 停止
- infeasible な解での早期停止を防止

### 変更ファイル

| ファイル | 変更内容 |
|---|---|
| `workspace.py` | `EeVelocityConfig`, EE 速度評価・制約関数 |
| `constraints.py` | `build_scipy_constraints` に `ee_velocity_config`, `site_name` 引数追加 |
| `optimizer.py` | `target_cond`, `ee_velocity_config` 対応, early stop の feasibility チェック |
| `configs.py` | 新デフォルト値, `ee_max_linear_velocity`, `dq_max`, `early_stop_target_cond` |
| `optimize_excitation_trajectory.py` | 新制約のワイヤリング, `dq_max` による JointLimits 上書き |
| `default.yaml` | 新デフォルト設定 |
| `test_cli_excitation.py` | デフォルト値アサーションの更新 |

### 最適化実行

新設定 (duration=3s, harmonics=3, dq_max=5deg/s, EE vel≤0.25m/s, target κ≤5) で
最適化を開始したが, 30 分以上経過しても最初の restart が完了せず, セッション時間内に結果を得られなかった.

### 残存課題

1. **1 iteration のプロファイリング**: 実測なしに推測で最適化するのは非効率
2. **制約の段階的評価**: 安い制約(joint limits, 純 NumPy)を先に評価し, 違反なら FK 制約をスキップ
3. **解析的 velocity/acceleration 上界**: Fourier 係数の三角不等式で FK ループを排除
4. **並列 Monte Carlo**: restart 数を増やして feasible 解の発見確率を向上

---

## 2026-03-13: 1 iteration プロファイリングと診断ラン

### プロファイリング結果

`scripts/profile_optimizer.py` を作成し, 各コンポーネントの所要時間を実測.

| コンポーネント | 1 回 | 1 iter (×37 FD) | 割合 |
|---|---|---|---|
| collision constraint | 67.2 ms | 2.49 s | 68.5% |
| objective (cond number) | 28.2 ms | 1.04 s | 28.8% |
| payload workspace | 1.05 ms | 39 ms | 1.1% |
| EE velocity | 1.01 ms | 37 ms | 1.0% |
| workspace | 0.56 ms | 21 ms | 0.6% |
| NumPy constraints (3つ) | 0.04 ms | 1.5 ms | ~0% |

- **1 SLSQP iteration = 3.6s** (プロファイルと実測が一致)
- collision constraint が全体の ~69% — Box-Capsule 距離計算が支配的
- EE velocity / dq_max 制約を除外しても 3.56s/iter — これらはボトルネックではない

### 診断ラン (constrained vs unconstrained 並列比較)

2 つの最適化を tmux で並列実行:
- **constrained**: dq_max=5°/s, EE vel≤0.25m/s, 全制約あり
- **unconstrained**: dq_max/EE vel 制約なし, workspace/collision/payload 制約のみ

共通設定: n_monte_carlo=5, max_iter=100, early_stop=true, patience=3, target_cond=10.0

#### 結果

| | unconstrained | constrained |
|---|---|---|
| 条件数 | **2.11** | **2.74** |
| feasible | No (margin=-0.004) | No (全 restart infeasible) |
| 最大関節速度 | 125°/s | 51.6°/s |
| 最大 EE 線速度 | 86.3 cm/s | 54.2 cm/s |
| wall time | 1596s (27min, 4 restart) | 2015s (34min, 5 restart) |
| best restart | #0 | #4 |

wandb ラン: `constrained-diag`, `unconstrained-diag` (project: `ur5e-excitation`)

#### constrained 各 restart

| restart | cond | margin | feasible | time |
|---|---|---|---|---|
| 1 | 7.28 | -0.243 | No | 401s |
| 2 | 3.73 | -0.530 | No | 405s |
| 3 | 6.78 | -0.589 | No | 403s |
| 4 | — | — | — | — |
| 5 | **2.74** | — | No | — |

#### unconstrained 各 restart

| restart | cond | margin | feasible | time |
|---|---|---|---|---|
| 1 | **2.11** | -0.004 | No | 397s |
| 2 | 2.20 | -0.024 | No | 398s |
| 3 | 2.12 | -0.078 | No | 396s |
| 4 (patience stop) | 2.26 | — | No | — |

### 分析

1. **SLSQP は制約を無視しているのではない**: constrained で dq_max=5°/s を要求 → 125°/s (無制約) から 51.6°/s まで低下. 制約を尊重しようとしているが 100 iteration では到達しない
2. **unconstrained でも全 restart infeasible**: collision/workspace 制約だけでも feasible 解を見つけられない (ただし margin=-0.004 と極めて僅差)
3. **SLSQP は本問題に不適切**: 有限差分ベースの局所勾配法では, 36 変数の非凸空間で狭い feasible 領域を探索できない

### 結論: 最適化アルゴリズムの変更が最優先

SLSQP の問題:
- 有限差分で 36+1=37 回/iter の関数評価 (うち collision が 69%)
- 局所探索のみ, restart 間の情報共有なし
- 制約付き非凸問題で feasible 領域に到達困難

候補アルゴリズム:
1. **COBYLA**: 有限差分不要 → 1 iter あたり ~37 倍速い iteration
2. **Differential Evolution**: scipy 組込み, 大域探索, 制約対応
3. **CMA-ES + augmented Lagrangian**: 文献でも excitation optimization に使用実績

---

## 2026-03-13: 最適化アルゴリズムの文献調査と適切性評価

### 調査動機

SLSQP が feasible 解を返さない問題に対し, アルゴリズム選択の妥当性を文献に照らして評価した.

### 問題の性質の整理

| 項目 | 内容 |
|---|---|
| 決定変数 | フーリエ係数 a_{j,k}, b_{j,k} → 2 × 6 × N_h 個 (N_h=5 で 60, N_h=3 で 36) |
| 軌道表現 | q(t) = q0 + w(t) × Σ[a sin(kωt) + b cos(kωt)], 窓関数で境界条件自動満足 |
| 目的関数 | cond(W) = σ_max / σ_min(リグレッサ行列の条件数) |
| 制約 | 関節位置/速度/加速度, ワークスペース, 衝突回避, EE 速度 — 非線形不等式制約 |
| 現行手法 | Multi-start SLSQP (有限差分勾配) |

### SLSQP の問題点(文献に基づく分析)

現行手法は [[Swevers1997_excitation]](../REFERENCES/MAIN.md#Swevers1997_excitation) の SQP アプローチを直接踏襲している. しかし以下の問題がある:

1. **条件数は非平滑**: σ_max/σ_min は特異値の"担い手"が入れ替わる点で微分不可能. 有限差分勾配がこの kink 付近で不正確になる
2. **有限差分コスト**: 36 変数で 1 勾配あたり 37 回の関数評価. collision constraint が 69% を占めるため 1 iter = 3.6s
3. **局所最適**: 非凸問題に対しマルチスタートで対処しているが効率が悪い

### 推奨改善策

#### 1. 目的関数を D-optimal 基準に変更(最優先)

`minimize -log det(W^T W) = -2 Σ log(σ_i)`

- [[Calafiore2001_calibration]](../REFERENCES/MAIN.md#Calafiore2001_calibration) が実験的有効性を実証
- [[Lee2021_excitation_geometric]](../REFERENCES/MAIN.md#Lee2021_excitation_geometric) が理論的正当性を示す
- 条件数と異なり全特異値が正である限り C^∞ で滑らか → 有限差分勾配の精度が向上
- 条件数はバリデーション指標として報告すればよい

#### 2. ソルバの改善

- **短期**: 目的関数を D-optimal に変えるだけで SLSQP の収束改善が見込める
- **中期**: [[Tian2024_virtual_constraints]](../REFERENCES/MAIN.md#Tian2024_virtual_constraints) が示すように CasADi + IPOPT で解析的勾配を供給すれば大幅に高品質な解を短時間で得られる (同論文: cond=51/10min vs メメティック法 57890/60min超)
- **代替**: CMA-ES でグローバル探索 → SLSQP でローカル研磨のハイブリッド

#### 3. ドロップイン改善

- `log(cond(W))` への変更 — スケール安定化
- `method='COBYLA'` — 有限差分不要, 非平滑目的に robust

### 参考文献

- [[Swevers1997_excitation]](../REFERENCES/MAIN.md#Swevers1997_excitation) — フーリエ級数 + SQP の原論文
- [[Lee2021_excitation_geometric]](../REFERENCES/MAIN.md#Lee2021_excitation_geometric) — 幾何学的基準・解析的勾配
- [[Tian2024_virtual_constraints]](../REFERENCES/MAIN.md#Tian2024_virtual_constraints) — グラミアン代理指標 + IPOPT
- [[Calafiore2001_calibration]](../REFERENCES/MAIN.md#Calafiore2001_calibration) — D-optimal 基準の実験的検証
- [[Kubus2008_rtls]](../REFERENCES/MAIN.md#Kubus2008_rtls) — 回帰行列構成の基礎文献

---

## 2026-03-13: D-optimal 目的関数の実装と診断ラン

### 実装内容

`objective.py` に D-optimal 目的関数を追加し, `optimizer.py` で `objective_type` による切替を実装した.

#### `d_optimal_objective()` — `objective.py`

- 目的関数値: `-2 * Σ log(σ_i)` (リグレッサの全特異値の対数和の符号反転)
- C^∞ で滑らか(条件数と異なり特異値交差点での kink がない)
- 特異値フロア `1e-30` で log(0) を回避

#### `d_optimal_with_cond()` — `objective.py`

- 1 回の SVD から D-optimal 値と条件数の両方を返すヘルパー
- `optimizer.py` が D-optimal で最適化しつつ条件数をバリデーション指標として報告するために使用

#### `optimizer.py` の変更

- `OptimizerConfig.objective_type`: `"d_optimal"` (デフォルト) / `"condition_number"`
- `optimize()` 内で `d_optimal_with_cond()` または `condition_number_objective()` を自動切替
- 条件数は `objective_type` に関わらず常にトラッキング

#### CLI — `configs.py`, `optimize_excitation_trajectory.py`

- `--objective` フラグ: `"d_optimal"` / `"condition_number"`
- デフォルトを `d_optimal` に変更

### 診断ラン結果: D-optimal vs condition_number

2 つの D-optimal 診断ランを tmux 並列実行(n_mc=5, max_iter=100, early_stop, patience=3):

#### D-optimal 目的関数

| | unconstrained | constrained (dq≤5°/s, EE≤25cm/s) |
|---|---|---|
| 条件数 | **10.72** | **6.68** |
| D-opt 値 | -157.3 | -153.0 |
| feasible | No (margin=-27.0) | No (margin=-60.3) |
| wall time | 1992s (33min) | 1754s (29min) |
| restarts | 5/5 | 5/5 (patience stop) |

#### 比較: condition_number 目的関数(前回)

| | unconstrained | constrained |
|---|---|---|
| 条件数 | **2.11** | **2.74** |
| feasible | No (margin=-0.004) | No (全 infeasible) |

### 制約別違反分析

D-optimal 結果に対し, 制約ごとのマージンを個別評価:

| 制約 | unconstrained margin | constrained margin |
|---|---|---|
| joint_position | +1.49 (OK) | +2.47 (OK) |
| joint_velocity | **-3.56** | **-5.72** |
| joint_acceleration | **-27.03** | **-60.29** |
| workspace_ee | **-0.04** | **-0.19** |
| collision | **-0.11** | **-0.11** |

### 根本原因の特定: 文献との差異分析

D-optimal 目的関数は文献(Calafiore 2001, Lee 2021)では成功しているが, 本プロジェクトでは全 restart が infeasible. 体系的に文献との差異を調査した結果, 以下が判明:

#### 1. 制約の種類が根本的に異なる

| 制約 | 文献 (Swevers, Lee, Calafiore, Kubus, Tian, Rackl) | 本プロジェクト |
|---|---|---|
| 関節位置 | Yes | Yes |
| 関節速度 | Yes (一部) | Yes |
| 関節加速度 | Yes (一部) | Yes |
| **EE ワークスペース変位** | **No** | **Yes** (≤0.5m) |
| **衝突回避** | **No** | **Yes** (自己衝突+地面+ペイロード) |

調査した6本の論文のいずれも衝突回避やワークスペース変位制約を使用していない.

#### 2. D-optimal が係数を制限の33倍まで押し込む

D-optimal は情報量(全特異値の積)を最大化するため, 大きな振幅を要求する. Fourier 第3高調波の加速度安全上限は係数 0.20 rad だが, 最適化結果は 6.64 rad まで到達(33倍超過).

#### 3. 根本原因の修正

前回セッションの結論"目的関数の非滑らかさが主因"は不正確であった. 真の根本原因は:

> **探索空間が制約に対して広すぎる**. SLSQP はバウンド制約なしの広い空間で, 文献にない衝突・ワークスペース制約の狭い feasible 領域を見つけられない. 目的関数の滑らかさ改善は副次的効果に留まる.

### 対策の優先順位再編

| 順位 | 施策 | 理由 |
|---|---|---|
| **1** | 解析的 Fourier 係数バウンド(三角不等式) | 加速度違反が最大の違反源. バウンドで vel/acc 違反を構造的に排除 |
| **2** | 制約の段階的評価(安い制約で早期棄却) | collision/workspace の非凸制約評価回数を削減 |
| **3** | collision constraint の高速化 | 全体の 69% を占めるボトルネック |
| **4** | アルゴリズム変更 (COBYLA/IPOPT) | バウンド導入後に再評価. 探索空間の縮小で SLSQP でも動く可能性 |
| **5** | 並列 Monte Carlo | 正解性ではなく速度の問題 |

D-optimal 目的関数は実装済みのため維持するが, feasibility 改善の主要因ではないことが判明した.

---

## 2026-03-13: harmonics=3 vs harmonics=5 比較実験

### 目的

Fourier バウンド (dq_max=2.0 rad/s) 下で高調波数の影響を比較する.

### 共通設定

- duration=5.0s, base_freq=1/3 Hz, dq_max=2.0 rad/s
- D-optimal 目的関数, Fourier bounds 有効, 加速度制約なし, EE 速度制約なし
- n_monte_carlo=20, max_iter=100, early_stop (patience=10, target_cond=5.0)

### 結果

| | h3 (harmonics=3) | h5 (harmonics=5) |
|---|---|---|
| wandb ラン | `h3-100iter` (q33flt3k) | `h5-100iter` (xkvsvxlk) |
| 決定変数 | 36 | 60 |
| **条件数** | **6.87** | **5.44** |
| D-opt 値 | -95.0 | -98.3 |
| feasible | No (margin=-3e-7) | **Yes** |
| wall time | 2913s (~49min) | 2792s (~47min) |
| restarts 完了 | 11/20 (patience stop) | 20/20 |
| evaluations | 17,286 | 16,382 |
| dq_max 実測 | 1.09 rad/s | 1.03 rad/s |
| ddq_max 実測 | 5.22 rad/s² | 7.29 rad/s² |
| collision margin | +0.007 | +0.025 |
| payload_workspace margin | -3e-7 | 0.000 |

### h5 制約マージン詳細

| 制約 | マージン | 状態 |
|---|---|---|
| joint_position | +4.53 | OK |
| workspace | +0.27 | OK |
| payload_workspace | 0.000 | OK (境界上) |
| collision | +0.025 | OK (2.5cm 余裕) |

### h5 軌道統計

| 指標 | 値 |
|---|---|
| dq_per_joint_max | [0.98, 0.71, 0.71, 0.71, 1.03, 0.85] rad/s |
| ddq_per_joint_max | [7.16, 5.00, 5.00, 5.00, 7.29, 6.54] rad/s² |

### 分析

1. **harmonics=5 が明確に優越**: 条件数 5.44 vs 6.87(21% 改善), collision margin 2.5cm vs 0.7cm
2. **計算時間は同等**: h5 の方がむしろ速い(47min vs 49min). 決定変数は多いが SLSQP が早く収束
3. **h3 は patience stop**: restart 11 で打ち切り. h5 は 20 restart 完走しより良い解を探索
4. **dq_max の利用率は約 50%**: バウンド 2.0 rad/s に対し実測 ~1.0 rad/s. 衝突・ワークスペース制約が支配的

### ワークスペース下端の修正

ワークスペース下端が地面から 3cm であり, collision safety_margin=5cm と不整合であることを発見.
下端を **7cm** に修正(`ur5e_with_box.xml`: center_z=0.479, half_z=0.409).

### 結論

- harmonics=5, dq_max=2.0, duration=5.0s を推奨設定として採用
- 現時点のベスト: **κ=5.44** (feasible, Kubus et al. の κ=7-8 を上回る)

## 2026-03-23: FT オフセット付き最適化 + フーリエバウンド拡張

### フーリエバウンドの加速度対応

`compute_fourier_bounds()` を拡張し, 速度と加速度の両方からバウンドを導出するようにした.
各係数 (j, k) について速度バウンドと加速度バウンドの厳しい方(min)を採用する.

### FT オフセットなし最適化(base_freq=0.2, duration=5s, n_workers=6)

| dq_max | 条件数 | 速度違反 | workspace違反 | feasible |
|---|---|---|---|---|
| 1.5 rad/s | **1.93** | 0.0014 | 0.0006 | No(僅差)|
| 1.0 rad/s | **2.20** | 0.0019 | 0.0002 | No(僅差)|

### FT オフセットあり + フーリエバウンドなし(中断)

dq_max=1.0, with_ft_offset, column_scale, n_workers=4 で実行.
フーリエバウンドなしのため加速度制約を大幅に違反(margin=-30 等)し, 中断.

### FT オフセットあり + フーリエバウンドあり(dq_max=1.0, n_workers=4)

| ddq_max (rad/s²) | 条件数 | feasible | 1リスタート時間 |
|---|---|---|---|
| 1.0 | **41.65** | True | ~60-80s |
| 2.5 | **18.94** | True | ~140s |
| 5.0 | **14.41** | True | ~130s |
| 7.5 | **13.67** | True | ~150s |

### 分析

1. **フーリエバウンドの効果が顕著**: バウンドなしでは全て infeasible, ありでは全て feasible
2. **劇的な高速化**: 30分/リスタート → 1-2分/リスタート
3. **ddq_max と条件数のトレードオフ**: 加速度制限を緩めるほど条件数改善. 5.0→7.5 では鈍化(14.41→13.67)
4. **ddq_max=5.0 が実用的な選択**: 条件数 14.4 で feasible, 7.5 との差は小さい

### FT オフセットあり + バウンドなし(duration=3s/5s, n_workers=6)

dq_max のみ制約(ddq_max=0), フーリエバウンドなし:

| duration | dq_max | 条件数 | feasible | 実測 dq | 実測 ddq | 違反 |
|---|---|---|---|---|---|---|
| 3s | 1.0 | **2.45** | False | 1.70 | 12.82 | 速度 0.70 (大幅) |
| 3s | 1.5 | **2.05** | False | 1.51 | 12.44 | 速度 0.005 (僅差) |
| 5s | 1.0 | **1.54** | False | 5.98 | 34.78 | 速度 4.98 (大幅) |
| 5s | 1.5 | **1.57** | False | 1.50 | 10.36 | 速度 0.002 (僅差) |

dq_max=1.0 ではバウンドなしだと SLSQP が制約を無視(実測 6 倍). dq_max=1.5 はほぼ feasible.

### FT オフセットあり + フーリエバウンドあり duration/ddq_max スイープ (dq_max=1.5, n_workers=3)

| ddq_max | 3s 条件数 | 3s feasible | 5s 条件数 | 5s feasible |
|---|---|---|---|---|
| 10.0 | 10.36 | True | **9.17** | True |
| 7.5 | 11.35 | False* | **9.64** | True |
| 5.0 | 14.20 | True | **10.74** | False* |
| 2.5 | 24.62 | True | **17.77** | True |

*payload_workspace 違反が実質ゼロ(数値誤差レベル)

### 分析

1. **5s は全条件で条件数が改善**(3s 比 10〜30%)
2. **dq_max=1.5 に対し実測は 0.3〜0.75**: フーリエバウンドが速度を保守的に制限
3. **ddq_max=10.0 でも実測は 3.15〜3.28**: 加速度バウンドより速度バウンドが支配的
4. infeasible 2件はいずれも payload_workspace の境界上(margin=-0.000000)


## 2026-05-07: シーン更新後の baseline と 3 仮説検証

### 実施内容

`scene_with_box.xml` のジオメトリを更新(base body Z 軸 +180° 回転, workspace_region を 50×40×45 cm に縮小, payload_box を 10×25×20 cm の薄板に変更)し, 新しい task setup で励起軌道最適化を実行."最適化結果が作業空間全体を活かしていない"という指摘に対し, 3 仮説を並列検証した.

### Baseline (3s, mc4, Fourier bounds, dq=1.5, ddq=10)

| 項目 | 値 |
|---|---|
| 条件数 | **7.40** |
| feasible | True (4 リスタート中 1) |
| payload_workspace margin | 0.0(境界張り付き)|
| 実測 dq_max | 0.70 (設定 1.5 の 47%) |
| 実測 ddq_max | 3.25 (設定 10 の 33%) |
| Wall time | 506 s |

### 3 仮説並列検証結果

| 仮説 | 設定 | 最良 cond | feasible | 結論 |
|---|---|---|---|---|
| baseline | 3s, mc4, fb | 7.40 | ✓ | 基準 |
| A: 局所最適 | mc20 (max_iter=100) | 7.57 (feasible 中) / 7.10 (infeasible) | 4/20 | わずかに悪化, baseline は実質大域最適 |
| B: bounds 影響 | `--no-use-fourier-bounds` | 2.04 | 全て ✗ (margin -0.63) | bounds は制約遵守に必須 |
| C: duration | 5s | 8.61 | 全て ✗ | duration 単独では改善せず |

### 分析

1. **baseline は実質大域最適**: mc20 (5 倍リスタート) でも最良 feasible cond は 7.57 で baseline 7.40 を超えられない. SLSQP は狭い AABB 内で条件数 7.4 を達成する解を効率的に見つけている.
2. **Fourier bounds の役割**: ペイロードの workspace 制約は順運動学が絡むため Fourier 係数 bounds に変換できず inequality のまま残る. bounds を外すと SLSQP の per-timestep 制約 (`min(margin)` 関数) は非滑らかで収束しづらく, cond は劇的に改善するが全例 infeasible になる.
3. **duration 延長が効かない理由**: 表現自由度は増えるが, 新しい狭い AABB (50×40×45 cm) に対し既に最適解が見つかっているため, 追加の自由度が活用されていない.
4. **"作業空間全体を活かさない"現象は仕様通り**: 目的関数は条件数最小化のみで, 空間カバレッジ項は含まれていない. 条件数 7.4 が達成可能なら SLSQP は AABB 全体を埋める動機を持たない. 空間カバレッジを増やしたい場合は目的関数に `λ · (1 / coverage)` 等を加える設計変更が必要.

### 残タスク

- `notes/TODO.md` に"目的関数へのワークスペースカバレッジ項導入の検討"を追加

## 2026-05-26: FT センサ方式への移行と慣性パラメータ同定検証

### 背景

これまでの wrench は解析 regressor (`compute_wrench_from_parameters`, `I·a` 逆動力学) で計算しており, 実機 FT センサと符号・成分順・物理量が一致しなかった. MuJoCo の force/torque sensor を導入し実機 FT に近づけた.

### 実施した変更

1. **MJCF にセンサ追加** ([assets/ur5e/mjcf/ur5e_with_box.xml](../../assets/ur5e/mjcf/ur5e_with_box.xml)): `payload_box_mount` 原点に `ft_sensor` site (tool0 と pos/quat 一致を確認), `<force>`/`<torque>` センサを定義.
2. **PD playback の 3 バグ修正** ([src/mjwarp_ur5e/identification/execution.py](../../src/mjwarp_ur5e/identification/execution.py)):
   - バグ1: `data.ctrl = tau` (トルク) → `data.ctrl = q_des` (目標角度). MJCF actuator は position-velocity servo (kp=2000/500, kd=400/100) で ctrl は目標角度を取る.
   - バグ2: 1 ステップ `mj_step` → `n_substeps`(=5) ループ. 軌道 dt=0.01s に対しモデル timestep=0.002s なので 5 回回さないと物理時間が 1/5 しか進まない.
   - バグ3: `ddq_meas = ddq_des` (理想) → `data.qacc` (実測).
3. **settling phase 追加**: 記録前に初期目標で 1s (`settle_time`) 整定. home keyframe は静的平衡でない (servo は重力補償しないため qacc=18.67 が湧く) ので, 整定しないと初期数ステップに大きな加速度過渡が乗る. 整定で frame0 の ddq_max が 2.97→0.003 に低減.
4. **wrench をセンサ読み取り + 符号反転**: `data.sensordata` から `[Fx,Fy,Fz,Mx,My,Mz]` (tool0 frame) を読む. MuJoCo 規約は parent→child の支持力 (静止 Fz=-9.81) なので, 実機 FT 規約 (child→parent, 荷重がフランジに及ぼす力) に合わせ全成分を符号反転 (静止 Fz=+9.81).

### センサが測る物理量 (コード+MuJoCo doc で確定)

- force/torque sensor は site が属する body (child=`payload_box_mount`) と親 (parent=`wrist_3_link`) 間の相互作用力. `mj_rnePostConstraint` で全力 (接触・拘束・外乱含む) を計算.
- frame: site frame = tool0. 基準点: site 原点 = フランジ面 (CoM ではない).
- 符号反転後は"荷重 (ペイロード) がフランジに及ぼす力・トルク"= 実機 FT 相当.

### 慣性パラメータ同定検証 (Kubus 2007 eq.5)

[[papers/Kubus-IROS2007-On-line_Rigid_Object/on-line-rigid-object-recognition-and-pose-estimation-based-on-inertial-parameters|Kubus+ 2007]] の eq.5 回帰行列 V (`[Sf;Sτ]` 順, sensor frame) を実装し, FT センサ実測 wrench を y として LS 同定.

| パラメータ | 真値 | LS 推定 | 誤差 |
|---|---|---|---|
| 質量 | 0.927 kg | 0.927 | **0.0%** |
| 重心 | [0, -0.1, 0.1] | [0, -0.1, 0.1] | **0.0%** |
| 残差ノルム | - | 0.0 | - |

### 重大な落とし穴 — 重力の二重計上 (FTA で特定)

当初 LS 同定で質量が**ちょうど半分** (0.464 kg) になった. 原因切り分け (fault-tree-debug 相当) の結果:

- **MuJoCo `mj_objectAcceleration` が返す線形加速度は重力を含む proper acceleration** (静止時 a_lin = -g_sensor).
- 論文 eq.1 の `f = m·(a - g)` の `(a - g)` に対応するのは, この `mj_objectAcceleration` の出力**そのもの**.
- なのに当初コードは `a_lin - g_sensor` とさらに重力を引き, `(-g) - g = -2g` の二重計上. force の m 列が 2 倍になり質量が半分に推定された (係数 2 で完全に説明).
- 修正: `mj_objectAcceleration` の線形成分をそのまま `(a-g)` として使う → 質量・重心が誤差 0% で復元.

### 分析

- 慣性対角推定値 [0.0265, 0.0131, 0.0149] は `m.body_inertia` [0.0079, 0.0039, 0.0056] と異なるが, これは前者がセンサ原点 (フランジ面) まわり, 後者が CoM まわりという基準点の違い. 平行軸定理で変換すれば一致するはず (残差ゼロが推定の正しさを保証).
- Kubus 2007 §IX が指摘する"シミュレーション条件数と実機条件数の乖離"(sim 6.5-8.2 vs exp 14.4-23.4) は, 当プロジェクトの"最適化軌道を追従させると wrench が想定と合わない"現象と同根の課題.

### 残課題

- FT センサ整合の regressor を `identification/` に正式実装 (現状は検証スクリプトのみ)
- 推定慣性の CoM まわりへの変換と真値一致確認
- PD 追従誤差 (max_pos_err≈0.097 rad) が同定精度に与える影響の評価

## 2026-05-26 ペイロードを Robotiq 2F-85 グリッパへ切替

box ペイロードを Menagerie の Robotiq 2F-85 グリッパに置き換え, 励起軌道を PD 追従させて FT wrench を記録・プロットし, 再生動画も生成した.

### 方針 (ユーザー確定)

- グリッパは多体・可動・自己接触構造だが, 指を全開で固定し 1 剛体相当として扱う.
- 実装は actuator を残し再生中 `ctrl=0` (全開) 保持. 指関節を完全拘束する案は不採用.
- 資産は google-deepmind/mujoco_menagerie の `robotiq_2f85/` を取得 (BSD-2-Clause).

### 成果物

- 資産: `assets/robotiq_2f85/` (2f85.xml + assets/ STL 8 種 + LICENSE).
- シーン: `assets/ur5e/mjcf/ur5e_with_gripper.xml`, `scene_with_gripper.xml`.
  - グローバル meshdir を廃し各 mesh で明示パス指定 (UR5e と robotiq の asset ディレクトリ共存のため).
  - `<option>` は `integrator=implicitfast` に 2F-85 由来の `cone=elliptic impratio=10` を併合.
  - グリッパの `base` ボディは UR5e の `base` と名前衝突するため `gripper_base` に改名 (contact exclude も追随).
  - 2F-85 の material (`black`/`gray` 等) は UR5e と衝突するため `gripper_*` に改名.
  - FT センサ site (`ft_sensor`) を `gripper_mount` 原点 (= attachment_site, tool0 整合) に配置.
  - keyframe home は qpos 14 (UR5e 6 + グリッパ 8 関節, 全 0=開) + ctrl 7 (UR5e 6 + グリッパ 1=0 開).
- npz: `results/playback_gripper.npz`, プロット `results/flange_wrench_gripper.png`,
  動画 `results/playback_gripper.mp4` (4 視点グリッド).

### コード修正

- `identification/execution.py`: 単一剛体慣性 `params` の解決を遅延化. FT センサ経路では
  `params` 不使用のため, FT センサがある場合は `body_inertial_parameters_from_model` を呼ばない.
  これにより `cfg.body_name` (=`payload_box_mount`, グリッパシーンでは不在) でも追従が走る.

### 検証

- モデルロード: `nu=7` (UR5e 6 + グリッパ 1), `nq=14`, `nsensor=2`.
- 静止時 FT バイアス `Fz=10.325 N` がグリッパサブツリー重量 (1.0526 kg × 9.81 = 10.326 N) と一致.
- 追従誤差 max_pos=0.0965 rad は box 時 (≈0.097) と同等.
- wrench プロファイルは滑らか (高周波振動なし). 指関節 qpos は全行程 0 (開) 維持.
- 関連テスト 14 件 pass, ruff clean.

### 留意点 (トレードオフ)

- 全開保持でもグリッパは可動 (コンプライアンス) のため, 厳密には単一剛体 10 パラメータ
  同定とは不整合. `body_inertial_parameters_from_model` はグリッパ全体の合成慣性を返さない.
  本作業は wrench 記録・動画が目的のため許容. 同定精度評価にはサブツリー合成慣性の算出が別途必要.

## FT センサ実測経路の同定破綻を FTA で診断・修正 (バグ A/B/C)

nested payload (赤+青, 0.9 kg) で `test_integration::test_playback_and_estimation_pipeline` の
復元質量が破綻 (0.0177 kg) したのを fault-tree-debug + 現物検証で 3 バグに分解し, B/C を修正.

- **バグ A — 誤診**:"regressor の重力二重計上"と当初診断したが, `set_model_state` は cacc を
  populate せず `mj_objectAcceleration` が重力を含まないため, `types.py:24` の手動 gravity 減算は
  正しい (静止で force = m·g を確認, 減算除去で 0 に退化). コード変更せず.
- **バグ B — 修正**: オープンループ FT 経路の `mj_forward` を `mj_inverse` に置換. `mj_forward` は
  cfrc_int 計算前に qacc を順動力学解で上書きし FT が ddq を無視していた. `mj_inverse` は qacc 保持.
- **バグ C — 修正**: `sampling.py` の手組み kinematics が 2 フレーム混在 (objVel/Acc を mjOBJ_BODY で
  呼ぶと xmat を body-x まわり -90° 回した frame を返す vs J@qacc/gravity は xmat). 解決: objVel/Acc を
  **mjOBJ_SITE (ft_sensor)** で取得 (FT と同フレーム・同基準点), classical→spatial 補正 `a_lin -= ω×v`,
  cacc は mj_inverse で populate (重力込み) ので手動 gravity 減算なし, 符号 `+[torque;force]`.
  site 基準経路は `site_name` オプションで追加し旧 body 経路は温存.

結果: `test_integration` の FT 同定が真値 0.9 kg を rel 0.1 で復元して PASS. 全 98 件中 pass, 残 3 件は
pre-existing (単一 payload 時代の stale 期待値 + 旧 body 経路の cacc 依存破綻, ISSUES に記録).

## 旧 body 経路を削除し ft_sensor site 基準に一本化 (frame A 根絶)

上記の pre-existing 3 件は旧 body 経路 (frame A + cacc 二重計上の 2 バグ) に起因. ユーザー判断で旧経路を
完全削除し ft_sensor site 基準に一本化した. `sample_body_kinematics` は `site_name` デフォルト `"ft_sensor"`
で常に site 経路 (`_sample_site_kinematics`) を通り, body 分岐と `_body_acceleration_from_qacc` を削除.
`site_name` デフォルト化で objective/optimizer/execution/data_buffer の明示配線は不要 (ft_site_name config は
YAGNI で省略). FT 無しモデルは site なし→ValueError で非対応. pre-existing 3 件も解消: 期待値を nested 構成へ
更新 (mass=0.9, first_moments=[0,0,0.1425], inertia=diag([0.0315,0.0275,0.0055]), payload_box_red
half=[0.05,0.15,0.05]/offset=[0,0,0.05]), `test_static_pose` は `set_model_state(qpos,0,0)` を挟んで site
経路へ乗せ force norm=8.829 を厳密照合. **全 101 テスト pass, ruff clean.**
