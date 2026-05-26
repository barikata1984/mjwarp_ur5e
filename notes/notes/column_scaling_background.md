# FT オフセット拡張時の列スケーリング — 数理的背景

## 問題設定

FT センサオフセットを含めた回帰モデル:

$$
\begin{bmatrix} f \\ \tau \end{bmatrix}_t = \begin{bmatrix} I_6 \mid V_t \end{bmatrix} \begin{bmatrix} b \\ \phi \end{bmatrix}
$$

- $b \in \mathbb{R}^6$: FT センサオフセット（力3成分 + トルク3成分）
- $\phi \in \mathbb{R}^{10}$: 慣性パラメータ（質量, 重心×3, 慣性テンソル×6）
- $V_t \in \mathbb{R}^{6 \times 10}$: 時刻 $t$ の回帰行列（運動状態に依存）
- $I_6$: 6×6 単位行列（オフセットは運動状態に依存しない定数項）

$T$ ステップ積み上げた回帰行列:

$$
W = \begin{bmatrix} I_6 & V_1 \\ I_6 & V_2 \\ \vdots & \vdots \\ I_6 & V_T \end{bmatrix} \in \mathbb{R}^{6T \times 16}
$$

## 問題: 列ノルムの不均衡

### オフセット列（左側6列）

$I_6$ が $T$ 回積み上がるので、各列の L2 ノルムは:

$$
\|w_j^{\text{offset}}\|_2 = \sqrt{T}, \quad j = 1, \ldots, 6
$$

### 物理列（右側10列）

ノルムは運動状態（角速度、角加速度、線形加速度）に依存:

$$
\|w_j^{\text{physics}}\|_2 = \mathcal{O}(1) \sim \mathcal{O}(10)
$$

### ノルム比の具体例

| $T$ (ステップ数) | $\sqrt{T}$ | 典型的な物理列ノルム | 比率 |
|---|---|---|---|
| 100 | 10.0 | ~5 | 2x |
| 500 | 22.4 | ~5 | 4.5x |
| 1000 | 31.6 | ~5 | 6.3x |

ステップ数が増えるほど不均衡が拡大する。

## 特異値分解（SVD）と特異値の算出

条件数の算出に用いる特異値は、特異値分解（SVD）によって得られる。

### SVD の定義

任意の行列 $W \in \mathbb{R}^{m \times n}$（$m \geq n$）は以下のように分解できる:

$$
W = U \Sigma V^T
$$

- $U \in \mathbb{R}^{m \times m}$: 左特異ベクトルの直交行列
- $\Sigma \in \mathbb{R}^{m \times n}$: 対角成分に特異値を持つ行列
- $V \in \mathbb{R}^{n \times n}$: 右特異ベクトルの直交行列

$\Sigma$ の対角成分が特異値であり、慣例として降順に並べる:

$$
\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_n \geq 0
$$

### 特異値の幾何学的意味

特異値は、行列 $W$ が空間を「引き伸ばす」度合いを表す。
$V$ の各列（右特異ベクトル）が入力空間の方向、$U$ の各列が出力空間の方向に対応し、
$\sigma_i$ はその方向に沿った伸縮の倍率である。

本実装の回帰行列 $W \in \mathbb{R}^{6T \times 16}$ の場合:
- 入力空間は 16 次元のパラメータ空間（6オフセット + 10慣性パラメータ）
- 出力空間は $6T$ 次元の力/トルク観測空間
- $\sigma_i$ はパラメータ空間の $i$ 番目の主方向がどれだけ観測に寄与するかを示す

### 特異値と列ノルムの関係

特異値は列ノルムによって上界が定まる。具体的には:

$$
\sigma_{\max} \leq \sqrt{\sum_{j=1}^{n} \|w_j\|_2^2}
$$

また、1つの列のノルムが他より極端に大きいと、最大特異値はその列に支配される一方、
最小特異値は他の列の情報量に依存する。この不均衡が条件数を悪化させる。

### 条件数の定義

条件数は最大特異値と最小特異値の比として定義される:

$$
\kappa(W) = \frac{\sigma_{\max}}{\sigma_{\min}} = \frac{\sigma_1}{\sigma_n}
$$

- $\kappa = 1$: 全方向の情報量が均等（理想的）
- $\kappa$ が大きい: 特定の方向の情報が乏しく、推定が不安定になる
- $\kappa = \infty$: ランク落ち（ある方向の情報が完全にゼロ）

### 本実装での計算

本実装（`regressor.py`）では、SVD を直接計算して特異値を取得している:

```python
singular_values = np.linalg.svd(matrix, compute_uv=False)
# compute_uv=False: 特異値のみ計算（U, V は不要）
condition_number = singular_values[0] / singular_values[-1]
```

`compute_uv=False` を指定することで、特異値のみを効率的に計算している
（$U$, $V$ の構築を省略できるため、計算量が削減される）。

## 条件数への影響

条件数 $\kappa(W) = \sigma_{\max} / \sigma_{\min}$ は列ノルムの比に強く依存する。

直感的には:
- ノルムが大きい列 → SVD で大きな特異値に寄与
- ノルムが小さい列 → 相対的に「埋もれる」

この結果、条件数が**パラメータのスケール差**によって人為的に悪化する。
これは物理的な情報不足（励起不十分）とは区別すべき**見かけの悪条件**である。

## 列スケーリングの操作（本実装で採用）

16列それぞれについて、その列の L2 ノルム（全行の二乗和の平方根）を計算し、
その列の全要素をそのノルムで除する。オフセット列か物理列かは区別せず同一の操作を適用する。

$$
\tilde{W} = W \cdot D^{-1}, \quad D = \text{diag}(\|w_1\|_2, \|w_2\|_2, \ldots, \|w_{16}\|_2)
$$

結果として全列が L2 ノルム = 1 に揃う。
これは等価的に、変換後のパラメータ $\tilde{\theta} = D\theta$ について最適化することに相当する。

### 効果

- 全列が単位ノルムに揃う → スケール差による人為的な悪条件が除去される
- 条件数が「各パラメータ方向の情報量の比」を正しく反映するようになる

## 数学的根拠と文献

### 1. 列均衡化 — 本実装で採用している手法

本実装の列スケーリングは、数値線形代数における列均衡化（column equilibration）に基づく。
Golub & Van Loan [[1]](#ref-golub2013) は、最小二乗問題において列均衡化が条件数改善のための
標準的な前処理であることを示している:

$$
\kappa(\tilde{W}) \leq \sqrt{n} \cdot \kappa_{\text{opt}}
$$

ここで $n$ は列数、$\kappa_{\text{opt}}$ は最適なスケーリングでの条件数。

ロボット動的パラメータ同定の文脈では、Swevers et al. [[2]](#ref-swevers1997) および
Gautier & Khalil [[3]](#ref-gautier1992) が、回帰行列の条件数最小化による最適励起軌道設計に
おいて列スケーリングを標準的な前処理として適用している。Gautier & Khalil はこの手法を
明示的に「列ノルム正規化後の条件数最小化」として定式化しており、本実装と同一のアプローチである。

### 2. D-optimal 設計 — 条件数最小化の代替手法

本実装では条件数最小化に加え、D-optimal 基準（$\max \det(W^T W)$）も目的関数として選択可能である。
Pukelsheim [[4]](#ref-pukelsheim1993) は、最適実験計画において情報行列 $M = W^T W$ の
行列式がパラメータのスケーリングに依存することを指摘しており、D-optimal 設計においても
列スケーリングの重要性が認識されている。

## 注意点

### D-optimal 目的関数とスケーリングの関係

D-optimal 目的関数 $\max \det(W^T W)$ は列スケールに対して**不変ではない**。
列をスケーリングすると行列式は以下のように変換される:

$$
\det(\tilde{W}^T \tilde{W}) = \det(D^{-1} W^T W D^{-1}) = \frac{\det(W^T W)}{\prod_j \|w_j\|_2^2}
$$

スケーリングなしの D-optimal は、ノルムの大きいオフセット列が存在するだけで
行列式が大きくなるため、物理列の情報量改善と区別がつかない。
したがって D-optimal を使用する場合も列スケーリングは必須の前処理である。

条件数（$\sigma_{\max}/\sigma_{\min}$）も厳密にはスケール不変ではないが、
比の構造を持つため D-optimal（積の構造）ほど極端にノルムに引きずられない傾向がある。

### スケーリングありの利点

- 最適化が安定する（D-optimal がオフセット列に引きずられない）
- 条件数がパラメータ間の情報バランスを正しく反映する

### スケーリングありの注意点

- 報告される条件数は**正規化後の値**であり、生の回帰行列の条件数とは異なる
- 最適化された軌道で実際に推定精度が改善するかは別途検証が必要
- スケーリングは軌道最適化の目的関数にのみ影響し、推定時のアルゴリズム（LS/TLS/RTLS）には影響しない

### スケーリングなしの問題

- $T$ が大きいとき、D-optimal 目的関数がオフセット列のノルム増大に引きずられる
- 物理パラメータの励起を犠牲にしてオフセット列のノルムをさらに増大させる方向に最適化が進み、制約違反方向に暴走する傾向がある

## 代替アプローチ（未実装）

docs/ISSUES.md に記載の代替案:

1. **重み付き D-optimal**: $\max \det(W^T \Lambda W)$ で重み行列 $\Lambda$ によりパラメータの重要度を指定
2. **ブロック対角前処理**: オフセットブロックと物理ブロックを別々にスケーリング
3. **オフセット列のサブサンプリング**: $I_6$ の繰り返しを間引いてノルムを直接制御

## References

<a id="ref-golub2013"></a>
[1] G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th ed. Baltimore: Johns Hopkins University Press, 2013.
[DOI: 10.1137/1.9781421407944](https://doi.org/10.1137/1.9781421407944)

<a id="ref-swevers1997"></a>
[2] J. Swevers, C. Ganseman, D. B. Tukel, J. De Schutter, and H. Van Brussel, "Optimal robot excitation and identification," *IEEE Trans. Robot. Autom.*, vol. 13, no. 5, pp. 730–740, 1997.
[DOI: 10.1109/70.631234](https://doi.org/10.1109/70.631234)

<a id="ref-gautier1992"></a>
[3] M. Gautier and W. Khalil, "Exciting trajectories for the identification of base inertial parameters of robots," *Int. J. Robot. Res.*, vol. 11, no. 4, pp. 362–375, 1992.
[DOI: 10.1177/027836499201100408](https://doi.org/10.1177/027836499201100408)

<a id="ref-pukelsheim1993"></a>
[4] F. Pukelsheim, *Optimal Design of Experiments*. New York: Wiley, 1993. Reprinted by SIAM, Classics in Applied Mathematics, No. 50, 2006.
[DOI: 10.1137/1.9780898719109](https://doi.org/10.1137/1.9780898719109)
