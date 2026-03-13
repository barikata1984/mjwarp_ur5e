# 参考文献マスタ

### Kubus2008_rtls

**On-Line Estimation of Inertial Parameters Using a Recursive Total Least-Squares Approach**
Kubus, D., Kroeger, T., Wahl, F. M. — IEEE/RSJ IROS, 2008
DOI: `10.1109/IROS.2008.4651060`

> 剛体ペイロードの 10 慣性パラメータをオンラインで推定する RTLS 手法を提案。data matrix にも誤差を考慮する TLS が LS/RIV より優れることを実験的に示した。本プロジェクトの回帰行列構成と励起軌道設計の基礎文献。

### Swevers1997_excitation

**Optimal robot excitation and identification**
Swevers, J., Ganseman, C., Tukel, D. B., de Schutter, J., Van Brussel, H. — IEEE Trans. Robotics and Automation, 1997
DOI: `10.1109/70.631234` | [URL](https://ieeexplore.ieee.org/document/631234/)

> フーリエ級数パラメトリゼーション + SQP による励起軌道最適化の原論文。条件数最小化を目的関数とし、関節制約下で最適軌道を設計する手法を確立した。本プロジェクトの SLSQP ベース最適化の直接的な先行研究。

### Lee2021_excitation_geometric

**Optimal excitation trajectories for mechanical systems identification**
Lee, T., Lee, B.-D., Park, F. C. — Automatica, 2021
DOI: `10.1016/j.automatica.2021.109773` | [URL](https://www.sciencedirect.com/science/article/abs/pii/S0005109821002934)

> 幾何学的基準による励起軌道最適化を提案。リグレッサの幾何学的性質に基づく解析的勾配を導出し、勾配ベース最適化の効率と頑健性を向上させた。D-optimal 基準の理論的正当性を示す。

### Tian2024_virtual_constraints

**Excitation Trajectory Optimization for Dynamic Parameter Identification Using Virtual Constraints in Hands-on Robotic System**
Tian, H., Huber, M., Mower, C. E., Han, Z., Li, C., Duan, X., Bergeles, C. — arXiv, 2024
DOI: なし | [URL](https://arxiv.org/abs/2401.16566)

> グラミアン行列ベースの条件数代理指標を提案し、CasADi + IPOPT で解析的勾配付き最適化を実現。SLSQP やメメティック法と比較して大幅に低い条件数を短時間で達成。本プロジェクトの IPOPT 移行の参考。

### Rackl2012_bspline_excitation

**Robot excitation trajectories for dynamic parameter estimation using optimized B-splines**
Rackl, W., Lampariello, R., Hirzinger, G. — IEEE ICRA, 2012
DOI: `10.1109/ICRA.2012.6225279` | [URL](https://ieeexplore.ieee.org/document/6225279/)

> B-spline パラメトリゼーションによる励起軌道設計を提案。フーリエ級数に対する代替的な軌道表現として、局所的な制御性の利点を示した。

### Calafiore2001_calibration

**Robot dynamic calibration: Optimal excitation trajectories and experimental parameter estimation**
Calafiore, G., Indri, M., Bona, B. — Journal of Robotic Systems, 2001
DOI: `10.1002/1097-4563(200102)18:2<55::AID-ROB1005>3.0.CO;2-O`

> D-optimal 設計基準（log det(W^T W) 最大化）による励起軌道最適化を実験的に検証。条件数最小化の代替として D-optimal 基準の有効性を実証した。
