---
Title: On-line Rigid Object Recognition and Pose Estimation Based on Inertial Parameters
Authors:
  - Kubus, Daniel
  - Kröger, Torsten
  - Wahl, Friedrich M.
Year: 2007
Venue: IROS
Tags:
  - "inertial-parameter-identification"
  - "excitation-trajectory"
  - "force-torque-sensor"
  - "object-recognition"
  - "pose-estimation"
  - "recursive-instrumental-variables"
PDF: "[[papers/Kubus-IROS2007-On-line_Rigid_Object/main.pdf|📃]]"
Import Date: "2026-05-26"
Read Date: 2026-05-26
Executive Summary: マニピュレータが把持した剛体の全慣性パラメータ (質量・重心・慣性行列の 10 要素) をオンライン推定し, 物体認識と把持姿勢推定に用いる手法。6D 力/トルク・6D 加速度・3D 角速度・関節角を融合し, Newton-Euler 方程式から構成した回帰行列で recursive instrumental variables (RIV) 法により偏りの少ない推定を行う。励起軌道はフーリエ級数で表し相関行列の条件数を最小化して設計, FT センサオフセットと取付プレートの慣性も補償する。質量・主慣性モーメントの並進・回転不変性を特徴量とし Kullback-Leibler ダイバージェンスで認識。質量・重心は相対誤差 1% 程度と高精度だが, シミュレーション条件数と実機条件数が 2-3 倍乖離する点を課題として指摘。
Citekey: Kubus-IROS2007-On-line_Rigid_Object
BibTeX Key: kubus2007online
DOI: 10.1109/IROS.2007.4399184
Relevance: 5
Repository: none
Category: note
Template Version: v2.3
---

## Executive Summary
マニピュレータが把持した剛体の全慣性パラメータ (質量・重心・慣性行列の 10 要素) をオンライン推定し, 物体認識と把持姿勢推定に用いる手法. 6D 力/トルク・6D 加速度・3D 角速度・関節角を融合し, Newton-Euler 方程式から構成した回帰行列で recursive instrumental variables (RIV) 法により偏りの少ない推定を行う. 励起軌道はフーリエ級数で表し相関行列の条件数を最小化して設計, FT センサオフセットと取付プレートの慣性も補償する. 質量・主慣性モーメントの並進・回転不変性を特徴量とし Kullback-Leibler ダイバージェンスで認識. 質量・重心は相対誤差 1% 程度と高精度だが, シミュレーション条件数と実機条件数が 2-3 倍乖離する点を課題として指摘.

---
## Summary

### この論文が答えた問い, あるいは解決した課題は何か?
視覚特徴では区別できない物体 (例: 同一外観だが内部に巣穴のある鋳物) を, 慣性パラメータという物理特徴で認識・識別したい, という課題に答えている. 把持物体を適切な軌道で動かして 10 個の慣性パラメータをオンライン推定し, それらから並進・回転不変な特徴量 (質量と主慣性モーメント) を抽出して物体認識を行い, さらに慣性行列の固有ベクトルから把持姿勢を推定する.

### 提案手法のアプローチと, その根幹をなす要素は何か?
把持物体を励起軌道で運動させ, センサ計測 (力/トルク・加速度・角速度) と関節角から慣性パラメータを線形回帰で推定し, その推定値を特徴量化して認識・姿勢推定に用いる, というセンサ融合パイプラインである. 根幹要素は次の通り:

- **Newton-Euler 線形回帰モデル (eq.1-5)**: センサ frame `S` で力 `Sf = mSa - mSg + Sα×mSc + Sω×(Sω×mSc)` とトルク `Sτ = SISα + Sω×(SISω) + mSc×Sa - mSc×Sg` を, 未知パラメータ `Sϕ = [m, mScx, mScy, mScz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz]^T` に対し線形な形 `[Sf; Sτ] = V Sϕ` に整理. 回帰行列 `V` は計測した `a, α, ω, g` から構成.
- **励起軌道のフーリエ級数表現と条件数最適化 (eq.14)**: 各関節を `qi(t) = Σ ρ sin(2πkft) + σ cos(2πkft) + qi,0` で表現 (本質的にジャーク制限). M 個の `V` を積み上げた相関行列 `Υ = Ṽ^T Ṽ` の条件数 `κ(Υ)` を最小化. Monte Carlo 探索で候補を見つけ Matlab `fmincon` で最適化.
- **recursive instrumental variables (RIV) 法 (eq.23-25)**: 相関ノイズ下でも偏りの少ない推定を得る. 識別変数行列 `Uk` と異なる信号源からの操作変数行列 `Wk` を用い (例: 角速度は角速度センサ, 操作変数は関節角由来), 相互相関を確保しノイズと無相関化.
- **誤差源の補償**: FT センサオフセット (ゼロ化 + 擬似重力ベクトル `Sginit`, または 16 パラメータへ拡張して直接推定), 取付プレート (distal sensor part) とグリッパの慣性を推定して減算 (`ÎObject = Îall - Idist - Igripper`).
- **不変特徴量と KL ダイバージェンス認識**: 質量と主慣性モーメント (慣性行列の固有値) は並進・回転不変. 特徴ベクトルとその共分散を用いた対称 KL ダイバージェンス `JKL` で物体を識別.

### 特に参考とした既存研究と, それらと比した提案手法の新規性は何か?
ロボット動力学パラメータ推定の最適励起軌道設計 (フーリエ級数 + 条件数最小化) の先行研究を慣性負荷推定に転用している. 新規性は次の点: (1) 視覚ではなく慣性特徴を物体認識・識別に用いる発想, (2) 複数センサ (力/トルク・加速度・角速度・関節角) を融合し, RIV 法で識別変数と操作変数を別信号源から取ることで相関ノイズ下の偏りを抑制, (3) 認識だけでなく慣性行列固有ベクトルからの把持姿勢推定まで統合. RLS に対し RIV を採ることで相関ノイズ下のバイアス低減を図る点が推定面での差別化.

### どのように訓練・最適化したのか?
- **損失関数 / 最適化目的**: パラメータ推定自体は最小二乗系 (加法誤差モデル `[Sf; Sτ] + e = V Sϕ`) を RIV 法で解く. 軌道設計の最適化目的は相関行列の条件数 `κ(Υ) = σmax/σmin` の最小化. 制約として最小/最大関節角・最大関節速度・最大関節加速度・各センサの最大定格・自己衝突回避を最適化各反復で検査.
- **データセット**: N/A (機械学習的な訓練データセットではない). 実機計測は 1kS/s で 1500 個の `W`/`U` 行列をサンプリング. テスト物体は幾何学的に単純で慣性パラメータは寸法から理論的に算出 (ground truth). グリッパの慣性は RIV 法で実測, distal sensor part は静的推定 (TLS) で質量・重心のみ算出 (質量 56.5g など).

### どのように検証したか? 指標と結果は?
Stäubli RX60 産業用マニピュレータ, JR3 製 6D 力/トルク+6D 加速度センサ, ADXRS300 角速度センサで実機検証. 指標は相対誤差 `erel(x̂) = ||x̂ - xth|| / ||xth|| × 100%`.

- **慣性パラメータ推定 (Table IV, V)**: 識別変数/操作変数の組合せ s4 (加速度=センサ, 角加速度=Kalman フィルタ後関節角由来, 操作変数も別信号源) が最良で, 質量誤差 0.42%, 重心 0.85-1.05%. 慣性行列要素は組合せにより 0.5-33% と幅があり, 対角要素 (Iyy, Izz) は精度良いが非対角要素や Ixx はノイズ感受性が高い. 角速度センサ由来の角加速度はノイズ過大で不適, エンコーダ由来が有望.
- **条件数 (Table I)**: N=3 の最適化軌道で, シミュレーション条件数 6.51-8.18 に対し実験条件数 14.44-23.42 と 2-3 倍乖離.
- **把持姿勢推定 (Table VII)**: 姿勢間の roll/pitch/yaw を推定. 理論 90°/90°/0° に対し 86.23°/81.71°/3.86° 等, 物体により誤差あり.
- **物体認識 (Table VIII)**: 質量が同一の 3 物体を異なる把持姿勢で対称 KL ダイバージェンスにより全て正しく認識.

### 検証結果に基づいた議論, 明らかになった課題はあるか?
- (§IX Conclusions より) **シミュレーション条件数と実機条件数の乖離**: 最適化はシミュレーション条件数で行うが, 実機軌道がシミュレーション軌道から逸脱する (未モデル化動力学のため事実上常に発生する) ため, シミュレーション条件数に基づく軌道最適化の妥当性に疑問がある, と明示.
- (§IX Conclusions より) 手首装着加速度センサ由来の角加速度信号はノイズ・外乱が大きく推定に不適, エンコーダ由来の角加速度が有望.
- (§IX Conclusions より) 把持姿勢推定精度は, 力/トルク計測の誤差だけでなく識別変数の誤差も考慮する推定法 (TLS の再帰版など) で改善し得る, と将来課題に言及.
- (§VIII より) distal sensor part は慣性行列要素が小さすぎて精度良く推定できず, 静的推定で質量・重心のみ算出.

---
## 自身の研究との関連
本論文は当プロジェクト (mjwarp_ur5e: UR5e ペイロード慣性同定のための最適励起軌道生成) の直接的な理論基盤である.

- **回帰行列とパラメータベクトル**: 本論文 eq.5 の `V` 行列と `Sϕ = [m, mScx, mScy, mScz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz]` は, プロジェクトの `rigid_body_wrench_regressor` / `InertialParameters.to_vector` と対応する 10 パラメータ定式化そのもの.
- **励起軌道と条件数最適化**: eq.14 のフーリエ級数軌道と `κ(Υ)` 最小化は, プロジェクトの `WindowedFourierTrajectory` と `condition_number_objective` の元ネタ. プロジェクトでは境界条件付き windowed Fourier に拡張している.
- **FT センサオフセット (16 パラメータ拡張)**: 本論文の"offset を直接推定し 10→16 パラメータに増やす"アプローチが, プロジェクトの `with_ft_offset` (回帰行列を `[I6 | V]` に拡張) と一致. 列スケーリングは本論文には無いプロジェクト独自の追加.
- **重要な相補/差分 — 条件数の sim-real ギャップ**: 本論文 §IX が指摘する"シミュレーション条件数と実機条件数の 2-3 倍乖離"は, 当セッションで観測した"最適化した軌道を実機 (シミュレータ) で追従させると wrench プロファイルが想定と合わない"現象と本質的に同根. 本論文は RIV 法 (識別変数と操作変数を別信号源から取る) で計測ノイズ起因のバイアスに対処しており, プロジェクトの単純な LS/TLS 同定で推定値がずれる問題への対処指針となる.
- **推定法**: 本論文は RIV を主軸とし TLS にも言及. プロジェクトは LS/TLS/RTLS を実装済みで, RIV は未実装. 相関ノイズ下のバイアス低減のため RIV 導入が今後の選択肢.

---
## 追加議論


---
## BibTex
<details>
<summary> Click to show/noshow the BibTex data </summary>

```bibtex
@inproceedings{kubus2007online,
  title     = {On-line Rigid Object Recognition and Pose Estimation Based on Inertial Parameters},
  author    = {Kubus, Daniel and Kr{\"o}ger, Torsten and Wahl, Friedrich M.},
  booktitle = {2007 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages     = {1402--1408},
  year      = {2007},
  address   = {San Diego, CA, USA},
  publisher = {IEEE},
  doi       = {10.1109/IROS.2007.4399184}
}
```
</details>
