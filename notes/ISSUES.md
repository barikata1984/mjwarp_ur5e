# 未解決の技術課題

解決したらエントリごと削除する.

## FT オフセット拡張時の列スケーリング妥当性

- 回帰行列 `[I₆ | V]` でオフセット列の L2 ノルムが `√T` 倍に膨張し, スケーリングなしでは D-optimal 目的関数が制約違反方向に暴走する
- スケーリングありでは条件数は良好だが, 正規化後の値であり実推定精度との対応は未検証
- スケーリングの代替案: 重み付き D-optimal, ブロック対角前処理, オフセット列のサブサンプリング

## 10s 条件での feasibility 問題

- duration=10s (1001 timesteps) で subsample_factor=1 のとき, 全 restart で feasible=False
- SLSQP が高次元制約空間で局所最適に陥り, 制約遵守と目的関数改善を両立できない
- 対策候補: subsample_factor 引き上げ, IPOPT 移行, restart 数/イテレーション数増加

## `EarlyStopConfig.min_improvement` 未使用

- `min_improvement: float = 1e-3` フィールドが定義されているが, early stopping ロジックでは厳密な改善 (`cond < best_cond`) のみを判定
- 微小改善を無視する閾値として `min_improvement` を組み込む必要がある

## 推定慣性の基準点が CoM でない

- FT センサ同定で得る慣性はセンサ原点 (フランジ面) まわり. `m.body_inertia` (CoM まわり) と直接比較できない
- 平行軸定理で CoM まわりへ変換する処理が未実装

## PD 追従誤差が同定精度に与える影響が未評価

- PD playback の追従誤差は max_pos_err≈0.097 rad (≈5.5°)
- 質量・重心は誤差 0% で復元できたが, 追従誤差が慣性推定や実機 sim-real ギャップにどう効くか未評価

## FT sim-real 比較で Fx/Fy/Mx/My のスケール不一致

- `mj_inverse` による FT 計算は Fz では span 比 1.03 だが, Fx 1.37, Fy 1.72, Mx 2.00, My 1.78
- tool0 位置・速度はサブミリ精度で一致 → パイプラインロジックの問題ではない
- 残存要因: (1) ddq の savgol 平滑化 (window=15, 179ms) がピーク加速度を 65–85% に抑制, (2) sim payload 質量パラメータ差 (cube 0.1 kg 含む vs 実機)
- 対策候補: savgol window 縮小, 実機ペイロード質量の正確な反映
