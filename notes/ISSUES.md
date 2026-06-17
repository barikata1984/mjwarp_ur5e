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

## 慣性パラメータ同定の sim-real 乖離 (MJCF/URDF モデル不一致)

- cube データで OLS+bias 物体慣性: 質量 +25.8%, Iyy +207%, Izz -144%
- FTA で排除した仮説:
  - ddq フィルタ不整合 (pipeline 10 Hz LPF vs replay raw forward diff): filtered ddq で再実行しても変化なし
  - wrench taring 不整合: 両方 frame0-tared で整合
- 確認された構造的問題: MJCF (MuJoCo) と URDF (Pinocchio) で FT センサ以降のボディ構成が異なる
  - MJCF 合計 ~1.195 kg vs URDF 合計 ~1.352 kg (FT300s の扱いが異なる)
  - グリッパーベース CoM Z: MJCF 0.0355 vs URDF 0.0315 (4 mm 差)
- FT300s 分はキャリブレーション済みで差分法で相殺されるが, グリッパー部分の差が残存
- 次のステップ: Pinocchio RNEA で URDF から直接レンチ生成 → pipeline self-consistency テスト

## MPC 励起プランナーに衝突・作業域制約が未接続

- `ExcitationPlanner` (`identification/mpc/planner.py`) に衝突制約・作業域制約が組み込まれていない
- 実行時にロボットがテーブルを貫通し, 危険な姿勢に到達するケースを確認
- 既存インフラは流用可能: `WorkspaceConstraintConfig`, `CollisionConfig`, `make_collision_constraint`, `make_workspace_constraint` (`optimizer.py` / `constraints.py`)
- これらを `ExcitationPlanner._build_constraints()` に追加することで対処できる

## FT sim-real 比較で Fx/Fy/Mx/My のスケール不一致

- 隣接二点差分 ddq + ft300s_mount 6mm シフト後: Fz 0.97, Fx 1.26, Fy 1.67, Mx 1.94, My 1.63
- ddq を savgol→隣接二点に変更で Fx/My が改善 (1.37→1.26, 1.78→1.63), ただし Fy/Mx は残存
- ft300s_mount 6mm シフトでは ratio にほぼ変化なし → センサ位置は支配要因ではない
- 残存要因: sim payload 質量パラメータ差 (cube 0.1 kg 含む vs 実機), wrench フレーム/キャリブレーション差
