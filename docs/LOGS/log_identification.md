# 慣性パラメータ同定・最適励起 実験記録

## 2026-03-12: regressor rank 9 問題の調査と修正

### 問題

`payload_box_mount` の stacked regressor が動的軌道でも rank 9 止まりで、10 パラメータすべてが独立に励起されなかった。

### 原因

`set_model_state` が `mj_forward` を呼んでおり、2 つの問題を引き起こしていた:

1. ユーザ指定の `qacc` (関節加速度) が MuJoCo の順動力学で上書きされていた
2. MuJoCo 3.6 のキネマティクスパイプラインでは `data.cacc` (body Cartesian 加速度) が未計算のため、`mj_objectAcceleration` が返す angular acceleration が常にゼロだった

null space 解析により、欠落方向は慣性テンソルのトレース `Ixx + Iyy + Izz` であることを特定。トレース方向の回帰列は `[alpha_body, 0]` に等しく、`alpha_body = 0` なので常にゼロ列となっていた。

### 修正

- `set_model_state`: `mj_forward` → `mj_kinematics` + `mj_comPos` + `mj_fwdVelocity`
- `sample_body_kinematics`: `mj_jacBody` で body Jacobian を取得し、`J @ qacc` で body frame の加速度を補正

### 結果

| 指標 | 修正前 | 修正後 |
|------|--------|--------|
| rank | 9 | **10** |
| 条件数 | inf | **~5** |

base parameter 化は不要であることを確認。
