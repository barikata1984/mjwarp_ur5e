# FT300s アセンブリモデル・軌道リプレイ ログ

## 2026-06-09: グリッピングシーン構築 + 軌道リプレイ + FT 検証

### グリッピングレンダリング

- `assets/ur5e/mjcf/scene_grasping.xml` を作成 (FT300s + 2F-85 + 5cm cube)
- 立方体は gripper_base の子ボディとして welded (freejoint だとキーフレーム nq 不一致 + 落下問題)
- グリッパ閉じは gravity-off で 3000 step シミュレーション → gripper_qpos を保存して再利用
- `scripts/render_grasping.py`: 一時ファイルで robot XML に cube body を注入し, キーフレーム互換を維持
- 出力: `ur5e_ft300s_2f85_grasping.png` (4 視点: overview + 3 close-up)

### 軌道リプレイ + FT 計測

- `scripts/replay_trajectory_video.py` を作成
- データ: `data/replay_recording_2026-06-09_08-41-21/recording.npz` (508 samples, 5.59s, ~84Hz)
  - フィールド: `joint_position`, `joint_velocity`, `wrench`, `tip_position` 等
- FT 計算: `mj_inverse` で各タイムステップの FT を計算, 符号反転 (child→parent)
- ddq: recording に含まれないため `savgol_filter(dq, window=15, poly=3, deriv=1)` で生成
- 動画: 4 視点グリッド (overview/front/top/side), 30fps, imageio+ffmpeg
- 出力: `results/replay/replay_4view.mp4`, `replay_ft.npz`, `ft_comparison.png`

### FTA による sim パイプライン検証

- **Top event**: Mx/My で sim と real の FT span が約 2 倍異なる
- **検証結果**:
  - tool0 位置: mean abs error < 0.6 mm → キネマティクス正常
  - tool0 速度: mean abs error < 0.3 mm/s → 正常
  - tool0 加速度: sim span が recording の 65–85% → ddq 平滑化が原因
  - `attachment_site` = recording の `tip_position` で一致確認
  - `ft_sensor` site / `pinch` site は `tip_position` と不一致 (期待通り)
- **結論**: パイプラインロジックに問題なし. ずれの要因は ddq 平滑化過剰 + 質量パラメータ差
- Fz span 比 1.03 (ほぼ一致), Fx 1.37, Fy 1.72, Mx 2.00, My 1.78
