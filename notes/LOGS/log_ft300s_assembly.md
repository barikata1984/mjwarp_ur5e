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

## 2026-06-10: ddq 差分方式変更 + FT300s 幾何ずれ調査・修正

### ddq 計算の変更

- savgol (window=15, 179ms) → 隣接二点差分 (`np.diff(dq) / dt`) に変更
- 改善: Fx 1.37→1.26, My 1.78→1.63. Fy/Mx は大きな変化なし

### FT300s-wrist3 間の幾何ずれ調査

- **ISO 9409-1-50-4-M6**: センタリングカラー径 25mm, 突出量 3mm, PCD 40mm, 4×M6
- **FT300s**: 全高 37.5mm (モデルの 42.2mm はカップリングアダプタ込み), 外径 75mm, 質量 300g
- **カップリングアダプタ**: 37.5 + 4.7 = 42.2mm (モデル値と整合)
- wrist3 メッシュの tool0 フランジ面 (遠位端) が ft300s_mount 原点より 1.06mm 近位にずれ
- FT300s メッシュのロボット側面は body 原点と一致 (0mm オフセット)
- UR フランジのセンタリングボス + カップリングアダプタの嵌め合い構成が未反映

### ft300s_mount 位置修正

- `ft300s_mount` body pos を y=0.1 → y=0.094 に変更 (6mm tool0 側シフト)
- 嵌め合い構成 (センタリングボス 3mm + アダプタ嵌入) を反映
- FT 比較: 6mm シフト後も ratio にほぼ変化なし → センサ位置は FT ずれの支配要因ではない
- 最終 ratio: Fz 0.97, Fx 1.26, Fy 1.67, Mx 1.94, My 1.63
