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

## 2026-06-10: シミュレーション慣性パラメータ同定パイプライン構築

### パイプライン

- `scripts/identify_from_sim.py` を新規作成
- `IdentificationPipeline` (ROS 非依存) を Pinocchio + URDF (`ur5e_ft300s_robotiq85.urdf`) で駆動
- 入力: `recording.npz` (実機関節軌道) + `replay_ft.npz` (MuJoCo `mj_inverse` wrench)
- sim / real 両方の同定を実行し, 比較テーブルと JSON/CSV を `results/replay/` に出力
- gripper 校正 (`gripper.json`) による差分法 (object inertia) にも対応

### FTA: sim vs real 慣性パラメータ乖離の原因分析

- **Top event**: sim 同定結果が real と系統的に乖離 (m -10%, hz -34% with 0.1 kg cube)
- **根本原因**:
  1. MuJoCo cube 0.1 kg が実物より軽い
  2. 実機 gripper_cal (0.907 kg) と MuJoCo グリッパー質量 (1.053 kg) の不一致 — 差分法で object 側にズレが乗る

### cube 質量をアルミ密度に変更

- 5 cm 角, 密度 2700 kg/m³ → 0.3375 kg, diaginertia 1.40625e-4 kg·m²
- `replay_trajectory_video.py` の `CUBE_BODY_XML` を更新
- 結果 (OLS+bias total): m 差 -10.1% → +7.9%, hz 差 -33.9% → +3.3% に改善

### グリッパー質量スケーリング

- Menagerie 配分を維持し, 全ボディを比率 0.907/1.053 = 0.862 でスケール
- 結果 (OLS+bias total): m 差 -3.5%, hz 差 -3.7% (良好)
- Iyy は +129% で乖離残存 — モデルの質量配置 (重心位置) の限界

### グリッパー質量配分の代替ソース調査

- Robotiq 公式: 925 g (カップリング込み) / 900 g (なし), CoM/慣性は画像のみ
- MuJoCo Menagerie: 0.900 kg, CAD 由来 (現モデルと同一)
- automaticaddison URDF: 0.921 kg, base 0.663 kg / fingers 0.258 kg (Menagerie は 0.777 / 0.125)
- automaticaddison 比率で再スケール → Iyy 悪化 (+151%), 改善せず
- **結論**: Menagerie 配分 + 総質量スケーリングが現状ベスト

### 最終設定

- Menagerie 配分, 総質量 0.907 kg スケール, アルミ cube 0.3375 kg
- 出力: `results/replay/identification_result.json`, `identification_comparison.csv`

## 2026-06-10: cube/pan データ対応 + FTA (sim-real 乖離の原因分析 第 2 ラウンド)

### replay_trajectory_video.py の拡張

- `--recording`, `--output-dir`, `--filter-ddq` CLI 引数を追加
- `--filter-ddq`: pipeline と同じ 10 Hz LPF で ddq を計算するオプション (NumericalDifferentiator 再利用)
- `_compute_ft()` が ft300s_mount 位置も返すよう拡張
- 比較プロットを 2x3 → 5x3 に拡張 (力/トルク/並進位置/速度/加速度)
- カメラアングルを `scene_with_box.xml` の payload_overview/view_x/view_y/view_z に統一
- side カメラを反対側 (-X) から撮影するよう変更

### cube データでの sim 同定実行

- 入力: `data/cube_2026-06-10_06-35-32/recording.npz`
- 出力: `results/cube/` (replay_ft.npz, replay_4view.mp4, ft_comparison.png, identification_result.json)
- OLS+bias 物体慣性比較 (sim vs real): 質量 +25.8%, hz +5.2%, Iyy +207%, Izz -144%

### FTA: sim-real 乖離の原因分析

- **Top event**: cube の OLS+bias 差分法で物体慣性パラメータに大きな乖離
- **排除した仮説**:
  1. ddq フィルタ不整合 (H1): pipeline は 10 Hz LPF, replay は raw forward diff. diff RMS は filtered の 50–68%. しかし filtered ddq で再実行しても結果ほぼ不変 → 主因ではない
  2. wrench taring 不整合 (H2): 両方 frame0-tared, sim[0]=[0,...,0], real[0] は identify 側で引く → 整合
- **確認された構造的問題** (H: MJCF vs URDF モデル不一致):
  - MJCF: ft300s_mount 0.3 kg あり, ft300_sensor なし, gripper_base 0.777 kg, 合計 ~1.195 kg
  - URDF: ft300s_mount なし, ft300_sensor 0.442 kg あり, gripper_base 0.788 kg, 合計 ~1.352 kg
  - gripper_base CoM Z: MJCF 0.0355 vs URDF 0.0315 m (4 mm 差)
  - FT300s 分はキャリブレーション済みだが, グリッパー部分の差が残存
- **次のステップ**: Pinocchio RNEA で URDF から直接レンチ生成 → pipeline self-consistency テスト

### 実行手順

```bash
# 1. pinocchio インストール (初回のみ)
pip install pin

# 2. sim wrench 生成 (replay_ft.npz がない場合, MuJoCo + imageio 必要)
python scripts/replay_trajectory_video.py

# 3. sim vs real 慣性パラメータ同定
python scripts/identify_from_sim.py

# オプション
#   --recording <path>    特定の recording.npz を指定 (デフォルト: data/ 配下の最新)
#   --wrench <path>       別の replay_ft.npz を指定
#   --gripper-cal <path>  別のグリッパー校正ファイル
#   --method OLS+bias     比較手法 (OLS / TLS / OLS+bias / TLS+bias)
```
