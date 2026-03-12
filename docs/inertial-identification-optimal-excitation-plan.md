# 慣性パラメータ推定向け最適励起軌道 実装計画

## 目的

このリポジトリ上で、UR5e + MuJoCo + Warp を用いた慣性パラメータ推定のための最適励起軌道生成基盤を実装する。
目標は、参照実装 `isaac-lab-inertia-identification` のうち有効な設計を取り込みつつ、現在の `mjwarp_ur5e` の責務分離に沿って、次の 3 点を段階的に成立させることにある。

- オフラインで最適励起軌道を生成できること
- MuJoCo モデル上で軌道の可行性を検証できること
- その先の慣性推定ループへ自然に接続できる API を持つこと

本メモは実装順序をフェーズ単位で固定し、各フェーズの成果物、責務、受け入れ条件、注意点を明確化する。

## 作業進捗

### フェーズ進捗一覧

- フェーズ 0: 完了
  - 数理対象、評価対象フレーム、初期シーン、最適化の基本方針を整理済み
- フェーズ 1: 完了
  - `src/mjwarp_ur5e/trajectories/` を追加済み
  - `base`, `fourier`, `window`, `windowed_fourier` を実装済み
  - 軌道生成テストを追加し、コンテナ内 `pytest` 通過済み
- フェーズ 2: 完了
  - `src/mjwarp_ur5e/identification/` を追加済み
  - MuJoCo body kinematics 抽出、payload 慣性パラメータ抽出、剛体 wrench regressor 構築を実装済み
  - 回帰行列テストを追加し、フェーズ 1 のテストと合わせてコンテナ内 `pytest` 通過済み
- フェーズ 3: 未着手
  - joint limit、workspace、collision の制約レイヤは未実装
- フェーズ 4: 未着手
  - 最適化器、objective、validation、結果保存は未実装
- フェーズ 5: 未着手
  - `tyro` ベースの励起軌道 CLI は未実装
- フェーズ 6: 未着手
  - MuJoCo 上での軌道再生と時系列計測基盤は未実装
- フェーズ 7: 未着手
  - LS / TLS / RTLS 推定器統合は未実装
- フェーズ 8: 未着手
  - Warp による高速化は未着手
- フェーズ 9: 継続中
  - 各フェーズに応じた unit test は追加中
  - integration test と smoke test は後続フェーズで整備する

### 現在位置

- 実装済みの最前線はフェーズ 2 まで
- 次の着手対象はフェーズ 3 の制約評価レイヤ
- 直近の主タスクは、trajectory から joint / workspace / collision 制約を評価できる形に API を固めること

### 現時点の技術メモ

- (解決済み) stacked regressor が rank 9 で止まっていた問題を修正し、動的軌道で **full rank 10** を達成した
- 原因: `set_model_state` が `mj_forward` を呼んでおり、ユーザ指定の `qacc` が順動力学で上書きされていた。さらに MuJoCo 3.6 のキネマティクスパイプラインでは `data.cacc` (body Cartesian 加速度) が未計算のため、`mj_objectAcceleration` が返す angular acceleration が常にゼロだった。結果として慣性テンソルのトレース方向 `Ixx + Iyy + Izz` が null space に落ちていた
- 修正: `mj_forward` を `mj_kinematics` + `mj_comPos` + `mj_fwdVelocity` に置き換え、body 加速度を Jacobian (`mj_jacBody`) 経由で `J @ qacc` として計算するようにした
- 修正後の条件数は動的軌道で約 5〜50 の有限値になり、10 パラメータすべてが独立に励起される
- base parameter 化は不要であることを確認した

## 前提と設計方針

- 現在のロボットモデルの正系は MuJoCo MJCF であり、最適化の可行性評価もまず MuJoCo ベースで行う
- Warp は将来の高速化や並列化に使うが、初期フェーズでは必須依存にしない
- 参照実装のうち再利用するのは「軌道の表現」「多点開始最適化の考え方」「制約構成」であり、Pinocchio と Isaac Sim 依存部分は直接移植しない
- 既存パッケージの責務を壊さず、薄い CLI と再利用可能なライブラリ層を維持する
- まずは payload 単体の 10 慣性パラメータ同定に必要な励起を主対象とし、全身同定や摩擦同定は後続とする

## 想定成果物

- `src/mjwarp_ur5e/trajectories/`
  - windowed Fourier 系の軌道表現
- `src/mjwarp_ur5e/identification/`
  - 回帰行列構築、条件数評価、最適化、制約、軌道入出力
- `src/mjwarp_ur5e/demos/`
  - 最適化実行 CLI
  - 軌道検証 CLI
- `tests/`
  - 軌道生成、制約、最適化、回帰行列構築、JSON 入出力のテスト
- `docs/`
  - 運用メモとアルゴリズム設計メモ

## フェーズ 0: 要件固定と数理対象の確定

### 目的

実装前に、何を最適化し、何を制約し、何を評価指標にするかを固定する。

### 実施内容

- 慣性推定対象を payload の 10 パラメータに限定する
- 評価対象フレームを `attachment_site` に固定する
- 最適化目的を、時系列で積み上げた回帰行列 $A$ の条件数最小化に置く
- 初期姿勢 `q0` を現在の実験姿勢系と整合させる
- ワークスペース制約を、すでに可視化済みの作業領域と整合させる
- ペイロード付き派生シーン `scene_with_box.xml` を、初期の検証対象シーンとして扱う

### 具体的な決定事項

- 最適化変数: 各関節の Fourier 正弦係数 + 余弦係数
- 軌道境界条件: $q(0)=q(T)=q_0$, $\dot q(0)=\dot q(T)=0$
- 制約:
  - 関節位置制限
  - 関節速度制限
  - 関節加速度制限
  - EE 作業領域制限
  - 自己干渉・床干渉・payload 干渉回避
- 初期実装の最適化手法: `scipy.optimize.minimize(..., method="SLSQP")`

### 完了条件

- このメモに沿って、以降のフェーズで必要な入力と出力がブレなくなること
- `q0`、シーン、対象フレーム、制約の意味が README か docs で参照可能になること

## フェーズ 1: 軌道表現レイヤの実装

### 目的

最適化器から独立した軌道表現を実装する。まずは windowed Fourier 系だけに絞る。

### 新規モジュール案

- `src/mjwarp_ur5e/trajectories/base.py`
- `src/mjwarp_ur5e/trajectories/window.py`
- `src/mjwarp_ur5e/trajectories/fourier.py`
- `src/mjwarp_ur5e/trajectories/windowed_fourier.py`
- `src/mjwarp_ur5e/trajectories/__init__.py`

### 実施内容

- dataclass ベースの config を定義する
- 時刻配列生成、軌道位置、速度、加速度計算を分離する
- `q`, `dq`, `ddq` をまとめて返す共通 API を持たせる
- 初期姿勢 `q0` を任意設定できるようにする
- 係数を JSON 化しやすい構造に保つ

### API 目標

- `WindowedFourierTrajectoryConfig`
- `WindowedFourierTrajectory(config).sample()`
- 戻り値は `TrajectorySample` のような構造体にまとめる

### 受け入れ条件

- ゼロ係数で静止軌道になること
- ランダム係数で中間時刻に非自明な運動が出ること
- 境界条件が数値誤差の範囲で満たされること

### テスト

- 出力 shape
- 境界条件
- ゼロ係数時の静止確認
- 係数シリアライズ/デシリアライズ

## フェーズ 2: MuJoCo ベースの運動学・回帰行列構築

### 目的

最適化の目的関数となる回帰行列 $A$ を、このリポジトリの MuJoCo モデルから構築できるようにする。

### 新規モジュール案

- `src/mjwarp_ur5e/identification/regressor.py`
- `src/mjwarp_ur5e/identification/sampling.py`
- `src/mjwarp_ur5e/identification/types.py`

### 実施内容

- 軌道サンプル `q`, `dq`, `ddq` から、各時刻の payload 慣性パラメータ回帰行列を構築する
- 必要であれば MuJoCo の内部量と空間ベクトルを整理する補助関数を追加する
- `attachment_site` またはその親 body を基準に、観測モデルの定義を固定する
- 行列積み上げロジックを最適化ループから分離する

### 注意点

- 参照実装の `PinocchioKinematics.compute_regressor()` は直接流用しない
- MuJoCo の記号系と参照文献の記号系の対応表をコードコメントか docs に残す
- まずは正しさを優先し、必要になった時点で Warp へのオフロードを検討する

### 受け入れ条件

- `A` の shape が期待通りであること
- 静止軌道では行列が退化し、条件数が大きくなること
- 異なる軌道係数で条件数が変化すること

### テスト

- stacked regressor の行列 shape
- 静止軌道での near-singular 判定
- ランダム軌道で有限条件数が得られること

## フェーズ 3: 制約評価レイヤの実装

### 目的

最適化器が扱う制約を、独立した関数群として実装する。

### 新規モジュール案

- `src/mjwarp_ur5e/identification/constraints.py`
- `src/mjwarp_ur5e/identification/collision.py`
- `src/mjwarp_ur5e/identification/workspace.py`

### 実施内容

- joint 位置、速度、加速度制約を定義する
- `attachment_site` の位置履歴から作業領域制約を定義する
- MuJoCo モデル上で、自己干渉・床干渉・payload 干渉を評価する
- 最適化変数から軌道を 1 回だけ再構築し、複数制約で共有できる cache を持つ

### 設計判断

- まずは安全側の近似制約で十分とする
- 衝突判定は初期フェーズでは幾何近似または最小距離近似でよい
- 高精度な接触判定は後続フェーズで MuJoCo 接触情報に置き換える

### 受け入れ条件

- 小振幅軌道が全制約を満たすこと
- 明らかに過激な軌道が少なくとも 1 制約で弾かれること
- 制約関数が `c(x) >= 0` 形式で統一されること

### テスト

- joint constraints の正負判定
- 作業領域逸脱の検出
- 衝突または地面侵入ケースの検出

## フェーズ 4: 最適化エンジンの実装

### 目的

軌道表現、回帰行列、制約を束ねて、多点開始の最適励起探索を行う。

### 新規モジュール案

- `src/mjwarp_ur5e/identification/optimizer.py`
- `src/mjwarp_ur5e/identification/objective.py`
- `src/mjwarp_ur5e/identification/io.py`

### 実施内容

- `OptimizerConfig` と `OptimizationResult` を定義する
- 評価関数で条件数を返す
- 初期点生成を Monte Carlo 方式で行う
- 各初期点から SLSQP を実行し、最良解を採用する
- フルサンプル検証 `validate_trajectory()` を用意する
- 結果を JSON に保存できるようにする

### 実装上の要点

- 例外時は大きい有限値を返し、最適化器を壊さない
- 評価回数、反復時間、最良 start index を記録する
- subsample 評価と full-resolution 検証を分ける

### 受け入れ条件

- ゼロ軌道より良い条件数の軌道が探索できること
- 結果 JSON から軌道を再構築できること
- 制約違反解を正しく検知できること

### テスト

- objective の有限値性
- 最適化結果の shape とメタデータ
- JSON round-trip

## フェーズ 5: CLI とユーザー導線の整備

### 目的

最適化を実際に回すための入口を、既存の `tyro` 方針で実装する。

### 新規モジュール案

- `src/mjwarp_ur5e/cli/configs.py` への config 追加
- `src/mjwarp_ur5e/demos/optimize_excitation_trajectory.py`
- `src/mjwarp_ur5e/demos/validate_excitation_trajectory.py`

### 実施内容

- harmonics、duration、base frequency、restarts、subsample、output path を CLI 化する
- `scene_with_box.xml` を既定検証シーンとして指定可能にする
- 結果 JSON の保存先を `debug/` か将来の `data/` 配下に統一する
- README と docs に実行例を追記する

### 受け入れ条件

- コンテナ内で単一コマンド実行できること
- 出力 JSON が他モジュールから再利用できること
- `--help` が現在の CLI スタイルと整合すること

## フェーズ 6: MuJoCo 上での軌道再生と計測基盤

### 目的

最適化済み軌道を MuJoCo シミュレーションで再生し、将来の推定ループに必要な時系列データを取得できるようにする。

### 新規モジュール案

- `src/mjwarp_ur5e/identification/execution.py`
- `src/mjwarp_ur5e/identification/data_buffer.py`
- `src/mjwarp_ur5e/demos/run_identification_playback.py`

### 実施内容

- 目標 `q`, `dq`, `ddq` に基づく追従再生を実装する
- 関節状態、EE 姿勢、必要なら wrench 相当量を時系列で収集する
- ノイズなし計測とノイズ付き計測を切り替え可能にする
- 軌道追従誤差を記録する

### 設計判断

- 初期はオープンループ再生 + 記録でもよい
- その後、PD 追従や inverse dynamics 補助へ発展させる

### 受け入れ条件

- JSON の軌道を読んで最後まで再生できること
- 計測データが一定の schema で保存されること
- 明らかな追従破綻を検出できること

## フェーズ 7: 慣性推定器との接続

### 目的

最適励起で得た時系列データを、payload 慣性パラメータ推定へ接続する。

### 新規モジュール案

- `src/mjwarp_ur5e/identification/estimators/`
  - `batch_ls.py`
  - `batch_tls.py`
  - `rtls.py`
- `src/mjwarp_ur5e/demos/run_inertial_identification.py`

### 実施内容

- まずは batch LS でオフライン検証を行う
- その後 TLS、RTLS の順で追加する
- 真値 payload パラメータとの比較レポートを出す
- 推定結果の信頼性指標として条件数、残差、誤差率を出力する

### 受け入れ条件

- 合成データで真値回収ができること
- ノイズ付きデータで劣化挙動が妥当であること
- 推定器 API がデータ収集層と分離されていること

## フェーズ 8: Warp 導入と高速化

### 目的

最適化の律速部を把握したうえで、必要箇所のみ Warp を使って高速化する。

### 候補

- 軌道サンプルの一括評価
- 回帰行列のバッチ構築
- Monte Carlo 初期点の並列評価

### 方針

- 先に Python/NumPy 実装で正解系を固める
- profiler を取ってから Warp 対象を決める
- 高速化のために API を壊さない

### 完了条件

- 同じ入力で NumPy 実装と整合すること
- 高速化の費用対効果が明確であること

## フェーズ 9: 品質保証と受け入れ基準

### テスト階層

- unit test
  - 軌道生成
  - 制約関数
  - 回帰行列構築
  - 推定器の数値安定性
- integration test
  - 最適化実行
  - JSON 保存/再読込
  - MuJoCo 上での軌道再生
- smoke test
  - CLI 実行
  - 代表シーンでの短時間最適化

### 最低受け入れライン

- `pytest` が通ること
- 最短構成の CLI で軌道最適化が 1 回成功すること
- 結果 JSON から軌道再構築と再検証ができること
- 作業領域と衝突制約が最低限機能すること

## 推奨実装順

実装順は次で固定する。

1. フェーズ 1
2. フェーズ 2
3. フェーズ 3
4. フェーズ 4
5. フェーズ 5
6. フェーズ 6
7. フェーズ 7
8. フェーズ 8

フェーズ 0 は今の時点で着手済みの整理フェーズとみなし、必要に応じて本メモを更新する。

## 初回スプリントの具体タスク

まず 1 スプリント目では、次だけをスコープに入れる。

- `trajectories` パッケージの新設
- `identification/regressor.py` の最小版作成
- `identification/optimizer.py` の最小版作成
- `demos/optimize_excitation_trajectory.py` の追加
- 軌道生成と objective のユニットテスト追加

この段階では、推定器本体、Warp 最適化、完全な接触ベース衝突判定までは入れない。

## 未解決事項

- ~~payload 回帰行列を MuJoCo のどの内部量から最も素直に構築するか~~ → 解決済み。`mj_jacBody` による Jacobian と `mj_objectVelocity` / `mj_objectAcceleration` の組合せで構築する
- `attachment_site` 基準での wrench 観測モデルをどう定義するか
- 最適化結果の保存先を `debug/` と `data/` のどちらに置くか
- 作業領域制約を base frame で持つか、初期 EE 基準で持つか
- 衝突判定を幾何近似で固定するか、MuJoCo 接触情報へ広げるか

## 参考

参照実装として、同一ワークスペース内の `isaac-lab-inertia-identification` を読む。
特に次の責務分けは有用である。

- 軌道表現と最適化器の分離
- 制約関数群の独立モジュール化
- objective 評価と validation の分離
- CLI を薄く保つ構成

ただし、Pinocchio 依存の回帰行列計算と Isaac Sim 実行系はそのまま移植せず、MuJoCo モデルに合わせて再設計する。