# MPC 型オンライン励起軌道最適化: 実装計画

## 概要

Zhang-RSS2025 の MPC 的オンライン励起軌道最適化を, MuJoCo シミュレーション上の UR5e で実装する.
ARMOUR / 区間演算 / 衝突回避は省略し, 関節制限のみを制約とする簡易版.

## アーキテクチャ

新規コードは `src/mjwarp_ur5e/` 以下に配置. 既存モジュールは変更しない.

```
src/mjwarp_ur5e/
  trajectories/
    quintic_spline.py              [Step 1] 軌道パラメータ化
  identification/
    mpc/
      __init__.py                  [Step 2 で作成, Step 6 で更新]
      config.py                    [Step 2] データクラス
      metrics.py                   [Step 3] 励起品質メトリクス
      planner.py                   [Step 4] 単一ホライズン最適化
      loop.py                      [Step 5] MPC ループ
  demos/
    run_mpc_identification.py      [Step 7] CLI デモ
```

## 依存関係グラフ

```
Step 1 (QuinticSpline) ─┐
Step 2 (Config)         ├──→ Step 4 (Planner) ──→ Step 5 (Loop) ──→ Step 6 (Exports)
Step 3 (Metrics)        ┘                                            ──→ Step 7 (CLI)
```

Wave 1 (並列可): Step 1 + Step 2 + Step 3
Wave 2: Step 4
Wave 3: Step 5
Wave 4 (並列可): Step 6 + Step 7

## パラメータ順序の注意

`mjwarp_ur5e` 側の慣性パラメータ順序: `[m, hx, hy, hz, Ixx, Iyy, Izz, Ixy, Ixz, Iyz]`
`iparam_identification` 側の順序: `[m, hx, hy, hz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz]`

MPC ループ内は `mjwarp_ur5e` の規約で閉じる. 結果を比較するときのみ変換が必要.

## wrench 順序

`mjwarp_ur5e` 側: `[τx, τy, τz, fx, fy, fz]` (torque-first)

## 各ステップの詳細

- [Step 1: QuinticSpline 軌道パラメータ化](step1_quintic_spline.md)
- [Step 2: MPC Config データ構造](step2_config.md)
- [Step 3: Metrics モジュール](step3_metrics.md)
- [Step 4: ExcitationPlanner](step4_planner.md)
- [Step 5: MPCLoop](step5_loop.md)
- [Step 6: パッケージエクスポート](step6_exports.md)
- [Step 7: CLI デモ](step7_cli.md)

## 参考論文

Zhang, Zhou, Vasudevan."Provably-Safe, Online System Identification" (RSS 2025)
- ホライズン長 T_h = 3.0 s, リプラン周期 T_replan = 1.5 s
- コスト: リグレッサ条件数 κ₂(W(k)) の最小化
- 軌道: Bezier 曲線 (本実装では quintic spline で代替)
