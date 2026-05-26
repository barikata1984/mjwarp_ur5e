# docs/ ガイド

このディレクトリはプロジェクトの設計文書、進捗管理、実験記録を管理する。

## 構成

| パス | 役割 |
|------|------|
| `TODO.md` | 進捗チェックリスト |
| `ISSUES.md` | 未解決の技術課題 |
| `LOGS/README.md` | 実験記録インデックス |
| `LOGS/log_*.md` | トピック別実験記録 |
| `REFERENCES/MAIN.md` | 参考文献マスタ |
| `REFERENCES/references.bib` | BibTeX (MAIN.md から自動生成) |
| `SURVEYS/README.md` | 文献調査インデックス |
| `SURVEYS/*.md` | サーベイ別レポート |
| `REPORTS/` | 日次報告 (`YYYY-MM-DD.md`) |
| `container-agent-notes.md` | コンテナ環境覚書 |
| `inertial-identification-optimal-excitation-plan.md` | 慣性パラメータ同定・最適励起 実装計画 |

## 運用規則

- LOGS は **追記のみ**。過去の記録は編集しない
- ISSUES は解決したら **エントリごと削除**
- 日付は ISO 形式 (`YYYY-MM-DD`)
- 参考文献の引用規約は `.claude/rules/references.md` を参照
