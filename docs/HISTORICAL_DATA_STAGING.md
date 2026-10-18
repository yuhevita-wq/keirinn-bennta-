# 旧リポジトリの原始データを新研究へ接続

**F1・S級の原始データは旧リポジトリに残す。** 新リポジトリには採用済みの1着重み、固定済み対象母数、検証結果、再現コードのみを保持する。

## GitHub Actionsで研究用データを生成

[Verify pinned 14,081 F1 S-class research cohort](../.github/workflows/verify-history.yml) は以下の処理を実行する。

1. 旧リポジトリの固定コミット `8da8f4f8e1125eb522df5546d07cd6657fab7dda` を取得。
2. F1・S級の過去12期間の独立勝率CSVを各SHA-256で照合。
3. 対象レース、確定1〜3着、予想ラインを厳密に結合。
4. 採用した1着確率 `q_i` と元の `p_i` を、1レース1行のJSONLに出力。
5. 14,081R、2着候補83,381件、3着候補69,300件、期間ごとの対象IDハッシュを検証。
6. JSONLと監査JSONを**GitHub Actionsの成果物**として保存。新リポジトリのGit履歴に大量のレース行を複製しない。

実行成功の検証値は [staged_history_audit.json](../reports/staged_history_audit.json) に記録。成果物は期限付きであり、**元ファイルが固定コミットに残っていれば再生成可能**。

## 自分のPCで生成する場合

旧リポジトリを固定コミットでチェックアウトしたディレクトリを用意する。

```bash
python scripts/stage_history.py --source-root ../keirin-market-analysis- --output stage/f1_s_class_14081.jsonl
```

旧ブランチの最新状態を暗黙に使うことはしない。GitコミットID、全12本のCSV SHA-256、対象レースID母数が一致しなければ処理は失敗する。

## 出力1行の内容

- `race_id`, `race_date`, `period`, `race_type`
- `predicted_line_formation`
- 確定した `first`, `second`, `third`
- 全出走選手の `car_no`, `raw_p`, `adopted_q`
- 分割の来歴（法則発見の標本を制限するためのものではない）

**未実施**：新しい2着の条件付き確率推定、3着の条件付き確率推定、市場オッズ統合、払戻・期待値研究。

このJSONLには確定着順が含まれるため、**数学的法則の発見・検証用の研究データ**。レース直前の予測入力として結果列を使ってはいけない。
