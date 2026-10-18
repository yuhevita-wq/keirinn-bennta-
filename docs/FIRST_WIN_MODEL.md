# F1・S級｜採用済み1着独立勝率の計算仕様

## 1. 入力

F1開催のS級レースの**出走表・選手属性・予想ライン**のみ。オッズ・確定着順・払戻・後日確定した数値を事前予測に入力しない。通常の養成所期数と、外国人選手区分 `term=999` を数値として混同しない。

- ベース17特徴量：`score, age, term, s_count, b_count, nige_count, makuri_count, sashi_count, mark_count, first_count, second_count, third_count, outside_count, win_rate, top2_rate, top3_rate, gear`。
- 19のライン・選手関係特徴量：旧リポジトリの `discover_v2.py` と同じ定義。
- 希少カテゴリ1特徴量：`term_999_foreign_category`（0/1のまま、標準偏差で割らない）。
- 正式採用パラメータの再現には [2026年9月モデルの凍結マニフェスト](../models/first_win_2026sep_manifest.json) を使用する。

## 2. 数式

37特徴量 `x_ij` の欠損を学習期間の中央値で埋め、通常の特徴量は学習期間の平均・標準偏差で標準化する。別途37個の欠損フラグを連結して、合計74係数の線形予測子を作る。

\[
z_{r,i}=\sum_{j=1}^{37}\beta_j\frac{\tilde{x}_{r,i,j}-\mu_j}{\sigma_j}
+\sum_{j=1}^{37}\beta_{37+j}\mathbf{1}[x_{r,i,j}\text{が欠損}]
\]

\[
p_{r,i}=\frac{\exp(z_{r,i})}{\sum_{h\in r}\exp(z_{r,h})}
\]

開発期間6,393レースで校正した係数
**\(\theta=0.6412587661746468\)** を使う。

\[
q_{r,i}=\frac{p_{r,i}\exp\left[-\theta\max(p_{r,i}-0.4,0)\right]}
{\sum_{h\in r} p_{r,h}\exp\left[-\theta\max(p_{r,h}-0.4,0)\right]}
\]

ここで \(q_{r,i}\) が採用済みの**1着独立勝率**。市場オッズは含まれない。

**\(\gamma,\delta,a,b\)**（2着・3着用係数）は新モデルでは未推定。旧V7の係数は流用しない。

## 3. 期間と再現性

- 凍結されている過去の14,215予測レースは**12個の時間区切りの異なるモデル**で計算した。1〜3着確定結果と結合できるのは14,081R。
- 新リポジトリ内の `first_win_2026sep_manifest.json` は**2026年8月30日までで学習した9月用スナップショット**。過去14,081Rの独立勝率を、この一つの係数で再生成してはいけない。
- 全期間の各CSVと対応する12個のモデル・SHA256は [source_ledger.json](../data/source_ledger.json) に固定。大きなCSVを新リポジトリに重複コピーしていない。
- レース情報の当時の事前スナップショット一致は未認証。締切15分前の投票への適用性能は別途検証が必要。

## 4. 手動での確率計算

以下のような実際のレースの出走表をJSONにして実行する。

```bash
python src/predict_first.py --input path/to/racecard.json
```

必須JSON属性：

```json
{
  "meeting_grade": "F1",
  "race_type": "Ｓ級予選",
  "predicted_line_formation": "1-2-3/4-5/6-7",
  "riders": [
    {"car_no": 1, "score": 102.4, "age": 30, "term": 105},
    {"car_no": 2, "score": 100.3, "age": 33, "term": 99},
    {"car_no": 3, "score": 98.4, "age": 37, "term": 87},
    {"car_no": 4, "score": 102.1, "age": 27, "term": 117},
    {"car_no": 5, "score": 99.2, "age": 34, "term": 103},
    {"car_no": 6, "score": 98.9, "age": 38, "term": 89},
    {"car_no": 7, "score": 97.6, "age": 40, "term": 85}
  ]
}
```

上は**入力形式だけを示す架空例**であり、他の未記載の特徴量は欠損として補完される。実レースでは17個の特徴量すべてを可能な限り入力すること。

## 5. ベンターとの比較は何を意味するか

論文の独立モデル擬似R²は0.1245、F1・S級採用1着モデルの2025年4月以降7,688Rの擬似R²は0.187018。ただし競技と対象の難易度が違うため、数字が高いだけで「ベンターの競馬モデルを上回る」とは断言しない。また市場統合の利益検証が済んだ数字でもない。
