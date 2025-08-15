# lgbm_multiclass_shap-mit

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-3.4+-green)](https://lightgbm.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-0.42+-orange)](https://github.com/slundberg/shap)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

LightGBM × SHAP を用いた多クラス分類モデルの予測と、個別サンプルごとの特徴量寄与度可視化ツールです。

---

## 対象者
- Python で機械学習モデル（LightGBM）を扱える方
- 多クラス分類の予測結果と特徴量寄与を確認したい方
- SHAPを使ってモデルの透明性・説明責任を向上させたい方
- 予測結果のトップ3候補や寄与特徴量を出力したい方

---

## 概要
本ツールでは、LightGBM による多クラス分類モデルを学習し、
- 予測クラスラベル
- クラスごとの予測確率
- Top-3予測候補
- 各サンプルごとのSHAP寄与特徴量（上位3つ）

を CSV として出力できます。

これにより、個別サンプルの予測根拠を定量的に可視化し、モデルの説明性・信頼性を向上させます。

---

## 特徴
- LightGBMによる多クラス分類
- 各サンプルごとのTop-3予測候補
- SHAPによる特徴量寄与度の可視化
- CSV出力で分析結果の保存・共有が可能
- クラスごとの平均SHAP値を確認してモデル全体の解釈も可能

---

## 実行方法
```bash
python shap_analysis.py
```
-  shap_analysis.py で学習 → 予測 → SHAP分析 → CSV出力まで一括実行
-  出力例:
  - predictions_with_top3.csv：個別サンプルの予測・Top3候補・SHAP寄与度
  - shap_analysis.py：クラスごとの平均SHAP値


# 入力データフォーマット例
2_data/sample.csv を参照ください。


# フォルダ構成例
```
├── 1_flow/
│   └── shap_analysis.py       # メインスクリプト
├── 2_data/
│   └── sample.csv              # 入力データ（縦持ち形式）
├── 3_output/
│   └──                        # 画像出力先
```

## CSV出力例（predictions_with_top3.csv）
 | customer\_id | predicted\_class | トップ1候補メーカー | トップ1確率 | 上位1特徴量          | 上位1寄与度 | ... |
| ------------ | ---------------- | ---------- | ------ | --------------- | ------ | --- |
| 1            | 5                | 5          | 0.97   | car\_preference | -1.39  | ... |




## 今後の拡張予定
- Top-N 候補の精度指標やグラフ可視化の追加
- 特徴量寄与のグローバル・ローカル解釈のUI表示
- 他モデル（XGBoost, CatBoostなど）への対応

## 貢献方法
プロジェクトへの貢献は以下の方法で歓迎します：
- バグ報告や機能追加の提案は Issues を通じて行ってください。
- コードの改善や新機能の追加は Pull Request を作成してください。
- ドキュメントの改善や翻訳も歓迎します。

## LICENSE
MIT License（詳細はLICENSEファイルをご参照ください）

#### 開発者： iwakazusuwa(Swatchp)















