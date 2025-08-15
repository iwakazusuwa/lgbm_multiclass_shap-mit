# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib  # 日本語対応
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier, plot_tree
import seaborn as sns
import lightgbm as lgb
import shap
import sys

# =========================
# ディレクトリ・ファイル設定
# =========================
INPUT_FOLDER = '2_data'
OUTPUT_FOLDER = '3_output'

# カラム名
ID = 'customer_id'
# 目的変数
target_col = 'manufacturer'
# 説明変数
numeric_cols = ["family", "age", "children", "income"]

# パス設定
parent_path = os.path.dirname(os.getcwd())
input_path = os.path.join(parent_path, INPUT_FOLDER, 'sample_car_data.csv')

output_path = os.path.join(parent_path, OUTPUT_FOLDER)
# フォルダがなければ作成
os.makedirs(output_path, exist_ok=True)  

save_Fe_path = os.path.join(output_path, "比較用_LightGBM_Feature.png")
save_lgb_path  = os.path.join(output_path, "正解クラスの予測順位.png")
save_Matrix_path  = os.path.join(output_path, "Confusion_Matrix.png")

save_name = os.path.join(output_path, "メーカー別_SHAP_特徴ランキング.csv")
save_name_3  = os.path.join(output_path, "predictions_with_top3.csv")


# =========================
# CSV読み込み
# =========================
try:
    df = pd.read_csv(input_path, encoding="utf-8")
except UnicodeDecodeError:
    df = pd.read_csv(input_path, encoding="cp932")

# %%
# =========================
# カテゴリ列自動判定
# =========================
exclude_cols = [ID, target_col]
categorical_cols = [col for col in df.columns if col not in exclude_cols + numeric_cols]
df[categorical_cols] = df[categorical_cols].astype("category")

# =========================
# 説明変数と目的変数
# =========================
X_df = df.drop([ID, target_col], axis=1)
y_df = df[target_col]

# =========================
# ラベルを0始まりに変換（LightGBM対応）
# =========================
le = LabelEncoder()
y_enc = le.fit_transform(y_df)
class_names = [str(c) for c in le.classes_]

# クラス数（カテゴリーの種類）を確認
classes = np.unique(y_df)
print("クラス:", classes)

# =========================
# データ分割（test_size指定なし → デフォルト25%）
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_df, y_enc, random_state=0, stratify=y_enc
)
print("訓練データ数:", len(X_train))
print("テストデータ数:", len(X_test))

# %%
# =========================
# LightGBMモデル（訓練/テスト分割で学習）
# =========================
objective = 'binary' if len(class_names) == 2 else 'multiclass'
metric = 'binary_error' if objective == 'binary' else 'multi_error'
params = {'objective': objective, 'metric': metric, 'verbose': -1}
if objective == 'multiclass':
    params['num_class'] = len(class_names)

# 訓練データで学習
lgb_train = lgb.Dataset(X_train, label=y_train, feature_name=X_df.columns.tolist())
lgb_model = lgb.train(params, lgb_train, num_boost_round=50)

# 予測
y_pred_lgb_prob = lgb_model.predict(X_test)
y_pred_lgb = (y_pred_lgb_prob > 0.5).astype(int) if objective=='binary' else np.argmax(y_pred_lgb_prob, axis=1)


if objective == 'binary':
    y_pred_lgb = (y_pred_lgb_prob > 0.5).astype(int)
else:
    y_pred_lgb = np.argmax(y_pred_lgb_prob, axis=1)
    
# ==============================
# SHAP値計算
# ==============================

# TreeExplainerを使う場合
explainer = shap.TreeExplainer(lgb_model)
shap_values = explainer.shap_values(X_df)  # これでNumPy配列が返る

# binary / multiclassで形状が変わる
if objective == "binary":
    print("SHAP values shape:", shap_values.shape)  # (n_samples, n_features)
else:
    print("SHAP values shape:", [v.shape for v in shap_values])  # クラスごとの配列

# %%
# =========================
# 精度評価
# =========================
print("【LightGBM】     Accuracy:", accuracy_score(y_test, y_pred_lgb))
print("【LightGBM】     F1 Score:", f1_score(y_test, y_pred_lgb, average='weighted'))

# =========================
# 混同行列
# =========================
fig, ax = plt.subplots(figsize=(6, 5))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_lgb, ax=ax)
ax.set_title("LightGBM Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(save_Matrix_path), dpi=300)
# plt.show()



# ==============================
# 特徴量重要度
# ==============================
ax = lgb.plot_importance(lgb_model, max_num_features=10)
plt.title("LightGBM Feature Importance")
plt.savefig(os.path.join(save_Fe_path), dpi=300)


# %%
# ==============================
# LightGBMパラメータ設定
# ==============================
params = {
    'objective': 'multiclass',
    'metric': 'multi_error',
    'num_class': len(classes),
    'verbose': -1
}

# ==============================
# 学習（100回で学習）
# ==============================
lgb_model = lgb.train(params, lgb_train, num_boost_round=100)

# ==============================
#  全データに対する予測
# ==============================
y_all_pred_prob = lgb_model.predict(X_df.values)  # 予測確率
y_all_pred = np.argmax(y_all_pred_prob, axis=1)  # 予測クラスラベル

# ==============================
# 上位3候補の抽出
# ==============================
top3_classes = np.argsort(y_all_pred_prob, axis=1)[:, ::-1][:, :3]
top3_probs = np.sort(y_all_pred_prob, axis=1)[:, ::-1][:, :3]


# ==============================
# 元データに結果を追加
# ==============================
df["predicted_manufacturer"] = y_all_pred

for i in range(y_all_pred_prob.shape[1]):
    df[f"prob_class_{i}"] = y_all_pred_prob[:, i]

# 上位3クラスと確率の列を追加
for i in range(3):
    df[f"top{i+1}_class"] = top3_classes[:, i]
    df[f"top{i+1}_prob"] = top3_probs[:, i]
    
print(df[[
    target_col,               # 正解ラベル
    "predicted_manufacturer",     # モデルの予測
    "top1_class", "top1_prob",    # 1位予測と確率
    "top2_class", "top2_prob",    # 2位予測と確率
    "top3_class", "top3_prob"     # 3位予測と確率
]].head())



# %%
# =======================================
# 　正解割合（Top-3 Accuracy）
# =======================================
top3_accuracy = np.mean([
    y_true in top3 for y_true, top3 in zip(df[target_col], top3_classes)
])
print(f"Top-3 Accuracy: {top3_accuracy:.3f}")

# =======================================
# 正解が topN の何番目
# =======================================
def top_rank(row):
    true_class = row[target_col]
    top_classes = [row["top1_class"], row["top2_class"], row["top3_class"]]
    return top_classes.index(true_class) + 1 if true_class in top_classes else None

df["correct_rank"] = df.apply(top_rank, axis=1)
print(df["correct_rank"].value_counts())


# =======================================
# 件数と割合の計算
# =======================================
rank_counts = df["correct_rank"].value_counts().sort_index()
rank_percent = (rank_counts / rank_counts.sum()) * 100
# =======================================
# 棒グラフの作成
# =======================================
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(rank_percent.index.astype(str), rank_percent.values, color='skyblue')

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 1,
            f"{height:.1f}%", ha='center', va='bottom')

ax.set_title("正解クラスの予測順位（Top-k Accuracy）", fontsize=14)
ax.set_xlabel("予測順位")
ax.set_ylabel("割合（%）")
plt.ylim(0, 105)
plt.tight_layout()
plt.savefig(os.path.join(save_lgb_path), dpi=300)
# plt.show()

# %%
# ==============================
# 上位3寄与特徴量抽出
# ==============================
results_top3 = []

for i in range(len(df)):
    pred_class = df.loc[i, "predicted_manufacturer"]

    # 多クラス対応
    if isinstance(shap_values, list):
        shap_vals = shap_values[pred_class][i, :]  # 予測クラスのSHAP値を使用
        features = X_df.columns.tolist()
    else:
        shap_vals = shap_values[i, :]
        features = X_df.columns.tolist()

    # 上位3特徴量の抽出
    top_idx = np.argsort(np.abs(shap_vals))[::-1][:3]
    top_features = [features[j] for j in top_idx]
    top_values = [shap_vals[j] for j in top_idx]
    # 結果をまとめる
    tmp = {
        "customer_id": df.loc[i, "customer_id"],
        "predicted_class": pred_class,
        "top1_feature": top_features[0],
        "top1_value": top_values[0],
        "top2_feature": top_features[1],
        "top2_value": top_values[1],
        "top3_feature": top_features[2],
        "top3_value": top_values[2]
    }
    results_top3.append(tmp)

top3_df = pd.DataFrame(results_top3)

# %%
#==============================
# 重複列を削除して安全化
#==============================
df = df.loc[:, ~df.columns.duplicated()]
top3_df = top3_df.loc[:, ~top3_df.columns.duplicated()]

# 列名を統一
df = df.rename(columns={"predicted_manufacturer": "predicted_class"})

# 安全にマージ
df_merged = pd.merge(
    df,
    top3_df,
    on=["customer_id", "predicted_class"],
    how="left",
    suffixes=("", "_top3")  # 同名列が残っても自動で _top3 が付く
)


# 列名リネーム　	
rename_dict = {
    "top1_feature": "上位1特徴量",
    "top1_value": "上位1寄与度",
    "top2_feature": "上位2特徴量",
    "top2_value": "上位2寄与度",
    "top3_feature": "上位3特徴量",
    "top3_value": "上位3寄与度",
    "prob_class_0": "メーカー0の確率",
    "prob_class_1": "メーカー1の確率",
    "prob_class_2": "メーカー2の確率",
    "prob_class_3": "メーカー3の確率",
    "prob_class_4": "メーカー4の確率",
    "prob_class_5": "メーカー5の確率",
    "prob_class_6": "メーカー6の確率",
    "top1_class": "トップ1候補メーカー",
    "top1_prob": "トップ1確率",
    "top2_class": "トップ2候補メーカー",
    "top2_prob": "トップ2確率",
    "top3_class": "トップ3候補メーカー",
    "top3_prob": "トップ3確率"
}
df_merged = df_merged.rename(columns=rename_dict)

# CSV出力
df_merged.to_csv(save_name_3, index=False, encoding="utf-8-sig")

# %%
# ==============================
# multiclass / binary 両対応で DataFrame化
# ==============================
if isinstance(shap_values, list):  # multiclass
    # 予測クラスに対応する SHAP 値を抽出
    shap_array = np.array([shap_values[pred][i, :]
                           for i, pred in enumerate(df["predicted_class"].values)])
else:  # binary
    shap_array = shap_values

# DataFrameに変換
shap_df = pd.DataFrame(shap_array, columns=X_df.columns)
shap_df["predicted_class"] = df["predicted_class"].values

# ==============================
# クラスごとの SHAP値平均（特徴量別）
# ==============================
summary_list = []
for cls in sorted(df["predicted_class"].unique()):
    shap_mean = shap_df[shap_df["predicted_class"] == cls].drop(columns="predicted_class").mean().abs()
    shap_mean_sorted = shap_mean.sort_values(ascending=False)
    
    for feature, value in shap_mean_sorted.items():
        summary_list.append({
            "メーカー": cls,
            "特徴量": feature,
            "平均SHAP値": round(value, 5)
        })

shap_summary_df = pd.DataFrame(summary_list)

# 上位N個
topN = 5
display_df = shap_summary_df.groupby("メーカー").head(topN)

# CSV保存
display_df.to_csv(save_name, index=False, encoding="utf-8-sig")
#display_df

# OS別で出力フォルダを開く
if sys.platform.startswith('win'):
    os.startfile(output_path)
elif sys.platform.startswith('darwin'):
    subprocess.run(['open', output_path])
else:
    subprocess.run(['xdg-open', output_path])

print(" 完了")
