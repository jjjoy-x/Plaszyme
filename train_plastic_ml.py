import os
import torch
import pandas as pd
import itertools
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

from plastic.mol_features.descriptors_rdkit import PlasticFeaturizer

# ========== 用户配置 ==========
SDF_DIR = "/Users/shulei/PycharmProjects/Plaszyme/plastic/mols_for_unimol_10_sdf"  # 修改为你的SDF文件夹
CONFIG_PATH = "/Users/shulei/PycharmProjects/Plaszyme/plastic/mol_features/rdkit_features.yaml"
CO_MATRIX_CSV = "/Users/shulei/PycharmProjects/Plaszyme/test/outputs/plastic_co_matrix.csv"
SIM_THRESHOLD = 0.01
TEST_SIZE = 0.3
RANDOM_STATE = 42
OUTPUT_DIR = "run/ml_results_2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== 特征提取 ==========
print("🧬 提取塑料结构描述符...")
featurizer = PlasticFeaturizer(CONFIG_PATH)
feature_dict, _ = featurizer.featurize_folder(SDF_DIR)

features_df = pd.DataFrame.from_dict(feature_dict, orient="index")
features_df = features_df.sort_index()

# ========== 加载共降解矩阵 ==========
co_matrix = pd.read_csv(CO_MATRIX_CSV, index_col=0)
plastics = features_df.index.intersection(co_matrix.index)
features_df = features_df.loc[plastics]
co_matrix = co_matrix.loc[plastics, plastics]

# ========== 构建样本对（二分类任务）==========
X, y = [], []
pairs = list(itertools.combinations(plastics, 2))
for p1, p2 in pairs:
    if pd.isna(co_matrix.loc[p1, p2]):
        continue
    label = 1 if co_matrix.loc[p1, p2] >= SIM_THRESHOLD else 0
    pair_feature = np.abs(features_df.loc[p1] - features_df.loc[p2])  # 可改为拼接或其他方式
    X.append(pair_feature.values)
    y.append(label)

X = np.array(X)
y = np.array(y)
feature_names = features_df.columns.tolist()

print(f"✅ 样本对数：{len(X)} (正样本: {sum(y)}, 负样本: {len(y) - sum(y)})")

# ========== 数据划分 ==========
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

# ========== 模型构建 ==========
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
}

for name, model in models.items():
    print(f"\n🚀 训练模型：{name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n📊 分类报告（{name}）:")
    print(classification_report(y_test, y_pred, digits=4))
    auc = roc_auc_score(y_test, y_proba)
    print(f"AUC: {auc:.4f}")

    # 特征重要性
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        continue

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    importance_df.to_csv(os.path.join(OUTPUT_DIR, f"{name}_importance.csv"), index=False)

    # 可视化
    plt.figure(figsize=(10, 5))
    sns.barplot(x="importance", y="feature", data=importance_df.head(15), palette="viridis")
    plt.title(f"Top 15 Feature Importance - {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{name}_top_features.png"))
    plt.close()

print("✅ 所有模型训练完成，结果保存在：", OUTPUT_DIR)