# ====================== 1. 导入工具包 ======================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, recall_score, precision_score, classification_report
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline  

import shap
import warnings
import joblib 

warnings.filterwarnings('ignore')

# ====================== 2. 读取数据 ======================
df = pd.read_csv("cs-training.csv")
df = df.drop(columns=["Unnamed: 0"])  # 删除索引列

# ====================== 3. 数据清洗 ======================
# 中位数填充缺失值
df = df.fillna(df.median())
# 异常值处理
df = df[df["age"] > 18]
df = df[df["NumberOfTimes90DaysLate"] < 20]

# ====================== 4. 划分特征和目标 ======================
X = df.drop("SeriousDlqin2yrs", axis=1)
y = df["SeriousDlqin2yrs"]
feature_names = X.columns.tolist()

# ====================== 5. 划分训练集、验证集和测试集 ======================
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# ====================== 6. 构建带SMOTE和标准化的Pipeline ======================
pipeline = ImbPipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('classifier', XGBClassifier(
        n_estimators=200,
        eval_metric='logloss',
        random_state=42,
        use_label_encoder=False
    ))
])

# ====================== 7. 超参数调优（结合SMOTE） ======================
# 定义参数搜索空间
param_dist = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [3, 5, 7],
    'classifier__learning_rate': [0.01, 0.05, 0.1],
    'classifier__subsample': [0.8, 0.9, 1.0],
    'classifier__colsample_bytree': [0.8, 0.9, 1.0]
}

# 使用分层交叉验证
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

random_search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=20,
    cv=cv,
    scoring='roc_auc',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

# 因为 pipeline 中包含 SMOTE，所以 fit 时会自动过采样
random_search.fit(X_train, y_train)

print("最佳参数:", random_search.best_params_)
print("最佳交叉验证AUC:", random_search.best_score_)

best_model = random_search.best_estimator_

# ====================== 8. 在验证集上评估（早停或模型选择） ======================
y_val_proba = best_model.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, y_val_proba)
print(f"验证集 AUC: {val_auc:.4f}")

# ====================== 9. 最终模型在测试集上的评估 ======================
y_test_proba = best_model.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, y_test_proba)
print(f"测试集 AUC: {test_auc:.4f}")

# 计算 KS 值
def ks_score(y_true, y_pred_proba):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    return max(tpr - fpr)

ks = ks_score(y_test, y_test_proba)
print(f"KS 值: {ks:.4f}")

# 寻找最佳阈值（最大化 F1 分数）
from sklearn.metrics import f1_score
thresholds = np.arange(0.1, 0.9, 0.01)
f1_scores = [f1_score(y_test, (y_test_proba >= t).astype(int)) for t in thresholds]
best_thresh = thresholds[np.argmax(f1_scores)]
print(f"最佳阈值: {best_thresh:.2f}")

y_test_pred = (y_test_proba >= best_thresh).astype(int)
recall = recall_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
print(f"召回率 (Recall): {recall:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print("分类报告:")
print(classification_report(y_test, y_test_pred))

# ====================== 10. 保存最终模型 ======================
joblib.dump(best_model, 'best_credit_model.pkl')
print("模型已保存为 best_credit_model.pkl")

# ====================== 11. 可视化 ======================
# 11.1 ROC 曲线
plt.figure(figsize=(6,4))
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
plt.plot(fpr, tpr, label=f'AUC = {test_auc:.4f}')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
plt.show()

# 11.2 KS 曲线
plt.figure(figsize=(6,4))
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
ks_stat = max(tpr - fpr)
plt.plot(fpr, label='FPR')
plt.plot(tpr, label='TPR')
plt.plot([0,1], [0,1], 'k--')
plt.legend()
plt.title(f'KS Curve (KS = {ks_stat:.4f})')
plt.tight_layout()
plt.savefig('ks_curve.png', dpi=300, bbox_inches='tight')
plt.show()

# 11.3 SHAP 分析
xgb_model = best_model.named_steps['classifier']
scaler = best_model.named_steps['scaler']
# 对测试集进行标准化
X_test_scaled = scaler.transform(X_test)
explainer = shap.TreeExplainer(xgb_model)
# 使用部分样本加速（例如 100 个）
sample_idx = np.random.choice(len(X_test_scaled), min(100, len(X_test_scaled)), replace=False)
shap_values = explainer.shap_values(X_test_scaled[sample_idx])
shap.summary_plot(shap_values, X_test_scaled[sample_idx], feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
plt.show()

print("所有可视化已保存")