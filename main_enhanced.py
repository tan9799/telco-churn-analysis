# -*- coding: utf-8 -*-
"""
电信客户流失数据分析及可视化 - 增强版
包含完整的 EDA、特征工程、模型训练（含调优）、SHAP 解释、交互式大屏
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 创建输出目录
os.makedirs('output', exist_ok=True)

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# -------------------- 机器学习相关库 --------------------
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve, confusion_matrix,
                             classification_report, make_scorer)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE

import shap
SHAP_AVAILABLE = True

# ============================================================
# 1. 数据加载与清洗
# ============================================================
print("="*60)
print("1. 加载与清洗数据...")
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
print(f"原始数据形状: {df.shape}")

# 处理 TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# 去除其他可能存在的 NaN（确保后续聚类安全）
df = df.dropna()

print(f"清洗后数据形状: {df.shape}")

# ============================================================
# 2. 描述性统计
# ============================================================
print("\n2. 描述性统计...")
print(df[['tenure', 'MonthlyCharges', 'TotalCharges']].describe())
print("\n流失分布:\n", df['Churn'].value_counts(normalize=True))

# ============================================================
# 3. 探索性数据分析 (EDA) - 增强可视化
# ============================================================
print("\n3. 生成 EDA 图表...")

# 3.1 流失率饼图
churn_counts = df['Churn'].value_counts()
plt.figure(figsize=(6,6))
plt.pie(churn_counts, labels=['留存 (No)', '流失 (Yes)'], autopct='%1.1f%%',
        colors=['#2ecc71', '#e74c3c'], startangle=90, explode=(0,0.05))
plt.title('图1 客户流失率分布')
plt.savefig('output/churn_rate_pie.png', dpi=300, bbox_inches='tight')
plt.close()

# 3.2 连续特征箱线图（流失 vs 留存）
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
features = ['tenure', 'MonthlyCharges', 'TotalCharges']
for i, feat in enumerate(features):
    sns.boxplot(x='Churn', y=feat, data=df, ax=axes[i], palette='Set2')
    axes[i].set_title(f'{feat} 分布 (流失 vs 留存)')
plt.tight_layout()
plt.savefig('output/boxplots_churn.png', dpi=300, bbox_inches='tight')
plt.close()

# 3.3 分类特征与流失率（合同、付款方式、互联网服务）
cat_features = ['Contract', 'PaymentMethod', 'InternetService']
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, cat in enumerate(cat_features):
    df.groupby(cat)['Churn'].apply(lambda x: (x=='Yes').mean()).plot(kind='bar', ax=axes[i], color='skyblue')
    axes[i].set_title(f'{cat} 流失率')
    axes[i].set_ylabel('流失率')
plt.tight_layout()
plt.savefig('output/categorical_churn.png', dpi=300, bbox_inches='tight')
plt.close()

# 3.4 相关性热力图（完整）
plt.figure(figsize=(10,8))
numeric_df = df[['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']].copy()
numeric_df['Churn_num'] = df['Churn'].map({'Yes':1, 'No':0})
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('图3 数值特征相关性热力图')
plt.savefig('output/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# 3.5 KMeans 聚类（修复 NaN 问题）
print("执行 KMeans 聚类...")
cluster_df = df[['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']].copy()
# 删除任何可能存在的 NaN（虽然已经 dropna，但保留此步骤以防万一）
cluster_df = cluster_df.dropna()
scaler_cluster = StandardScaler()
cluster_scaled = scaler_cluster.fit_transform(cluster_df)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(cluster_scaled)
pca = PCA(n_components=2)
cluster_pca = pca.fit_transform(cluster_scaled)
plt.figure(figsize=(8,6))
scatter = plt.scatter(cluster_pca[:,0], cluster_pca[:,1], c=clusters, cmap='viridis', alpha=0.6)
plt.title('图4 KMeans 客户分群 (PCA)')
plt.xlabel('主成分1')
plt.ylabel('主成分2')
plt.colorbar(scatter, label='聚类')
plt.savefig('output/customer_clusters.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# 4. 特征工程与预处理
# ============================================================
print("\n4. 特征工程...")
data = df.copy()
# 二分类编码
binary_cols = ['Churn', 'gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
for col in binary_cols:
    data[col] = data[col].map({'Yes':1, 'No':0})

# One-Hot 编码
categorical_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                    'Contract', 'PaymentMethod']
for col in categorical_cols:
    data[col] = data[col].fillna('Unknown')
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
data.drop('customerID', axis=1, inplace=True)

X = data.drop('Churn', axis=1)
y = data['Churn']

# 确保所有数值
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

# 划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# 标准化（用于线性模型和SVM）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

# 树模型用原始数值
X_train_tree = X_train_res.values
X_test_tree = X_test.values
feature_names = X.columns.tolist()

# ============================================================
# 5. 模型训练（含交叉验证与调优）
# ============================================================
print("\n5. 模型训练与超参数调优...")

models_grid = {
    'Logistic Regression': {
        'model': LogisticRegression(max_iter=1000, random_state=42),
        'params': {'C': [0.1, 1, 10]},
        'use_scaled': True
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=42, n_jobs=-1),
        'params': {'n_estimators': [50, 100], 'max_depth': [None, 10]},
        'use_scaled': False
    },
    'XGBoost': {
        'model': xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42),
        'params': {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 6]},
        'use_scaled': False
    },
    'LightGBM': {
        'model': lgb.LGBMClassifier(random_state=42, n_jobs=-1),
        'params': {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1], 'num_leaves': [31, 50]},
        'use_scaled': False
    },
    'SVM': {
        'model': SVC(probability=True, random_state=42),
        'params': {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']},
        'use_scaled': True
    }
}

best_models = {}
results = []

for name, cfg in models_grid.items():
    print(f"\n>>> 调优 {name}")
    model = cfg['model']
    param_grid = cfg['params']
    use_scaled = cfg['use_scaled']
    
    if use_scaled:
        X_train_use = X_train_scaled
        X_test_use = X_test_scaled
    else:
        X_train_use = X_train_tree
        X_test_use = X_test_tree
    
    gs = GridSearchCV(model, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=0)
    gs.fit(X_train_use, y_train_res)
    best_model = gs.best_estimator_
    best_models[name] = best_model
    print(f"  最佳参数: {gs.best_params_}")
    
    y_pred = best_model.predict(X_test_use)
    y_prob = best_model.predict_proba(X_test_use)[:, 1] if hasattr(best_model, 'predict_proba') else None
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0
    
    results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'ROC-AUC': auc
    })
    print(f"  F1={f1:.4f}, AUC={auc:.4f}")

results_df = pd.DataFrame(results).round(4)
print("\n【模型性能汇总】")
print(results_df.to_string(index=False))
results_df.to_csv('output/model_performance.csv', index=False)

best_model_name = results_df.loc[results_df['F1-Score'].idxmax(), 'Model']
best_clf = best_models[best_model_name]
print(f"\n最佳模型: {best_model_name}")

# ============================================================
# 6. 可视化评估
# ============================================================
print("\n6. 生成评估图表...")

# 6.1 模型性能对比条形图
fig, ax = plt.subplots(figsize=(12,6))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x = np.arange(len(results_df['Model']))
width = 0.2
for i, metric in enumerate(metrics):
    ax.bar(x + i*width, results_df[metric], width, label=metric)
ax.set_xlabel('模型')
ax.set_ylabel('分数')
ax.set_title('图5 各模型性能对比')
ax.set_xticks(x + width*1.5)
ax.set_xticklabels(results_df['Model'], rotation=15)
ax.legend(loc='lower right')
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('output/model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 6.2 最佳模型混淆矩阵
if best_model_name in ['Logistic Regression', 'SVM']:
    X_test_best = X_test_scaled
else:
    X_test_best = X_test_tree
y_pred_best = best_clf.predict(X_test_best)
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['留存 (No)', '流失 (Yes)'],
            yticklabels=['留存 (No)', '流失 (Yes)'])
plt.title(f'图6 {best_model_name} 混淆矩阵')
plt.ylabel('实际标签')
plt.xlabel('预测标签')
plt.savefig('output/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# 6.3 ROC 曲线
plt.figure(figsize=(8,6))
for name, cfg in models_grid.items():
    model = best_models[name]
    if hasattr(model, 'predict_proba'):
        if cfg['use_scaled']:
            X_plot = X_test_scaled
        else:
            X_plot = X_test_tree
        y_prob_plot = model.predict_proba(X_plot)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob_plot)
        auc_score = roc_auc_score(y_test, y_prob_plot)
        plt.plot(fpr, tpr, label=f'{name} (AUC={auc_score:.3f})')
plt.plot([0,1], [0,1], 'k--', label='随机猜测')
plt.xlabel('假阳性率 (FPR)')
plt.ylabel('真阳性率 (TPR)')
plt.title('图7 ROC曲线对比')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.savefig('output/roc_curves.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# 7. SHAP 可解释性（针对最佳树模型）
# ============================================================
if SHAP_AVAILABLE and best_model_name not in ['Logistic Regression', 'SVM']:
    print("\n7. SHAP 可解释性分析...")
    explainer = shap.TreeExplainer(best_clf)
    X_sample = X_test_tree[:100]
    shap_values = explainer.shap_values(X_sample)
    
    plt.figure(figsize=(10,6))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type="bar", show=False)
    plt.title('图8 SHAP 特征重要性 (条形图)')
    plt.tight_layout()
    plt.savefig('output/shap_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10,8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.title('图9 SHAP 特征影响分布图')
    plt.tight_layout()
    plt.savefig('output/shap_summary_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================
# 8. Pyecharts 交互式大屏（增强版）
# ============================================================
try:
    from pyecharts.charts import Pie, Bar, Page
    from pyecharts import options as opts
    from pyecharts.globals import ThemeType
    
    print("\n8. 生成 Pyecharts 交互式大屏...")
    
    churn_data = [['留存 (No)', int(churn_counts['No'])], ['流失 (Yes)', int(churn_counts['Yes'])]]
    pie = (Pie(init_opts=opts.InitOpts(theme=ThemeType.VINTAGE, width="600px", height="400px"))
           .add("", churn_data, radius=["40%", "70%"])
           .set_global_opts(title_opts=opts.TitleOpts(title="客户流失率分布"))
           .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {d}%")))
    
    models_list = results_df['Model'].tolist()
    acc_list = [round(v*100,1) for v in results_df['Accuracy']]
    f1_list = [round(v*100,1) for v in results_df['F1-Score']]
    bar = (Bar(init_opts=opts.InitOpts(theme=ThemeType.VINTAGE, width="700px", height="400px"))
           .add_xaxis(models_list)
           .add_yaxis("准确率", acc_list)
           .add_yaxis("F1分数", f1_list)
           .set_global_opts(title_opts=opts.TitleOpts(title="模型性能对比"),
                            legend_opts=opts.LegendOpts(pos_top="5%"))
           .set_series_opts(label_opts=opts.LabelOpts(is_show=True, position="top")))
    
    contract_churn = df.groupby('Contract')['Churn'].apply(lambda x: (x=='Yes').mean()*100).reset_index()
    contract_bar = (Bar(init_opts=opts.InitOpts(theme=ThemeType.VINTAGE, width="600px", height="400px"))
                    .add_xaxis(contract_churn['Contract'].tolist())
                    .add_yaxis("流失率 (%)", contract_churn['Churn'].round(1).tolist())
                    .set_global_opts(title_opts=opts.TitleOpts(title="不同合同类型流失率"),
                                     xaxis_opts=opts.AxisOpts(name="合同类型"),
                                     yaxis_opts=opts.AxisOpts(name="流失率 (%)"))
                    .set_series_opts(label_opts=opts.LabelOpts(is_show=True, position="top")))
    
    page = Page(layout=Page.DraggablePageLayout)
    page.add(pie)
    page.add(bar)
    page.add(contract_bar)
    page.render("output/churn_dashboard_enhanced.html")
    print("✅ 增强版交互大屏已生成: output/churn_dashboard_enhanced.html")
    
except ImportError as e:
    print(f"⚠️ pyecharts 未安装或出错: {e}")

# ============================================================
# 9. 输出总结
# ============================================================
print("\n" + "="*60)
print("🎉 项目运行完毕！生成文件列表：")
print("  output/ 目录下包含所有图表和报告。")
print("="*60)