import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# -------------------- 机器学习相关库 --------------------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve, confusion_matrix,
                             silhouette_score)

import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE

# 可选：SHAP用于可解释性
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("注意: shap库未安装，跳过SHAP分析。")

# ============================================================
# 1. 数据加载与清洗
# ============================================================
print("="*60)
print("1. 正在加载与清洗数据...")
print("="*60)

df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
print(f"原始数据形状: {df.shape}")

# 处理TotalCharges：转为数值，空值填充中位数
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
median_val = df['TotalCharges'].median()
df['TotalCharges'] = df['TotalCharges'].fillna(median_val)
print(f"TotalCharges 缺失值处理完成，剩余 NaN: {df['TotalCharges'].isnull().sum()}")

print("数据清洗完成。")

# ============================================================
# 2. 描述性统计特征（显式输出，满足项目要求）
# ============================================================
print("\n" + "="*60)
print("2. 描述性统计特征...")
print("="*60)

print("\n【数值型特征描述性统计】")
print(df[['tenure', 'MonthlyCharges', 'TotalCharges']].describe())

print("\n【分类特征分布】")
print(f"\n客户流失分布:\n{df['Churn'].value_counts()}")
print(f"\n合同类型分布:\n{df['Contract'].value_counts()}")
print(f"\n互联网服务分布:\n{df['InternetService'].value_counts()}")
print(f"\n付款方式分布:\n{df['PaymentMethod'].value_counts()}")

print("\n【流失客户 vs 留存客户 特征对比】")
churn_yes = df[df['Churn'] == 'Yes']
churn_no = df[df['Churn'] == 'No']
print(f"流失客户数量: {len(churn_yes)} ({len(churn_yes)/len(df)*100:.1f}%)")
print(f"留存客户数量: {len(churn_no)} ({len(churn_no)/len(df)*100:.1f}%)")
print(f"\n流失客户平均在网时长: {churn_yes['tenure'].mean():.2f}个月")
print(f"留存客户平均在网时长: {churn_no['tenure'].mean():.2f}个月")
print(f"\n流失客户平均月费用: {churn_yes['MonthlyCharges'].mean():.2f}元")
print(f"留存客户平均月费用: {churn_no['MonthlyCharges'].mean():.2f}元")

# ============================================================
# 3. 探索性数据分析 (EDA) & 可视化
# ============================================================
print("\n" + "="*60)
print("3. 探索性数据分析 (EDA) & 可视化...")
print("="*60)

# 3.1 流失率饼图
churn_counts = df['Churn'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(churn_counts, labels=['留存 (No)', '流失 (Yes)'], autopct='%1.1f%%', 
        colors=['#2ecc71', '#e74c3c'], startangle=90, explode=(0, 0.1))
plt.title('图1 客户流失率分布')
plt.savefig('churn_rate_pie.png', dpi=300, bbox_inches='tight')
plt.show()

# 3.2 关键特征与流失关系（合同类型、在网时长、月费用）
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

contract_order = ['Month-to-month', 'One year', 'Two year']
sns.barplot(x='Contract', y='Churn', data=df, order=contract_order, ax=axes[0], palette='viridis')
axes[0].set_title('图2-a 不同合同类型下的流失率')
axes[0].set_ylabel('流失率')

df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 72, 100], 
                            labels=['0-1年', '1-2年', '2-4年', '4-6年', '6年以上'])
sns.barplot(x='tenure_group', y='Churn', data=df, ax=axes[1], palette='magma')
axes[1].set_title('图2-b 不同在网时长下的流失率')
axes[1].set_xlabel('在网时长')

df['charge_group'] = pd.cut(df['MonthlyCharges'], bins=[0, 30, 60, 90, 120], 
                            labels=['低 (0-30)', '中低 (30-60)', '中高 (60-90)', '高 (90-120)'])
sns.barplot(x='charge_group', y='Churn', data=df, ax=axes[2], palette='coolwarm')
axes[2].set_title('图2-c 不同月费用下的流失率')
axes[2].set_xlabel('月费用区间')

plt.tight_layout()
plt.savefig('key_features_churn.png', dpi=300, bbox_inches='tight')
plt.show()

# 3.3 相关性热力图
plt.figure(figsize=(8, 6))
numeric_df = df[['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']]
temp_df = df.copy()
temp_df['Churn_num'] = temp_df['Churn'].map({'Yes': 1, 'No': 0})
corr_data = pd.concat([numeric_df, temp_df['Churn_num']], axis=1)
sns.heatmap(corr_data.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('图3 数值特征相关性热力图')
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# 删除临时列
df.drop(['tenure_group', 'charge_group'], axis=1, inplace=True)

# ============================================================
# 4. 特征工程与预处理（稳健方案：强制转数值 + 填充0）
# ============================================================
print("\n" + "="*60)
print("4. 特征工程与预处理...")
print("="*60)

data = df.copy()

# 4.1 二分类Yes/No转为0/1
binary_cols = ['Churn', 'gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
for col in binary_cols:
    data[col] = data[col].map({'Yes': 1, 'No': 0})

# 4.2 多分类特征One-Hot编码（先填充 NaN 为 'Unknown' 以防编码后出现 NaN）
categorical_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                    'Contract', 'PaymentMethod']

for col in categorical_cols:
    data[col] = data[col].fillna('Unknown')

data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# 4.3 删除customerID
data.drop('customerID', axis=1, inplace=True)

# 4.4 划分特征X和目标y
X = data.drop('Churn', axis=1)
y = data['Churn']

# 【核心修复】强制所有列为数值，无法转换的变为 NaN，然后全部填充 0
X = X.apply(pd.to_numeric, errors='coerce')
nan_count_before = X.isnull().sum().sum()
print(f"转数值前 X 中 NaN 数量: {nan_count_before}")
X = X.fillna(0)  # 所有 NaN 用 0 填充
print(f"填充0后 X 中 NaN 数量: {X.isnull().sum().sum()}")
print(f"特征维度: {X.shape}")

# 4.5 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4.6 SMOTE过采样（现在 X_train 已无 NaN）
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"SMOTE后训练集形状: {X_train_res.shape}")
print(f"正负样本比例: {np.bincount(y_train_res)}")

# 4.7 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

X_train_tree = X_train_res.values
X_test_tree = X_test.values
feature_names = X.columns.tolist()

# ============================================================
# 5. KMeans 聚类分析（客户分群，满足算法多样性要求）
# ============================================================
print("\n" + "="*60)
print("5. KMeans 聚类分析 (客户分群)...")
print("="*60)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_train_scaled)

sil_score = silhouette_score(X_train_scaled, clusters)
print(f"轮廓系数: {sil_score:.4f}")

cluster_counts = np.bincount(clusters)
print(f"聚类分布: 类别0={cluster_counts[0]}, 类别1={cluster_counts[1]}, 类别2={cluster_counts[2]}")

# 可视化聚类结果（PCA降维）
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
plt.title('图4 KMeans客户分群可视化 (PCA降维)')
plt.xlabel('主成分1')
plt.ylabel('主成分2')
plt.colorbar(scatter, label='聚类标签')
plt.savefig('customer_clusters.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n【各聚类中心特征值对比（Top 5特征）】")
cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=feature_names)
for i in range(3):
    top_features = cluster_centers.iloc[i].abs().sort_values(ascending=False).head(5)
    print(f"聚类 {i} 的特征: {top_features.index.tolist()}")

# ============================================================
# 6. 模型训练与评估
# ============================================================
print("\n" + "="*60)
print("6. 训练多种机器学习模型...")
print("="*60)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost': xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, 
                                 random_state=42, eval_metric='logloss', use_label_encoder=False),
    'LightGBM': lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, 
                                   random_state=42, n_jobs=-1),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42)
}

results = []
predictions = {}

for name, model in models.items():
    print(f"\n>>> 正在训练: {name}")
    
    if name in ['Logistic Regression', 'SVM']:
        X_train_use = X_train_scaled
        X_test_use = X_test_scaled
    else:
        X_train_use = X_train_tree
        X_test_use = X_test_tree
    
    model.fit(X_train_use, y_train_res)
    
    y_pred = model.predict(X_test_use)
    y_prob = model.predict_proba(X_test_use)[:, 1] if hasattr(model, 'predict_proba') else None
    
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
    
    predictions[name] = (y_pred, y_prob)
    print(f"  准确率: {acc:.4f}, 召回率: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

results_df = pd.DataFrame(results).round(4)
print("\n" + "-"*50)
print("【模型性能汇总表】")
print(results_df.to_string(index=False))
results_df.to_csv('model_performance.csv', index=False)

best_model = results_df.loc[results_df['F1-Score'].idxmax(), 'Model']
print(f"\n⭐ 表现最好的模型: {best_model} (F1={results_df['F1-Score'].max():.4f})")

# ============================================================
# 7. 结果可视化（模型对比、混淆矩阵、ROC曲线）
# ============================================================
print("\n" + "="*60)
print("7. 生成评估可视化图表...")
print("="*60)

# 7.1 模型性能对比条形图
fig, ax = plt.subplots(figsize=(12, 6))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x = np.arange(len(results_df['Model']))
width = 0.2

for i, metric in enumerate(metrics):
    ax.bar(x + i*width, results_df[metric], width, label=metric)

ax.set_xlabel('模型')
ax.set_ylabel('分数')
ax.set_title('图5 各模型性能对比')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(results_df['Model'], rotation=15)
ax.legend(loc='lower right')
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 7.2 混淆矩阵（最佳模型）
if best_model in ['Logistic Regression', 'SVM']:
    best_clf = models[best_model]
    best_clf.fit(X_train_scaled, y_train_res)
    y_pred_best = best_clf.predict(X_test_scaled)
else:
    best_clf = models[best_model]
    best_clf.fit(X_train_tree, y_train_res)
    y_pred_best = best_clf.predict(X_test_tree)

cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['留存 (No)', '流失 (Yes)'],
            yticklabels=['留存 (No)', '流失 (Yes)'])
plt.title(f'图6 {best_model} 混淆矩阵')
plt.ylabel('实际标签')
plt.xlabel('预测标签')
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# 7.3 ROC曲线
plt.figure(figsize=(8, 6))
for name, (y_pred, y_prob) in predictions.items():
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'{name} (AUC={auc_score:.3f})')

plt.plot([0, 1], [0, 1], 'k--', label='随机猜测')
plt.xlabel('假阳性率 (FPR)')
plt.ylabel('真阳性率 (TPR)')
plt.title('图7 ROC曲线对比')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================
# 8. SHAP可解释性分析
# ============================================================
if SHAP_AVAILABLE:
    print("\n" + "="*60)
    print("8. SHAP 模型可解释性分析 (以XGBoost为例)...")
    print("="*60)
    
    xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=42, 
                                  eval_metric='logloss', use_label_encoder=False)
    xgb_model.fit(X_train_tree, y_train_res)
    
    explainer = shap.TreeExplainer(xgb_model)
    X_test_sample = X_test_tree[:100]
    shap_values = explainer.shap_values(X_test_sample)
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_sample, feature_names=feature_names, 
                      plot_type="bar", show=False)
    plt.title('图8 SHAP 特征重要性 (XGBoost)')
    plt.tight_layout()
    plt.savefig('shap_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_sample, feature_names=feature_names, 
                      show=False)
    plt.title('图9 SHAP 特征影响分布图')
    plt.tight_layout()
    plt.savefig('shap_summary_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("SHAP分析图表已生成！")

# ============================================================
# 9. Pyecharts 交互式组合大屏
# ============================================================
try:
    from pyecharts.charts import Pie, Bar, Page
    from pyecharts import options as opts
    from pyecharts.globals import ThemeType
    
    print("\n" + "="*60)
    print("9. 生成 Pyecharts 交互式大屏 (HTML)...")
    print("="*60)
    
    # 【修复】使用 .iloc 按位置取值，避免 KeyError
    churn_data = [
        ['留存 (No)', int(churn_counts.iloc[0])],
        ['流失 (Yes)', int(churn_counts.iloc[1])]
    ]
    
    pie_chart = (
        Pie(init_opts=opts.InitOpts(theme=ThemeType.VINTAGE))
        .add("", churn_data, radius=["40%", "70%"])
        .set_global_opts(title_opts=opts.TitleOpts(title="客户流失率分布"))
        .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {d}%"))
    )
    
    models_list = results_df['Model'].tolist()
    acc_list = results_df['Accuracy'].tolist()
    f1_list = results_df['F1-Score'].tolist()
    
    bar_chart = (
        Bar(init_opts=opts.InitOpts(theme=ThemeType.VINTAGE))
        .add_xaxis(models_list)
        .add_yaxis("准确率", [round(v*100, 1) for v in acc_list])
        .add_yaxis("F1分数", [round(v*100, 1) for v in f1_list])
        .set_global_opts(
            title_opts=opts.TitleOpts(title="模型性能对比"),
            legend_opts=opts.LegendOpts(pos_top="5%")
        )
    )
    
    page = Page(layout=Page.DraggablePageLayout)
    page.add(pie_chart)
    page.add(bar_chart)
    page.render("churn_dashboard.html")
    print("✅ 交互式大屏已生成: churn_dashboard.html (请用浏览器打开)")
    
except ImportError:
    print("⚠️ pyecharts未安装，跳过交互式大屏生成。")

# ============================================================
# 10. 总结输出
# ============================================================
print("\n" + "="*60)
print("🎉 项目运行完毕！")
print("="*60)
print("\n【生成的文件列表】")
files = [
    "churn_rate_pie.png - 图1 流失率饼图",
    "key_features_churn.png - 图2 关键特征与流失关系",
    "correlation_heatmap.png - 图3 相关性热力图",
    "customer_clusters.png - 图4 KMeans客户分群",
    "model_comparison.png - 图5 模型性能对比",
    "confusion_matrix.png - 图6 最佳模型混淆矩阵",
    "roc_curves.png - 图7 ROC曲线对比",
]
if SHAP_AVAILABLE:
    files.append("shap_feature_importance.png - 图8 SHAP特征重要性")
    files.append("shap_summary_plot.png - 图9 SHAP特征分布")
files.append("churn_dashboard.html - 交互式大屏")
files.append("model_performance.csv - 模型性能数据")

for f in files:
    print(f"  ✅ {f}")

print("\n【关键结论】")
print(f"  - 流失率: {len(churn_yes)/len(df)*100:.1f}%")
print(f"  - 最佳模型: {best_model} (F1={results_df['F1-Score'].max():.4f})")
if 'sil_score' in locals():
    print(f"  - KMeans轮廓系数: {sil_score:.4f}")
print("="*60)
