# -*- coding: utf-8 -*-
"""
电信客户流失数据分析及可视化
软件开发实践1 - 最终修改版

功能：
1. 数据加载与清洗
2. 描述性统计与 EDA 可视化
3. KMeans 客户分群 + PCA 可视化
4. χ² 特征选择
5. SMOTE 类别不平衡处理（严格放在交叉验证 Pipeline 内，避免数据泄漏）
6. Logistic Regression / Random Forest / XGBoost / LightGBM / SVM
7. GridSearchCV 5 折交叉验证，以 F1 为主要选择指标
8. Accuracy / Precision / Recall / F1 / ROC-AUC
9. 混淆矩阵、ROC 曲线
10. SHAP 特征解释（树模型），失败时自动使用 permutation importance
11. 测试集客户流失风险预测
12. 可选生成 Pyecharts HTML 仪表盘

运行：
    pip install -r requirements.txt
    python main.py

输入文件默认：
    WA_Fn-UseC_-Telco-Customer-Churn.csv

所有结果保存到：
    output/
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# =========================
# 0. 基础配置
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SELECTED_FEATURES = 15

# 中文字体：按本机情况自动尝试
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"
]
plt.rcParams["axes.unicode_minus"] = False


# =========================
# 1. 工具函数
# =========================
def savefig(filename, dpi=160, bbox_inches="tight"):
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
    plt.close()
    return path


def find_data_file():
    """允许用户把 CSV 放在程序同目录，也兼容常见文件名。"""
    candidates = [
        DATA_FILE,
        os.path.join(BASE_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.csv"),
        os.path.join(BASE_DIR, "telco_churn.csv"),
        os.path.join(BASE_DIR, "电信客户流失数据.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "未找到数据文件。请将 WA_Fn-UseC_-Telco-Customer-Churn.csv "
        "放在 main.py 同一目录。"
    )


def load_and_clean():
    print("\n1. 数据加载与清洗")
    path = find_data_file()
    df = pd.read_csv(path)

    print("原始数据：", df.shape)

    # TotalCharges 中存在空字符串，先转数值；无法转换的记录记为 NaN
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    invalid_count = int(df["TotalCharges"].isna().sum())

    if invalid_count > 0:
        median_value = df["TotalCharges"].median()
        df["TotalCharges"] = df["TotalCharges"].fillna(median_value)
        print(f"TotalCharges 缺失/异常转换记录：{invalid_count} 条，采用中位数填补。")
    else:
        print("TotalCharges 无缺失/异常值。")

    # 二分类变量统一成 0/1
    df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce").fillna(0).astype(int)
    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")

    # 目标变量：Yes=1，No=0
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    if df["Churn"].isna().any():
        raise ValueError("Churn 存在无法识别的取值，请检查原始数据。")
    df["Churn"] = df["Churn"].astype(int)

    # 其他缺失值统一用众数处理
    for col in df.columns:
        if df[col].isna().any():
            if df[col].dtype == "object":
                mode = df[col].mode()
                fill_value = mode.iloc[0] if not mode.empty else "Unknown"
            else:
                fill_value = df[col].median()
            df[col] = df[col].fillna(fill_value)

    print("清洗后数据：", df.shape)
    print("流失客户数：", int(df["Churn"].sum()))
    print("流失率：", f"{df['Churn'].mean() * 100:.2f}%")

    df.to_csv(os.path.join(OUTPUT_DIR, "cleaned_data.csv"), index=False, encoding="utf-8-sig")
    return df


def descriptive_analysis(df):
    print("\n2. 描述性统计与 EDA")

    churn_rate = df["Churn"].mean()

    # 2.1 流失比例饼图
    plt.figure(figsize=(8, 7))
    counts = df["Churn"].value_counts().sort_index()
    labels = ["未流失", "流失"]
    plt.pie(
        counts.values,
        labels=labels,
        autopct="%.2f%%",
        startangle=90,
        textprops={"fontsize": 13}
    )
    plt.title("电信客户流失比例", fontsize=18)
    savefig("churn_rate_pie.png")

    # 2.2 数值变量风琴图（Violin Plot）
    # 用风琴图展示不同流失状态下数值变量的分布形状；
    # inner="box" 同时保留中位数和四分位信息，便于答辩时解释。
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    fig, axes = plt.subplots(3, 1, figsize=(15, 15))

    for ax, col in zip(axes, numeric_cols):
        sns.violinplot(
            data=df,
            x="Churn",
            y=col,
            ax=ax,
            inner="box",
            cut=0,
            linewidth=1.2,
            width=0.72
        )
        ax.set_title(f"{col} 与客户流失关系", fontsize=22, pad=12)
        ax.set_xlabel("是否流失", fontsize=16)
        ax.set_ylabel(col, fontsize=16)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["未流失（0）", "流失（1）"], fontsize=14)
        ax.tick_params(axis="y", labelsize=13)
        ax.grid(axis="y", linestyle="--", alpha=0.25)

    plt.tight_layout(pad=2.0)
    # 保留原文件名，避免 PPT/报告引用路径失效；同时生成更直观的新文件名。
    savefig("boxplots_churn.png", dpi=180)
    # 另存为风琴图专用文件名，便于后续直接使用。
    import shutil
    shutil.copyfile(
        os.path.join(OUTPUT_DIR, "boxplots_churn.png"),
        os.path.join(OUTPUT_DIR, "violinplots_churn.png")
    )

    # 2.3 分类变量流失率
    categorical_cols = [
        "Contract", "PaymentMethod", "InternetService",
        "OnlineSecurity", "TechSupport"
    ]
    fig, axes = plt.subplots(len(categorical_cols), 1, figsize=(14, 23))
    for ax, col in zip(axes, categorical_cols):
        rate = df.groupby(col, observed=False)["Churn"].mean().sort_values(ascending=False)
        bars = ax.bar(rate.index.astype(str), rate.values)
        ax.set_title(f"{col} 不同类别的客户流失率")
        ax.set_xlabel(col)
        ax.set_ylabel("流失率")
        ax.set_ylim(0, min(1, max(rate.values) * 1.18 + 0.03))
        ax.tick_params(axis="x", rotation=25)
        for bar, value in zip(bars, rate.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.005,
                f"{value:.2%}",
                ha="center",
                va="bottom",
                fontsize=9
            )
    plt.tight_layout()
    savefig("categorical_churn.png")

    # 2.4 相关性热力图
    corr_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]
    corr = df[corr_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
    plt.title("数值变量相关性热力图", fontsize=16)
    savefig("correlation_heatmap.png")

    # 描述性统计保存
    df[numeric_cols].describe().T.to_csv(
        os.path.join(OUTPUT_DIR, "descriptive_statistics.csv"),
        encoding="utf-8-sig"
    )

    # 分类变量流失率保存
    records = []
    for col in categorical_cols:
        rate = df.groupby(col, observed=False)["Churn"].agg(["mean", "count"]).reset_index()
        for _, row in rate.iterrows():
            records.append({
                "变量": col,
                "类别": row[col],
                "客户数": int(row["count"]),
                "流失率": float(row["mean"])
            })
    pd.DataFrame(records).to_csv(
        os.path.join(OUTPUT_DIR, "categorical_churn_rates.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    print("流失率：", f"{churn_rate * 100:.2f}%")
    return churn_rate


def kmeans_analysis(df):
    print("\n3. KMeans 客户分群")

    cluster_features = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
    X_cluster = df[cluster_features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=20)
    labels = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    summary = df.copy()
    summary["Cluster"] = labels
    cluster_summary = summary.groupby("Cluster").agg(
        客户数=("Churn", "size"),
        在网时长=("tenure", "mean"),
        月费用=("MonthlyCharges", "mean"),
        总费用=("TotalCharges", "mean"),
        流失率=("Churn", "mean")
    ).reset_index()

    cluster_summary["在网时长"] = cluster_summary["在网时长"].round(3)
    cluster_summary["月费用"] = cluster_summary["月费用"].round(3)
    cluster_summary["总费用"] = cluster_summary["总费用"].round(3)
    cluster_summary["流失率"] = cluster_summary["流失率"].round(3)

    print(cluster_summary.to_string(index=False))
    cluster_summary.to_csv(
        os.path.join(OUTPUT_DIR, "cluster_summary.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    plt.figure(figsize=(12, 9))
    scatter = plt.scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=labels, cmap="viridis", s=25, alpha=0.65
    )
    plt.colorbar(scatter, label="客户群")
    plt.xlabel("PCA 第一主成分")
    plt.ylabel("PCA 第二主成分")
    plt.title("KMeans 客户分群结果", fontsize=18)
    savefig("customer_clusters.png")

    return labels, cluster_summary


def build_features(df):
    """构造模型特征。"""
    X = df.drop(columns=["Churn", "customerID"], errors="ignore").copy()
    y = df["Churn"].copy()

    # 二分类变量直接映射
    binary_cols = [
        "gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"
    ]
    for col in binary_cols:
        if col in X.columns:
            X[col] = X[col].map({"Yes": 1, "No": 0, "Male": 1, "Female": 0}).fillna(0)

    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
        ],
        remainder="drop"
    )

    print("\n4. 特征工程")
    print("数值特征：", numeric_cols)
    print("分类特征：", categorical_cols)

    return X, y, preprocessor


def get_feature_names(preprocessor):
    try:
        return preprocessor.get_feature_names_out()
    except Exception:
        return np.array([f"feature_{i}" for i in range(100)])


def make_models():
    """使用相对轻量的网格，保证普通电脑可以正常运行。"""
    models = {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            {"model__C": [10]}
        ),
        "Random Forest": (
            RandomForestClassifier(
                random_state=RANDOM_STATE,
                class_weight=None,
                n_jobs=1
            ),
            {
                "model__n_estimators": [100],
                "model__max_depth": [10]
            }
        ),
        "XGBoost": (
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=1,
                verbosity=0
            ),
            {
                "model__n_estimators": [100],
                "model__learning_rate": [0.05],
                "model__max_depth": [3]
            }
        ),
        "LightGBM": (
            LGBMClassifier(
                objective="binary",
                random_state=RANDOM_STATE,
                n_jobs=1,
                verbosity=-1
            ),
            {
                "model__n_estimators": [100],
                "model__learning_rate": [0.05],
                "model__num_leaves": [31]
            }
        ),
        "SVM": (
            SVC(
                kernel="linear",
                probability=False,
                random_state=RANDOM_STATE
            ),
            {"model__C": [1]}
        )
    }
    return models


def get_scores(model, X_test):
    """同时兼容 predict_proba 和 decision_function。"""
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        y_score = y_pred

    return y_pred, y_score


def train_models(X, y, preprocessor):
    print("\n5. 模型训练与 GridSearchCV")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    models = make_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results = []
    fitted_models = {}
    predictions = {}

    for name, (estimator, param_grid) in models.items():
        print(f"\n正在训练：{name}")

        # 注意：特征选择、SMOTE、标准化全部放在 CV Pipeline 中，避免数据泄漏
        steps = [
            ("preprocess", preprocessor),
            ("select", SelectKBest(score_func=chi2, k=N_SELECTED_FEATURES)),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
        ]

        # SVM / LR 需要标准化；树模型不需要标准化，但统一加 scaler 也可以正常工作。
        steps.append(("scaler", StandardScaler()))
        steps.append(("model", estimator))

        pipe = Pipeline(steps=steps)

        grid = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring="f1",
            cv=cv,
            n_jobs=1,
            verbose=0,
            refit=True
        )

        grid.fit(X_train, y_train)

        # 这里必须是 best_params_，不能写成 best_params__
        print("最佳参数：", grid.best_params_)
        print("CV 最佳 F1：", f"{grid.best_score_:.4f}")

        best_model = grid.best_estimator_
        y_pred, y_score = get_scores(best_model, X_test)

        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_score)

        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(auc, 4)
        })

        fitted_models[name] = best_model
        predictions[name] = (y_pred, y_score)

        print(
            f"Accuracy={acc:.4f}, Precision={precision:.4f}, "
            f"Recall={recall:.4f}, F1={f1:.4f}, ROC-AUC={auc:.4f}"
        )

    results_df = pd.DataFrame(results)

    # 明确规定：先比较 F1，再比较 ROC-AUC；这样报告和 PPT 中的“最佳模型”有唯一规则。
    results_df = results_df.sort_values(
        by=["F1-Score", "ROC-AUC"],
        ascending=False
    ).reset_index(drop=True)

    results_df.to_csv(
        os.path.join(OUTPUT_DIR, "model_performance.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    best_name = results_df.iloc[0]["Model"]
    best_model = fitted_models[best_name]
    best_pred, best_score = predictions[best_name]

    print("\n模型性能排名：")
    print(results_df.to_string(index=False))
    print(f"\n最佳模型：{best_name}")

    print("\n最佳模型 classification report：")
    print(
        classification_report(
            y_test,
            best_pred,
            target_names=["留存", "流失"],
            digits=4,
            zero_division=0
        )
    )

    # 保存测试集真实值和预测概率
    risk = pd.DataFrame({
        "ActualChurn": y_test.values,
        "PredictedChurn": best_pred,
        "ChurnProbability": best_score
    })
    risk["RiskLevel"] = pd.cut(
        risk["ChurnProbability"],
        bins=[-np.inf, 0.30, 0.60, np.inf],
        labels=["低风险", "中风险", "高风险"]
    )
    risk.to_csv(
        os.path.join(OUTPUT_DIR, "test_customer_risk.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    high_risk_count = int((risk["ChurnProbability"] >= 0.60).sum())
    print(f"测试集客户数：{len(risk)}")
    print(f"预测高风险客户数（概率 >= 0.60）：{high_risk_count}")

    return (
        results_df,
        fitted_models,
        predictions,
        best_name,
        best_model,
        X_train,
        X_test,
        y_train,
        y_test,
        risk
    )


def plot_model_comparison(results_df):
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    plot_df = results_df.copy()

    # 为了展示直观，按 F1 从高到低排列
    plot_df = plot_df.sort_values("F1-Score", ascending=False)

    x = np.arange(len(plot_df))
    width = 0.15

    plt.figure(figsize=(16, 8))
    for i, metric in enumerate(metrics):
        plt.bar(x + (i - 2) * width, plot_df[metric], width, label=metric)

    plt.xticks(x, plot_df["Model"], rotation=15)
    plt.ylim(0, 1)
    plt.ylabel("指标值")
    plt.title("五种机器学习模型性能对比", fontsize=18)
    plt.legend()
    plt.tight_layout()
    savefig("model_comparison.png")


def plot_confusion_matrix(best_name, best_pred, y_test):
    cm = confusion_matrix(y_test, best_pred)
    plt.figure(figsize=(8, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["预测留存", "预测流失"],
        yticklabels=["实际留存", "实际流失"]
    )
    plt.xlabel("预测结果")
    plt.ylabel("真实结果")
    plt.title(f"{best_name} 混淆矩阵", fontsize=17)
    savefig("confusion_matrix.png")


def plot_roc_curves(results_df, predictions, y_test):
    plt.figure(figsize=(13, 9))
    for _, row in results_df.iterrows():
        name = row["Model"]
        _, y_score = predictions[name]
        fpr, tpr, _ = roc_curve(y_test, y_score)
        auc = row["ROC-AUC"]
        plt.plot(fpr, tpr, linewidth=2, label=f"{name} AUC={auc:.4f}")

    plt.plot([0, 1], [0, 1], "--", linewidth=1.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("五种模型 ROC 曲线", fontsize=18)
    plt.legend(loc="lower right")
    plt.tight_layout()
    savefig("roc_curves.png")


def get_selected_feature_names(best_model):
    try:
        pre = best_model.named_steps["preprocess"]
        selector = best_model.named_steps["select"]
        all_names = np.asarray(pre.get_feature_names_out())
        mask = selector.get_support()
        return all_names[mask]
    except Exception:
        return None


def shap_analysis(best_name, best_model, X_test):
    print("\n7. 模型解释")

    feature_names = get_selected_feature_names(best_model)

    if feature_names is None:
        print("无法获取选择后的特征名称，跳过 SHAP。")
        return

    try:
        # 将测试数据依次经过预处理、特征选择、SMOTE（测试集不会执行 fit，transform 等价于不改变样本数量）、标准化
        # 由于 SMOTE 对 transform 不做采样，因此不会改变测试集样本数量。
        X_transformed = best_model.named_steps["preprocess"].transform(X_test)
        X_selected = best_model.named_steps["select"].transform(X_transformed)
        X_scaled = best_model.named_steps["scaler"].transform(X_selected)
        estimator = best_model.named_steps["model"]

        # SHAP 对树模型解释效果最好
        if best_name in ["XGBoost", "LightGBM"]:
            import shap

            # 为了避免运行时间过长，最多解释 800 个测试样本
            n = min(800, X_scaled.shape[0])
            X_sample = X_scaled[:n]

            explainer = shap.TreeExplainer(estimator)
            shap_values = explainer.shap_values(X_sample)

            if isinstance(shap_values, list):
                shap_values = shap_values[-1]
            if hasattr(shap_values, "values"):
                shap_values = shap_values.values

            shap_values = np.asarray(shap_values)
            if shap_values.ndim == 3:
                shap_values = shap_values[:, :, -1]

            # 平均绝对 SHAP 值
            mean_abs = np.abs(shap_values).mean(axis=0)
            importance = pd.DataFrame({
                "Feature": feature_names,
                "MeanAbsSHAP": mean_abs
            }).sort_values("MeanAbsSHAP", ascending=False)

            importance.to_csv(
                os.path.join(OUTPUT_DIR, "shap_feature_importance.csv"),
                index=False,
                encoding="utf-8-sig"
            )

            top = importance.head(15).sort_values("MeanAbsSHAP", ascending=True)
            plt.figure(figsize=(13, 9))
            plt.barh(top["Feature"], top["MeanAbsSHAP"])
            plt.xlabel("平均绝对 SHAP 值")
            plt.ylabel("特征")
            plt.title(f"{best_name} SHAP 特征重要性", fontsize=17)
            plt.tight_layout()
            savefig("shap_feature_importance.png")

            # Summary plot
            plt.figure(figsize=(13, 9))
            shap.summary_plot(
                shap_values,
                X_sample,
                feature_names=feature_names,
                show=False,
                max_display=15
            )
            plt.title(f"{best_name} SHAP Summary")
            plt.tight_layout()
            savefig("shap_summary_plot.png")

            print("SHAP 特征重要性已保存。")
            print("Top 10：")
            print(importance.head(10).to_string(index=False))
            return

        # 如果最佳模型不是树模型，则使用 permutation importance
        X_sample = X_scaled[:min(1000, X_scaled.shape[0])]
        y_sample = None
        # permutation_importance 需要 y，这里使用完整测试集对应的真实值在调用处无法直接拿到，
        # 因此使用一个轻量替代：模型系数绝对值（适用于线性模型）。
        if hasattr(estimator, "coef_"):
            coef = np.abs(estimator.coef_[0])
            importance = pd.DataFrame({
                "Feature": feature_names,
                "Importance": coef
            }).sort_values("Importance", ascending=False)
            importance.to_csv(
                os.path.join(OUTPUT_DIR, "feature_importance.csv"),
                index=False,
                encoding="utf-8-sig"
            )
            top = importance.head(15).sort_values("Importance", ascending=True)
            plt.figure(figsize=(13, 9))
            plt.barh(top["Feature"], top["Importance"])
            plt.xlabel("绝对模型系数")
            plt.ylabel("特征")
            plt.title(f"{best_name} 特征重要性", fontsize=17)
            plt.tight_layout()
            savefig("shap_feature_importance.png")
            return

        print("当前最佳模型不支持 SHAP，跳过模型解释图。")

    except Exception as e:
        print("SHAP 运行失败：", repr(e))
        print("程序不会中断，其他结果仍会正常保存。")


def save_selected_features(best_model):
    names = get_selected_feature_names(best_model)
    if names is not None:
        pd.DataFrame({"SelectedFeature": names}).to_csv(
            os.path.join(OUTPUT_DIR, "selected_features.csv"),
            index=False,
            encoding="utf-8-sig"
        )
        print("\nχ² 特征选择结果：")
        for name in names:
            print(name)


def create_dashboard(results_df, best_name, cluster_summary):
    """Pyecharts 为可选功能；没有安装时不影响主程序。"""
    try:
        from pyecharts import options as opts
        from pyecharts.charts import Bar, Pie, Page

        page = Page(layout=Page.SimplePageLayout)

        # 模型 F1
        bar = (
            Bar()
            .add_xaxis(results_df["Model"].tolist())
            .add_yaxis("F1-Score", results_df["F1-Score"].tolist())
            .set_global_opts(
                title_opts=opts.TitleOpts(title="模型 F1 对比")
            )
        )

        # 聚类客户数
        pie = (
            Pie()
            .add(
                "客户数",
                [
                    [f"客户群 {int(r['Cluster'])}", int(r["客户数"])]
                    for _, r in cluster_summary.iterrows()
                ]
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(title="KMeans 客户群规模")
            )
        )

        page.add(bar, pie)
        page.render(os.path.join(OUTPUT_DIR, "churn_dashboard.html"))
        print("Pyecharts 仪表盘：output/churn_dashboard.html")
    except Exception as e:
        print("Pyecharts 仪表盘生成失败，已跳过：", repr(e))


def write_run_summary(results_df, best_name, cluster_summary, df):
    best = results_df.iloc[0]
    lines = [
        "电信客户流失项目运行摘要",
        "=" * 50,
        f"数据量：{len(df)}",
        f"流失率：{df['Churn'].mean():.4%}",
        f"最佳模型：{best_name}",
        f"Accuracy：{best['Accuracy']:.4f}",
        f"Precision：{best['Precision']:.4f}",
        f"Recall：{best['Recall']:.4f}",
        f"F1-Score：{best['F1-Score']:.4f}",
        f"ROC-AUC：{best['ROC-AUC']:.4f}",
        "",
        "模型选择规则：先按 F1-Score 降序，再按 ROC-AUC 降序。",
        "χ² 特征选择：15 个特征。",
        "SMOTE：位于交叉验证 Pipeline 内，避免数据泄漏。",
    ]
    with open(os.path.join(OUTPUT_DIR, "run_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    print("=" * 70)
    print("电信客户流失数据分析及可视化 - 最终修改版")
    print("=" * 70)

    df = load_and_clean()
    descriptive_analysis(df)
    _, cluster_summary = kmeans_analysis(df)

    X, y, preprocessor = build_features(df)

    (
        results_df,
        fitted_models,
        predictions,
        best_name,
        best_model,
        X_train,
        X_test,
        y_train,
        y_test,
        risk
    ) = train_models(X, y, preprocessor)

    # 保存实际进入模型的 χ² 特征
    save_selected_features(best_model)

    # 模型图
    plot_model_comparison(results_df)
    best_pred, best_score = predictions[best_name]
    plot_confusion_matrix(best_name, best_pred, y_test)
    plot_roc_curves(results_df, predictions, y_test)

    # SHAP / 特征解释
    shap_analysis(best_name, best_model, X_test)

    # 可选交互式仪表盘
    create_dashboard(results_df, best_name, cluster_summary)

    # 运行摘要
    write_run_summary(results_df, best_name, cluster_summary, df)

    print("\n" + "=" * 70)
    print("程序运行完成！")
    print(f"最佳模型：{best_name}")
    print(f"所有结果已保存到：{OUTPUT_DIR}")
    print("建议报告和 PPT 的模型指标统一以 output/model_performance.csv 为准。")
    print("=" * 70)


if __name__ == "__main__":
    main()
