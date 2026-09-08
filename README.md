# 电信客户流失数据分析及可视化（优化版）

## 一、项目目标
按照《软件开发实践1》项目要求，完成电信客户数据的**数据清洗、描述性统计、可视化、数据挖掘/机器学习、结果评估与系统展示**，形成完整的数据分析闭环。

## 二、环境
- Python 3.9+
- VS Code / PyCharm / Jupyter Notebook 均可
- 依赖：见 `requirements.txt`

安装：
```bash
pip install -r requirements.txt
```

## 三、运行
将 `WA_Fn-UseC_-Telco-Customer-Churn.csv` 与 `main.py` 放在同一目录：
```bash
python main.py
```

## 四、核心功能
1. **数据预处理**：TotalCharges 类型转换、异常/缺失处理、重复值处理、分类变量编码。
2. **EDA**：流失率、连续变量分布、合同/付款/网络类型流失率、相关性热力图。
3. **客户分群**：KMeans + PCA，对客户进行三类画像划分。
4. **特征工程**：One-Hot Encoding + 卡方检验 Top-K 特征选择。
5. **类别不平衡**：SMOTE 仅在训练集及交叉验证内部执行，避免数据泄漏。
6. **模型预测**：逻辑回归、随机森林、XGBoost、LightGBM、SVM。
7. **超参数调优**：GridSearchCV + 5 折交叉验证，以 F1 为优化目标。
8. **模型评估**：Accuracy、Precision、Recall、F1、ROC-AUC、混淆矩阵、ROC 曲线。
9. **模型解释**：树模型使用 SHAP；非树模型使用置换特征重要性。
10. **风险识别**：输出测试集客户流失概率及低/中/高风险等级。
11. **交互展示**：Pyecharts 生成 `output/churn_dashboard.html`。

## 五、输出文件
| 文件 | 用途 |
|---|---|
| churn_rate_pie.png | 客户流失率 |
| boxplots_churn.png | 流失/留存连续特征对比 |
| categorical_churn.png | 分类变量流失率 |
| correlation_heatmap.png | 相关性分析 |
| customer_clusters.png | KMeans + PCA 分群 |
| cluster_summary.csv | 各客户群统计 |
| model_performance.csv | 五种模型性能 |
| model_comparison.png | 模型指标对比 |
| confusion_matrix.png | 最佳模型混淆矩阵 |
| roc_curves.png | ROC 对比 |
| shap_feature_importance.png | 特征重要性 |
| shap_summary_plot.png | SHAP 影响分布（树模型最佳时） |
| test_customer_risk.csv | 测试集风险概率 |
| churn_dashboard.html | Pyecharts 交互大屏 |

## 六、方法学优化说明
原项目将 SMOTE 后的数据直接送入 GridSearchCV，容易造成交叉验证折之间的信息泄漏。优化版将 `SelectKBest → SMOTE → StandardScaler（如需要）→ Model` 放入 Pipeline，使每个 CV fold 独立完成特征选择、过采样和标准化，再训练模型。

此外，项目报告中的“最佳模型”和具体指标必须以本脚本**实际运行生成的 `model_performance.csv` 为准**，不要手工填写旧结果。这样可以避免 PPT、期末报告和代码之间出现指标不一致。

## 七、提交前检查
- [ ] 报告中的 Python 版本、IDE、库版本与实际环境一致
- [ ] 报告中的缺失值处理与代码一致
- [ ] 报告中的特征工程与代码一致
- [ ] 报告中的模型指标来自最新 `model_performance.csv`
- [ ] 图序、图名、表序、表名连续且正文有引用
- [ ] 根目录包含 README.md
- [ ] ZIP 小于 30 MB
- [ ] 删除不必要的缓存、临时文件
