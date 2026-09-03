# 电信客户流失数据分析及可视化

## 项目简介

本项目针对 IBM Telco Customer Churn 数据集，完成从数据清洗、探索性分析、特征工程、机器学习建模到结果可视化的完整流程。项目综合运用逻辑回归、随机森林、XGBoost、LightGBM 和 SVM 五种模型进行客户流失预测，并通过 SHAP 进行模型可解释性分析，识别客户流失的关键驱动因素。

## 数据集说明

- **数据来源**：Kaggle (blastchar/telco-customer-churn)
- **样本数量**：7,043 条客户记录
- **特征数量**：21 个特征（含目标变量 `Churn`）
- **文件要求**：将数据集文件 `WA_Fn-UseC_-Telco-Customer-Churn.csv` 放置于项目根目录下

> 下载地址：https://www.kaggle.com/datasets/blastchar/telco-customer-churn

## 环境要求

- Python 3.9 或更高版本
- 操作系统：Windows / macOS / Linux

## 安装与配置

### 1. 克隆或下载项目

```bash
git clone <repository-url>
cd 电信客户流失数据分析及可视化
2. 安装依赖库
使用 pip 安装所需依赖（推荐使用清华镜像源加速）：

bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost lightgbm shap pyecharts jupyter -i https://pypi.tuna.tsinghua.edu.cn/simple
3. 准备数据集
将 WA_Fn-UseC_-Telco-Customer-Churn.csv 文件放入项目根目录。

运行项目
在项目根目录下执行：

bash
python main.py
或指定 Python 解释器路径（如使用非系统默认 Python）：

bash
E:\python\python.exe main.py
项目输出
运行完成后，将在项目根目录生成以下文件：

文件名	说明
churn_rate_pie.png	图1：客户流失率饼图
key_features_churn.png	图2：关键特征与流失关系图
correlation_heatmap.png	图3：数值特征相关性热力图
customer_clusters.png	图4：KMeans 客户分群可视化
model_comparison.png	图5：各模型性能对比图
confusion_matrix.png	图6：最佳模型混淆矩阵
roc_curves.png	图7：ROC 曲线对比图
shap_feature_importance.png	图8：SHAP 特征重要性（需安装 shap）
shap_summary_plot.png	图9：SHAP 特征影响分布图（需安装 shap）
churn_dashboard.html	交互式可视化大屏（需安装 pyecharts）
model_performance.csv	模型性能评估数据表
项目结构
text
电信客户流失数据分析及可视化/
├── main.py                              # 主程序
├── README.md                            # 项目说明文档
├── WA_Fn-UseC_-Telco-Customer-Churn.csv # 数据集（需自行下载）
├── churn_rate_pie.png                   # 输出图片
├── key_features_churn.png               # 输出图片
├── correlation_heatmap.png              # 输出图片
├── customer_clusters.png                # 输出图片
├── model_comparison.png                 # 输出图片
├── confusion_matrix.png                 # 输出图片
├── roc_curves.png                       # 输出图片
├── shap_feature_importance.png          # 输出图片（需 shap）
├── shap_summary_plot.png                # 输出图片（需 shap）
├── churn_dashboard.html                 # 交互式大屏（需 pyecharts）
└── model_performance.csv                # 模型性能数据
功能模块
数据预处理：缺失值处理、数据类型转换、特征编码

探索性数据分析（EDA） ：流失率统计、特征分布分析、相关性分析

特征工程：SMOTE 过采样处理类别不平衡、数据标准化

机器学习建模：逻辑回归、随机森林、XGBoost、LightGBM、SVM

模型可解释性：SHAP 值分析

可视化展示：Matplotlib/Seaborn 静态图表 + Pyecharts 交互式大屏

常见问题
1. 提示 ModuleNotFoundError: No module named 'xxx'
请确保已安装所有依赖库，可重新执行安装命令：

bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
（如没有 requirements.txt，请参考安装与配置章节逐个安装）

2. 提示 FileNotFoundError: WA_Fn-UseC_-Telco-Customer-Churn.csv
请将数据集文件放置于项目根目录，确保文件名与代码中一致。

3. shap 库安装失败
可跳过 SHAP 分析，不影响其他功能运行。如需要 SHAP 图表，可尝试安装较低版本：

bash
pip install shap==0.42.1
4. pyecharts 生成的 HTML 大屏无法显示
使用 Chrome 或 Edge 浏览器打开 churn_dashboard.html 文件即可。

技术栈
数据处理：Pandas, NumPy

数据可视化：Matplotlib, Seaborn, Pyecharts

机器学习：Scikit-learn, XGBoost, LightGBM

类别不平衡处理：imbalanced-learn (SMOTE)

模型可解释性：SHAP

参考资料
IBM Telco Customer Churn Dataset: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

SHAP 文档: https://shap.readthedocs.io/

XGBoost 文档: https://xgboost.readthedocs.io/

LightGBM 文档: https://lightgbm.readthedocs.io/


版权声明
本项目仅供学习交流使用，数据集版权归原始作者所有。

text

---

### 使用说明

1. 将上述内容保存为 `README.md` 文件，放在你的项目根目录 `e:\电信客户流失数据分析及可视化\` 下。
2. 如果需要 `requirements.txt`（便于一键安装依赖），可额外创建并写入以下内容：
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
imbalanced-learn>=0.9.0
xgboost>=1.5.0
lightgbm>=3.3.0
shap>=0.40.0
pyecharts>=2.0.0
jupyter>=1.0.0

text

然后用户就可以用 `pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple` 一键安装所有依赖。