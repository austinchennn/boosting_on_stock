# 1 组执行手册

本文档说明项目结构、数据准备方式以及各任务的运行方法。

---

## 1. 项目结构

```
├── backend/
│   ├── baseline_regression_model/   # 任务一：Baseline 回归模型
│   ├── situation_awareness_classification_model/  # 任务二：态势识别分类模型
│   └── system/                      # 任务三：模型系统编排（SystemController 等）
├── frontend/
│   └── app.py                       # Streamlit 前端（任务三/五入口）
├── final_dataset_with_time_features/  # 数据集目录（需自行准备）
├── outputs/                         # 模型、预测结果、配置等输出
└── requirements.txt
```

> **注意**：所有导入使用绝对包路径（如 `backend.system...`），运行时须确保项目根目录在 `PYTHONPATH` 中。

---

## 2. 数据集准备

本项目 **不包含数据集文件**，请自行获取并放置到正确位置。

### 2.1 文件要求

在项目根目录下创建 `final_dataset_with_time_features/` 文件夹，并将数据文件放入其中：

```
final_dataset_with_time_features/
└── final_dataset_with_time_features.parquet
```

- **格式**：Apache Parquet（`.parquet`）
- **文件名**：必须为 `final_dataset_with_time_features.parquet`

### 2.2 数据结构

每一行代表某只股票在某个交易日的记录，即 `(date, stock, features)` 的格式。每个交易日为一个横截面，每行表示该交易日中某只股票及其全部因子特征。

### 2.3 必需列

| 列名 | 说明 | 用途 |
|---|---|---|
| `date` | 交易日期（可被 `pd.to_datetime` 解析） | 索引 |
| `stock` 或 `ticker` | 股票代码（若列名为 `ticker`，系统自动重命名为 `stock`） | 索引 |
| `y_ret_5` | 未来 5 日收益率 | 任务一回归目标 |
| `y_ret_10` | 未来 10 日收益率 | 任务一 |
| `y_ret_20` | 未来 20 日收益率 | 任务一 |
| `r_future_5` | 未来 5 日收益率 | 任务二分类标签构建 |
| `r_past_10` | 过去 10 日收益率 | 任务二分类标签构建 |

### 2.4 特征列

数据集还应包含以下 **数值型因子/特征列**（系统会自动识别所有数值列作为模型输入特征）：

**核心特征（任务一二必须包含）：**
- `ret_5d`、`ret_10d` — 短期与中期收益
- `momentum_change`（= `ret_5d` - `ret_10d`）— 趋势变化
- `bias_60`、`roc_20` — 基础因子
- `r_past_10` — 历史收益

**可扩展特征（第二阶段可加入）：**
- `ret_1d`、`ret_3d`
- PCA 因子 `F1`–`F5`
- 行业与板块因子
- 滞后特征（`close_lag1`、`close_lag3`、`close_lag5`）

### 2.5 时间范围

| 集合 | 时间 |
|---|---|
| 训练集 | 2016–2022 |
| 测试集 | 2023–2026 |

数据必须按时间顺序划分，并按 `(date, stock)` 组织。

---

## 3. 安装依赖

```bash
pip install -r requirements.txt
```

---

## 4. 运行方式

### 4.1 运行前端（推荐，任务三/五入口）

在 **项目根目录** 执行：

```bash
streamlit run frontend/app.py
```

前端功能对应操作指南中的任务三和任务五：

| 功能 | 说明 |
|---|---|
| **模型训练** | 选择模型类型（LightGBM / XGBoost）、任务类型（回归 / 分类）、标签类型（收益 / 态势），一键训练 |
| **预测与排序** | 输入指定日期，输出该日所有股票预测结果，支持 Top 5% / 10% 筛选与排序 |
| **可视化** | Rank IC 时间序列曲线、累计收益曲线、饼图等 |
| **结果保存** | 下载模型文件、预测结果（date, stock, score）、实验配置（特征、参数、时间区间） |

### 4.2 运行后端 CLI（可选）

```bash
export PYTHONPATH=$PYTHONPATH:.
python backend/system/main_task3.py
```

---

## 5. 任务说明

### 任务一：Baseline 回归模型（必须完成）

- **模型**：LightGBM（回归），可选 XGBoost 对比
- **预测目标**：未来收益 `y_ret_5`（连续数值）
- **特征处理**：每个交易日内横截面 Z-score 标准化（可选行业中性化）
- **训练方式**：全市场数据训练（所有股票一起训练，不逐股建模）
- **输出**：每只股票预测值（连续数值），保存每日预测结果 `(date, stock, prediction)`
- **评估指标**：
  - Rank IC（每日横截面相关系数）
  - ICIR（IC 均值 / 标准差）
  - 多空组合收益（Top 10% vs Bottom 10%）

对应代码：`backend/baseline_regression_model/`

### 任务二：态势识别分类模型（重点任务）

- **标签定义**（$T_1=10, T_2=5, \theta_1=1\%, \theta_2=2\%$）：
  - **0**：$r_{future} \leq 1\%$
  - **1**：$r_{future} > 1\%$ 且 $r_{past} \leq 2\%$（初涨）
  - **2**：$r_{future} > 1\%$ 且 $r_{past} > 2\%$（中涨）
- **特征处理**：对收益类特征（`ret_*`）截断 clip 至 $[-1, 1]$，再横截面 Z-score 标准化
- **模型**：LightGBM 多分类
- **输出**：每只股票的类别概率 (P0, P1, P2)
- **评分公式**：$score = P_1 + 0.5 \times P_2$
- **选股**：按 score 排序，选取 Top 5% 或 Top 10%
- **评估指标**：
  - Top 5% / 10% 命中率
  - Precision（重点关注"初涨"类别）
  - 简单组合收益

对应代码：`backend/situation_awareness_classification_model/`

### 任务三：模型系统实现（必须完成）

通过 Streamlit 前端（`frontend/app.py`）实现，具体见第 4.1 节。

### 任务四：策略与可视化（验证阶段，后期可先不做）

用于验证模型输出是否具有实际选股能力：
- **Rank IC 时间序列曲线**：验证排序能力
- **简单策略累计收益曲线**：验证选股有效性（每日选取 Top 5%/10%，等权构建组合）

> 本阶段策略仅用于验证，不涉及交易成本、换手率等复杂因素。

### 任务五：系统交互（简化实现）

用于结果展示，已集成在 Streamlit 前端中（见第 4.1 节）：
- 选择模型类型（回归 / 分类）、时间区间
- 展示 IC 曲线、策略收益曲线
- 后续可逐步扩展交互功能
# boosting_on_stock
