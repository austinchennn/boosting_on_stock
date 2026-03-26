import pandas as pd
import numpy as np
import os
import sys

# 假设 data_processor 和 model_trainer 在同一目录下
# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 将当前目录添加到 sys.path
sys.path.append(current_dir)

from data_processor import SituationDataProcessor
from model_trainer import SituationModelTrainer
from evaluator import SituationEvaluator
from task2visualizer import Task2Visualizer

def run_task2_pipeline():
    """
    任务二主流水线：
    1. 读取数据并构建 0/1/2 态势标签
    2. 执行特征工程与标准化
    3. 按时间划分训练集 (2016-2022) 与测试集 (2023-2026)
    4. 进行多分类训练并输出类别概率
    5. 构建自定义评分并进行横截面排序
    6. 计算评估指标并保存结果
    7. 可视化分析 (饼图 + 收益曲线)
    """
    # 配置路径
    # 假设数据在 workspace 的 final_dataset_with_time_features 目录下
    # main_task2.py -> backend/situation_awareness_classification_model
    # WORKSPACE_ROOT (root) -> parent(parent(current_dir))
    WORKSPACE_ROOT = os.path.dirname(os.path.dirname(current_dir))
    DATA_PATH = os.path.join(WORKSPACE_ROOT, "final_dataset_with_time_features", "final_dataset_with_time_features.parquet")
    OUTPUT_DIR = os.path.join(current_dir, "outputs")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("步骤 1: 加载数据并构建标签...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"数据文件未找到: {DATA_PATH}")

    # 1. 读取数据
    df = pd.read_parquet(DATA_PATH)
    print(f"已加载数据形状: {df.shape}")

    # 初始化处理器
    processor = SituationDataProcessor()

    # 2. 构建态势标签 (0/1/2)
    # 逻辑：必须先判断 r_future > 1%，再根据 r_past 区分初涨/中涨
    df = processor.construct_labels(df)
    print("标签构建完成。")
    print("标签分布:\n", df['label'].value_counts(normalize=True))

    # 3. 特征工程
    print("步骤 2: 特征工程与预处理...")
    df = processor.feature_engineering(df)
    
    # 获取核心特征列表
    feature_cols = processor.core_features
    print(f"核心特征: {feature_cols}")

    # 4. 特征标准化 (2.4 节要求)
    # 注意：收益类特征截断 + 每日横截面 Z-score
    # 标签绝对不能参与标准化
    df = processor.apply_preprocessing(df, feature_cols)
    print("预处理完成。")

    # 5. 数据划分 (1.4 节要求)
    # 按时间顺序划分：训练集 (2016-2022)，测试集 (2023-2026)
    # 严禁 Shuffle
    print("步骤 3: 划分数据...")
    train_df, test_df = processor.split_data_by_time(df)
    print(f"训练集形状: {train_df.shape} ({train_df['date'].min()} - {train_df['date'].max()})")
    print(f"测试集形状: {test_df.shape} ({test_df['date'].min()} - {test_df['date'].max()})")

    # 6. 模型训练 (LightGBM 多分类)
    print("步骤 4: 训练 LightGBM 模型...")
    trainer = SituationModelTrainer(num_class=3) # 显式指定3分类
    
    # 训练模型
    # 注意：这里我们使用 multi_logloss 作为评估指标
    model = trainer.train(train_df, feature_cols, label_col='label')
    print("模型训练完成。")

    # 7. 预测与概率输出
    print("步骤 5: 生成预测 (概率)...")
    # 预测测试集
    # 输出应该是 (N, 3) 的矩阵，分别为 P0, P1, P2
    probs = trainer.predict_proba(model, test_df, feature_cols)
    
    # 结果保存
    # 务必确认列顺序：0, 1, 2
    # P1 (初涨) 权重高， P2 (中涨) 权重次之
    # 必须保留真实收益率 'r_future_5' 以计算多空收益
    cols_to_keep = ['date', 'ticker', 'label']
    # 优先检查 r_future_5 (Task 2 label source), 其次 y_ret_5 (Task 1 label source)
    target_return_col = None
    if 'r_future_5' in test_df.columns:
        target_return_col = 'r_future_5'
    elif 'y_ret_5' in test_df.columns:
        target_return_col = 'y_ret_5'
        
    if target_return_col:
        cols_to_keep.append(target_return_col)
    
    result_df = test_df[cols_to_keep].copy()
    result_df['prob_0'] = probs[:, 0]
    result_df['prob_1'] = probs[:, 1]
    result_df['prob_2'] = probs[:, 2]

    # 计算自定义评分 score = P1 + 0.5 * P2
    evaluator = SituationEvaluator()
    result_df['score'] = evaluator.calculate_custom_score(result_df)
    
    # 保存结果
    RESULT_PATH = os.path.join(OUTPUT_DIR, "task2_predictions.parquet")
    result_df.to_parquet(RESULT_PATH)
    print(f"预测结果已保存至 {RESULT_PATH}")
    
    # 简单查看一下 Top Score 的情况
    print("Top 5 评分样本:")
    print(result_df.sort_values('score', ascending=False).head(5))

    # 8. 评估模型精度 (Precision of Top K)
    print("步骤 6: 评估模型精度 (Top K 命中率)...")
    metrics = evaluator.evaluate_precision(result_df, top_k_list=[0.05, 0.1])
    print("评估指标:", metrics)

    # 9. 可视化分析
    print("步骤 7: 可视化分析...")
    visualizer = Task2Visualizer(output_dir=os.path.join(OUTPUT_DIR, "plots"))
    # 9.1 饼图
    visualizer.plot_precision_pie_charts(metrics)
    
    # 9.2 多空收益曲线
    if target_return_col and target_return_col in result_df.columns:
        print(f"计算多空组合收益 (Long Top 5% - Short Bottom 5%) using {target_return_col}...")
        long_short_rets = evaluator.calculate_long_short_daily_returns(result_df, target_col=target_return_col, top_k=0.05)
        visualizer.plot_long_short_returns(long_short_rets)
    else:
        print("警告: 缺少真实收益率列 (r_future_5 或 y_ret_5)，跳过收益曲线绘制。")

if __name__ == "__main__":
    run_task2_pipeline()