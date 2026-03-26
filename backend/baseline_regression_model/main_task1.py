from data_handler import DataProcessor
from model_trainer import BaselineModel
from evaluator import ModelEvaluator
from task1visualizer import Task1Visualizer


def _select_feature_columns(df):
    """选择可用于训练的数值因子列，剔除索引列与明显标签/未来信息列。"""
    exclude_cols = {
        "date",
        "stock",
        "y_ret_5",
        "y_ret_10",
        "y_ret_20",
        "y_ret_ma3_15",
        "future_ma3_15",
        "r_future_5",
        "label_up_15",
    }

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in exclude_cols and not col.startswith("y_")]
    if not feature_cols:
        raise ValueError("未找到可用特征列，请检查输入数据")
    return feature_cols

def run_task1_pipeline():
    """
    任务一主执行流程：
    1. 实例化 DataProcessor，加载数据并划分训练/测试集。
    2. 对训练集和测试集执行横截面 Z-score 特征标准化。
    3. 实例化 BaselineModel，选取目标特征和标签 (y_ret_5) 进行全市场训练。
    4. 对测试集进行预测，横截面排序，并保存 (date, stock, prediction) 结果。
    5. 实例化 ModelEvaluator，传入预测结果与真实标签，计算并打印 Rank IC, ICIR 和多空收益。
    """
    from pathlib import Path

    # Correct path for backend/baseline_regression_model/main_task1.py (depth 2)
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DATA_PATH = PROJECT_ROOT / "final_dataset_with_time_features" / "final_dataset_with_time_features.parquet"
    OUTPUT_DIR = Path(__file__).parent / "outputs"
    PRED_SAVE_PATH = OUTPUT_DIR / "task1_predictions.parquet"
    PLOT_DIR = OUTPUT_DIR / "plots"

    processor = DataProcessor(str(DATA_PATH))
    full_df = processor.load_data()
    train_df, test_df = processor.split_data_by_time(full_df)

    feature_cols = _select_feature_columns(train_df)

    train_std = processor.cross_sectional_zscore(train_df, feature_cols)
    test_std = processor.cross_sectional_zscore(test_df, feature_cols)

    train_ready = train_std.dropna(subset=feature_cols + ["y_ret_5"]).reset_index(drop=True)
    test_ready = test_std.dropna(subset=feature_cols + ["y_ret_5"]).reset_index(drop=True)

    model = BaselineModel(model_type="lightgbm")
    model.train(train_ready[feature_cols], train_ready["y_ret_5"])

    predictions = model.predict(test_ready[feature_cols])
    pred_df = model.generate_predictions_df(test_ready[["date", "stock"]], predictions)
    model.save_predictions(pred_df, str(PRED_SAVE_PATH))

    eval_input_df = pred_df.merge(
        test_df[["date", "stock", "y_ret_5"]],
        on=["date", "stock"],
        how="inner",
    )
    evaluator = ModelEvaluator(eval_input_df, target_col="y_ret_5")
    report = evaluator.generate_evaluation_report()

    visualizer = Task1Visualizer(report, save_dir=str(PLOT_DIR))
    visualizer.calculate_benchmarks(eval_input_df)
    rank_ic_plot_path = visualizer.plot_rank_ic_ts()
    cumulative_return_plot_path = visualizer.plot_cumulative_return()
    ic_distribution_plot_path = visualizer.plot_ic_distribution()

    print("任务一流程执行完成。")
    print(f"训练集样本数: {len(train_ready)} | 测试集样本数: {len(test_ready)}")
    print(f"特征数量: {len(feature_cols)}")
    print(f"预测结果文件: {PRED_SAVE_PATH}")
    print(f"Rank IC 曲线文件: {rank_ic_plot_path}")
    print(f"累计收益曲线文件: {cumulative_return_plot_path}")
    print(f"IC 分布图文件: {ic_distribution_plot_path}")
    print(f"平均 Rank IC: {report['mean_rank_ic']:.6f}" if report["mean_rank_ic"] == report["mean_rank_ic"] else "平均 Rank IC: nan")
    print(f"ICIR 稳定性指标: {report['icir']:.6f}" if report["icir"] == report["icir"] else "ICIR 稳定性指标: nan")
    print(f"多空组合累计收益: {report['long_short_return']:.6f}" if report["long_short_return"] == report["long_short_return"] else "多空组合累计收益: nan")

if __name__ == "__main__":
    run_task1_pipeline()