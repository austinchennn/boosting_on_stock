from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd

class BaselineModel:
    """
    基线模型类：负责 LightGBM 模型的初始化、全市场训练、预测及结果保存。
    """
    
    def __init__(self, model_type: str = 'lightgbm', params: dict = None):
        """
        初始化模型。
        
        参数:
        - model_type: 模型类型，支持 'lightgbm' 或 'xgboost'
        - params: 外部传入的超参数字典
        """
        self.model_type = model_type.lower()
        if self.model_type not in ["lightgbm", "xgboost"]:
            raise ValueError("任务一必须使用 LightGBM 或 XGBoost 回归模型")

        self.params = params if params else {}
        self.is_fitted = False
        
        if self.model_type == 'lightgbm':
            # 设置 LightGBM 的默认回归参数
            default_params = {
                "objective": "regression",      # 回归任务：预测未来收益连续值
                "learning_rate": 0.05,
                "n_estimators": 300,
                "num_leaves": 31,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "n_jobs": -1,
            }
            # 合并自定义参数
            merged_params = default_params.copy()
            merged_params.update(self.params)
            # 核心：无论外部传入什么，强制设定为回归任务
            merged_params["objective"] = "regression"
            self.model = lgb.LGBMRegressor(**merged_params)
            
        elif self.model_type == 'xgboost':
            try:
                import xgboost as xgb
                default_params = {
                    "objective": "reg:squarederror",
                    "learning_rate": 0.05,
                    "n_estimators": 300,
                    "max_depth": 6,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "random_state": 42,
                    "n_jobs": -1,
                }
                merged_params = default_params.copy()
                merged_params.update(self.params)
                self.model = xgb.XGBRegressor(**merged_params)
            except ImportError:
                raise ImportError("请先安装 xgboost 库: pip install xgboost")


    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        使用全市场数据训练回归模型。
        不进行逐股票建模，所有样本统一学习。
        
        参数:
        - X_train: 训练集特征矩阵 (通常为 2016-2022 年数据)
        - y_train: 训练集预测目标 (必须是 y_ret_5) 
        """
        # 训练前的安全性检查
        if X_train.empty:
            raise ValueError("X_train 为空，无法训练模型")
        if y_train.empty:
            raise ValueError("y_train 为空，无法训练模型")
        if len(X_train) != len(y_train):
            raise ValueError("X_train 与 y_train 行数不一致")

        # 调用 fit 接口进行全市场数据训练
        self.model.fit(X_train, y_train)
        self.is_fitted = True

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        对测试集进行预测，输出每只股票的连续数值预测结果
        
        参数:
        - X_test: 测试集特征 (通常为 2023-2026 年数据)
        
        返回:
        - 预测出的未来收益连续数值数组
        """
        if not self.is_fitted:
            raise RuntimeError("模型尚未训练，请先调用 train")
        
        if X_test.empty:
            return np.array([], dtype=float)

        # 获取模型预测的原始数值
        predictions = self.model.predict(X_test)
        return np.asarray(predictions, dtype=float)

    def generate_predictions_df(self, test_df: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
        """
        将预测结果与日期、股票拼接，并执行关键的横截面排序。
        
        参数:
        - test_df: 包含原始索引 (date, stock) 的测试集数据框
        - predictions: predict 函数生成的数值数组
        
        返回:
        - 格式为 (date, stock, prediction) 的排序后结果
        """
        required_cols = {"date", "stock"}
        missing_cols = required_cols.difference(test_df.columns)
        if missing_cols:
            missing_cols_str = ", ".join(sorted(missing_cols))
            raise ValueError(f"test_df 缺少必要列: {missing_cols_str}")

        if len(test_df) != len(predictions):
            raise ValueError("predictions 长度与 test_df 行数不一致")

        # 1. 抽取 date 和 stock，挂载预测值
        pred_df = test_df.loc[:, ["date", "stock"]].copy()
        pred_df["prediction"] = predictions

        # 2. 执行横截面排序 ：
        # 按日期升序排列，在同一日期（横截面）内，按预测收益降序排列
        # 这一步是为了方便后续筛选 Top 10% 的股票 [cite: 35]
        pred_df = pred_df.sort_values(["date", "prediction"], ascending=[True, False]).reset_index(drop=True)
        return pred_df

    def save_predictions(self, pred_df: pd.DataFrame, save_path: str):
        """
        按照指南要求的格式保存每日预测结果 。
        
        参数:
        - pred_df: 包含 (date, stock, prediction) 的结果数据框
        - save_path: 文件保存路径 (支持 .csv 或 .parquet)
        """
        required_cols = ["date", "stock", "prediction"]
        missing_cols = [col for col in required_cols if col not in pred_df.columns]
        if missing_cols:
            missing_cols_str = ", ".join(missing_cols)
            raise ValueError(f"pred_df 缺少必要列: {missing_cols_str}")

        # 确保保存目录存在
        save_target = Path(save_path).expanduser()
        save_target.parent.mkdir(parents=True, exist_ok=True)

        # 导出结果
        export_df = pred_df.loc[:, required_cols]
        if save_target.suffix.lower() == ".parquet":
            export_df.to_parquet(save_target, index=False)
        else:
            export_df.to_csv(save_target, index=False)