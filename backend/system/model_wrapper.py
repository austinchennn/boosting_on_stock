import joblib
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from backend.baseline_regression_model.model_trainer import BaselineModel
from backend.situation_awareness_classification_model.model_trainer import SituationModelTrainer
from backend.system.config import TaskType, ModelType

class ModelWrapper:
    def __init__(self, task_type: TaskType, model_type: ModelType, params: dict = None):
        self.task_type = task_type
        self.model_type = model_type
        self.params = params if params else {}
        self.model = None
        self.trainer = None
        
        # 初始化相应的训练器
        if self.task_type == TaskType.REGRESSION:
            self.trainer = BaselineModel(model_type=self.model_type.value, params=self.params)
        elif self.task_type == TaskType.CLASSIFICATION:
            self.trainer = SituationModelTrainer(num_class=3, model_type=self.model_type.value)

    def train(self, df: pd.DataFrame, feature_cols: list, label_col: str):
        """
        统一训练方法。
        """
        print(f"开始训练 {self.task_type.value} 模型，类型: {self.model_type.value}...")
        
        if self.task_type == TaskType.REGRESSION:
            X = df[feature_cols]
            y = df[label_col]
            self.trainer.train(X, y)
            self.model = self.trainer.model
            
        elif self.task_type == TaskType.CLASSIFICATION:
            # Task 2 训练器返回模型对象
            self.model = self.trainer.train(df, feature_cols, label_col, params=self.params)
            
        print("训练完成。")
        return self.model

    def predict(self, df: pd.DataFrame, feature_cols: list) -> pd.Series:
        """
        统一预测方法。
        返回标准化的 'score' 序列。
        - 回归: 原始预测收益率。
        - 分类: 加权得分 (P1*0.5 + P2*1.0) 代表看涨程度。
        """
        if self.model is None:
            raise ValueError("模型尚未训练或加载。")
            
        if self.task_type == TaskType.REGRESSION:
            # Task 1 包装器通常期望 dataframe，但为了安全起见
            return self.trainer.predict(df[feature_cols])
            
        elif self.task_type == TaskType.CLASSIFICATION:
            # Task 2 需要传递模型对象
            probs = self.trainer.predict_proba(self.model, df, feature_cols)
            # probs 是 (N, 3) -> [类别 0 (看空), 类别 1 (初涨), 类别 2 (中涨)]
            # 我们构建一个综合得分用于排序: score = P1 + 0.5 * P2
            # P1 = probs[:, 1], P2 = probs[:, 2]
            score = probs[:, 1] * 1.0 + probs[:, 2] * 0.5
            return score

    def save(self, path: str):
        joblib.dump(self.model, path)
        print(f"模型已保存至 {path}")

    def load(self, path: str):
        self.model = joblib.load(path)
        # 对于 Task 1，如果想使用训练器方法，需要将其挂载回训练器
        if self.task_type == TaskType.REGRESSION:
            self.trainer.model = self.model
            self.trainer.is_fitted = True
        print(f"模型已从 {path} 加载")

    def get_params(self):
        """
        获取底层模型的参数。
        """
        if self.model is not None and hasattr(self.model, 'get_params'):
            return self.model.get_params()
        
        # 如果模型未训练但训练器已初始化参数，则作为后备
        if self.task_type == TaskType.REGRESSION and hasattr(self.trainer, 'params'):
            return self.trainer.params
            
        return self.params
