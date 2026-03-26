import lightgbm as lgb
import pandas as pd
import numpy as np

class SituationModelTrainer:
    """
    基于 LightGBM 的多分类模型训练器。
    对应手册：2.5 模型训练流程
    """

    def __init__(self, num_class: int = 3, model_type: str = 'lightgbm'):
        self.num_class = num_class
        self.model_type = model_type.lower()
        
        # 强制性核心参数 (LightGBM)
        self.lgb_default_params = {
            'objective': 'multiclass',
            'num_class': num_class,
            'metric': 'multi_logloss',
            'n_jobs': -1,
            'verbose': -1,
            'class_weight': 'balanced', # 应对样本不平衡
            'random_state': 42
        }

    def train(self, train_df: pd.DataFrame, feature_cols: list, label_col: str = 'label', params: dict = None):
        """
        训练模型 (Scikit-learn API)。支持 LightGBM 和 XGBoost。
        """
        # 准备数据
        X_train = train_df[feature_cols]
        y_train = train_df[label_col]

        if self.model_type == 'lightgbm':
            # 参数合并
            train_params = self.lgb_default_params.copy()
            if params:
                train_params.update(params)

            # 初始化并训练模型
            print(f"开始训练 LightGBM 模型，参数配置: {train_params}")
            model = lgb.LGBMClassifier(**train_params)
            model.fit(X_train, y_train)
            return model
            
        elif self.model_type == 'xgboost':
            try:
                import xgboost as xgb
                # XGBoost default params
                xgb_default_params = {
                    'objective': 'multi:softprob',
                    'num_class': self.num_class,
                    'eval_metric': 'mlogloss',
                    'n_jobs': -1,
                    'random_state': 42,
                    # XGBoost doesn't have 'class_weight' param directly like sklearn/lgb, 
                    # usually handled via sample_weight or scale_pos_weight (binary only).
                    # For multiclass, passing sample_weight to fit is better if needed.
                }
                train_params = xgb_default_params.copy()
                if params:
                    train_params.update(params)
                
                print(f"开始训练 XGBoost 模型，参数配置: {train_params}")
                model = xgb.XGBClassifier(**train_params)
                model.fit(X_train, y_train)
                return model
            except ImportError:
                raise ImportError("请先安装 xgboost 库: pip install xgboost")
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")

    def predict_proba(self, model, test_df: pd.DataFrame, feature_cols: list) -> np.ndarray:
        """
        预测每只股票属于三类 (0, 1, 2) 的概率。
        返回: (N, 3) 的 numpy 数组，列顺序对应 Label 0, 1, 2。
        
        注意：LightGBM 的 predict_proba 默认按标签值排序返回列。
        我们的标签是 0, 1, 2，所以:
        - Column 0 -> P(Label=0)
        - Column 1 -> P(Label=1) (初涨)
        - Column 2 -> P(Label=2) (中涨)
        """
        X_test = test_df[feature_cols]
        
        # 必须使用 predict_proba 获取概率，严禁使用 predict
        probs = model.predict_proba(X_test)
        
        return probs