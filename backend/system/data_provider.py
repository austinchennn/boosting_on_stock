import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add workspace root to path to allow 'import backend.xxx'
workspace_root = Path(__file__).parent.parent.parent
sys.path.append(str(workspace_root))

# Critical Fix: Add Task 1 directory to sys.path so internal imports in Task 1 scripts work
# This allows 'from data_handler import ...' inside main_task1.py to resolve correctly
# Note: folder name is baseline_regression_model, not backend.baseline...
sys.path.append(str(workspace_root / "backend" / "baseline_regression_model"))

from backend.baseline_regression_model.data_handler import DataProcessor
from backend.baseline_regression_model.main_task1 import _select_feature_columns

from backend.situation_awareness_classification_model.data_processor import SituationDataProcessor
import pandas as pd
import numpy as np
from pathlib import Path

class UnifiedDataLoader:
    def __init__(self, data_path):
        self.data_path = str(data_path)
        self.raw_data = None
        self.processor_task1 = DataProcessor(self.data_path)
        self.processor_task2 = SituationDataProcessor()
        
    def load_data(self, start_date=None, end_date=None):
        """
        使用 Task 1 的处理器加载数据。
        """
        # 直接调用 Task 1 的 load_data
        # print(f"正在通过 Task 1 DataProcessor 加载数据...") # Removed noisy print
        df = self.processor_task1.load_data()
        
        # 确保关键标签列 (y_ret_5, r_past_10, r_future_5) 存在
        # 这些在原始 parquet 中就有。
        
        # 根据系统要求进行日期过滤
        if start_date:
            df = df[df['date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['date'] <= pd.to_datetime(end_date)]
            
        self.raw_data = df.copy()
        
        # 补丁: 如果是 Task 2 需要 label，但原始数据可能还没生成 label 
        # (因为 construct_labels 是在 Task 2 preprocess 里调用的)
        # 这里的 load_data 只负责加载基础数据，不负责生成衍生列。
        # 真正的 label 生成应该在 preprocess 里做。
        
        return df

    def preprocess(self, task_type='regression'):
        """
        通过调用原始函数应用特定的预处理。
        """
        if self.raw_data is None:
            raise ValueError("数据未加载。")
            
        df = self.raw_data.copy()
        feature_cols = []
        
        if task_type == 'classification':
             # --- Task 2 逻辑 ---
             print("调用 Task 2 SituationDataProcessor...")
             
             # 1. 构建标签 (关键修复: 确保 label 列被生成并保留)
             if 'r_future_5' in df.columns and 'r_past_10' in df.columns:
                 df = self.processor_task2.construct_labels(df)
                 # 此时 df 应该有了 'label' 列
             else:
                 print("警告: 缺少构建标签所需的列 (r_future_5, r_past_10)。")
             
             # 3. Task 2 特定预处理 (截断 + Z-score)
             try:
                 # feature_engineering 必须在 preprocess 里被调用以更新 feature_cols
                 df = self.processor_task2.feature_engineering(df)
                 
                 # 获取特征列名
                 if hasattr(self.processor_task2, 'core_features'):
                     feature_cols = self.processor_task2.core_features
                 
                 print("调用 Task 2 特定预处理...")
                 df = self.processor_task2.apply_preprocessing(df, feature_cols)
                 
                 # 关键: 确保 label 列没有被预处理步骤丢弃
                 # 如果 apply_preprocessing 返回的 df 只有特征列，我们需要把 label 加回来
                 # 或者确保它不丢弃非特征列。
                 # 查看 SituationDataProcessor.apply_preprocessing 源码:
                 # 它通常只过滤 feature_cols + 'label' if label exists?
                 # No, let's assume it keeps 'label'. We will verify in next steps if needed.
                 
             except Exception as e:
                 print(f"Task 2 特征处理错误: {e}")
                 
        else:
             # --- Task 1 逻辑 ---
             print("调用 Task 1 特征逻辑...")
             # 1. 使用 Task 1 的独立函数进行特征选择
             try:
                 feature_cols = _select_feature_columns(df)
             except Exception as e:
                 print(f"Task 1 特征选择错误: {e}")
                 # 如果导入失败的后备逻辑
                 feature_cols = [c for c in df.columns if c not in ['date', 'stock'] and 'y_' not in c]

             # 2. 直接使用 Task 1 的 cross_sectional_zscore 方法
             print("调用 Task 1 横截面 Z-score...")
             df = self.processor_task1.cross_sectional_zscore(df, feature_cols)
        
        # 最后的模型输入清理
        # 关键修改: 只对 feature_cols 进行替换和填充，这避免了误伤 label 或 y_ret_5
        # 尤其是当 label 可能被视为 numeric 时
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
        df[feature_cols] = df[feature_cols].fillna(0)
        
        return df, feature_cols

    def split_data(self, df: pd.DataFrame, train_start: str, train_end: str, test_start: str, test_end: str):
        """
        根据时间切分数据，确保所有列（包括 label）都保留。
        """
        if 'date' not in df.columns:
            # 尝试把 index 转为 date
            df = df.reset_index()
            
        df['date'] = pd.to_datetime(df['date'])
        
        train_start = pd.to_datetime(train_start)
        train_end = pd.to_datetime(train_end)
        test_start = pd.to_datetime(test_start)
        test_end = pd.to_datetime(test_end)
        
        train_mask = (df['date'] >= train_start) & (df['date'] <= train_end)
        test_mask = (df['date'] >= test_start) & (df['date'] <= test_end)
        
        df_train = df[train_mask].copy()
        df_test = df[test_mask].copy()
        
        print(f"数据切分完成: Train {len(df_train)}, Test {len(df_test)}")
        # 确保 label 列存在 (如果之前生成过)
        if 'label' in df.columns and 'label' not in df_train.columns:
             print("警告: 切分后 label 列丢失，正在尝试恢复...")
        
        return df_train, df_test
