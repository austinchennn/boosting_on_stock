import pandas as pd
import numpy as np

class SituationDataProcessor:
    """
    负责态势识别任务的数据处理，包括三分类标签构建和特征清洗。
    """

    def construct_labels(self, df: pd.DataFrame, theta1: float = 0.01, theta2: float = 0.02) -> pd.DataFrame:
        """
        根据手册 2.1 节逻辑构建三分类态势标签。
        标签定义：
        - 0: r_future <= 1%
        - 1 (初涨): r_future > 1% 且 r_past <= 2%
        - 2 (中涨): r_future > 1% 且 r_past > 2%
        
        参数:
        - df: 原始 DataFrame，需包含 r_past_10 (r_past) 和 r_future_5 (r_future)
        - theta1: 未来收益阈值 (1%)
        - theta2: 过去收益阈值 (2%)
        """
        # 1. 提取核心列名（对应手册 2.1）
        r_past = df['r_past_10']  # 过去10日收益
        r_future = df['r_future_5']  # 未来5日收益

        # 2. 定义分类条件
        # 条件1：满足初涨 (Label 1)
        cond_1 = (r_future > theta1) & (r_past <= theta2)

        # 条件2：满足中涨 (Label 2)
        cond_2 = (r_future > theta1) & (r_past > theta2)

        # 3. 使用 np.select 快速赋值（默认值为 0）
        conditions = [cond_1, cond_2]
        choices = [1, 2]

        df['label'] = np.select(conditions, choices, default=0)

        return df

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        根据手册 2.2 节构建核心特征：
        - ret_5d, ret_10d
        - momentum_change (ret_5d - ret_10d)
        - bias_60, roc_20
        - r_past_10
        """
        # 计算趋势变化特征 momentum_change
        if 'ret_5d' not in df.columns or 'ret_10d' not in df.columns:
            raise ValueError(f"特征工程需要 'ret_5d' 和 'ret_10d'，但发现: {list(df.columns)}")

        df['momentum_change'] = df['ret_5d'] - df['ret_10d']
        

        # 确保 r_past_10 存在于 DataFrame 中
        # 注意：r_past_10 通常由原始数据提供或在加载时计算
        if 'r_past_10' not in df.columns:
             raise ValueError("r_past_10 缺失。")

        # 定义第一版必须使用的特征列表
        self.core_features = [
            'ret_5d', 'ret_10d', 
            'momentum_change', 
            'bias_60', 'roc_20', 
            'r_past_10'
        ]
        
        return df
    
    #默认不用加可选特征
    # def add_expandable_features(self, df: pd.DataFrame, 
    #                             include_pca: bool = False, 
    #                             include_industry: bool = False, 
    #                             include_lag: bool = False) -> tuple[pd.DataFrame, list]:
    #     """
    #     根据手册 2.3 节，在基础模型稳定后，可选择性地将已有列加入特征名单。
        
    #     参数:
    #     - df: 原始 DataFrame，假设已包含所有扩展列。
    #     - include_pca: 是否启用 PCA 因子 F1-F5。
    #     - include_industry: 是否启用行业与板块因子 。
    #     - include_lag: 是否启用收盘价滞后特征。
        
    #     返回:
    #     - (df, expandable_list): 返回原始数据及本次新增的特征名称列表。
    #     """
    #     expandable_list = []

    #     # 1. 加入短期收益率特征
    #     expandable_list.append('ret_1d')
    #     expandable_list.append('ret_3d')

    #     # 2. 加入 PCA 因子 (F1-F5)
    #     if include_pca:
    #         expandable_list.extend(['F1', 'F2', 'F3', 'F4', 'F5'])

    #     # 3. 加入行业与板块因子 
    #     if include_industry:
    #         # 假设你的列名就是 industry_code 或类似的标识
    #         # 这里你可以根据实际 Parquet 里的行业列名进行修改
    #         expandable_list.append('industry_code') 
    #         expandable_list.append('sector_code')

    #     # 4. 加入滞后特征 
    #     if include_lag:
    #         expandable_list.extend(['close_lag1', 'close_lag3', 'close_lag5'])

    #     return df, expandable_list

    def apply_preprocessing(self, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
        """
        执行手册 2.4 节要求的特征处理：
        1. 对收益类特征 (ret_*) 进行截断 (clip) 。
        2. 执行每日横截面 Z-score 标准化 。
        """
        # 1. 收益率特征截断 (Clipping)
        # 根据手册要求，将 ret_* 特征限制在合理区间 [-1, 1] 以去除极端异常值
        ret_cols = [col for col in feature_cols if isinstance(col, str) and col.startswith('ret_')]
        if ret_cols:
            df[ret_cols] = df[ret_cols].clip(lower=-1.0, upper=1.0)
            
        # 2. 每日横截面 Z-score 标准化
        # [Manual 2.4] Cross-sectional Z-score standardization per day
        if 'date' not in df.columns:
            raise ValueError("横截面预处理需要 'date' 列。")

        # 使用 groupby().transform() 计算均值和标准差
        # 这种矢量化操作比 apply(lambda) 快很多
        grouped = df.groupby('date')[feature_cols]
        mean = grouped.transform('mean')
        std = grouped.transform('std')
        
        # Z-score 公式: (x - mean) / std
        # 添加 1e-8 防止除以零
        df[feature_cols] = (df[feature_cols] - mean) / (std + 1e-8)
        
        # 填充可能产生的缺失值 (例如某些日期只有一个样本导致 std 为 NaN)
        df[feature_cols] = df[feature_cols].fillna(0)
        
        return df

    def split_data_by_time(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        根据手册 1.4 节执行数据划分：
        - 训练集：2016–2022
        - 测试集：2023–2026
        - 必须按时间顺序划分
        """
        # 确保 'date' 列为 datetime 类型
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            try:
                df['date'] = pd.to_datetime(df['date'])
            except Exception as e:
                raise ValueError(f"无法将 'date' 列转换为 datetime 类型: {e}")

        # 定义划分区间
        # 训练集: 2016-2022
        train_mask = (df['date'].dt.year >= 2016) & (df['date'].dt.year <= 2022)
        
        # 测试集: 2023-2026
        test_mask = (df['date'].dt.year >= 2023) & (df['date'].dt.year <= 2026)

        # 划分并按时间排序
        train_df = df[train_mask].sort_values('date').reset_index(drop=True)
        test_df = df[test_mask].sort_values('date').reset_index(drop=True)

        return train_df, test_df