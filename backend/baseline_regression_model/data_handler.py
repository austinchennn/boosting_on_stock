from pathlib import Path

import pandas as pd

class DataProcessor:
    """
    数据处理类：负责数据的加载、时间划分与横截面特征标准化。
    数据格式要求为: (date, stock, features)
    """
    
    def __init__(self, data_path: str):
        """
        初始化数据处理器。
        
        参数:
        - data_path: 原始数据文件的路径
        """
        self.data_path = Path(data_path).expanduser()

    def load_data(self) -> pd.DataFrame:
        """
        加载原始数据集。
        
        返回:
        - 包含所有历史数据的 DataFrame
        """
        # 1. 检查文件路径是否存在，防止读取空路径报错
        if not self.data_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {self.data_path}")

        # 2. 校验文件格式，确保是项目要求的 parquet 格式
        if self.data_path.suffix.lower() != ".parquet":
            raise ValueError(f"仅支持 parquet 文件，当前收到: {self.data_path.suffix}")

        # 3. 正式读取数据
        df = pd.read_parquet(self.data_path)

        # 4. 【核心逻辑】列名兼容性处理：将 ticker 转换为 stock
        # 逻辑：如果原始数据列里有 'ticker' 且没有 'stock'，则进行重命名
        # 这是为了确保数据符合 (date, stock, features) 的标准格式
        if "ticker" in df.columns and "stock" not in df.columns:
            df = df.rename(columns={"ticker": "stock"})

        # 5. 强制校验：检查是否包含必须的 'date' 和 'stock' 列
        required_cols = {"date", "stock"}
        missing_cols = required_cols.difference(df.columns)
        if missing_cols:
            missing_cols_str = ", ".join(sorted(missing_cols))
            raise ValueError(f"数据不符合 (date, stock, features) 格式，缺少列: {missing_cols_str}")

        # 6. 时间格式标准化：将 date 列转换为 pandas 的 datetime 类型
        if not pd.api.types.is_datetime64_any_dtype(df["date"]):
            df = df.copy() # 避免 SettingWithCopyWarning 警告
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # 7. 检查是否存在非法日期（如格式错误导致解析为 NaT 的值）
        if df["date"].isna().any():
            raise ValueError("date 列存在无法解析的日期值")

        # 8. 自动识别特征列：除了 date 和 stock 以外的所有列都视为因子特征 
        feature_cols = [col for col in df.columns if col not in {"date", "stock"}]
        if not feature_cols:
            raise ValueError("数据不符合 (date, stock, features) 格式，未检测到任何特征列")

        # 9. 数据重排与清洗：
        # - 重新排列列顺序：date, stock, 然后是所有特征
        # - 按日期和股票代码排序，确保每个交易日为一个横截面 
        # - 重置索引，保证索引连续
        ordered_cols = ["date", "stock", *feature_cols]
        df = df.loc[:, ordered_cols].sort_values(["date", "stock"]).reset_index(drop=True)

        return df

    def split_data_by_time(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        按时间顺序划分训练集与测试集。
        - 训练集: 2016-2022
        - 测试集: 2023-2026
        """
        # 1. 安全检查：确保输入数据包含核心索引列
        required_cols = {"date", "stock"}
        missing_cols = required_cols.difference(df.columns)
        if missing_cols:
            missing_cols_str = ", ".join(sorted(missing_cols))
            raise ValueError(f"输入数据缺少必要列: {missing_cols_str}")

        # 2. 格式强制转换：确保日期列是 datetime 格式，方便提取年份
        if not pd.api.types.is_datetime64_any_dtype(df["date"]):
            df = df.copy()
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        if df["date"].isna().any():
            raise ValueError("date 列存在无法解析的日期值")

        # 3. 时间维度提取：获取每行数据的年份
        year_series = df["date"].dt.year
        
        # 4. 根据文档要求定义掩码（Mask）
        # 训练集范围：2016年1月1日 至 2022年12月31日
        train_mask = year_series.between(2016, 2022)
        # 测试集范围：2023年1月1日 至 2026年12月31日
        test_mask = year_series.between(2023, 2026)

        # 5. 执行切分并排序：
        # sort_values 确保每个集合内部也是按时间轴线性排列的
        train_df = df.loc[train_mask].sort_values(["date", "stock"]).reset_index(drop=True)
        test_df = df.loc[test_mask].sort_values(["date", "stock"]).reset_index(drop=True)

        # 6. 空值检查：防止因原始数据时间范围不对导致后续训练崩溃
        if train_df.empty:
            raise ValueError("训练集为空，请检查 2016-2022 年是否有数据")

        if test_df.empty:
            raise ValueError("测试集为空，请检查 2023-2026 年是否有数据")

        return train_df, test_df
    
    def cross_sectional_zscore(self, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
        """
        在每个交易日内进行横截面 Z-score 标准化 。
        目的是消除不同交易日之间市场整体波动幅度的差异，只保留股票间的相对强弱关系
        """
        # 1. 基础校验逻辑
        required_cols = {"date", "stock"}
        missing_base_cols = required_cols.difference(df.columns)
        if missing_base_cols:
            missing_cols_str = ", ".join(sorted(missing_base_cols))
            raise ValueError(f"输入数据缺少必要列: {missing_cols_str}")

        if not feature_cols:
            raise ValueError("feature_cols 不能为空")

        # 2. 检查传入的特征列是否在 DataFrame 中以及是否为数值类型
        missing_feature_cols = [col for col in feature_cols if col not in df.columns]
        if missing_feature_cols:
            missing_cols_str = ", ".join(missing_feature_cols)
            raise ValueError(f"待标准化特征列不存在: {missing_cols_str}")

        non_numeric_cols = [col for col in feature_cols if not pd.api.types.is_numeric_dtype(df[col])]
        if non_numeric_cols:
            non_numeric_cols_str = ", ".join(non_numeric_cols)
            raise ValueError(f"以下特征列不是数值类型，无法进行 Z-score 标准化: {non_numeric_cols_str}")

        # 3. 创建副本避免直接修改原始数据
        standardized_df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(standardized_df["date"]):
            standardized_df["date"] = pd.to_datetime(standardized_df["date"], errors="coerce")

        if standardized_df["date"].isna().any():
            raise ValueError("date 列存在无法解析的日期值")

        # 4. 【核心计算逻辑】：实现横截面处理 [cite: 8]
        # 只提取特征部分进行计算
        feature_frame = standardized_df.loc[:, feature_cols]
        
        # 关键：按 'date' 分组。groupby(date) 确保均值和标准差是“每日”独立计算的 
        grouped = feature_frame.groupby(standardized_df["date"])
        
        # transform("mean") 会生成一个与原表行数一致的表，每行填充该日该特征的均值
        feature_means = grouped.transform("mean")
        feature_stds = grouped.transform("std")
        
        # 处理异常：如果某天所有股票该特征都一样（标准差为0），会导致除以0报错
        zero_std_mask = feature_stds.eq(0)

        # 5. 执行 Z-score 公式： (x - mean) / std
        # .mask 处理标准差为 0 的情况，防止 Inf 出现
        zscore_features = (feature_frame - feature_means) / feature_stds.mask(zero_std_mask)
        
        # 6. 细节清洗：
        # - 保持原有的空值（NaN）不变
        # - 如果标准差为 0 且原本有值，则将该特征值设为 0（表示该股票处于平均水平）
        zscore_features = zscore_features.where(feature_frame.notna())
        zscore_features = zscore_features.mask(zero_std_mask & feature_frame.notna(), 0.0)

        # 7. 写回数据：将标准化后的特征替换回 DataFrame 的对应位置
        standardized_df.loc[:, feature_cols] = zscore_features

        return standardized_df