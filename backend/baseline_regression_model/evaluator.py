import pandas as pd
import numpy as np

class ModelEvaluator:
    """
    模型评估类：负责计算 Rank IC、ICIR 以及多空组合收益等指标。
    """
    
    def __init__(self, pred_df: pd.DataFrame, target_col: str = 'y_ret_5'):
        """
        初始化评估器。
        
        参数:
        - pred_df: 包含 date, stock, prediction 和真实标签的 DataFrame
        - target_col: 真实的未来收益目标列名（默认为 y_ret_5）
        """
        self.target_col = target_col

        # 1. 检查必要列：确保数据包含日期、股票代码、预测值和真实收益列
        required_cols = {"date", "stock", "prediction", self.target_col}
        missing_cols = required_cols.difference(pred_df.columns)
        if missing_cols:
            missing_cols_str = ", ".join(sorted(missing_cols))
            raise ValueError(
                f"pred_df 缺少必要列: {missing_cols_str}。"
                "请先按 (date, stock) 将预测结果与测试集真实标签做内连接。"
            )

        # 2. 数据清洗与对齐：只提取需要的列并建立副本，避免影响原始数据
        aligned_df = pred_df.loc[:, ["date", "stock", "prediction", self.target_col]].copy()
        
        # 3. 时间格式标准化：确保日期列为 datetime 格式以便分组
        if not pd.api.types.is_datetime64_any_dtype(aligned_df["date"]):
            aligned_df["date"] = pd.to_datetime(aligned_df["date"], errors="coerce")

        # 4. 异常值处理：剔除含有空日期的行、无预测值的行或无真实收益的行
        aligned_df = aligned_df.dropna(subset=["date", "prediction", self.target_col])
        
        # 5. 去重：确保每个交易日每只股票只有一条记录，保留最新的一条
        aligned_df = aligned_df.drop_duplicates(subset=["date", "stock"], keep="last")
        
        # 6. 排序：按时间顺序重新组织数据
        aligned_df = aligned_df.sort_values(["date", "stock"]).reset_index(drop=True)

        if aligned_df.empty:
            raise ValueError("对齐后的评估数据为空，请检查预测结果与真实标签的连接逻辑")

        self.pred_df = aligned_df

    def _calculate_long_short_daily_returns(self) -> pd.Series:
        """计算按日期索引的每日多空收益序列。"""
        # 1. 初始化字典，用于存放每一个交易日计算出的多空对冲收益
        daily_returns = {}

        # 2. 按日期（date）进行分组，即在每一个“横截面”内独立执行筛选逻辑
        for date, group in self.pred_df.groupby("date", sort=True):
            # 3. 数据清洗：确保该日期下，预测值（prediction）和真实收益率（target_col）同时存在
            valid = group[["prediction", self.target_col]].dropna()
            n = len(valid)
            
            # 4. 样本量检查：如果当天有效股票不足 2 只，则无法构建多空对冲，直接跳过
            if n < 2:
                continue

            # 5. 确定分箱大小：按照指南 1.8 要求选取 Top 10% 和 Bottom 10% 的股票
            # 使用 np.ceil 向上取整并确保至少包含 1 只股票
            bucket_size = max(int(np.ceil(n * 0.1)), 1)
            
            # 6. 提取多头端（Long）：
            # 选取模型预测得分最高的前 10% 股票，计算它们真实的平均收益率
            long_mean = valid.nlargest(bucket_size, "prediction")[self.target_col].mean()
            
            # 7. 提取空头端（Short）：
            # 选取模型预测得分最低的后 10% 股票，计算它们真实的平均收益率
            short_mean = valid.nsmallest(bucket_size, "prediction")[self.target_col].mean()
            
            # 8. 计算每日收益差额：多头组平均收益 - 空头组平均收益
            # 如果模型有选股能力，此差值应长期为正
            daily_returns[date] = long_mean - short_mean

        return pd.Series(daily_returns, dtype=float, name="ls_return").sort_index()

    def calculate_rank_ic(self) -> pd.Series:
        """
        计算每日横截面 Rank IC (预测值与真实收益的 Spearman 相关系数)。
        
        返回:
        - 按日期索引的每日 Rank IC 序列
        """
        def _daily_rank_ic(group: pd.DataFrame) -> float:
            # 提取当前日期内预测值和真实值都不为空的样本
            valid = group[["prediction", self.target_col]].dropna()
            if len(valid) < 2: # 样本数过少无法计算相关系数
                return np.nan

            # 计算 Rank IC：先将数值转为排名，再计算 Pearson 相关系数，等同于 Spearman 相关系数
            pred_rank = valid["prediction"].rank(method="average")
            target_rank = valid[self.target_col].rank(method="average")
            return pred_rank.corr(target_rank, method="pearson")

        # 按天分组计算，得到每日 IC 时间序列
        # include_groups=False 防止 FutureWarning
        rank_ic_series = self.pred_df.groupby("date", sort=True).apply(_daily_rank_ic, include_groups=False)
        rank_ic_series.name = "rank_ic"
        return rank_ic_series

    def calculate_icir(self, rank_ic_series: pd.Series) -> float:
        """
        计算 ICIR (Rank IC 均值 / Rank IC 标准差)。
        衡量模型预测能力的稳定性。
        
        参数:
        - rank_ic_series: calculate_rank_ic 输出的每日序列
        """
        clean_ic = rank_ic_series.dropna()
        if clean_ic.empty:
            return np.nan

        ic_mean = clean_ic.mean()
        # 计算标准差（使用样本标准差 ddof=1）
        ic_std = clean_ic.std(ddof=1)
        
        # 如果标准差接近 0，说明 IC 没有波动，无法计算 IR
        if pd.isna(ic_std) or np.isclose(ic_std, 0.0):
            return np.nan

        return float(ic_mean / ic_std)

    def calculate_long_short_return(self) -> float:
        """
        计算多空组合收益 (做多 Top 10% 股票，做空 Bottom 10% 股票)。
        
        返回:
        - 测试期内多空组合的累计收益率
        """
        daily_returns_series = self._calculate_long_short_daily_returns()
        if daily_returns_series.empty:
            return np.nan

        cumulative_return = daily_returns_series.sum()
        return float(cumulative_return)

    def generate_evaluation_report(self) -> dict:
        """
        汇总所有指标，生成最终的评估报告字典。
        包含：Rank IC 均值、ICIR、多空累计收益以及原始 IC 序列。
        """
        rank_ic_series = self.calculate_rank_ic()
        ls_return_series = self._calculate_long_short_daily_returns()
        mean_rank_ic = float(rank_ic_series.mean()) if not rank_ic_series.dropna().empty else np.nan
        icir = self.calculate_icir(rank_ic_series)
        long_short_return = float(ls_return_series.sum()) if not ls_return_series.empty else np.nan

        return {
            "mean_rank_ic": mean_rank_ic,
            "icir": icir,
            "long_short_return": long_short_return,
            "rank_ic_series": rank_ic_series,
            "ls_return_series": ls_return_series,
        }