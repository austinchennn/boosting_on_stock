import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

class Task1Visualizer:
    """
    任务一可视化器：负责生成 Rank IC 曲线和策略累计收益曲线。
    """

    def __init__(self, eval_report: dict, save_dir: str = "plots"):
        """
        初始化可视化器。
        
        参数:
        - eval_report: ModelEvaluator.generate_evaluation_report() 返回的字典。
        - save_dir: 图表保存目录。
        """
        self.rank_ic_series = eval_report.get("rank_ic_series")
        self.ls_return_series = eval_report.get("ls_return_series") # 需确保 evaluator 输出了每日收益序列
        self.market_avg_series = None
        self.top_return_series = None
        self.bottom_return_series = None
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 字体设置
        import platform
        sys_str = platform.system()
        if sys_str == "Windows":
             plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        elif sys_str == "Darwin":
             plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC']
        else:
             plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
        
        plt.rcParams['axes.unicode_minus'] = False # 负号

    def _validate_series(self, series: pd.Series, series_name: str) -> pd.Series:
        if series is None:
            raise ValueError(f"{series_name} 不存在，请检查评估报告内容")

        clean_series = pd.Series(series).dropna()
        if clean_series.empty:
            raise ValueError(f"{series_name} 为空，无法绘图")

        clean_series.index = pd.to_datetime(clean_series.index)
        clean_series = clean_series.sort_index()
        return clean_series

    def plot_rank_ic_ts(self, rolling_window: int = 20):
        """
        绘制 Rank IC 时间序列曲线。
        对应要求 4.1: Rank IC时间序列曲线 (用于验证排序能力) [cite: 104]。
        同时展示原始 IC 和 20 日移动平均线。
        """
        rank_ic_series = self._validate_series(self.rank_ic_series, "rank_ic_series")
        rolling_ic = rank_ic_series.rolling(window=rolling_window, min_periods=1).mean()

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(rank_ic_series.index, rank_ic_series.values, label="Daily Rank IC", linewidth=1.0, alpha=0.7)
        ax.plot(rolling_ic.index, rolling_ic.values, label=f"{rolling_window}-Day MA", linewidth=2.0)
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
        ax.set_title("Rank IC Time Series")
        ax.set_xlabel("Date")
        ax.set_ylabel("Rank IC")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()

        save_path = self.save_dir / "rank_ic_timeseries.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def plot_rank_ic_distribution(self):
        """
        绘制 Rank IC 分布直方图。
        展示 IC 值的分布情况 (均值、偏度等特征)。
        """
        rank_ic_series = self._validate_series(self.rank_ic_series, "rank_ic_series")
        ic_mean = rank_ic_series.mean()
        ic_std = rank_ic_series.std()

        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 直方图
        ax.hist(rank_ic_series.dropna(), bins=30, color='skyblue', edgecolor='black', alpha=0.7, density=True, label='Frequency')
        
        # 密度曲线 (KDE)
        try:
            rank_ic_series.plot(kind='kde', ax=ax, color='red', linewidth=2, label='KDE')
        except:
             pass # 数据过少可能失败

        # 标注均值线
        ax.axvline(ic_mean, color='green', linestyle='--', linewidth=2, label=f'Mean IC: {ic_mean:.3f}')
        ax.axvline(0, color='gray', linestyle='-', linewidth=1)

        ax.set_title(f"Rank IC Distribution (Mean={ic_mean:.3f}, Std={ic_std:.3f})")
        ax.set_xlabel("Rank IC Value")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(alpha=0.3)
        
        fig.tight_layout()
        save_path = self.save_dir / "rank_ic_distribution.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path
    
    def calculate_benchmarks(self, test_df: pd.DataFrame):
        """
        计算对比基准序列。
        
        逻辑：
        1. 计算每日全市场 y_ret_5 的平均值作为 Market_Avg。
        2. 计算每日 Top 10% 和 Bottom 10% 的平均收益。
        """
        required_cols = {"date", "prediction", "y_ret_5"}
        missing_cols = required_cols.difference(test_df.columns)
        if missing_cols:
            missing_cols_str = ", ".join(sorted(missing_cols))
            raise ValueError(f"test_df 缺少基准计算所需列: {missing_cols_str}")

        benchmark_df = test_df.loc[:, ["date", "prediction", "y_ret_5"]].copy()
        if not pd.api.types.is_datetime64_any_dtype(benchmark_df["date"]):
            benchmark_df["date"] = pd.to_datetime(benchmark_df["date"], errors="coerce")

        benchmark_df = benchmark_df.dropna(subset=["date", "prediction", "y_ret_5"])
        if benchmark_df.empty:
            raise ValueError("test_df 为空，无法计算对比基准序列")

        market_avg_dict = {}
        top_return_dict = {}
        bottom_return_dict = {}

        for date, group in benchmark_df.groupby("date", sort=True):
            market_avg_dict[date] = group["y_ret_5"].mean()

            bucket_size = max(int(np.ceil(len(group) * 0.1)), 1)
            top_return_dict[date] = group.nlargest(bucket_size, "prediction")["y_ret_5"].mean()
            bottom_return_dict[date] = group.nsmallest(bucket_size, "prediction")["y_ret_5"].mean()

        self.market_avg_series = pd.Series(market_avg_dict, dtype=float, name="Market_Avg").sort_index()
        self.top_return_series = pd.Series(top_return_dict, dtype=float, name="Top_10pct").sort_index()
        self.bottom_return_series = pd.Series(bottom_return_dict, dtype=float, name="Bottom_10pct").sort_index()

        return {
            "market_avg_series": self.market_avg_series,
            "top_return_series": self.top_return_series,
            "bottom_return_series": self.bottom_return_series,
        }

    def plot_cumulative_return(self):
        """
        绘制简单策略累计收益曲线。
        对应要求 4.1: 简单策略累计收益曲线 (用于验证选股有效性) [cite: 105]。
        计算公式参考 4.2: 每日选取 Top 10% 股票并等权构建组合 [cite: 107, 108]。
        
        修改说明：采用算术累计收益（cumsum），避免高频调仓下的复利爆炸问题。
        """
        ls_return_series = self._validate_series(self.ls_return_series, "ls_return_series")
        # 改为算术累加
        cumulative_return = ls_return_series.cumsum()

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(cumulative_return.index, cumulative_return.values, color="tab:green", linewidth=2.0, label="Long-Short (Arithmetic)")

        if self.market_avg_series is not None:
            market_avg_series = self._validate_series(self.market_avg_series, "market_avg_series")
            # 同样改为算术累加
            market_cumulative_return = market_avg_series.cumsum()
            ax.plot(
                market_cumulative_return.index,
                market_cumulative_return.values,
                color="gray",
                linewidth=1.8,
                label="Market_Avg (Arithmetic)",
            )

        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
        ax.set_title("Long-Short Cumulative Return (Arithmetic)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Return (Sum)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()

        save_path = self.save_dir / "long_short_cumulative_return.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def plot_ic_distribution(self):
        """
        绘制 IC 值的分布直方图。
        用于辅助分析 IC 的稳定性 (ICIR) [cite: 34]。
        """
        rank_ic_series = self._validate_series(self.rank_ic_series, "rank_ic_series")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(rank_ic_series.values, bins=min(30, max(len(rank_ic_series) // 5, 10)), color="tab:blue", alpha=0.75, edgecolor="black")
        ax.axvline(rank_ic_series.mean(), color="tab:red", linestyle="--", linewidth=1.5, label="Mean Rank IC")
        ax.set_title("Rank IC Distribution")
        ax.set_xlabel("Rank IC")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()

        save_path = self.save_dir / "rank_ic_distribution.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def create_dashboard(self):
        """
        综合展示图表。
        对应任务五要求: 在系统交互界面展示 IC 曲线和策略收益曲线 [cite: 121, 122]。
        """
        pass