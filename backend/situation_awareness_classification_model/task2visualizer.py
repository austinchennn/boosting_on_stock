import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

class Task2Visualizer:
    """
    负责 Task 2 态势识别模型的评估结果可视化。
    包含：
    1. Precision (Top 5% vs Top 10%) 对比图
    2. 多空组合累计收益曲线
    """

    def __init__(self, output_dir="."):
        self.output_dir = output_dir
        # 如果目录不存在，自动创建
        if output_dir and output_dir != ".":
             os.makedirs(output_dir, exist_ok=True)
             
        # 设置中文字体 (尝试兼容 Mac 和 Windows)
        # 注意: 生产环境可能需更稳健的字体检测
        import platform
        sys_str = platform.system()
        if sys_str == "Windows":
             plt.rcParams['font.sans-serif'] = ['SimHei'] # Windows 黑体
        elif sys_str == "Darwin":
             plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC'] # Mac 字体
        else:
             plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei'] # Linux 备选
             
        plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

    def plot_precision_pie_charts(self, metrics: dict, filename: str = "precision_pie.png"):
        """
        绘制 Top 5% 和 Top 10% 命中率的对比饼图。
        展示：预测上涨股票中，【实际上涨】 vs 【实际未涨】的比例。
        命中率 = Precision (从 metrics 获取)
        """
        # 准备数据
        # metrics 包含 top_5%_hit_rate, top_10%_hit_rate
        # Hit Rate = Precision = 实际上涨 / 预测总数
        
        # 1. Top 5% 数据
        hit_rate_5 = metrics.get('top_5%_hit_rate', 0)
        # 饼图数据: [命中部分, 未命中部分]
        sizes_5 = [hit_rate_5, 1 - hit_rate_5]
        labels_5 = [f'命中 (Hit)\n{hit_rate_5:.1%}', f'未中 (Miss)\n{1-hit_rate_5:.1%}']
        
        # 2. Top 10% 数据
        hit_rate_10 = metrics.get('top_10%_hit_rate', 0)
        sizes_10 = [hit_rate_10, 1 - hit_rate_10]
        labels_10 = [f'命中 (Hit)\n{hit_rate_10:.1%}', f'未中 (Miss)\n{1-hit_rate_10:.1%}']
        
        # 绘图设置
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7)) # 加宽
        colors = ['#ff9999','#66b3ff'] # 红涨蓝跌
        explode = (0.05, 0)  # 突出显示命中部分
        
        # 子图 1: Top 5%
        ax1.pie(sizes_5, labels=labels_5, autopct=None, startangle=90, colors=colors, explode=explode, shadow=True)
        ax1.set_title(f'Top 5% 预测精度 (Precision)', fontsize=14, fontweight='bold')
        
        # 子图 2: Top 10%
        ax2.pie(sizes_10, labels=labels_10, autopct=None, startangle=90, colors=colors, explode=explode, shadow=True)
        ax2.set_title(f'Top 10% 预测精度 (Precision)', fontsize=14, fontweight='bold')
        
        plt.suptitle("态势识别模型精度对比 (Precision of Top K)", fontsize=18)
        
        # 保存或显示
        if self.output_dir:
            save_path = os.path.join(self.output_dir, filename)
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"图表已保存至: {save_path}")
        else:
            plt.show()
        plt.close()

    def plot_long_short_returns(self, daily_returns: pd.Series, filename: str = "ls_returns.png"):
        """
        绘制多空组合累计收益曲线 (Cumulative Long-Short Returns)。
        逻辑参考 Task 1 的可视化。
        
        参数:
        - daily_returns: 按日期索引的每日收益差序列。
        """
        if daily_returns.empty:
            print("警告: 无多空收益数据，跳过绘图。")
            return
            
        # 计算累计收益
        # 假设每日收益可以直接累加 (对数收益) 或 累乘 (简单收益)
        # 这里沿用 Task 1 逻辑，通常做累计求和
        cum_returns = daily_returns.cumsum()
        
        plt.figure(figsize=(12, 6))
        
        # 绘制主收益曲线
        plt.plot(cum_returns.index, cum_returns.values, label='累计多空收益 (Cum L-S Return)', color='#d62728', linewidth=2.5)
        
        # 添加填充区域 (Area chart style)
        plt.fill_between(cum_returns.index, cum_returns.values, 0, color='#d62728', alpha=0.1)
        
        # 添加辅助线 y=0
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.8)
        
        # 样式优化
        plt.title('多空组合累计收益 (Long Top 5% - Short Bottom 5%)', fontsize=16, fontweight='bold')
        plt.xlabel('日期 (Date)', fontsize=12)
        plt.ylabel('累计收益率 (Cumulative Return)', fontsize=12)
        plt.legend(loc='upper left', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        if self.output_dir:
            save_path = os.path.join(self.output_dir, filename)
            plt.savefig(save_path, dpi=150)
            print(f"收益曲线已保存至: {save_path}")
        else:
            plt.show()
        plt.close()