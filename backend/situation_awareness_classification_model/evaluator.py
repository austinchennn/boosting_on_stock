import pandas as pd
import numpy as np

class SituationEvaluator:
    """
    态势识别模型专用评估器。
    """

    def calculate_custom_score(self, prob_df: pd.DataFrame) -> pd.Series:
        """
        根据手册 2.6 节构建评分公式：
        score = P_1 + 0.5 * P_2
        其中 P1 为“初涨”概率，P2 为“中涨”概率。
        
        参数:
        - prob_df: 必须包含 prob_1 和 prob_2 两列。
          prob_1: 预测为'初涨'的概率
          prob_2: 预测为'中涨'的概率
          
        返回:
        - pd.Series: 每个样本的自定义评分
        """
        # 手册 2.6: P1 + 0.5 * P2
        if 'prob_1' not in prob_df.columns or 'prob_2' not in prob_df.columns:
            raise ValueError("概率表必须包含 'prob_1' 和 'prob_2' 列。")
            
        score = prob_df['prob_1'] + 0.5 * prob_df['prob_2']
        return score

    def select_top_stocks(self, df: pd.DataFrame, top_k: float = 0.05) -> pd.DataFrame:
        """
        根据 score 对股票进行每日横截面排序，并筛选出排名靠前的股票。
        
        参数:
        - df: 包含 'date', 'score' 的 DataFrame。
        - top_k: 筛选比例，默认 0.05 (Top 5%)。
        
        返回:
        - pd.DataFrame: 筛选后的 Top K 股票集合。
        """
        if 'score' not in df.columns:
            raise ValueError("输入 DataFrame 必须包含 'score' 列。")
            
        # 如果有日期列，则进行每日横截面排序
        if 'date' in df.columns:
            # 定义每日筛选逻辑
            # 使用 groupby().apply() 可能会慢，但对于评估目的通常可以接受
            # 优化：可以考虑先排序然后根据 rank 筛选，但这需要处理每个日期的不同总数
            
            def get_daily_top(group):
                n = int(np.ceil(len(group) * top_k)) # 使用 ceil 确保至少选出一个
                if n < 1: n = 1 
                return group.nlargest(n, 'score')
            
            # 必须设置 group_keys=False 避免索引层级增加
            return df.groupby('date', group_keys=False).apply(get_daily_top)
            
        else:
            # 如果没有日期列，则对整体数据进行排序筛选
            n = int(np.ceil(len(df) * top_k))
            return df.nlargest(n, 'score')

    def evaluate_precision(self, df: pd.DataFrame, top_k_list: list) -> dict:
        """
        计算手册 2.7 要求的指标：Top K 命中率 (Precision of Top K)。
        
        逻辑定义：
        1. 预测上涨股票 ('Predicted Up'): 根据评分筛选出的 Top K% 股票。
        2. 实际上涨股票 ('Actually Up'): 真实标签为 1 (初涨) 或 2 (中涨) 的股票。
        3. 命中率 = (预测上涨 ∩ 实际上涨) / 预测上涨
        
        参数:
        - df: 包含 'date', 'score', 'label' 的 DataFrame。
        - top_k_list: 需要评估的 Top K 比例列表，默认 [0.05, 0.1]。
        
        返回:
        - dict: 包含各 Top K 的命中率及统计计数。
        """
        if 'label' not in df.columns:
             raise ValueError("评估需要真实标签 'label' 列。")

        metrics = {}
        
        for k in top_k_list:
            # 1. 找出预测上涨股票 (Predict Top K)
            # 即根据模型评分选出的 Top K% 股票
            top_df = self.select_top_stocks(df, top_k=k)
            
            selected_count = len(top_df)
            if selected_count == 0:
                print(f"警告: Top {k} 筛选结果为空。")
                metrics[f'top_{int(k*100)}%_hit_rate'] = 0.0
                continue
            
            # 2. 找出这些票中实际上涨的 (Label > 0)
            # 注意：这里只统计【选中的 Top K 股票中】实际上涨的数量，而非全市场
            # 这是 Precision 的定义，不是 Recall
            # Label 1: 初涨, Label 2: 中涨 -> 两者都算“上涨”
            hit_mask = top_df['label'] > 0
            hit_count = hit_mask.sum()
            
            # 3. 计算比率 (Precision)
            # Precision = (Top K 中实际涨的) / (Top K 总数)
            hit_rate = hit_count / selected_count
            
            # 额外指标：初涨命中率 (Top K 中有多少是 Label 1)
            early_rise_count = (top_df['label'] == 1).sum()
            early_rise_rate = early_rise_count / selected_count

            # 记录详细指标
            k_pct = int(k*100)
            metrics[f'top_{k_pct}%_hit_rate'] = hit_rate
            metrics[f'top_{k_pct}%_count_selected'] = selected_count # 预测上涨总数
            metrics[f'top_{k_pct}%_count_hit'] = hit_count           # 实际上涨总数
            metrics[f'top_{k_pct}%_early_rise_rate'] = early_rise_rate
            
        return metrics

    def calculate_long_short_daily_returns(self, df: pd.DataFrame, target_col: str, top_k: float = 0.05) -> pd.Series:
        """
        计算每日多空组合收益（Situation Awareness Task 逻辑）。
        逻辑：
        - 多头 (Long): 模型预测为"上涨" (Top K score) 的股票。
        - 空头 (Short): 模型预测为"非上涨" (Bottom K score) 的股票。
        - 收益: Long_Mean - Short_Mean (每日)
        
        参数:
        - df: 包含 'date', 'score' 和 真实收益列 (target_col) 的 DataFrame。
        - target_col: 真实收益率列名 (例如 'y_ret_5')。
        - top_k: 多/空头的筛选比例 (默认 5%)。
        
        返回:
        - pd.Series: 按日期索引的每日多空收益差。
        """
        if target_col not in df.columns:
            raise ValueError(f"缺少真实收益列: {target_col}")

        daily_returns = {}
        
        for date, group in df.groupby('date'):
            # 移除缺失值
            valid = group.dropna(subset=['score', target_col])
            n = len(valid)
            if n < 2: 
                continue
                
            # 确定多空头数量
            n_select = max(int(np.ceil(n * top_k)), 1)
            
            # 多头: Score 最高的 Top K
            long_ret = valid.nlargest(n_select, 'score')[target_col].mean()
            
            # 空头: Score 最低的 Bottom K (或者也可以定义为 score 最得低的)
            short_ret = valid.nsmallest(n_select, 'score')[target_col].mean()
            
            daily_returns[date] = long_ret - short_ret
            
        return pd.Series(daily_returns, name='ls_return').sort_index()