import json
import datetime
from pathlib import Path
import pandas as pd
from backend.system.config import *
# Lazy imports used inside methods

class SystemController:
    def __init__(self):
        self._data_loader = None
        self.current_model = None
        self.current_config = {}
        # 新增: 缓存全量预测结果 DataFrame (date, stock, score)
        self.cached_predictions = None

    @property
    def data_loader(self):
        if self._data_loader is None:
            print("正在初始化数据加载器...")
            # Lazy import to avoid startup lag
            from backend.system.data_provider import UnifiedDataLoader
            self._data_loader = UnifiedDataLoader(DATA_PATH)
        return self._data_loader

    def run_training(self, 
                     task_type_str: str, 
                     model_type_str: str, 
                     start_date: str = "2016-01-01", 
                     end_date: str = "2022-12-31",
                     label_col: str = "y_ret_5"):
        
        # Local imports
        import pandas as pd
        from backend.system.model_wrapper import ModelWrapper
        
        # 1. 解析枚举类型
        try:
            task_type = TaskType(task_type_str)
            model_type = ModelType(model_type_str)
        except ValueError as e:
            return f"错误: 无效类型。{e}"

        # 2. 加载与预处理训练数据
        print(f"步骤 1: 加载训练数据 ({start_date} 至 {end_date})...")
        df_train = self.data_loader.load_data(start_date, end_date)
        if df_train.empty:
            return "错误: 未找到训练数据。"
            
        print("步骤 2: 预处理训练数据...")
        df_train_proc, feature_cols = self.data_loader.preprocess(task_type=task_type_str)
        
        # 3. 训练模型
        print("步骤 3: 正在训练模型...")
        self.current_model = ModelWrapper(task_type, model_type)
        self.current_model.train(df_train_proc, feature_cols, label_col)
        
        # 4. 保存模型产物
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{task_type_str}_{model_type_str}_{timestamp}.pkl"
        conf_filename = f"config_{timestamp}.json"
        
        save_path = MODEL_SAVE_PATH / model_filename
        self.current_model.save(save_path)
        
        # 5. 自动对测试集 (2023-2026) 进行全量推断
        test_start = "2023-01-01"
        test_end = "2026-12-31" 
        print(f"步骤 4: 正在对测试集进行全量推断 ({test_start} 至 {test_end})...")
        
        # 准备本次运行的独立结果目录 (统一存放 predictions 和 plots)
        run_save_dir = OUTPUT_PATH / "runs" / f"{task_type_str}_{model_type_str}_{timestamp}"
        run_save_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载测试数据
        df_test = self.data_loader.load_data(test_start, test_end)
        
        pred_full_path = None
        
        if not df_test.empty:
            # 预处理测试数据
            print(f"正在为 Task 2 生成标签 (Preprocess)... Task Type: {task_type_str}")
            df_test_proc, _ = self.data_loader.preprocess(task_type=task_type_str)
            
            # DEBUG: 打印列名
            print(f"DEBUG: df_test columns: {list(df_test.columns)}")
            print(f"DEBUG: df_test_proc columns: {list(df_test_proc.columns)}")
            
            # 预测
            scores = self.current_model.predict(df_test_proc, feature_cols)
            
            # 创建结果 DataFrame
            pred_df = df_test[['date', 'stock']].copy()
            pred_df['score'] = scores
            
            # 【修复】将真实标签也一并存入预测结果，便于后续统计
            # 回归: y_ret_5
            if 'y_ret_5' in df_test.columns:
                 pred_df['y_ret_5'] = df_test['y_ret_5']
            
            # 分类: label, r_future_5
            # 优先从处理后的数据中获取 label，因为它是动态构建的
            if 'label' in df_test_proc.columns: 
                 pred_df['label'] = df_test_proc['label']
            elif 'label' in df_test.columns:
                 pred_df['label'] = df_test['label']
            else:
                 print("警告: 无法在 df_test_proc 或 df_test 中找到 label 列")
            
            if 'r_future_5' in df_test.columns:
                 pred_df['r_future_5'] = df_test['r_future_5']
            
            # 本地缓存
            self.cached_predictions = pred_df
            
            # 保存预测结果 (现在存到统一的 run 目录)
            # 同时也保留在 predictions 目录存一份副本方便查找? 
            # 或者直接修改逻辑只存一份。为了整洁，存到 run 目录。
            pred_filename = f"predictions.parquet"
            pred_save_path = run_save_dir / pred_filename
            pred_df.to_parquet(pred_save_path)
            pred_full_path = str(pred_save_path)
            print(f"预测结果已保存至 {pred_save_path}")

            # 6. 调用可视化模块 (传入 run 目录)
            # 必须传入包含 label 的数据 (pred_df 里已经有了)
            # 或者把 label 合并给 df_test
            vis_df = df_test.copy()
            if 'label' in pred_df.columns:
                vis_df['label'] = pred_df['label']
                
            self._run_visualization(task_type_str, vis_df, scores, output_dir=run_save_dir)

        else:
            print("警告: 未找到测试数据，跳过预测。")

        # 保存配置 (包含预测路径)
        config = {
            "task_type": task_type_str,
            "model_type": model_type_str,
            "train_start": start_date,
            "train_end": end_date,
            "features": feature_cols,
            "label": label_col,
            "parameters": self.current_model.get_params(),
            "timestamp": timestamp,
            "model_path": str(save_path),
            "prediction_path": pred_full_path,
            "run_dir": str(run_save_dir)
        }
        with open(CONFIG_SAVE_PATH / conf_filename, 'w') as f:
            json.dump(config, f, indent=4, default=str)

        self.current_config = config
        return f"训练与推断完成！结果已保存至 {run_save_dir}"

    def predict_daily(self, date: str, model_path: str = None, top_k_percent: float = 0.1):
        """
        获取指定日期的预测结果（优先从缓存/文件读取）。
        如果无缓存，尝试回退到手动推断。
        """
        import pandas as pd
        from backend.system.model_wrapper import ModelWrapper

        target_date = pd.to_datetime(date)
        
        # 情况 A: 内存中已有缓存预测结果 (来自最近一次训练)
        if self.cached_predictions is not None:
             print("使用缓存的预测结果...")
             daily_df = self.cached_predictions[self.cached_predictions['date'] == target_date].copy()
             if not daily_df.empty:
                 return self._format_daily_output(daily_df, top_k_percent)
        
        # 情况 B: 尝试从磁盘加载 (如果配置存在)
        if 'prediction_path' in self.current_config and self.current_config['prediction_path']:
             pred_path = Path(self.current_config['prediction_path'])
             if pred_path.exists():
                 print(f"正在从 {pred_path} 加载预测结果...")
                 # 加载全量 parquet 可能会慢，生产环境需优化
                 full_preds = pd.read_parquet(pred_path)
                 self.cached_predictions = full_preds # 缓存以供下次使用
                 daily_df = full_preds[full_preds['date'] == target_date].copy()
                 if not daily_df.empty:
                     return self._format_daily_output(daily_df, top_k_percent)

        # 情况 C: 回退到手动推断 (旧逻辑)
        print("未找到预计算的预测结果，正在为该日期执行手动推断...")
        
        if self.current_model is None:
             # 如果提供了路径，尝试加载模型
             if model_path and Path(model_path).exists():
                # 注意: 如果不知道具体类型，回退通过配置可优化，这里默认回归演示
                self.current_model = ModelWrapper(TaskType.REGRESSION, ModelType.LIGHTGBM) 
                self.current_model.load(model_path)
             else:
                 return None, "错误: 未加载模型且未提供有效路径，无法生成预测。"

        # 加载特定日期数据
        self.data_loader.load_data(date, date)
        if self.data_loader.raw_data is None or self.data_loader.raw_data.empty:
            return None, f"错误: 未找到日期 {date} 的数据"

        # 预处理
        task_type = self.current_config.get('task_type', 'regression') # 默认回退
        df, inferred_cols = self.data_loader.preprocess(task_type=task_type)
        
        # 特征
        feature_cols = self.current_config.get('features', inferred_cols)
        
        # 预测
        try:
            scores = self.current_model.predict(df, feature_cols)
        except Exception as e:
            return None, f"预测错误: {e}"

        daily_df = df[['date', 'stock']].copy()
        
        # 区分命名：回归任务叫 'predicted_return'，分类叫 'score'
        # 但前端暂时只认识 score 用于高亮。
        # 为了满足用户需求 "回归任务...根本只有y_ret_5"，我们这里把预测值命名为 predicted_ret
        # 注意: y_ret_5 是真实值，预测阶段是没有的！这里只能给预测值。
        
        # 检查是否是回归任务 (Fallback for manual inference)
        task_type_str = self.current_config.get('task_type', 'unknown')
        if task_type_str == 'regression':
             col_name = 'y_ret_5'
        else:
             col_name = 'score'
             
        daily_df[col_name] = scores
        
        return self._format_daily_output(daily_df, top_k_percent, sort_col=col_name)

    def _format_daily_output(self, df, top_k_percent, sort_col='score'):
        task_type_str = self.current_config.get('task_type', 'unknown')

        # 如果是回归任务
        if task_type_str == 'regression' or ('y_ret_5' in df.columns and 'label' not in df.columns):
            # 1. 确定预测列和真实列
            # 预测列通常是 'score' (缓存中) 或 'y_ret_5' (手动推断中)
            pred_col = 'score' if 'score' in df.columns else 'y_ret_5'
            true_col = 'y_ret_5_real' # 临时名
            
            # 如果 df 中有真实的 y_ret_5 (来自缓存的 'y_ret_5')
            # 注意: 手动推断时 'y_ret_5' 是预测值。缓存时 'y_ret_5' 是真实值 ('score'是预测)。
            # 如何区分? 缓存模式下 df 同时有 'score' 和 'y_ret_5'。
            has_true_label = False
            if 'score' in df.columns and 'y_ret_5' in df.columns:
                has_true_label = True
                # 重命名真实列，防止冲突
                df = df.rename(columns={'y_ret_5': 'y_ret_5_real'})
                true_col = 'y_ret_5_real'
            
            # 2. 计算 Rank IC
            if has_true_label:
                valid = df[[pred_col, true_col]].dropna()
                if len(valid) > 2:
                    ic = valid[pred_col].rank().corr(valid[true_col].rank())
                    df['Rank IC'] = ic
                else:
                    df['Rank IC'] = np.nan
            
            # 3. 重命名预测列为 'y_ret_5' (满足用户需求: 表头应该是 y_ret_5)
            if pred_col != 'y_ret_5':
                df = df.rename(columns={pred_col: 'y_ret_5'})
            
            # 4. 确保排序列名正确
            sort_col = 'y_ret_5'

        # 如果是分类任务
        elif task_type_str == 'classification' or 'score' in df.columns:
            sort_col = 'score'
            # 1. 检查是否有真实标签 'label' (0, 1, 2)
            if 'label' in df.columns:
                # 2. "是否真的涨了" -> 改为具体的 "Real State"
                # Map 1 -> 'Early Up', 2 -> 'Mid Up', 0 -> 'Flat/Down'
                label_map = {0: 'Flat/Down', 1: 'Early Up', 2: 'Mid Up'}
                df['Real State'] = df['label'].map(label_map)
                
                # 3. "是否预测中" (Are We Right?)
                # 逻辑: 只要在 Top K 列表中 (都是高分股,预期涨) 且真的涨了(1或2)，就算预测中
                is_up = df['label'].isin([1, 2])
                df['Are We Right?'] = is_up.map({True: 'Yes', False: 'No'})
            
            # 清理不必要的回归列
            cols_to_drop = ['y_ret_5', 'y_ret_5_real', 'Rank IC', 'prediction', 'Is Real Up?', '是否真的涨了']
            df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

            # 确保 score 列不改名
            # 如果没有 'score' 但有其他预测列名，可能需要重命名
            # 手动推断时可能只有 'score'
            if 'score' not in df.columns:
                 # 寻找预测列
                 pass # 假设一定有 score

        # 排序
        if sort_col in df.columns:
            df = df.sort_values(by=sort_col, ascending=False)
        
        # 截取 Top K
        n_top = int(max(1, len(df) * top_k_percent))
        return df.head(n_top), "成功"

    def _run_visualization(self, task_type_str, df_test, scores, output_dir: Path):
        """
        根据任务类型调用对应的 Visualizer。
        """
        print(f"\n步骤 5: 生成可视化图表 (任务类型: {task_type_str})...")

        # 准备数据: 创建包含预测和标签的 DataFrame
        vis_df = df_test.copy()
        
        # 准备输出目录
        plot_dir = output_dir
        print(f"图表保存目录: {plot_dir}")

        try:
            if task_type_str == "regression":
                # --- Task 1 可视化逻辑 ---
                vis_df['prediction'] = scores
                
                # 检查必要列：y_ret_5
                target_col = 'y_ret_5'
                if target_col not in vis_df.columns:
                    print(f"警告: 测试数据缺少收益率标签列 '{target_col}'，无法生成 Task 1 回归评估图表。")
                    return

                # 延迟导入 Task 1 模块
                from backend.baseline_regression_model.evaluator import ModelEvaluator
                from backend.baseline_regression_model.task1visualizer import Task1Visualizer

                print("正在调用 Task 1 ModelEvaluator...")
                # ModelEvaluator 内部有 dropna 逻辑
                evaluator = ModelEvaluator(vis_df, target_col=target_col)
                report = evaluator.generate_evaluation_report()
                
                print("正在调用 Task 1 Visualizer...")
                visualizer = Task1Visualizer(report, save_dir=str(plot_dir))
                
                # 保存评估报告
                report_df = pd.DataFrame([report])
                report_path = plot_dir / "evaluation_report.csv"
                report_df.to_csv(report_path, index=False)
                
                # 1. 绘制 Rank IC (需要 rolling_window，默认 20)
                visualizer.plot_rank_ic_ts()
                
                # 1.5 绘制 Rank IC 分布直方图
                visualizer.plot_rank_ic_distribution()
                
                # 2. 绘制累计收益曲线 (需先计算基准)
                # calculate_benchmarks 需要 df 有 prediction 和 y_ret_5
                visualizer.calculate_benchmarks(vis_df)
                visualizer.plot_cumulative_return() 
                print("Task 1 可视化生成完毕。")

            elif task_type_str == "classification":
                # --- Task 2 可视化逻辑 ---
                vis_df['score'] = scores
                
                # 检查必要列: label
                if 'label' not in vis_df.columns:
                     print("警告: 测试数据缺少 label 列，无法进行 Task 2 精度评估。")
                     return
                
                from backend.situation_awareness_classification_model.evaluator import SituationEvaluator
                from backend.situation_awareness_classification_model.task2visualizer import Task2Visualizer

                print("正在调用 Task 2 Evaluator...")
                evaluator = SituationEvaluator()
                visualizer = Task2Visualizer(output_dir=str(plot_dir))
                
                # 精度评估 (Pie Chart)
                metrics = evaluator.evaluate_precision(vis_df, top_k_list=[0.05, 0.1])
                visualizer.plot_precision_pie_charts(metrics)
                
                # 保存评估指标
                metrics_df = pd.DataFrame([metrics])
                metrics_path = plot_dir / "evaluation_metrics.csv"
                metrics_df.to_csv(metrics_path, index=False)
                
                # 收益评估 (Long-Short Curve)
                # 优先寻找 r_future_5, 其次 y_ret_5
                target_return_col = 'r_future_5' if 'r_future_5' in vis_df.columns else 'y_ret_5'
                
                if target_return_col in vis_df.columns:
                    print(f"计算多空组合收益 (Target: {target_return_col})...")
                    long_short_rets = evaluator.calculate_long_short_daily_returns(vis_df, target_col=target_return_col, top_k=0.05)
                    visualizer.plot_long_short_returns(long_short_rets)
                    
                    # 保存多空收益序列
                    long_short_path = plot_dir / "long_short_returns.csv"
                    # 转为 DataFrame 并命名表头，避免 Unnamed: 0 问题
                    ls_df_to_save = long_short_rets.to_frame(name='ls_return')
                    ls_df_to_save.to_csv(long_short_path, index_label='date')
                else:
                    print("警告: 缺少收益率列 (r_future_5 或 y_ret_5)，跳过 Task 2 收益曲线。")
                
                print("Task 2 可视化生成完毕。")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"可视化过程发生错误: {e}")

    def get_eval_metrics(self):
        pass
