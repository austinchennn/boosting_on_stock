import streamlit as st
import sys
import os
import pandas as pd
from pathlib import Path
import time
import glob

# Ensure backend can be imported
# Assumes running from project root: streamlit run frontend/app.py
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

try:
    from backend.system.system_controller import SystemController
    from backend.system.config import OUTPUT_PATH, TaskType, ModelType
except ImportError as e:
    st.error(f"Import Error: {e}. Please run streamlit from the project root directory.")
    st.stop()

# Page Configuration
st.set_page_config(
    page_title="量化模型训练与分析系统",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if 'controller' not in st.session_state:
    st.session_state.controller = SystemController()

if 'training_complete' not in st.session_state:
    st.session_state.training_complete = False

if 'current_run_dir' not in st.session_state:
    st.session_state.current_run_dir = None

def render_dashboard():
    st.title("量化模型训练与分析系统")
    st.markdown("### 系统实现与展示")

    # Sidebar: Configuration
    with st.sidebar:
        st.header("配置选项")
        
        task_type_label = "任务类型 (回归/分类)"
        task_type = st.selectbox(
            task_type_label,
            ["regression", "classification"],
            format_func=lambda x: "回归任务 (预测未来收益)" if x == "regression" else "分类任务 (态势识别)",
            help="选择回归 (Regression) 或分类 (Classification) 任务"
        )
        
        model_type_label = "模型类型"
        model_type = st.selectbox(
            model_type_label,
            ["lightgbm", "xgboost"],
            help="选择机器学习算法"
        )
        
        st.divider()
        st.markdown("**训练周期:**\n\n`2016-01-01` 至 `2022-12-31` (固定)")
        
        # Train Button
        if st.button("开始训练", type="primary"):
            with st.spinner("正在训练中... 请稍候。"):
                # Capture standard output to show logs
                import io
                from contextlib import redirect_stdout
                
                log_buffer = io.StringIO()
                with redirect_stdout(log_buffer):
                    label_col = "y_ret_5" if task_type == "regression" else "label"
                    
                    # Store start time
                    start_time = time.time()
                    
                    # Run Training
                    msg = st.session_state.controller.run_training(
                        task_type_str=task_type,
                        model_type_str=model_type,
                        start_date="2016-01-01",
                        end_date="2022-12-31",
                        label_col=label_col
                    )
                    
                # Update Session State
                st.session_state.training_complete = True
                # Parse output path from message or use latest run dir logic
                # For simplicity, we search for the latest run dir
                # Assuming standard naming convention in SystemController
                # We can also get it from controller.current_config['run_dir']
                if st.session_state.controller.current_config:
                    st.session_state.current_run_dir = st.session_state.controller.current_config.get('run_dir')
                
                st.success(f"训练完成！耗时: {time.time() - start_time:.2f}秒")
                st.text_area("训练日志", log_buffer.getvalue(), height=200)

    # Main Area: Results & Analysis
    if st.session_state.training_complete and st.session_state.current_run_dir:
        run_dir = Path(st.session_state.current_run_dir)
        
        tab1, tab2 = st.tabs(["可视化分析", "每日预测查询"])
        
        # Tab 1: Visualization
        with tab1:
            st.header("模型性能评估")
            
            # --- Load Metrics Tables ---
            # 1. Task 1 Report
            report_path = run_dir / "evaluation_report.csv"
            if report_path.exists():
                st.subheader("回归任务评估报告")
                st.dataframe(pd.read_csv(report_path))
            
            # 2. Task 2 Metrics
            metrics_path = run_dir / "evaluation_metrics.csv"
            if metrics_path.exists():
                st.subheader("分类任务评估指标 (Top K Precision)")
                st.dataframe(pd.read_csv(metrics_path))
                
            # 3. Task 2 Long-Short Returns
            ls_path = run_dir / "long_short_returns.csv"
            if ls_path.exists():
                st.subheader("多空组合收益数据预览")
                st.dataframe(pd.read_csv(ls_path).head()) 
                st.download_button(
                    label="下载多空收益数据 (CSV)",
                    data=open(ls_path, "rb").read(),
                    file_name="long_short_returns.csv",
                    mime="text/csv"
                )

            st.divider()
            st.header("可视化图表")
            
            # Look for png files in run_dir
            png_files = list(run_dir.glob("*.png"))
            
            if png_files:
                for png_path in png_files:
                    st.image(str(png_path), caption=png_path.name)
            else:
                st.warning("输出目录中未找到可视化图表。")
                
            # Full Predictions Download
            pred_file = list(run_dir.glob("*.parquet"))
            if pred_file:
                 st.download_button(
                     label="下载完整预测结果 (Parquet)",
                     data=open(pred_file[0], "rb").read(),
                     file_name=pred_file[0].name,
                     mime="application/octet-stream"
                 )

        # Tab 2: Daily Predictions
        with tab2:
            st.header("每日个股预测查询")
            
            col1, col2 = st.columns(2)
            with col1:
                query_date = st.date_input(
                    "选择日期",
                    value=pd.to_datetime("2023-01-05"),
                    min_value=pd.to_datetime("2023-01-01"),
                    max_value=pd.to_datetime("2026-12-31")
                )
            
            with col2:
                top_k_option = st.selectbox(
                    "筛选范围",
                    ["前 5%", "前 10%", "全部股票 (100%)"]
                )
                
            # Map selection to float
            top_k_map = {
                "前 5%": 0.05,
                "前 10%": 0.10,
                "全部股票 (100%)": 1.0
            }
            top_k_val = top_k_map[top_k_option]
            
            if st.button("开始查询"):
                date_str = query_date.strftime("%Y-%m-%d")
                df_res, status = st.session_state.controller.predict_daily(
                    date=date_str,
                    top_k_percent=top_k_val
                )
                
                if df_res is not None and not df_res.empty:
                    st.write(f"### {date_str} 预测结果")
                    
                    # 确定高亮列
                    # 如果结果中有 'y_ret_5' (回归任务), 则按其高亮
                    # 如果有 'score' (分类任务), 则按其高亮
                    highlight_col = 'score'
                    if 'y_ret_5' in df_res.columns:
                        highlight_col = 'y_ret_5'
                    elif 'score' in df_res.columns:
                        highlight_col = 'score'
                        
                    try:
                        st.dataframe(df_res.style.highlight_max(axis=0, subset=[highlight_col]))
                    except Exception:
                        # Fallback if styling fails
                        st.dataframe(df_res)
                    
                    # CSV Download for Daily Data
                    csv = df_res.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label=f"下载 {date_str} 数据 (CSV)",
                        data=csv,
                        file_name=f"pred_{date_str}_top{int(top_k_val*100)}.csv",
                        mime="text/csv",
                    )
                else:
                    st.error(f"未找到 {date_str} 的数据或预测失败。")
    
    else:
        st.info("请先在侧边栏配置并训练模型。")

def render_introduction():
    st.title("任务与模型详细说明")
    
    st.header("回归任务 (预测未来收益)")
    st.markdown("""
    该任务旨在通过机器学习模型捕捉股价在短周期内的变化幅度。
    
    - **预测目标**: 个股未来 5 日收益率 (`y_ret_5`)。
    - **特征逻辑**: 使用量价因子、技术指标等作为输入特征。
    - **训练逻辑**: 
        - 采用 LightGBM 或 XGBoost 等梯度提升树模型进行回归训练。
        - 模型通过最小化预测收益率与真实收益率之间的误差 (如 MSE) 来优化参数。
        - 最终输出为具体的收益率预测值，用于排序选股。
    """)
    
    st.divider()
    
    st.header("分类任务 (态势识别)")
    st.markdown("""
    该任务旨在识别个股当前所处的市场态势（如初涨、中涨、滞涨/下跌），从而构建更加稳健的评分系统。
    
    - **预测目标**: 识别个股所属的态势类别 (0, 1, 2)。
    - **类别定义**:
        - **类别 0**: 下跌、震荡或滞涨。
        - **类别 1 (P1)**: 初涨 (Initial Rise) —— 股价刚开始启动。
        - **类别 2 (P2)**: 中涨 (Intermediate Rise) —— 股价处于上升通道中段。
    
    - **模型输出与评分构建**:
        模型输出每只股票属于各类的概率 $P_0, P_1, P_2$。最终评分 ($Score$) 由以下公式计算：
        $$
        Score = P_1 + 0.5 \\times P_2
        $$
        *解释：赋予“初涨”态势更高的权重，以捕捉启动阶段的个股；“中涨”态势给予一半权重以维持趋势跟踪。*
            
    - **训练逻辑**: 
        - 使用多分类 (Multi-class Classification) 目标函数 (如 `multi_logloss`)。
        - 模型学习不同市场特征与股价态势之间的非线性映射关系。
    """)

def main():
    st.sidebar.title("控制台导航")
    page = st.sidebar.radio("选择页面", ["模型训练与展示", "任务与模型说明"])
    
    if page == "模型训练与展示":
        render_dashboard()
    else:
        render_introduction()

if __name__ == "__main__":
    main()
