import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from backend.system.system_controller import SystemController

def main():
    controller = SystemController()

    print("\n=======================================================")
    print("      任务3: 模型系统 (CLI 版本) v1.0         ")
    print("      注意: 训练数据集已固定 (2016-2022)     ")
    print("=======================================================")
    
    while True:
        print("\n请选择操作:")
        print("1. 训练模型 (Train Model) - 训练并生成2023-2026全量预测")
        print("2. 每日预测结果查询 (Predict Daily) - 查看某日推荐股票")
        print("3. 退出 (Exit)")
        
        choice = input("请输入选项数字: ").strip()
        
        if choice == '1':
            print("\n-------------------------------------------------------")
            print(">>> 模型训练配置")
            print("-------------------------------------------------------")
            
            print("\n请选择预测目标:")
            print("1. 收益率预测 (Regression) - [任务 1]")
            print("2. 态势识别 (Classification) - [任务 2]")
            
            target_choice = input("请输入选项数字 [默认: 1]: ").strip() or "1"
            
            if target_choice == '1':
                task_type_str = "regression"
                label_col = "y_ret_5"
                print(f"已选择: 收益率预测 (目标列: {label_col})")
            elif target_choice == '2':
                task_type_str = "classification"
                label_col = "label"
                print(f"已选择: 态势识别 (目标列: {label_col})")
            else:
                print("无效选择，默认使用收益率预测")
                task_type_str = "regression"
                label_col = "y_ret_5"

            model_type_str = input("\n请输入模型类型 (lightgbm/xgboost) [默认: lightgbm]: ").strip().lower() or "lightgbm"
            
            print("\n注意：本项目规定训练集固定为 2016-01-01 至 2022-12-31")
            
            # 固定日期范围
            start_date = "2016-01-01"
            end_date = "2022-12-31"
            
            print(f"\n正在启动训练 (任务: {task_type_str}, 模型: {model_type_str}, 范围: {start_date} -> {end_date})...")
            # 注意: controller.run_training 参数顺序需与定义一致
            msg = controller.run_training(task_type_str, model_type_str, start_date, end_date, label_col)
            print(msg)
            
        elif choice == '2':
            print("\n-------------------------------------------------------")
            print(">>> 每日预测查询")
            print("-------------------------------------------------------")
            
            # 检查模型状态
            if controller.cached_predictions is None and controller.current_model is None:
                print("警告: 当前内存中没有已加载的预测结果或模型。")
                print("通常建议先执行步骤 '1. 训练模型' 来生成全量预测结果。")
                input("按回车键继续...")
            else:   
                print(f"当前任务类型: {controller.current_config.get('task_type', '未知')}")
                
            print("\n请输入查询日期 (测试集范围通常为 2023-01-01 至 2026-12-31)")
            date = input("日期 (YYYY-MM-DD) [默认: 2023-01-05]: ").strip() or "2023-01-05"
            
            top_k_input = input("Top K 百分比 (例如 0.05 代表前5%) [默认: 0.05]: ").strip() or "0.05"
            try:
                top_k = float(top_k_input)
            except ValueError:
                top_k = 0.05
            
            print(f"\n正在获取 {date} 的预测结果...")
            # 修改: top_k_percent 设为 1.0 (100%) 以获取该日所有股票
            df_res, msg = controller.predict_daily(date, model_path=None, top_k_percent=1.0)
            
            if df_res is not None and not df_res.empty:
                total_stocks = len(df_res)
                print(f"\n查询成功! {date} 共有 {total_stocks} 只股票的预测结果。")
                print("-" * 50)
                
                # 仅展示前 Top K (用户指定) 和 后 5 (展示全貌)
                n_display = int(max(1, total_stocks * top_k))
                print(f"展示前 {top_k*100}% ({n_display}只) 股票:")
                print(df_res[['date', 'stock', 'score']].head(n_display).to_string(index=False))
                
                print("\n...... (中间省略) ......")
                print("\n排名靠后的 5 只股票:")
                print(df_res[['date', 'stock', 'score']].tail(5).to_string(index=False))
                print("-" * 50)
                
                save_option = input("是否保存该日【所有股票】结果到CSV? (y/n) [默认: n]: ").strip().lower()
                if save_option == 'y':
                    save_file = f"pred_{date}_full.csv"
                    df_res.to_csv(save_file, index=False)
                    print(f"已保存所有结果至 {save_file}")
            else:
                print(f"查询失败: {msg}")
                
        elif choice == '3':
            print("正在退出...")
            break
        else:
            print("无效选项，请重新输入。")

if __name__ == "__main__":
    main()
