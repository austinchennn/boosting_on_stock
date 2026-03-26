from enum import Enum
from pathlib import Path

class TaskType(Enum):
    REGRESSION = "regression"       # Task 1
    CLASSIFICATION = "classification" # Task 2

class ModelType(Enum):
    LIGHTGBM = "lightgbm"
    XGBOOST = "xgboost"

class LabelType(Enum):
    RETURN = "return"        # 收益
    SITUATION = "situation"  # 态势

# 默认路径配置
# 使用基于配置文件的绝对路径
# 确保无论从何处执行脚本都能正常工作
# backend/system/config.py -> parent=system -> parent=backend -> parent=root
PROJECT_ROOT = Path(__file__).parent.parent.parent
# 直接指向 parquet 文件
DATA_PATH = PROJECT_ROOT / "final_dataset_with_time_features" / "final_dataset_with_time_features.parquet"
OUTPUT_PATH = Path(__file__).parent / "outputs"

MODEL_SAVE_PATH = OUTPUT_PATH / "models"
PRED_SAVE_PATH = OUTPUT_PATH / "predictions"
CONFIG_SAVE_PATH = OUTPUT_PATH / "configs"

# 确保目录存在
MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
PRED_SAVE_PATH.mkdir(parents=True, exist_ok=True)
CONFIG_SAVE_PATH.mkdir(parents=True, exist_ok=True)
