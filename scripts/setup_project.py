# scripts/setup_project.py
import sys
from pathlib import Path

# 添加项目根目录到 sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cal_network_distance.generate_objects import generate_network_data

if __name__ == "__main__":
    print("设置项目：生成必要的数据文件")
    data = generate_network_data()
    print("项目设置完成！")