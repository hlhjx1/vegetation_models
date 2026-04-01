import os
import json

p = r"g:\我的云端硬盘\vegetation_models_v2\daima\对比实验_三数据集_8模型训练.ipynb"
print(f"File exists: {os.path.exists(p)}")

if os.path.exists(p):
    with open(p, "r", encoding="utf-8") as f:
        nb = json.load(f)
        print(f'Total cells: {len(nb["cells"])}')
        print(f"\nCell types:")
        for i, cell in enumerate(nb["cells"]):
            print(f'  Cell {i}: {cell["cell_type"]}')
