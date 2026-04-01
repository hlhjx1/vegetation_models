import json

p = r"g:\我的云端硬盘\vegetation_models_v2\daima\对比实验_三数据集_8模型训练.ipynb"
with open(p, "r", encoding="utf-8") as f:
    nb = json.load(f)
    print("=== Last 3 Cells ===")
    for i in range(max(0, len(nb["cells"]) - 3), len(nb["cells"])):
        cell = nb["cells"][i]
        print(f"\nCell {i} ({cell['cell_type']}):")
        source = cell.get("source", [])
        if isinstance(source, list):
            content = "".join(source)
        else:
            content = source
        print(content[:500] if len(content) > 500 else content)
