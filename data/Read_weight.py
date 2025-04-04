import os
import pickle
import numpy as np
import pandas as pd

# 数据集名称列表
datasets = ["ETH", "NBA", "Hotel", "students03", "Sim", "Sim_2", "Sim_5","ETH_"]

# 用于保存所有结果的列表
results = []

for dataset in datasets:
    print(f"=== 数据集：{dataset} ===")
    dataset_path = f"./{dataset}"

    files = {
        "train": os.path.join(dataset_path, "labels_train.pkl"),
        "valid": os.path.join(dataset_path, "labels_valid.pkl"),
        "test": os.path.join(dataset_path, "labels_test.pkl")
    }

    dataset_pos = 0
    dataset_neg = 0

    for name, path in files.items():
        if not os.path.exists(path):
            print(f"[跳过] 未找到文件：{path}")
            results.append({
                "Dataset": dataset,
                "Split": name,
                "Positives": "File Not Found",
                "Negatives": "File Not Found",
                "Group Weight": "N/A",
                "NG Weight": "N/A"
            })
            continue

        with open(path, 'rb') as f:
            labels = pickle.load(f)

        all_labels = np.concatenate(labels)
        num_pos = np.sum(all_labels == 1)
        num_neg = np.sum(all_labels == 0)

        dataset_pos += num_pos
        dataset_neg += num_neg

        group_weight = (num_pos + num_neg) / (2 * num_pos + 1e-8) if num_pos > 0 else float('inf')
        ng_weight = (num_pos + num_neg) / (2 * num_neg + 1e-8) if num_neg > 0 else float('inf')

        print(f"[{name.upper()}]")
        print(f"正样本数量:{num_pos},负样本数量:{num_neg}")
        print(f"group_weight:{group_weight:.4f},ng_weight:{ng_weight:.4f}")

        results.append({
            "Dataset": dataset,
            "Split": name,
            "Positives": num_pos,
            "Negatives": num_neg,
            "Group Weight": round(group_weight, 4),
            "NG Weight": round(ng_weight, 4)
        })

    print(f"[总计]")
    print(f"正样本数量:{dataset_pos},负样本数量:{dataset_neg}")
    if dataset_pos > 0 and dataset_neg > 0:
        total_group_weight = (dataset_pos + dataset_neg) / (2 * dataset_pos + 1e-8)
        total_ng_weight = (dataset_pos + dataset_neg) / (2 * dataset_neg + 1e-8)
        print(f"总group_weight: {total_group_weight:.4f},总ng_weight: {total_ng_weight:.4f}")

        results.append({
            "Dataset": dataset,
            "Split": "Total",
            "Positives": dataset_pos,
            "Negatives": dataset_neg,
            "Group Weight": round(total_group_weight, 4),
            "NG Weight": round(total_ng_weight, 4)
        })
    else:
        print("样本数不足，无法计算总权重")
        results.append({
            "Dataset": dataset,
            "Split": "Total",
            "Positives": dataset_pos,
            "Negatives": dataset_neg,
            "Group Weight": "Insufficient Data",
            "NG Weight": "Insufficient Data"
        })

    print()

# 保存为 Excel 表格
df = pd.DataFrame(results)
output_path = "./dataset_weight.csv"
df.to_csv(output_path, index=False)
print(f"结果已保存为：{output_path}")
