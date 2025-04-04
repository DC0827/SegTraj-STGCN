import os
import pickle

# 数据集名称列表
DATASET_NAMES = ["ETH", "NBA", "Hotel", "students03", "Sim"]

# 遍历每个数据集
for DATASET_NAME in DATASET_NAMES:
    # 输入文件夹路径
    input_folder = f'./code/STGroup/data/{DATASET_NAME}'
    # 输出文件夹路径
    output_folder = f'./code/STGroup/data/{DATASET_NAME}'

    # 如果输出文件夹不存在，则创建
    if not os.path.exists(output_folder):
        print(f'{DATASET_NAME}不存在')
        continue

    print(f"处理{DATASET_NAME}数据集中：")
    # 获取所有 .pkl 文件
    pkl_files = [f for f in os.listdir(input_folder) if f.endswith('.pkl')]

    # 遍历所有 .pkl 文件并写入 .txt 文件
    for pkl_file in pkl_files:
        input_path = os.path.join(input_folder, pkl_file)  # 输入文件完整路径
        output_path = os.path.join(output_folder, pkl_file.replace('.pkl', '.txt'))  # 输出文件完整路径

        # 读取 .pkl 文件
        with open(input_path, 'rb') as file:
            data = pickle.load(file)

        # 将内容写入 .txt 文件
        with open(output_path, 'w', encoding='utf-8') as txt_file:
            if isinstance(data, (list, dict, set)):
                # 如果是列表、字典或集合，逐行写入每个元素
                for item in data:
                    txt_file.write(str(item) + '\n')
            else:
                # 如果是其他格式，直接写入
                txt_file.write(str(data))

        print(f"Converted {pkl_file} to {output_path}")

    print(f"All .pkl files in {DATASET_NAME} have been successfully converted to .txt files.")
