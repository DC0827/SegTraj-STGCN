import torch
import networkx as nx
import pickle
import os

def tensor_to_array_groups(y, num_nodes):
    """
    根据展开的邻接矩阵（去掉对角线）计算分组表示
    :y: Tensor, 去掉对角线的邻接矩阵，形状为 [num_nodes * (num_nodes - 1)]
    :num_nodes: int, 节点数量
    :return: list, 预测的组表示，例如 [[0, 1, 3, 4], [2, 5], [6]]
    """
    # 恢复邻接矩阵（去掉对角线）
    adjacency_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.long)
    idx = 0
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:  # 排除对角线
                adjacency_matrix[i, j] = y[idx]
                idx += 1

    # 构建无向图
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    edges = torch.nonzero(adjacency_matrix, as_tuple=False).tolist()
    G.add_edges_from(edges)

    # 提取连通分量
    groups = [list(component) for component in nx.connected_components(G)]
    return groups

def process_pkl_file(input_file, output_file):
    """
    处理单个pkl文件，将邻接矩阵转换为群组划分并保存
    :input_file: str, 输入pkl文件路径
    :output_file: str, 输出pkl文件路径
    """
    with open(input_file, 'rb') as f:
        labels = pickle.load(f)  # 加载邻接矩阵列表

    clusters = []
    for label in labels:
        num_nodes = int((len(label) ** 0.5) + 1)  # 计算节点数量
        groups = tensor_to_array_groups(label, num_nodes)  # 转换为群组划分
        clusters.append(groups)

    # 保存结果
    with open(output_file, 'wb') as f:
        pickle.dump(clusters, f)

def process_folder(folder_path):
    """
    处理单个文件夹中的pkl文件
    :folder_path: str, 文件夹路径
    """
    # 定义输入和输出文件名
    input_files = ['labels_train.pkl', 'labels_test.pkl', 'labels_valid.pkl']
    output_files = ['clusters_train.pkl', 'clusters_test.pkl', 'clusters_valid.pkl']

    for input_file, output_file in zip(input_files, output_files):
        input_path = os.path.join(folder_path, input_file)
        output_path = os.path.join(folder_path, output_file)

        if os.path.exists(input_path):  # 检查文件是否存在
            process_pkl_file(input_path, output_path)
            print(f"处理完成：{input_path} -> {output_path}")
        else:
            print(f"文件不存在：{input_path}")

# 定义需要处理的文件夹列表
folders = ['ETH', 'Hotel', 'NBA', 'students03','Sim']

# 遍历每个文件夹并处理
for folder in folders:
    print(f"正在处理文件夹：{folder}")
    folder_path = f'./{folder}'
    process_folder(folder_path)

print("所有文件夹处理完成！")
