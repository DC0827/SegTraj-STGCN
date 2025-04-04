import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import networkx as nx
import os
from scipy import sparse
from sknetwork.clustering import Louvain
import numpy as np
os.environ["OMP_NUM_THREADS"] = "1"

from sklearn.manifold import TSNE
from sklearn.manifold import MDS
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import pandas as pd
from pandas.plotting import parallel_coordinates

import matplotlib.pyplot as plt
from matplotlib import gridspec
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader




class DisjointSet:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [1] * n      #结点的秩(高度)，用于减少合并后查找的如咋读

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            if self.rank[rootX] > self.rank[rootY]:
                self.parent[rootY] = rootX
            elif self.rank[rootX] < self.rank[rootY]:
                self.parent[rootX] = rootY
            else:
                self.parent[rootY] = rootX    #只有两个高度相同的合并才会增加秩
                self.rank[rootX] += 1


# 计算G-MITRE评价指标
def Compute_GM(y_bar, y):

    y_bar = groups_to_labels(y_bar)
    y = groups_to_labels(y)
    
    n = len(y_bar)
    
    #初始化两个并查集
    y_bar_ds = DisjointSet(2*n)
    y_ds = DisjointSet(2*n)

    # 统计真实分组y中标签到索引的映射
    label_to_indices = {}
    for idx, label in enumerate(y):
        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(idx)
    #构建y并查集
    for indices in label_to_indices.values():
        for i in range(1, len(indices)):
            y_ds.union(indices[0], indices[i])


    # 统计预测分组y_bar中标签到索引的映射
    label_to_indices = {}
    for idx, label in enumerate(y_bar):
        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(idx)
    #构建y_bar并查集
    for indices in label_to_indices.values():
        for i in range(1, len(indices)):
            y_bar_ds.union(indices[0], indices[i])


    # 获取连通分量
    connected_y = {y_ds.find(i) for i in range(n)}
    connected_ybar = {y_bar_ds.find(i) for i in range(n)}

    # 处理单元素连通分量，将其与虚拟对应物连接
    for i in range(n):
        if sum(y_ds.find(i) == y_ds.find(j) for j in range(n)) == 1:
            y_ds.union(i, n + i)
        if sum(y_bar_ds.find(i) == y_bar_ds.find(j) for j in range(n)) == 1:
            y_bar_ds.union(i, n + i)

    # 计算Recall
    num = 0
    den = 0
    for root in connected_y:
        S = [i for i in range(2*n) if y_ds.find(i) == root]
        num += len(S) - len(set(y_bar_ds.find(i) for i in S))
        den += len(S) - 1
    R = num / den if den != 0 else 1

    # 计算Precision
    num = 0
    den = 0
    for root in connected_ybar:
        S = [i for i in range(2*n) if y_bar_ds.find(i) == root]
        num += len(S) - len(set(y_ds.find(i) for i in S))
        den += len(S) - 1
    P = num / den if den != 0 else 1

    # 计算 F 值
    F = 2 * P * R / (P + R) if (P + R) != 0 else 0

    return P, R,F


def groups_to_labels(groups):
    """
    将分组形式转换为每个位置的群组类别形式
    :param groups: list of lists, 分组形式，例如 [[0, 1, 4], [2, 3]]
    :return: list, 每个位置的群组类别，例如 [0, 0, 0, 1, 1]
    """
    nodes_num = nodes_num = sum(len(group) for group in groups)  # 使用集合去重
    labels = [-1] * nodes_num  # 初始化为未分组
    for group_id, group in enumerate(groups):
        for node in group:
            labels[node] = group_id
    return labels


def tensor_to_array_groups(y, num_nodes):
    """
    根据展开的邻接矩阵（去掉对角线）计算分组表示
    :y: Tensor,去掉对角线的邻接矩阵，形状为 [num_nodes * (num_nodes - 1)]
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



def find_groups_method1(adjacency_matrix):
    """
    通过 KMeans 聚类为每个样本自动选择最佳聚类数量，并返回群组划分结果。
    参数:
        adjacency_matrix (torch.Tensor): 形状为 [batch_size, N, N] 的邻接矩阵。
        max_clusters (int): 最大聚类数量。
    返回:
        all_groups (list): 每个样本的群组划分，形如 [[群组1, 群组2], ...]。
        all_best_k (list): 每个样本的最佳聚类数量。
    """
    batch_size, N, _ = adjacency_matrix.size()
    all_groups = []
    all_best_k = []

    for b in range(batch_size):
        #获取当前样本的邻接矩阵
        adj = adjacency_matrix[b]
        embedding = adj.detach().cpu().numpy()  # 假设直接使用邻接矩阵作为特征

        # 特殊情况处理
        if N == 2:  # 如果只有两个点，按邻接矩阵值分类
            if adj[0, 1] > 0.9:  # 共享一个类
                groups = {0: [0, 1]}
            else:  # 分为两个类
                groups = {0: [0], 1: [1]}
            all_groups.append(list(groups.values()))
            all_best_k.append(len(groups))
            continue

        # 3. 自动确定聚类数量
        best_k = 1
        best_score = -1
        best_labels = None

        for k in range(2, N):  
           
            kmeans = KMeans(n_clusters=k, random_state=0)
            labels = kmeans.fit_predict(embedding)
           
            score = silhouette_score(embedding, labels)
            
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels

        # 如果未找到最佳聚类数，默认所有点归为一个类
        if best_labels is None:
            best_labels = [0] * N

        # 4. 转换为群组关系
        groups = {}
        for idx, label in enumerate(best_labels):
            groups.setdefault(label, []).append(idx)

        all_groups.append(list(groups.values()))
        all_best_k.append(best_k)

    return all_groups




def find_groups_method2(adjacency_matrix):
    """
    通过 Louvain 算法进行社区发现，并返回群组划分结果。
    参数:
        adjacency_matrix (torch.Tensor): 形状为 [batch_size, N, N] 的邻接矩阵。
    返回:
        all_groups (list): 每个样本的群组划分，形如 [[群组1, 群组2], ...]。
    """
    batch_size, N, _ = adjacency_matrix.size()
    all_groups = []



    louvain = Louvain()

    for b in range(batch_size):
        
        adj = adjacency_matrix[b]
        # 对称化邻接矩阵
        adj = 0.5*(adj + adj.transpose(-1, -2)) 


        adj = adj.cpu().detach().numpy()
        # 去掉自环
        np.fill_diagonal(adj, 0)

        # print("adj:",adj)


        # 二值化邻接矩阵
        percentile_75 = np.percentile(adj, 75)
        percentile_80 = np.percentile(adj, 80)
        percentile_90 = np.percentile(adj, 90)
        group_prob = (adj > 0.5).astype(int)

        
        # 特殊处理：只有两个点时直接判断它们是否是一组
        if N == 2:
            if group_prob[0, 1] > 0.5 and group_prob[1, 0] > 0.5:
                all_groups.append([[0, 1]])
            else:
                all_groups.append([[0], [1]])
            continue
        
        # 检测场景内全为孤立个体
        if group_prob.sum()==0:
            pred_gIDs = np.arange(N)
        else:
            # 转化为稀疏矩阵
            group_prob = sparse.csr_matrix(group_prob)
            # 应用 Louvain 聚类算法
            pred_gIDs = louvain.fit_predict(group_prob)
           

            
        # 标签转换为群组关系形式
        groups = {}
        for idx, label in enumerate(pred_gIDs.tolist()):  # 转换为普通列表
            groups.setdefault(label, []).append(idx)


    
        all_groups.append(list(groups.values()))

    return all_groups







def create_counterPart(a):
    """
    add fake counter parts for each agent
    args:
      a: list of groups; e.g. a=[[0,1],[2],[3,4]]
    """
    a_p = []
    for group in a:
        if len(group)==1:#singleton
            element = group[0]
            element_counter = -(element+1)#assume element is non-negative
            new_group = [element, element_counter]
            a_p.append(new_group)
        else:
            a_p.append(group)
            for element in group:
                element_counter = -(element+1)
                a_p.append([element_counter])
    return a_p





def compute_mitre(a, b):
    """
    compute mitre 
    more details: https://aclanthology.org/M95-1005.pdf
    args:
      a,b: list of groups; e.g. a=[[1.2],[3],[4]], b=[[1,2,3],[4]]
    Return: 
      mitreLoss a_b
      
    """

    total_m = 0  # total missing links
    total_c = 0  # total correct links

    # Convert b to a dictionary for faster lookup
    b_dict = {}
    for idx, group in enumerate(b):
        for element in group:
            b_dict[element] = idx

    for group_a in a:
        size_a = len(group_a)
        partitions = set()
        for element in group_a:
            if element in b_dict:
                partitions.add(b_dict[element])
        total_c += size_a - 1
        total_m += len(partitions) - 1

    return (total_c - total_m) / total_c if total_c != 0 else 0



# 将群组列表转换为每个节点的标签
def groups_to_labels(groups, num_nodes):
    """
    将群组列表转换为每个节点的标签。

    参数：
    - groups (list of lists): 每个子列表表示一个群组，包含节点索引。
    - num_nodes (int): 节点的总数。

    返回：
    - labels (numpy.ndarray): 长度为 num_nodes 的数组，表示每个节点的群组标签。
    """
    labels = np.full(num_nodes, -1, dtype=int)  # 初始化为 -1，表示未分配
    for group_id, group in enumerate(groups):
        for node in group:
            if labels[node] != -1:
                print(f"Warning: Node {node} assigned to multiple groups.")
            labels[node] = group_id
    return labels



# 计算群组MITRE
def compute_groupMitre(target, predict):
    """
    compute group mitre
    args: 
      target,predict: list of groups; [[0,1],[2],[3,4]]
    return: recall, precision, F1
    """
    #create fake counter agents
    target_p = create_counterPart(target)
    predict_p = create_counterPart(predict)
    recall = compute_mitre(target_p, predict_p)
    precision = compute_mitre(predict_p, target_p)
    if recall==0 or precision==0:
        F1 = 0
    else:
        F1 = 2*recall*precision/(recall+precision)
    return precision,recall,F1
# 计算群组MITRE损失


def compute_gmitre_loss(target, predict):
    _,_, F1 = compute_groupMitre(target, predict)
    return 1-F1





def compute_pairwise_accuracy(target, predict):
    """
    计算成对关系预测正确的百分比。
    args:
        target: 真实分组，例如 [[0, 1], [2], [3, 4]]。
        predict: 预测分组，例如 [[0, 1, 2], [3, 4]]。
    returns:
        成对关系预测正确的百分比。
    """
    # 将所有样本提取到一个列表中
    all_elements = sorted([element for group in target for element in group])
    n = len(all_elements)
    total_pairs = n * (n - 1) // 2  # 所有可能的成对关系数量
    correct_pairs = 0  # 预测正确的成对关系数量

    # 将分组转换为集合字典，方便快速查找
    target_dict = {}
    for idx, group in enumerate(target):
        for element in group:
            target_dict[element] = idx

    predict_dict = {}
    for idx, group in enumerate(predict):
        for element in group:
            predict_dict[element] = idx

    # 遍历所有成对关系
    for i in range(n):
        for j in range(i + 1, n):
            element_i = all_elements[i]
            element_j = all_elements[j]
            # 检查在真实分组和预测分组中是否属于同一个组
            if (target_dict[element_i] == target_dict[element_j]) == (predict_dict[element_i] == predict_dict[element_j]):
                correct_pairs += 1

    # 计算正确百分比
    accuracy = (correct_pairs / total_pairs) * 100
    return accuracy

def nll_gaussian(preds, target, variance, add_const=False):
    """
    loss function
    copied from https://github.com/ethanfetaya/NRI/blob/master/utils.py
    """
    neg_log_p = ((preds - target) ** 2 / (2 * variance))
    if add_const:
        const = 0.5 * np.log(2 * np.pi * variance)
        neg_log_p += const
    return neg_log_p.sum() / (target.size(0) * target.size(1))




# 用t-SNE 可视化结点嵌入
def visualize_embeddings_tsen(node_embeddings, labels=None, save_dir='visualization', sample_id=0, epoch=0):
    """
    使用 t-SNE 可视化单个样本的节点嵌入并保存为图片。

    参数：
    - node_embeddings (torch.Tensor): 形状为 [num_node, hidden_dim] 的张量。
    - labels (list or numpy.ndarray, 可选): 用于着色的标签，长度应为 num_node。
    - save_dir (str): 保存图片的目录路径。
    - sample_id (int): 当前样本的索引，用于命名图片文件。
    - epoch (int): 当前训练的 epoch,用于命名图片文件。
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 将张量移动到 CPU 并转换为 NumPy 数组
    embeddings = node_embeddings.cpu().detach().numpy()

    num_nodes = embeddings.shape[0]
    
    # 动态设置 perplexity
    if num_nodes <= 5:
        perplexity = 1
    else:
        perplexity = min(30, num_nodes - 2)
    
    # 应用 t-SNE
    tsne = TSNE(n_components=2, random_state=42,perplexity=perplexity)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # 创建绘图
    plt.figure(figsize=(8, 6))
   
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis', s=50, alpha=0.8)
    plt.colorbar(label='Cluster Label')
  
    plt.title(f't-SNE Visualization (Epoch {epoch}, Sample {sample_id})')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.tight_layout()
    
    # 保存图片 矢量图   位图
    save_path = os.path.join(save_dir, f'tsne_epoch_{epoch}_sample_{sample_id}.pdf')
    plt.savefig(save_path, format='pdf')  # 矢量图
    save_path = os.path.join(save_dir, f'tsne_epoch_{epoch}_sample_{sample_id}.png')
    plt.savefig(save_path, dpi=400)  # 位图
    plt.close()


# 使用 MDS可视化节点嵌入
def visualize_embeddings_mds(node_embeddings, labels=None, save_dir='visualization',
                             sample_id=0, epoch=0):

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 将张量移动到 CPU 并转换为 NumPy 数组
    embeddings = node_embeddings.cpu().detach().numpy()
    num_nodes = embeddings.shape[0]
    
    # 应用 MDS
    mds = MDS(n_components=2, random_state=42)
    embeddings_2d = mds.fit_transform(embeddings)
    
    # 创建绘图
    plt.figure(figsize=(8, 6))
   
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis', s=50, alpha=0.8)
    plt.colorbar(scatter)  #label='Cluster Label'
    
    plt.title(f'MDS Visualization (Epoch {epoch}, Sample {sample_id})')
    # plt.xlabel('Dimension 1')
    # plt.ylabel('Dimension 2')
    plt.tight_layout()
    
    # 保存图片  # 矢量图
    pdf_path = os.path.join(save_dir, f'mds_epoch_{epoch}_sample_{sample_id}.pdf')
    plt.savefig(pdf_path, format='pdf')  
    
    png_path = os.path.join(save_dir, f'mds_epoch_{epoch}_sample_{sample_id}.png')
    plt.savefig(png_path, dpi=400)  # 位图，分辨率 300 dpi
    
    plt.close()



#使用热力图可视化节点嵌入的相似度矩阵
def visualize_embeddings_heatmap(node_embeddings, labels=None, save_dir='visualization',
                                 sample_id=0, epoch=0):

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 将张量移动到 CPU 并转换为 NumPy 数组
    embeddings = node_embeddings.cpu().detach().numpy()
    num_nodes = embeddings.shape[0]

    # 构建真实邻接矩阵：同一标签的节点之间为1，不同标签为0
    real_adj = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if labels[i] == labels[j]:
                real_adj[i, j] = 1
            else:
                real_adj[i, j] = 0


    # 计算相似度矩阵（例如使用余弦相似度）
    similarity_matrix = cosine_similarity(embeddings)

     # 创建 DataFrame 以便更好地使用 Seaborn
    node_names = [f'Node {i}' for i in range(num_nodes)]
    real_df = pd.DataFrame(real_adj, index=node_names, columns=node_names)
    similarity_df = pd.DataFrame(similarity_matrix, index=node_names, columns=node_names)
    
    # 将节点按标签排序，以便相同标签的节点聚集
    sorted_indices = np.argsort(labels)
    sorted_labels = np.array(labels)[sorted_indices]
    sorted_node_names = np.array(node_names)[sorted_indices]
    
    # 重新排序 DataFrame
    real_df_sorted = real_df.iloc[sorted_indices, sorted_indices]
    similarity_df_sorted = similarity_df.iloc[sorted_indices, sorted_indices]
    
    # 设置绘图风格
    sns.set_theme(style='white')
    
    # 确定颜色映射的一致性
    vmin = min(real_adj.min(), similarity_matrix.min())
    vmax = max(real_adj.max(), similarity_matrix.max())
    
    # 创建一个大的画布，包含两个子图
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # 绘制真实的邻接矩阵热力图
    sns.heatmap(real_df_sorted, ax=axes[0], cmap='Reds', vmin=vmin, vmax=vmax,
                cbar=False, annot=False, linewidths=.5, square=True)
    axes[0].set_title('Real Adjacency Matrix', fontsize=18)
    # axes[0].set_xlabel('Nodes')
    # axes[0].set_ylabel('Nodes')
    
    # 绘制基于嵌入的邻接矩阵热力图
    sns.heatmap(similarity_df_sorted, ax=axes[1], cmap='Reds', vmin=vmin, vmax=vmax,
                cbar=False, annot=False, linewidths=.5, square=True)
    axes[1].set_title('Constructed Adjacency Matrix', fontsize=18)
    # axes[1].set_xlabel('Nodes')
    # axes[1].set_ylabel('Nodes')

    # 添加统一的颜色条
    # 使用 Matplotlib 的 colorbar 方法为整个图表添加一个颜色条
    # 这里使用 ScalarMappable 来创建一个与热力图相同的颜色映射
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap='Reds', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.04)
    # cbar.set_label('Value', rotation=270, labelpad=15, fontsize=12)

    
    plt.suptitle(f'Adjacency Matrix Comparison (Epoch {epoch}, Sample {sample_id})', fontsize=22)

    # 调整子图间的间距
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    
    # # 创建热力图
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(similarity_matrix, annot=True, fmt=".2f", cmap='Reds',
    #             xticklabels=labels if labels is not None else range(len(embeddings)),
    #             yticklabels=labels if labels is not None else range(len(embeddings)))
    
    # plt.title(f'Similarity Heatmap (Epoch {epoch}, Sample {sample_id})')
    # plt.tight_layout()
    
    # 保存图片
    pdf_path = os.path.join(save_dir, f'Adjacency Matrix Comparison_{epoch}_sample_{sample_id}.pdf')
    plt.savefig(pdf_path, format='pdf')  # 矢量图
    
    png_path = os.path.join(save_dir, f'Adjacency Matrix Comparison_{epoch}_sample_{sample_id}.png')
    plt.savefig(png_path, dpi=400)  # 位图，分辨率 400 dpi
    
    plt.close()



# 使用平行坐标图可视化节点嵌入
def visualize_embeddings_parallel_coordinates(node_embeddings, labels=None, save_dir='visualization',
                                   sample_id=0, epoch=0):
    
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 将张量移动到 CPU 并转换为 NumPy 数组
    embeddings = node_embeddings.cpu().detach().numpy()
    
    # 创建 DataFrame
    df = pd.DataFrame(embeddings)
    if labels is not None:
        df['Label'] = labels
    else:
        df['Label'] = 'All'
    
    # 创建平行坐标图
    plt.figure(figsize=(10, 8))
    if labels is not None:
        parallel_coordinates(df, 'Label', color=plt.cm.viridis(np.linspace(0, 1, len(set(labels)))))
    else:
        parallel_coordinates(df, 'Label', color=['b'])
    
    plt.title(f'Parallel Coordinates (Epoch {epoch}, Sample {sample_id})')
    plt.xlabel('Dimensions')
    plt.ylabel('Value')
    plt.tight_layout()
    
    # 保存图片
    pdf_path = os.path.join(save_dir, f'parallel_epoch_{epoch}_sample_{sample_id}.pdf')
    plt.savefig(pdf_path, format='pdf')  # 矢量图
  
    png_path = os.path.join(save_dir, f'parallel_epoch_{epoch}_sample_{sample_id}.png')
    plt.savefig(png_path, dpi=400)  # 位图，分辨率 400 dpi
    
    plt.close()



# 使用 PCA 可视化节点嵌入
def visualize_embeddings_pca(node_embeddings, labels=None, save_dir='visualization',
                             sample_id=0, epoch=0):
    
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 将张量移动到 CPU 并转换为 NumPy 数组
    embeddings = node_embeddings.cpu().detach().numpy()
    num_nodes = embeddings.shape[0]
    
    # 应用 PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    
    # 创建绘图
    plt.figure(figsize=(8, 6))
    if labels is not None:
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis', s=100, alpha=0.7)
        plt.colorbar(scatter)  #label='Cluster Label'
    else:
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=100, alpha=0.7)
    
    plt.title(f'PCA Visualization (Epoch {epoch}, Sample {sample_id})')
    # plt.xlabel('Principal Component 1')
    # plt.ylabel('Principal Component 2')
    plt.tight_layout()
    
    # 保存为矢量图格式 (PDF)
   
    pdf_path = os.path.join(save_dir, f'pca_epoch_{epoch}_sample_{sample_id}.pdf')
    plt.savefig(pdf_path, format='pdf')  # 矢量图
    
    #保存为高分辨率 PNG
  
    png_path = os.path.join(save_dir, f'pca_epoch_{epoch}_sample_{sample_id}.png')
    plt.savefig(png_path, dpi=400)  # 位图，分辨率 400 dpi
    
    plt.close()


def load_spring_sim(batch_size=1, suffix='', label_rate=0.02, save_folder="data/simulation/spring_simulation",
              load_folder=None, normalize=True):
    if load_folder is not None:
        #load saved data
        train_loader_path = os.path.join(load_folder, "train_data_loader"+suffix+".pth")
        valid_loader_path = os.path.join(load_folder, "valid_data_loader"+suffix+".pth")
        test_loader_path = os.path.join(load_folder, "test_data_loader"+suffix+".pth")
        datainfo_file = "datainfo"+suffix+".npy"
        datainfo_path = os.path.join(load_folder, datainfo_file)
        
        train_data_loader = torch.load(train_loader_path)
        valid_data_loader = torch.load(valid_loader_path)
        test_data_loader = torch.load(test_loader_path)
        datainfo = np.load(datainfo_path)
        return train_data_loader, valid_data_loader, test_data_loader,\
               datainfo[0], datainfo[1], datainfo[2], datainfo[3]
    
    
    
    loc_all = np.load('data/simulation/spring_simulation/loc_sampled_all_sim_group' + suffix + '.npy')
    vel_all = np.load('data/simulation/spring_simulation/vel_sampled_all_sim_group' + suffix + '.npy')
    edges_all = np.load('data/simulation/spring_simulation/gr_sim_group' + suffix + '.npy')
    
    num_sims = loc_all.shape[0]
    indices = np.arange(num_sims)
    np.random.shuffle(indices)
    train_idx = int(num_sims*0.6)
    valid_idx = int(num_sims*0.8)
    train_indices = indices[:train_idx]
    valid_indices = indices[train_idx:valid_idx]
    test_indices = indices[valid_idx:]
    
    
    
    loc_train = loc_all[train_indices]
    vel_train = vel_all[train_indices]
    edges_train = edges_all[train_indices]

    loc_valid = loc_all[valid_indices]
    vel_valid = vel_all[valid_indices]
    edges_valid = edges_all[valid_indices]

    loc_test = loc_all[test_indices]
    vel_test = vel_all[test_indices]
    edges_test = edges_all[test_indices]

    # [num_samples, num_timesteps, num_dims, num_atoms]
    num_atoms = loc_train.shape[3]

    loc_max = loc_train.max()
    loc_min = loc_train.min()
    vel_max = vel_train.max()
    vel_min = vel_train.min()
    
    datainfo = np.array([loc_max, loc_min, vel_max, vel_min])
    datainfo_file = "datainfo"+suffix+".npy"

    if normalize:
      # Normalize to [-1, 1]
      loc_train = (loc_train - loc_min) * 2 / (loc_max - loc_min) - 1
      vel_train = (vel_train - vel_min) * 2 / (vel_max - vel_min) - 1

      loc_valid = (loc_valid - loc_min) * 2 / (loc_max - loc_min) - 1
      vel_valid = (vel_valid - vel_min) * 2 / (vel_max - vel_min) - 1

      loc_test = (loc_test - loc_min) * 2 / (loc_max - loc_min) - 1
      vel_test = (vel_test - vel_min) * 2 / (vel_max - vel_min) - 1

    # Reshape to: [num_sims, num_atoms, num_timesteps, num_dims]
    loc_train = np.transpose(loc_train, [0, 3, 1, 2])
    vel_train = np.transpose(vel_train, [0, 3, 1, 2])
    feat_train = np.concatenate([loc_train, vel_train], axis=3)
    edges_train = np.reshape(edges_train, [-1, num_atoms ** 2])
    edges_train = np.array((edges_train + 1) / 2, dtype=np.int64)

    loc_valid = np.transpose(loc_valid, [0, 3, 1, 2])
    vel_valid = np.transpose(vel_valid, [0, 3, 1, 2])
    feat_valid = np.concatenate([loc_valid, vel_valid], axis=3)
    edges_valid = np.reshape(edges_valid, [-1, num_atoms ** 2])
    edges_valid = np.array((edges_valid + 1) / 2, dtype=np.int64)

    loc_test = np.transpose(loc_test, [0, 3, 1, 2])
    vel_test = np.transpose(vel_test, [0, 3, 1, 2])
    feat_test = np.concatenate([loc_test, vel_test], axis=3)
    edges_test = np.reshape(edges_test, [-1, num_atoms ** 2])
    edges_test = np.array((edges_test + 1) / 2, dtype=np.int64)

    feat_train = torch.FloatTensor(feat_train)
    edges_train = torch.LongTensor(edges_train)
    feat_valid = torch.FloatTensor(feat_valid)
    edges_valid = torch.LongTensor(edges_valid)
    feat_test = torch.FloatTensor(feat_test)
    edges_test = torch.LongTensor(edges_test)

    # Exclude self edges
    off_diag_idx = np.ravel_multi_index(
        np.where(np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)),
        [num_atoms, num_atoms])
    edges_train = edges_train[:, off_diag_idx]
    edges_valid = edges_valid[:, off_diag_idx]
    edges_test = edges_test[:, off_diag_idx]
    
    #create mask for training and validation data
    edges_train_masked = edges_train.clone()
    edges_train_masked[edges_train_masked==0]=-1
    edges_valid_masked = edges_valid.clone()
    edges_valid_masked[edges_valid_masked==0]=-1
    mask_train = np.random.choice(a=[1,0], size=edges_train_masked.size(),
                       p=[label_rate, 1-label_rate])
    mask_valid = np.random.choice(a=[1,0], size=edges_valid_masked.size(),
                       p=[label_rate, 1-label_rate])
    mask_train = torch.LongTensor(mask_train)
    mask_valid = torch.LongTensor(mask_valid)
    edges_train_masked = edges_train_masked*mask_train
    edges_valid_masked = edges_valid_masked*mask_valid
    
    edges_train_stack = torch.stack([edges_train, edges_train_masked], dim=-1)
    edges_valid_stack = torch.stack([edges_valid, edges_valid_masked], dim=-1)
    
    

    train_data = TensorDataset(feat_train, edges_train_stack)
    valid_data = TensorDataset(feat_valid, edges_valid_stack)
    test_data = TensorDataset(feat_test, edges_test)

    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=1)
    
    train_loader_path = os.path.join(save_folder, "train_data_loader"+suffix+".pth")
    valid_loader_path = os.path.join(save_folder, "valid_data_loader"+suffix+".pth")
    test_loader_path = os.path.join(save_folder, "test_data_loader"+suffix+".pth")
    datainfo_path = os.path.join(save_folder, datainfo_file)
    
    #save dataloader and datainfo array
    torch.save(train_data_loader, train_loader_path)
    torch.save(valid_data_loader, valid_loader_path)
    torch.save(test_data_loader, test_loader_path)
    np.save(datainfo_path, datainfo)
    

    return train_data_loader, valid_data_loader, test_data_loader, loc_max, loc_min, vel_max, vel_min   
    



def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def create_edgeNode_relation(num_nodes, self_loops=False):
    if self_loops:
        indices = np.ones([num_nodes, num_nodes])
    else:
        indices = np.ones([num_nodes, num_nodes]) - np.eye(num_nodes)
    rel_rec = np.array(encode_onehot(np.where(indices)[0]), dtype=np.float32)
    rel_send = np.array(encode_onehot(np.where(indices)[1]), dtype=np.float32)
    rel_rec = torch.from_numpy(rel_rec)
    rel_send = torch.from_numpy(rel_send)
    
    return rel_rec, rel_send


def indices_to_clusters(l):
    """
    args:
        l: indices of clusters, e.g.. [0,0,1,1]
    return: clusters, e.g. [(0,1),(2,3)]
    """
    d = dict()
    for i,v in enumerate(l):
        d[v] = d.get(v,[])
        d[v].append(i)
    clusters = list(d.values())
    return clusters

def compute_groupMitre_labels(target, predict):
    """
    compute group mitre given indices
    args: target, predict: list of indices of groups
       e.g. [0,0,1,1]
    """
    target = indices_to_clusters(target)
    predict = indices_to_clusters(predict)
    recall, precision, F1 = compute_groupMitre(target, predict)
    return recall, precision, F1

def compute_pairwise_accuracy_labels(target, predict):
    """
    compute group mitre given indices
    args: target, predict: list of indices of groups
       e.g. [0,0,1,1]
    """
    target = indices_to_clusters(target)
    predict = indices_to_clusters(predict)
    acc = compute_pairwise_accuracy(target, predict)
    return acc



if __name__ == '__main__':
    
    # 示例用法
    ybar = [[0, 1, 2, 6, 7, 9], [3, 4, 5, 8]]#[[0,1],[2,3,4],[5]]  # 预测结果
    y = [[0, 2, 6, 7, 9], [1, 3, 4, 5, 8]]#[[0],[1],[2,3],[4,5]] # 原始标签

  


    ybar=[[0,1,2,3,4,5,6,7,8,9]]
    y= [[0,1,2,3,4], [5,6,7,8,9]]


    ybar=[[0, 1, 2, 3, 4, 5, 6, 7, 8]]
    y= [[0, 3], [1, 2], [4], [5], [8, 6, 7]]

    # ybar=[[0,1]]
    # y= [[0], [1]]


    # P, R, F = Compute_GM(ybar, y)
    # print(f" Precision: {P}, Recall: {R}, F1: {F}") 
    P, R, F = compute_groupMitre(y,ybar) 
    print(f" Precision: {P}, Recall: {R}, F1: {F}") 
    # values = [1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0]
    # y = torch.tensor(values)
    # print(tensor_to_array_groups(y, 5))



