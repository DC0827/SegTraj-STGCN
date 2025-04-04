import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import pickle
from multiprocessing import Pool
from sklearn.preprocessing import MinMaxScaler

#建立热图网格基准
def build_ground(examples_train):
    """
    build ground for heatmap based on training examples
    args:
      examples_train: training examples
    """
    max_train_x = -np.inf
    min_train_x = np.inf
    max_train_y = -np.inf
    min_train_y = np.inf
    
    for example in examples_train:
        max_example_x = example[:,:,0].max()
        max_example_y = example[:,:,1].max()
        min_example_x = example[:,:,0].min()
        min_example_y = example[:,:,1].min()
        if max_example_x > max_train_x:
            max_train_x = max_example_x
        if max_example_y > max_train_y:
            max_train_y  = max_example_y
        if min_example_x < min_train_x:
            min_train_x = min_example_x
        if min_example_y < min_train_y:
            min_train_y = min_example_y
    
    ground = [min_train_x,max_train_x,min_train_y,max_train_y]
        
    return ground



# 单条轨迹热图生成函数
def heatMap(X,ground):
    # X是[timestep  Postions]
    # 初始化参数
    C, k_p, k_t = 1, 0.15, 0.1
    cellSide = 0.5

    x_min, x_max, y_min, y_max = ground[:4]

    numberOfCellForSide = [int((x_max-x_min) / cellSide), int((y_max-y_min) / cellSide)]
    
    #  x行  Y列  读的时候反过来
    H_X = np.zeros((numberOfCellForSide[1], numberOfCellForSide[0]))  

    # print(X)

    # 起始帧和结束帧
    frame_start = 0
    frame_end = len(X)-1
    # print(frame_start)
    # print(frame_end)

    # 记录每个单元格被点亮的起始和结束帧
    gridStart = np.full_like(H_X, np.nan)
    gridEnd = np.full_like(H_X, np.nan)
    for f in range(frame_start, frame_end + 1):
        # print('f:',f)

        grid_x = int(min(max((int(X[f, 0] - x_min) / cellSide), 1), numberOfCellForSide[0]) - 1)
        grid_y = int(min(max((int(X[f, 1] - y_min) / cellSide), 1), numberOfCellForSide[1]) - 1)
        # print(grid_x)

        if np.isnan(gridStart[grid_y, grid_x]):
            gridStart[grid_y, grid_x] = f
            gridEnd[grid_y, grid_x] = f
        else:
            gridEnd[grid_y, grid_x] = f

    # print(gridEnd)
    # print(gridStart)

    # 对轨迹经过的单元格进行热量计算(热量累积和衰减)
    for i in range(numberOfCellForSide[0]):
        for j in range(numberOfCellForSide[1]):
            if not np.isnan(gridStart[j, i]):
                Ebar = C / k_t * (1 - np.exp(-k_t * (gridEnd[j, i] - gridStart[j, i] + 1)))
                H_X[j, i] = Ebar * np.exp(-k_t * (frame_end - gridEnd[j, i] + 1))
    
    # print('H_X:',H_X)
    
            
    # 计算全局平均值
    mean_H_X = np.mean(H_X)

    # 查找满足条件的 (j, i)
    indices = np.argwhere(H_X > mean_H_X)

    # 遍历满足 H_X[j, i] > mean_H_X 的索引对
    for i, j in indices:
        # 查找所有满足 H_X[m, n] < H_X[i, j] 的网格位置 (n, m)
        candidates = np.argwhere(H_X < H_X[i, j])  # 进一步筛选

        for m, n in candidates:
            dist = np.sqrt((i - m)**2 + (j - n)**2)  # 计算距离
            H_X[m, n] += H_X[i, j] * np.exp(-k_p * dist)  # 更新 H_X[n, m]

    # print(f'np.min(H_X): {np.min(H_X):.4f},np.max(H_X): {np.max(H_X):.4f}')

     # 对每个热图归一化到 [0, 1]
    if np.max(H_X) > 0:  # 避免除以零
        H_X = (H_X - np.min(H_X)) / (np.max(H_X) - np.min(H_X))
    
    
    return H_X



# 计算整体热图
def calculate_overall_heatmap(heatmaps):
    overall_heatmap = np.zeros_like(next(iter(heatmaps.values())))
    for heatmap in heatmaps.values():
        overall_heatmap += heatmap
    
    #归一化整体热图
    mean_H = np.mean(overall_heatmap)
    std_H = np.std(overall_heatmap)
    if std_H > 0:
        overall_heatmap = (overall_heatmap - mean_H) / std_H
    return overall_heatmap


def calculate_global_heatmaps(tracks, ground, timesteps):
    """
    计算每个时间点的全局热图。
    参数：

        tracks: dict, 每个人的轨迹数据，键为人 ID,值为轨迹矩阵。
        ground: np.ndarray, 地面边界。
        timesteps: int, 时间步数。

    返回：
        global_heatmaps: list, 每个时间点的全局热图。
    """
    global_heatmaps = []
    for time_idx in range(timesteps):
        # 计算每个人在当前时间点的热图
        individual_heatmaps = {
            person_id: heatMap(track[:time_idx+1], ground) for person_id, track in tracks.items()
           }
        # 合成全局热图
        overall_heatmap = calculate_overall_heatmap(individual_heatmaps)

        global_heatmaps.append(overall_heatmap)
    return global_heatmaps


def compute_kl_divergence(p, q):
    """
    计算两个热图之间的 KL 散度。
    参数:
        p: 个体热图 (已归一化为概率分布)
        q: 群体热图 (已归一化为概率分布)
    返回:
        KL散度值
    """
    epsilon = 1e-10
    p = p.flatten() + epsilon
    q = q.flatten() + epsilon
    p = p / np.sum(p)
    q = q / np.sum(q)
    return np.sum(p * np.log(p / q))





 # 计算近点人数特征


def calculate_nearby(current_coord, other_coords):           
   
    """
    计算当前时间点的近点人数。

    参数：
        current_coord: np.ndarray, 当前点的坐标，形状为 (2,)。
        other_coords: np.ndarray, 当前时间点其他人的坐标，形状为 (N, 2)。

    返回：
        近点人数: int, 距离小于阈值的点的数量。
    """
    distances = np.sqrt(np.sum((other_coords - current_coord) ** 2, axis=1))
    return (np.sum(distances < 5) -1)   # 距离阈值为 5,同时去除自己





def calculate_velocity_acceleration(coords):
    """
    计算速度和加速度。

    参数：
        coords: np.ndarray, 形状为 (时间点, 2)，表示单人二维坐标随时间的变化。

    返回：
        扩展后的数组，形状为 (时间点, 6)，包括 x, y, x速度, y速度, x加速度, y加速度。
    """
    velocities = np.diff(coords, axis=0, prepend=coords[:1])  # 一阶差分，计算速度
    velocities[0] = coords[1] - coords[0]  # 填充初始速度
    accelerations = np.diff(velocities, axis=0, prepend=velocities[:1])  # 二阶差分，计算加速度
    accelerations[0] = velocities[1] - velocities[0]
    return np.hstack((coords, velocities, accelerations))



# 处理单个样本
def process_single_sample(args):

    sample_idx, sample, ground = args

    # 当前样本的人数
    people_num = len(sample)
    timesteps = len(sample[0])

    # 初始化轨迹数据
    tracks = {person_idx: sample[person_idx] for person_idx in range(people_num)}

    # 计算全局热图(包含每个时间点)
    global_heatmaps = calculate_global_heatmaps(tracks, ground, timesteps)
    
    # 用于存储当前样本扩展后的数据
    sample_extended = []

    for person_idx in range(len(sample)):
        
        coords_2 = sample[person_idx]  # 当前人的二维轨迹 [时间点, 2]

        # 计算速度和加速度
        expanded_coords = calculate_velocity_acceleration(coords_2)  # [时间点, 6]

        
        # 计算近点人数
        nearby_counts = np.array([
            calculate_nearby(coords_2[time_idx], sample[:, time_idx])
            for time_idx in range(timesteps)
        ])  


        # 获取热图KL散度特征
        # 获取当前人的热图点积特征
        # dot_product_features = np.array([
        #     np.sum(global_heatmaps[time_idx] * heatMap(coords_2[:time_idx+1], ground))
        #     for time_idx in range(timesteps)
        # ])


        dot_product_features = []
        kl_divergence_features = []

        for time_idx in range(timesteps):
            individual_heat = heatMap(coords_2[:time_idx+1], ground)
            global_heat = global_heatmaps[time_idx]

            # 点积特征
            dot_product = np.sum(global_heat * individual_heat)
            dot_product_features.append(dot_product)

            # KL散度特征
            # kl_div = compute_kl_divergence(individual_heat, global_heat)
            # kl_divergence_features.append(kl_div)

        dot_product_features = np.array(dot_product_features)
        # kl_divergence_features = np.array(kl_divergence_features)


        # print(dot_product_features)

        # 合并数据
        person_extended = np.hstack((expanded_coords, nearby_counts[:, None], dot_product_features[:, None]))  #,kl_divergence_features[:, None]
        sample_extended.append(person_extended)


    return sample_idx, sample_extended




def process_data_parallel(data, ground,num_workers=16):
    """
    使用多线程并行处理数据，保持结果顺序。
    """
    samples_num =  len(data)
    extended_data = [None] * samples_num  # 初始化结果列表

  # 创建进程池
    with Pool(processes=num_workers) as pool:
        # 将索引、样本和 ground 打包为元组列表
        args = [(sample_idx, data[sample_idx], ground) for sample_idx in range(samples_num)]


        # 使用 tqdm 包装生成器以显示进度条
        for sample_idx, sample_result in tqdm(pool.imap_unordered(process_single_sample, args), 
                                              total=samples_num, desc="Processing Samples"):
            extended_data[sample_idx] = sample_result  # 按索引存储结果

    return  extended_data



# 示例用法
if __name__ == "__main__":

    # 可变参数,待处理数据集的名称
    DATASET_NAME = "ETH_"  #  NBA ETH   Hotel  students03  Sim



    # 处理train
    print("开始处理 train 数据...")
    with open(f'./{DATASET_NAME}/examples_train_unnormalized.pkl', 'rb') as f:
        train_data = pickle.load(f)  # 形状为 [样本数, 人数, 时间点, 2]
    train_ground = build_ground(train_data)
    print('train_ground',train_ground)
    train = process_data_parallel(train_data, train_ground)


    # 处理valid
    print("开始处理 valid 数据...")
    with open(f'./{DATASET_NAME}/examples_valid_unnormalized.pkl', 'rb') as f:
        valid_data = pickle.load(f)  # 形状为 [样本数, 人数, 时间点, 2]
    valid_ground = build_ground(valid_data)
    print('valid_ground',valid_ground)
    valid = process_data_parallel(valid_data, valid_ground)

    # 处理test
    print("开始处理 test 数据...")
    with open(f'./{DATASET_NAME}/examples_test_unnormalized.pkl', 'rb') as f:
        test_data = pickle.load(f)  # 形状为 [样本数, 人数, 时间点, 2]
    test_ground = build_ground(test_data)
    print('test_ground',test_ground)
    test = process_data_parallel(test_data, test_ground)



    train = [np.array(sample) for sample in train]
    valid = [np.array(sample) for sample in valid]
    test = [np.array(sample) for sample in test]

    print('train_len:',len(train))
    print('valid_len:',len(valid))
    print('test_len:',len(test))



    # 数据归一化
    all_samples = np.concatenate(
        [sample.reshape(-1, 8) for sample in train], axis=0    #+ valid + test只能用训练集来归一化
    )  # 形状 [总人数 * 时间点, 8]

    # 计算训练集均值和标准差
    feature_mean = np.mean(all_samples, axis=0)  # (8,)
    feature_std = np.std(all_samples, axis=0)    # (8,)
    # 避免除零错误
    feature_std[feature_std == 0] = 1.0

    train_normalized = [(sample - feature_mean) / feature_std for sample in train]
    valid_normalized = [(sample - feature_mean) / feature_std for sample in valid]
    test_normalized = [(sample - feature_mean) / feature_std for sample in test]
 
    with open(f'./{DATASET_NAME}/tensors_train.pkl', "wb") as f:
        pickle.dump(train_normalized, f)

    with open(f'./{DATASET_NAME}/tensors_valid.pkl', "wb") as f:
        pickle.dump(valid_normalized, f)

    with open(f'./{DATASET_NAME}/tensors_test.pkl', "wb") as f:
        pickle.dump(test_normalized, f)

  