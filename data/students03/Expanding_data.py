import pickle
import torch
import random
import itertools
import networkx as nx

def load_data(example_path, label_path):
    with open(example_path, 'rb') as f:
        examples = pickle.load(f)
    with open(label_path, 'rb') as f:
        labels = pickle.load(f)
    return examples, labels

def label_to_adj_matrix(label, N):
    adj = torch.ones((N, N))
    label_index = 0
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            adj[i, j] = label[label_index]
            label_index += 1
    return adj

def extract_groups_from_adj(adj):
    G = nx.Graph()
    N = adj.shape[0]
    G.add_nodes_from(range(N))
    for i in range(N):
        for j in range(N):
            if i != j and adj[i, j] == 1:
                G.add_edge(i, j)
    groups = list(nx.connected_components(G))
    return groups

def generate_subsamples(data, adj, max_subsamples=10, person_limit=15):
    N = data.shape[0]
    subsamples = []

    # 原始样本一定保留
    full_data = data
    full_label = [int(adj[i, j].item()) for i in range(N) for j in range(N) if i != j]
    full_label = torch.tensor(full_label)

    # 如果样本人数少或全是孤立点，直接返回原始样本
    if N <= person_limit or torch.sum(adj) - torch.sum(torch.diag(adj)) == 0:
        return [(full_data, full_label)]

    groups = extract_groups_from_adj(adj)
    used_indices_set = set()

    for _ in range(100):  # 最多尝试 100 次生成
        if len(subsamples) >= max_subsamples:
            break

        # 随机选 person_limit 个人
        sampled = random.sample(list(range(N)), person_limit)

        # 补全他们所在的整个群组
        selected_indices = set(sampled)
        for g in groups:
            if selected_indices & set(g):  # 有交集就补全
                selected_indices.update(g)

        selected_indices = sorted(selected_indices)
        indices_tuple = tuple(selected_indices)
        if indices_tuple in used_indices_set:
            continue
        used_indices_set.add(indices_tuple)

        # 构建子样本
        sub_data = data[selected_indices]
        sub_adj = adj[torch.tensor(selected_indices)[:, None], torch.tensor(selected_indices)]
        sub_label = [int(sub_adj[i, j].item()) for i in range(len(selected_indices)) for j in range(len(selected_indices)) if i != j]
        sub_label = torch.tensor(sub_label)
        # 加入子样本
        subsamples.append((sub_data, sub_label))

    # 一定加入原始完整样本
    subsamples.append((full_data, full_label))
    return subsamples



def save_dataset(samples, prefix):
    examples = [x[0] for x in samples]
    labels = [x[1] for x in samples]
    with open(f'examples_{prefix}_unnormalized.pkl', 'wb') as f:
        pickle.dump(examples, f)
    with open(f'labels_{prefix}.pkl', 'wb') as f:
        pickle.dump(labels, f)

# === 主处理流程 ===

# 加载原始数据
all_data, all_labels = [], []
for set_name in ['train', 'valid', 'test']:
    exs, labs = load_data(f'examples_{set_name}_unnormalized_1.pkl', f'labels_{set_name}_1.pkl')
    all_data.extend(exs)
    all_labels.extend(labs)

print('原始样本总数:', len(all_data))


combined = list(zip(all_data, all_labels))
random.shuffle(combined)
all_data, all_labels = zip(*combined)
all_data, all_labels = list(all_data), list(all_labels)

print('已打乱原始样本顺序')

# 对所有样本扩展（保留来源 ID）
all_groups = []
for idx, (data, label) in enumerate(zip(all_data, all_labels)):
    N = data.shape[0]
    adj = label_to_adj_matrix(label, N)
    subs = generate_subsamples(data, adj, max_subsamples=10,person_limit=15)
    all_groups.append((idx, subs))  # idx 是来源编号

print('原始来源数量（将保持不拆分）:', len(all_groups))

# 优先分配扩展数量多的来源（贪心）
all_groups.sort(key=lambda x: len(x[1]), reverse=True)

train_groups, valid_groups, test_groups = [], [], []
train_count = valid_count = test_count = 0
target_ratio = {'train': 0.6, 'valid': 0.2, 'test': 0.2}

# 贪心分配组（保持每组一起）
for source_id, subs in all_groups:
    total = train_count + valid_count + test_count + 1e-8
    ratio_train = train_count / target_ratio['train']
    ratio_valid = valid_count / target_ratio['valid']
    ratio_test  = test_count  / target_ratio['test']

    # 分配到当前最缺的位置
    if min(ratio_train, ratio_valid, ratio_test) == ratio_train:
        train_groups.append((source_id, subs))
        train_count += len(subs)
    elif min(ratio_train, ratio_valid, ratio_test) == ratio_valid:
        valid_groups.append((source_id, subs))
        valid_count += len(subs)
    else:
        test_groups.append((source_id, subs))
        test_count += len(subs)

# 展开所有子样本
def flatten_groups(groups):
    samples = []
    source_ids = []
    for source_id, subs in groups:
        samples.extend(subs)
        source_ids.extend([source_id] * len(subs))
    return samples, source_ids

train_samples, train_ids = flatten_groups(train_groups)
valid_samples, valid_ids = flatten_groups(valid_groups)
test_samples,  test_ids  = flatten_groups(test_groups)

# 输出信息
print(f'最终划分样本数: train={len(train_samples)}, valid={len(valid_samples)}, test={len(test_samples)}')

# 保存数据
save_dataset(train_samples,'train')
save_dataset(valid_samples, 'valid')
save_dataset(test_samples, 'test')
