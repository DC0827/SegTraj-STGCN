import torch
import os
import numpy as np
import pickle

data_dir = r'../data/Sim'
output_dir = data_dir


# 加载 .pkl 文件
def load_pkl(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

# 读取数据
train_data = load_pkl(os.path.join(data_dir+ r'/tensors_train.pkl'))
# print(len(train_data))
# print(train_data[0].shape)
train_labels =  load_pkl(os.path.join(data_dir + r'/labels_train.pkl'))
val_data =  load_pkl(os.path.join(data_dir + r'/tensors_valid.pkl'))
val_labels =  load_pkl(os.path.join(data_dir + r'/labels_valid.pkl'))
test_data =  load_pkl(os.path.join(data_dir + r'/tensors_test.pkl'))
test_labels =  load_pkl(os.path.join(data_dir + r'/labels_test.pkl'))


# 过滤形状不一致的张量
def filter_data(data, labels):

    data = [torch.tensor(d) for d in data] 
    labels = [torch.tensor(d) for d in labels]

    target_shape = data[0].shape  # 以第一个张量的形状为基准
    filtered_data = []
    filtered_labels = []

    for i, t in enumerate(data):
        if t.shape == target_shape:
            filtered_data.append(t)
            filtered_labels.append(labels[i])

    return filtered_data, filtered_labels


# 过滤数据
train_data, train_labels = filter_data(train_data, train_labels)
val_data, val_labels = filter_data(val_data, val_labels)
test_data, test_labels = filter_data(test_data, test_labels)

print(len(train_data))
print(len(train_labels))
print(len(val_data))
print(len(val_labels))
print(len(test_data))
print(len(test_labels))

dat_dict = dict()
dat_dict["samples"] = torch.stack(train_data)
dat_dict["labels"] = torch.stack(train_labels)
torch.save(dat_dict, os.path.join(output_dir, "train.pt")) 

dat_dict = dict()
dat_dict["samples"] = torch.stack(val_data)
dat_dict["labels"] = torch.stack(val_labels)
torch.save(dat_dict, os.path.join(output_dir, "val.pt"))

dat_dict = dict()
dat_dict["samples"] = torch.stack(test_data)
dat_dict["labels"] = torch.stack(test_labels)
torch.save(dat_dict, os.path.join(output_dir, "test.pt"))

print(torch.stack(train_data).size())
print(torch.stack(train_labels).size())
print(torch.stack(val_data).size())
print(torch.stack(val_labels).size())