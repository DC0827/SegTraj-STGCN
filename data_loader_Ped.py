import os
import pickle
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
from sklearn.model_selection import train_test_split

class VarLenDataset(Dataset):
    def __init__(self, data, labels,args):
        self.data = data
        self.labels = labels
        self.args = args 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = torch.tensor(self.data[idx]).float()
        # 去掉加速度特征
        # sample = sample[:, :, [0, 1, 2, 3, 6, 7]]
        label = torch.as_tensor(self.labels[idx]).long()

        num_nodes, time_steps, feat_dim = sample.shape
        patch_size = self.args.patch_size
        patch_num = time_steps // patch_size

        # 进行reshape和转置：
        sample = sample.view(num_nodes, patch_num, patch_size, feat_dim)     
        sample = sample.permute(1, 0, 2, 3)                                  
        return sample, label


def data_generator(data_folder,args):
    with open(os.path.join(data_folder, "tensors_train.pkl"), 'rb') as f:
        train_data = pickle.load(f)
    with open(os.path.join(data_folder, "labels_train.pkl"), 'rb') as f:
        train_label = pickle.load(f)
    with open(os.path.join(data_folder, "tensors_valid.pkl"), 'rb') as f:
        valid_data = pickle.load(f)
    with open(os.path.join(data_folder, "labels_valid.pkl"), 'rb') as f:
        valid_label = pickle.load(f)
    with open(os.path.join(data_folder, "tensors_test.pkl"),'rb') as f:
        test_data = pickle.load(f)
    with open(os.path.join(data_folder, "labels_test.pkl"), 'rb') as f:
        test_label = pickle.load(f)

    # 构建数据集
    train_dataset = VarLenDataset(train_data, train_label, args)
    valid_dataset = VarLenDataset(valid_data, valid_label, args)
    test_dataset  = VarLenDataset(test_data,  test_label,args)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_dataset,  batch_size=1, shuffle=False, drop_last=False)

    return train_loader, valid_loader, test_loader
