import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import numpy as np


class Load_Dataset(Dataset):
    # Initialize your data
    def __init__(self, dataset, args):
        
        super(Load_Dataset, self).__init__()

        X_train = dataset["samples"]
        y_train = dataset["labels"]
        # breakpoint()
        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

   

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train.float()
            self.y_data = y_train.long()


        self.len = X_train.shape[0]

  
        # 去掉位置
        # self.x_data = self.x_data[:, :, :, 2:8]  

        # 去掉速度
        # self.x_data = torch.cat([
        #     self.x_data[:, :, :,  0:2],  
        #     self.x_data[:, :, :,  4:8]  
        #     ], dim=-1)  

        # 去掉加速度                            
        self.x_data = torch.cat([
            self.x_data[:, :, :,  0:4],
            self.x_data[:, :, :,  6:]
            ], dim=-1)
        
        # 去掉近点人数
        # self.x_data = torch.cat([
        #     self.x_data[:, :, :,  0:6],
        #     self.x_data[:, :, :,  7:]
        #     ], dim=-1)

        # 去掉热图 
        # self.x_data = self.x_data[:, :, :,  0:7]


        shape = self.x_data.size()
        self.x_data = self.x_data.reshape(shape[0],shape[1],args.time_denpen_len, args.patch_size , shape[3])
        self.x_data = torch.transpose(self.x_data, 1,2)


    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


def data_generator(data_path, args):

    train_dataset = torch.load(os.path.join(data_path, "train.pt"),weights_only=True)
    valid_dataset = torch.load(os.path.join(data_path, "val.pt"),weights_only=True)
    test_dataset = torch.load(os.path.join(data_path, "test.pt"),weights_only=True)

    train_dataset = Load_Dataset(train_dataset, args)
    valid_dataset = Load_Dataset(valid_dataset, args)
    test_dataset = Load_Dataset(test_dataset, args)


    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, 
        batch_size=args.batch_size,
        shuffle=True, 
        # 这个参数设置为True，那么最后一个batch的大小可能会小于batch_size
        drop_last=args.drop_last,
        num_workers=0)
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset, 
        batch_size=args.batch_size,
        shuffle=False, 
        drop_last=args.drop_last,
        num_workers=0)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, 
        batch_size=args.batch_size,
        shuffle=False, 
        drop_last=False,
        num_workers=0)

    return train_loader, valid_loader, test_loader
