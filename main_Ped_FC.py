import os

from data_loader_Ped import data_generator
from utils import *

import torch as tr
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import os
import Model
import argparse
import random
from torch.optim import AdamW


class Train():
    def __init__(self, args):


        self.train, self.valid, self.test = data_generator(f"./data/{DATASET_NAME}/", args=args)


        self.args = args
    

        self.net = Model.FC_STGNN_Pedestrians_FC(
            args.indim_fea,
            args.conv_out, 
            args.lstmhidden_dim, 
            args.lstmout_dim,
            args.conv_kernel,
            args.hidden_dim,
            args.time_denpen_len, 
            args.num_windows,
            args.moving_window,
            args.stride, 
            args.decay, 
            args.pool_choice, 
            args.num_heads
        )

        self.net = self.net.cuda() if tr.cuda.is_available() else self.net
        self.loss_function = nn.CosineSimilarity(dim=-1)     #使用余弦相似度
        self.optim = optim.AdamW(self.net.parameters(), lr=args.lr, weight_decay=1e-4)  # AdamW 替换 Adam
    
        self.scheduler = StepLR(self.optim, step_size=args.lr_decay, gamma=args.gamma)  


        # 记录损失
        self.train_losses = []
        self.val_losses = []

    def compute_loss(self, prediction, label,epsilon=1e-7):
        
        bs = prediction.size(0)
        # 标签转换为onehot向量
        label_onehot = F.one_hot(label.long(), num_classes=2).float()
        # 使用余弦相似度计算损失
        cosine_sim = F.cosine_similarity(prediction.view(bs, -1), label_onehot.view(bs, -1),dim=1)  # 计算余弦相似度
        # 计算二元交叉熵损失（BCE Loss）
        # bce_loss = F.binary_cross_entropy(prediction, label)
        bce_loss = F.cross_entropy(
            prediction.view(-1, 2),                 # [bs * N*(N-1), 2]
            label.view(-1).long(),                    # [bs * N*(N-1)]
            weight=tr.tensor([ng_w, g_w],device=prediction.device)
        )

        loss = bce_loss  + 1 - cosine_sim.mean() 
        return loss
        
        # prediction = torch.clamp(prediction, 0, 1)
        # tp = (label * prediction).sum(dim=1)
        # fp = ((1 - label) * prediction).sum(dim=1)
        # fn = (label * (1 - prediction)).sum(dim=1)
        # soft_f1 = (2 * tp + epsilon) / (2 * tp + fp + fn + epsilon)
        # return 1 - soft_f1.mean()


    def Train_batch(self,epoch):
        
        
        self.train, self.valid, self.test = data_generator(f"./data/{DATASET_NAME}/", args=args)
        self.net.train()
        loss_ = 0

        self.optim.zero_grad()
        accum_steps = min(args.accum_steps,len(self.train))  # 梯度累积步数

        for i,(data, label) in enumerate(self.train):
            data = data.cuda() if tr.cuda.is_available() else data
            label = label.cuda() if tr.cuda.is_available() else label
            
            _,prediction,_ = self.net(data)
            # 计算余弦相似度

            loss = train.compute_loss(prediction, label)   

            loss.backward()   #计算梯度

            if (i+1)%accum_steps == 0:
                for param in self.net.parameters():
                    if param.grad is not None:
                        param.grad /= accum_steps
                #更新参数
                self.optim.step()
                self.optim.zero_grad()
                accum_steps = min(args.accum_steps,len(self.train)-i-1)  # 更新梯度累积步数

            loss_ +=  loss.item()
        

        return loss_ / len(self.train)  # 计算epoch内 loss 的均值

    def Train_model(self):
        epoch = self.args.epoch
        test_P_ = []
        test_R_ = []
        test_F1_ = []
        Best_F1 = 0  # 初始化最好F1

        # 初始化 test_P, test_R, test_F1
        test_P, test_R, test_F1 = 0.0, 0.0, 0.0


        for i in range(epoch):

            # 执行训练批次并计算损失
            loss = self.Train_batch(i)

       

            self.train_losses.append(loss)  # 记录训练损失

            # 计算验证集的F1值、成对准确率和验证损失
            F1_val, pairwise_accuracy, val_loss = self.valid_GM()
            self.val_losses.append(val_loss)  # 记录验证损失

            # 格式化信息
            log_message = (f"In the {i}th epoch, Train LOSS is {loss:.5f}.\n"
                        f"Val Results: val_loss is {val_loss:.5f},F1_val is {F1_val:.5f}, val_pairwise_accuracy is {pairwise_accuracy:.3f}%.")

            # 打印当前epoch的所有信息
            print(log_message)

            # 将所有信息写入日志文件
            with open(args.log_file_path, "a") as log_file:
                log_file.write(log_message + "\n")


            # 调整学习率
            prev_lr = self.optim.param_groups[0]['lr']  # 记录更新前的学习率
            self.scheduler.step()  # 调整学习率
            current_lr = self.optim.param_groups[0]['lr']  # 获取更新后的学习率
            # 如果学习率发生变化，打印日志
            if current_lr != prev_lr:
                print(f"!!!-----Learning rate changed at epoch {i}: {prev_lr} -> {current_lr}")
            

            #只有在验证集上表现更好才记录结果和保存模型
            if F1_val > Best_F1:  

                #更新当前最优的F1
                Best_F1 = F1_val

                # 保存模型
                model_filename = os.path.join(MODEL_SAVE_PATH, f"best_model_{DATASET_NAME}.pth")
                torch.save(self.net.state_dict(), model_filename)

             
                test_P, test_R, test_F1,Pairwise_Accuracy = self.test_GM()
                message = (f"*****Better Model Find! Test Results: P={test_P:.5f}, R={test_R:.5f}, F1={test_F1:.5f}, Pairwise Accuracy= {Pairwise_Accuracy:.3f}%*****")
                
                print(message)


                # 打开文件并直接打印内容
                with open(args.log_file_path, "a") as log_file:  
                    print(message, file=log_file)
                    log_file.flush()  # 立即将内容写入磁盘
                test_P_.append(test_P)
                test_R_.append(test_R)
                test_F1_.append(test_F1)

            # 每个epoch训练完成后绘制损失曲线
            self.plot_loss()


   

    # 放入cuda中
    def cuda_(self, x):
        x = tr.Tensor(np.array(x))
        if tr.cuda.is_available():
            return x.cuda()
        else:
            return x


    #在验证集上验证
    def valid_GM(self):
        self.net.eval()      #网络设置为评估模式
        # 初始化存储预测结果和真实标签
        prediction_ = []
        label_ = []

        loss_ = 0

        batch_count = 0  # 记录批次数量
        for data, label in self.valid:
            data = data.cuda() if tr.cuda.is_available() else data
            label = label.cuda() if tr.cuda.is_available() else label
            adjacency_matrix,prediction,_ = self.net(data)

            val_loss = self.compute_loss(prediction, label)   #计算余弦相似度
            loss_ = loss_ + val_loss.item()

            prediction = find_groups_method2(adjacency_matrix)

           
            batch_count += 1  # 统计 batch 数量
           
            prediction_.extend(prediction)
            label_.extend(label)
        
        # 定义评价指标
        total_P, total_R, total_F = 0, 0, 0
        n = len(prediction_)
        total_pairwise_accuracy = 0
        # print(prediction_[0])
        # breakpoint()


        for y_bar, y in zip(prediction_, label_):

            # y从张量更改为数组形式
            nodes_num = sum(len(group) for group in y_bar)
            y = tensor_to_array_groups(y, nodes_num)

            # 计算△GM评价指标
            P, R, F = compute_groupMitre(y,y_bar)
            # 计算pairwise准确率 
            pairwise_accuracy = compute_pairwise_accuracy(y,y_bar)
          
            total_P += P
            total_R += R
            total_F += F
            total_pairwise_accuracy += pairwise_accuracy


        avg_F = total_F / n
        avg_pairwise_accuracy = total_pairwise_accuracy / n


        return avg_F, avg_pairwise_accuracy,loss_/ batch_count
    

    # 在测试集上评估模型性能
    def test_GM(self):
        '''
        This is to predict the results for testing dataset
        :return:
        '''
        self.net.eval()
        prediction_ = []
        label_ = []
        for data, label in self.test:
            data = data.cuda() if tr.cuda.is_available() else data
            label_.extend(label)
            adjacency_matrix,_,_= self.net(data)
            # prediction = find_groups_method1(adjacency_matrix)
            prediction = find_groups_method2(adjacency_matrix)
            prediction_.extend(prediction)
        
        # 定义评价指标
        total_P, total_R, total_F,total_Pairwise_Accuracy = 0, 0, 0,0
        n = len(prediction_)

 
        for y_bar, y in zip(prediction_, label_):
            
            nodes_num = sum(len(group) for group in y_bar)

            # y从张量更改为数组形式
            y = tensor_to_array_groups(y, nodes_num)
            
            P, R, F = compute_groupMitre(y,y_bar)
            Pairwise_Accuracy = compute_pairwise_accuracy(y,y_bar)
           
            total_P += P
            total_R += R
            total_F += F
            total_Pairwise_Accuracy += Pairwise_Accuracy

        avg_P = total_P / n
        avg_R = total_R / n
        avg_F = total_F / n
        total_Pairwise_Accuracy = total_Pairwise_Accuracy / n

        return avg_P, avg_R, avg_F,total_Pairwise_Accuracy

  

    # 绘制loss曲线
    def plot_loss(self):
        """ 绘制训练损失和验证损失曲线 """
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.train_losses)), self.train_losses, label="Train Loss", color="blue")
        plt.plot(range(0, len(self.val_losses) * self.args.show_interval, self.args.show_interval), self.val_losses, label="Validation Loss", color="red")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss Over Epochs")
        plt.legend()
        plt.grid(True)

        # 保存图片
        save_path = os.path.join("./logs", f"{DATASET_NAME}/{DATASET_NAME}_注意力头数{args.num_heads}_学习率{args.lr}_批次{args.accum_steps}_{args.time0}_阈值0.9_有加速度_FC方式.png")
    
        plt.savefig(save_path)
        print(f"Loss curve saved to {save_path}")



def args_config_Ped(args):
        
    # 获取当前时间
    current_time = time.localtime()
    # 格式化为 月 日 小时 分钟
    args.time0 = time.strftime("%m-%d_%H:%M:%S", current_time)
    # print(f"Current time: {time0}")
    # breakpoint()

    args.epoch = 300
    args.k = 1  
    args.window_sample = 15   #原始样本在时间窗口上的长度

    args.decay = 0.7
    args.pool_choice = 'mean'     #池化方式为平均池化
    args.moving_window = [2, 3]   #移动时间窗口大小
    args.stride = [1, 2]          #窗口步长
    args.lr = 1e-3               #学习率
    args.gamma=0.9                #学习率衰减系数
    args.lr_decay = 10         #经过多少个epoch学习率衰减
    
    
    args.batch_size = 1
    args.accum_steps = 50  #真实的batch_size
    args.indim_fea = 8       #输入特征维度

    
    # 划分的patch大小
    args.patch_size = 3
    args.time_denpen_len = int(args.window_sample / args.patch_size)
    
    args.num_windows = 6      # 总的窗口个数



    args.conv_time_CNN = 6

    args.conv_kernel = 3
    args.lstmout_dim = 16     #1DCNN
    args.lstmhidden_dim = 16   #1DCNN输出维度
    args.conv_out = 1 
    
    args.hidden_dim = 16      #处理过程中的隐藏层维度

    # 堆叠注意力头数
    args.num_heads = 6

    return args


if __name__ == '__main__':
   


    MODEL_SAVE_PATH = "./saved_models"
    # 可变参数,待处理数据集的名称
    DATASET_NAME = "ETH_"  #  NBA ETH   Hotel  students03   Sim  Sim_2  Sim_5

    # 数据集对应的权重设置（正样本权重, 负样本权重）
    weight_map = {
        "ETH": (1.87, 0.68),
        "ETH_": (2.00, 0.67), 
        "Hotel": (4.46, 0.56),
        "students03": (7.43, 0.54),
        "Sim": (2.48, 0.63),
        "Sim_2": (5.04, 0.56),
        "Sim_5": (5.04, 0.55),
        "NBA": (1.125, 0.9)
    }

    # 获取当前数据集的权重
    g_w, ng_w = weight_map.get(DATASET_NAME, (1.0, 1.0))  # 默认权重为 (1.0, 1.0)

    
    MODEL_SAVE_PATH = "./saved_models"

    from args import args
    args = args()
    args = args_config_Ped(args)
    args.log_file_path = f"./logs/{DATASET_NAME}/{DATASET_NAME}_注意力头数{args.num_heads}_学习率{args.lr}_批次{args.accum_steps}_{ args.time0}_阈值0.9_有加速度_FC方式.txt"
    train = Train(args)
    train.Train_model()


    # CUDA_VISIBLE_DEVICES=5 python main_Ped_FC.py
