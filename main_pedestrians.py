import os


from utils import *

import torch as tr
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import math
import os
import Model
import argparse
import random
import pickle
from args import args
import matplotlib.ticker as mticker



class Train():
    def __init__(self, args):

        self.train_data, self.valid_data, self.test_data,self.train_label, self.valid_label, self.test_label = data_generator(f'./data/{DATASET_NAME}/')
    
        self.args = args
        self.net = Model.FC_STGNN_Pedestrians(
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
        # 优化器
        self.optim = optim.AdamW(self.net.parameters(),lr=args.lr,weight_decay=1e-4)
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=args.lr_decay, gamma=args.gamma)
        
        # # 优化器
        # self.optim = optim.SGD(self.net.parameters(), lr=args.lr, momentum=0, weight_decay=0)
        # # 学习率调度器
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=args.lr_decay, gamma=args.gamma)

        self.best_pairwise_accuracy = 0  # 初始化最好的pairwise_accuracy

        # 记录损失
        self.train_losses = []
        self.val_losses = []

    def compute_loss(self, prediction, label):
     
        # 使用余弦相似度计算损失
        label = label.float()
        # cosine_sim = self.loss_function(prediction, label)  # 计算余弦相似度
       
        bce_loss = F.binary_cross_entropy(prediction, label)   # 计算二元交叉熵损失（BCE Loss）


        # loss = (1 - cosine_sim).mean() + 0.01*bce_loss
        loss = bce_loss
        return loss


    def Train_batch(self,epoch):
       
       
        self.net.train()
        loss_train = []
        
        # 创建索引表，再将样本打乱
        training_indices = np.arange(len(self.train_data))
        np.random.shuffle(training_indices)
        
        self.optim.zero_grad()

        batch_loss = []  #初始化batch总损失
        accumulation_steps = min(args.batch_size, len(self.train_data)) 
        
        # 遍历样本训练模型
        for i,idx in enumerate(training_indices):

          
            train_example = self.train_data[idx]
            label_example = self.train_label[idx]
    
            
            if isinstance(train_example, list):
                train_example = np.array(train_example, dtype=np.float32)  # 列表转换为 NumPy 数组

        
            
            #add batch dimension  增加一个batch的维度
            train_example = torch.from_numpy(train_example).unsqueeze(0)
            label_example = label_example.unsqueeze(0)


            
            # 动态获取节点数目  第二个维度     get number of atoms
            # num_sensor = train_example.size(1) 
            # print(num_sensor)

            train_example = train_example.cuda() if tr.cuda.is_available() else train_example
            label_example = label_example.cuda() if tr.cuda.is_available() else label_example
            
            
            train_example = train_example.view(
                train_example.shape[0],  # batch size 
                train_example.shape[2]//args.patch_size,    # patch_num
                train_example.shape[1],  # num nodes
                args.patch_size,         # patch_size  
                train_example.shape[3]   #
            )

            
         
            _,prediction,_ = self.net(train_example.float())


            #计算损失
            loss = train.compute_loss(prediction, label_example) 

            loss_train.append(loss.item())

            batch_loss.append(loss.item())


            # 归一化loss
            (loss / accumulation_steps).backward()
 
            # 累积batch_size个样本的梯度后更新参数
            if (i+1) % args.batch_size == 0 or (i+1) == len(self.train_data):

                # 梯度信息调试
                max_grad = 0
                min_grad = float('inf')
                for name, param in self.net.named_parameters():
                    if param.grad is not None:
                        # print(f"{name} grad: {param.grad}")
                        # 手动对梯度进行归一化
                        grad_norm = param.grad.norm().item()
                        max_grad = max(max_grad, grad_norm)
                        min_grad = min(min_grad, grad_norm)
                print(f"Max grad norm: {max_grad}, Min grad norm: {min_grad}")
        
                with open("./Grad/Ped/grad_log.txt", "a") as f:
                    f.write(f"Epoch {epoch}, Batch {(i+1)//args.batch_size}, Loss: {np.mean(batch_loss)}\n")

                    grad_values = []  # 存储梯度 norm 以便可视化
                    grad_names = []   # 存储参数名称

                    for name, param in self.net.named_parameters():
                        if param.grad is not None:
                            grad_norm = param.grad.norm().item()
                            f.write(f"{name}: {grad_norm}\n")
                            grad_values.append(grad_norm)
                            grad_names.append(name)
                        else:
                            f.write(f"{name}: None\n")

                    # 绘制梯度 norm 的柱状图
                    if grad_values:
                        plt.figure(figsize=(18, 6))  # 增加图片尺寸，减少重叠
                        bars = plt.bar(grad_names, grad_values)  # 绘制柱状图

                        plt.xticks(rotation=45, ha="right", fontsize=10)  # 旋转 X 轴标签
                        plt.xlabel("Parameters")
                        plt.ylabel("Gradient Norm")
                        plt.title(f"Gradient Norms at Epoch {epoch}, Batch {(i+1) // args.batch_size}")
                        plt.grid(axis="y")

                        # 标注梯度值，倾斜显示
                        for bar, value in zip(bars, grad_values):
                            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.05, 
                                    f"{value:.2e}", ha='center', va='bottom', fontsize=8, rotation=45)  # 旋转 45°

                        # 保存图片
                        plt.savefig(f"./Grad/Ped/gradients_epoch{epoch}_batch{(i+1) // args.batch_size}.png", bbox_inches="tight")
                        plt.close()
                # breakpoint()
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                self.optim.step()
                self.optim.zero_grad()

                accumulation_steps = min(args.batch_size, len(self.train_data)-i-1)
                batch_loss = []  # 重置batch损失

        # 数据全部训练一次后在验证集上进行模型评估
        self.net.eval()
        valid_indices = np.arange(len(self.valid_data))
        valid_F = []
        valid_loss = []
        valid_pairwise_accuracy = []
        
        with torch.no_grad():
            for idx in valid_indices:
                valid_example = self.valid_data[idx]
                label_example = self.valid_label[idx]
                if isinstance(valid_example, list):
                   valid_example = np.array(valid_example, dtype=np.float32)  # 转换为 NumPy 数组
                # 增加batch维度
                valid_example = torch.from_numpy(valid_example).unsqueeze(0)
                label_example = label_example.unsqueeze(0)

                valid_example = valid_example.cuda() if tr.cuda.is_available() else valid_example
                label_example = label_example.cuda() if tr.cuda.is_available() else label_example

                # patch划分
                valid_example = valid_example.view(
                    valid_example.shape[0],  # batch size 
                    valid_example.shape[2]//args.patch_size,   
                    valid_example.shape[1],  # num nodes    # 
                    args.patch_size,         # patch_size 
                    valid_example.shape[3]   
                )
            
                adjacency_matrix,prediction,node_embeddings= self.net(valid_example.float())

                loss = train.compute_loss(prediction, label_example) 
                valid_loss.append(loss.item())

            
                prediction = find_groups_method2(adjacency_matrix)
               
                
                # y从张量更改为数组形式
                nodes_num = sum(len(group) for group in prediction[0])
                
                label_example = tensor_to_array_groups(label_example[0], nodes_num)

               
                P, R, F = compute_groupMitre(label_example,prediction[0])
                pairwise_accuracy = compute_pairwise_accuracy(label_example,prediction[0])

                valid_pairwise_accuracy.append(pairwise_accuracy)
                valid_F.append(F)

                # 选择结点数目较多的进行可视化
                # if nodes_num > 5:
                if False:
                    sample_id = idx
                    visualize_embeddings_tsen(
                        node_embeddings.squeeze(0),  # [num_node, hidden_dim]
                        labels = groups_to_labels(label_example,nodes_num),        
                        save_dir=f'visualization/{DATASET_NAME}/tsne_visualizations/validation',
                        sample_id=sample_id,
                        epoch=epoch
                    )
                    visualize_embeddings_mds(
                        node_embeddings.squeeze(0),  # [num_node, hidden_dim]
                        labels = groups_to_labels(label_example,nodes_num),        
                        save_dir=f'visualization//{DATASET_NAME}/mds_visualizations/validation',
                        sample_id=sample_id,
                        epoch=epoch
                    )
                    visualize_embeddings_heatmap(
                        node_embeddings.squeeze(0),  # [num_node, hidden_dim]
                        labels = groups_to_labels(label_example,nodes_num),        
                        save_dir=f'visualization//{DATASET_NAME}/heatmap_visualizations/validation',
                        sample_id=sample_id,
                        epoch=epoch
                    )
                    visualize_embeddings_parallel_coordinates(
                        node_embeddings.squeeze(0),  # [num_node, hidden_dim]
                        labels = groups_to_labels(label_example,nodes_num),        
                        save_dir=f'visualization//{DATASET_NAME}/parallel_coordinates_visualizations/validation',
                        sample_id=sample_id,
                        epoch=epoch
                    )
                    visualize_embeddings_pca(
                        node_embeddings.squeeze(0),  # [num_node, hidden_dim]
                        labels = groups_to_labels(label_example,nodes_num),        
                        save_dir=f'visualization//{DATASET_NAME}/pca_visualizations/validation',
                        sample_id=sample_id,
                        epoch=epoch
                    )


        self.train_losses.append(np.mean(loss_train))  # 记录每个epoch训练损失
        self.val_losses.append(np.mean(valid_loss))    # 记录每个epoch验证损失
        # 格式化信息
        log_message = (f"In the {epoch}th epoch, The train LOSS is {np.mean(loss_train):.5f}.\n"
                    f"Val Results: val_loss is {np.mean(valid_loss):.5f}, val_F1 is{np.mean(valid_F):.5f},val_pairwise_accuracy is {np.mean(valid_pairwise_accuracy):.3f}%.")

        # 打印当前epoch的所有信息
        print(log_message)

        # 将所有信息写入日志文件
        with open(args.log_file_path, "a") as log_file:
            log_file.write(log_message + "\n")

  


        # 验证集上效果好再在测试集上测试并保存模型   
        if np.mean(valid_pairwise_accuracy) > self.best_pairwise_accuracy:
       
            # 保存模型
            model_filename = os.path.join(MODEL_SAVE_PATH, f"best_model_{DATASET_NAME}.pth")
            torch.save(self.net.state_dict(), model_filename)
            
            # 更新最好的pairwise_accuracy
            self.best_pairwise_accuracy = np.mean(valid_pairwise_accuracy)
          
            # 数据在测试集上评估
            self.net.eval()
            test_indices = np.arange(len(self.test_data))

            test_F = []
            test_P = []
            test_R = []
            test_pairwise_accuracy = []
            
            with torch.no_grad():
                for idx in test_indices:
                    test_example = self.test_data[idx]
                    label_example = self.test_label[idx]
                    if isinstance(test_example, list):
                        test_example = np.array(test_example, dtype=np.float32)  # 转换为 NumPy 数组

                    test_example = torch.from_numpy(test_example).unsqueeze(0)
                    label_example = label_example.unsqueeze(0)

                    test_example = test_example.cuda() if tr.cuda.is_available() else test_example
                    
                
                    # patch划分
                    test_example = test_example.view(
                        test_example.shape[0],  # batch size 
                        test_example.shape[2]//args.patch_size, 
                        test_example.shape[1],  # num nodes
                        args.patch_size,         # patch_size    # 
                        test_example.shape[3]   
                    )
                    
                    adjacency_matrix,_ ,node_embeddings= self.net(test_example.float())
                    # print( 'adjacency_matrix', adjacency_matrix)
                    # breakpoint()


                    # prediction = find_groups_method1(adjacency_matrix)
                    prediction = find_groups_method2(adjacency_matrix)
                    
                    # y从张量更改为数组形式
                    nodes_num = sum(len(group) for group in prediction[0])
                    label_example = tensor_to_array_groups(label_example[0], nodes_num)

                    
                    P, R, F = compute_groupMitre(label_example,prediction[0]) 
                    pairwise_accuracy = compute_pairwise_accuracy(label_example,prediction[0])
                    # print(prediction[0], label_example)
                    test_F.append(F)
                    test_P.append(P)
                    test_R.append(R)
                    test_pairwise_accuracy.append(pairwise_accuracy)
            

            print(
                f"Better model updated and saved to. The evaluation indicators on the test set are as follows: \n",
                "Average Precision: {:.5f}".format(np.mean(test_P)),
                "Average Recall: {:.5f}".format(np.mean(test_R)),
                "Average F1: {:.5f}".format(np.mean(test_F)),
                )
            
            # 打开文件并直接打印内容
            with open(args.log_file_path, "a") as log_file:  
                print(f"****** Better Model Find. ******", file=log_file)
                print(f"TESTING is: P = {np.mean(test_P):.5f}, R = {np.mean(test_R):.5f}, F1 = {np.mean(test_F):.5f},pairwise_accuracy is {np.mean(test_pairwise_accuracy):.3f}%\n", file=log_file)
                log_file.flush()  
        return 


    def Train_model(self):
        for i in range(args.epoch):
            self.Train_batch(i)
            # 调整学习率
            prev_lr = self.optim.param_groups[0]['lr']  # 记录更新前的学习率
            self.scheduler.step()  # 调整学习率
            current_lr = self.optim.param_groups[0]['lr']  # 获取更新后的学习率
            # 如果学习率发生变化，打印日志
            if current_lr != prev_lr:
                print(f"!!!-----Learning rate changed at epoch {i}: {prev_lr} -> {current_lr}")
            # 每个epoch训练完成后绘制损失曲线
            plot_loss(self)

 

  
# 数据加载
def data_generator(data_folder):
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
    return train_data,valid_data,test_data,train_label, valid_label,test_label


# 绘制loss曲线
def plot_loss(self):
    """ 绘制训练损失和验证损失曲线 """
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(self.train_losses)), self.train_losses, label="Train Loss", color="blue")
    plt.plot(range(0, len(self.val_losses) * self.args.show_interval, self.args.show_interval), self.val_losses, label="Validation Loss", color="red")
   
    # **强制 y 轴用普通数值格式显示**
    formatter = mticker.ScalarFormatter(useOffset=False)  # 关闭偏移量
    formatter.set_scientific(False)  # 禁用科学计数法
    plt.gca().yaxis.set_major_formatter(formatter)  # 应用到 y 轴

   
   
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.grid(True)

    # 保存图片
    save_path = os.path.join("./logs", f"Ped_{DATASET_NAME}_heads{args.num_heads}_batch{args.batch_size}_epoch{args.epoch}_{args.time0}.png")
    plt.savefig(save_path)
    print(f"Loss curve saved to {save_path}")
    # plt.show()





def args_config_ped(args):

    # 获取当前时间
    current_time = time.localtime()
    # 格式化为 月 日 小时 分钟
    args.time0 = time.strftime("%m-%d_%H_%M_%S", current_time)


    args.epoch = 200
    args.k = 1
    args.window_sample = 15   #原始样本在时间窗口上的长度

    args.decay = 0.7
    args.pool_choice = 'mean'     #池化方式为平均池化
    args.moving_window = [2, 3]   #滑动窗口的大小
    args.stride = [1, 2]          #窗口步长
    args.lr = 1e-2              #学习率
    args.lr_decay = 10            #经过多少个epoch学习率衰减
    args.gamma = 0.9             #衰减因子
    
    args.batch_size = 600
    args.indim_fea = 8  # 输入特征维度


    args.conv_kernel = 3
    
    args.patch_size = 3  # 划分的patch大小
    args.time_denpen_len = int(args.window_sample / args.patch_size)
    args.conv_out = 1   
    
    args.num_windows = 6      # 总的窗口个数

    args.conv_time_CNN = 6

    args.lstmout_dim = 16     #LSTN输出层维度
    args.hidden_dim = 16
    args.lstmhidden_dim = 16   #LSTM隐藏层维度


    # 堆叠注意力头数
    args.num_heads = 6

    args.log_file_path = f"./logs/Ped_{DATASET_NAME}_heads{args.num_heads}_batch{args.batch_size}_epoch{args.epoch}_{args.time0}.txt"
    
    return args


if __name__ == '__main__':


    # 可变参数,待处理数据集的名称
    DATASET_NAME = "ETH"  #  NBA ETH   Hotel  students03 Total
    
    MODEL_SAVE_PATH = "./saved_models"
   

    args = args()

    # 记录最好的F1分数
    best_F = 0



    args = args_config_ped(args)
    train = Train(args)
    train.Train_model()


    # CUDA_VISIBLE_DEVICES=6 python main_pedestrians.py
