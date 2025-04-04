import torch as tr
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
import math
from collections import OrderedDict
from Model_Base import *




#### Best for RUL
class FC_STGNN_RUL(nn.Module):
    def __init__(self, indim_fea, Conv_out, lstmhidden_dim, lstmout_dim, conv_kernel,hidden_dim, time_length, num_node, num_windows, moving_window,stride,decay, pooling_choice, n_class):
        super(FC_STGNN_RUL, self).__init__()
        # graph_construction_type = args.graph_construction_type
        self.nonlin_map = Feature_extractor_1DCNN_RUL(1, lstmhidden_dim, lstmout_dim,kernel_size=conv_kernel)
        self.nonlin_map2 = nn.Sequential(
            nn.Linear(lstmout_dim*Conv_out, 2*hidden_dim),
            nn.BatchNorm1d(2*hidden_dim)
        )

        self.positional_encoding = PositionalEncoding(2*hidden_dim,0.1,max_len=5000)

        self.MPNN1 = GraphConvpoolMPNN_block_v6(2*hidden_dim, hidden_dim, num_node, time_length, time_window_size=moving_window[0], stride=stride[0], decay = decay, pool_choice=pooling_choice)
        self.MPNN2 = GraphConvpoolMPNN_block_v6(2*hidden_dim, hidden_dim, num_node, time_length, time_window_size=moving_window[1], stride=stride[1], decay = decay, pool_choice=pooling_choice)


        self.fc = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(hidden_dim * num_windows * num_node, 2*hidden_dim)),
            ('relu1', nn.ReLU(inplace=True)),
            ('fc2', nn.Linear(2*hidden_dim, 2*hidden_dim)),
            ('relu2', nn.ReLU(inplace=True)),
            ('fc3', nn.Linear(2*hidden_dim, hidden_dim)),
            ('relu3', nn.ReLU(inplace=True)),
            ('fc4', nn.Linear(hidden_dim, n_class)),

        ]))



    def forward(self, X):
        # print(X.size())
        bs, tlen, num_node, dimension = X.size() ### tlen = 1
        print(bs, tlen, num_node, dimension)


        ### Graph Generation
        A_input = tr.reshape(X, [bs*tlen*num_node, dimension, 1])
        A_input_ = self.nonlin_map(A_input)
        A_input_ = tr.reshape(A_input_, [bs*tlen*num_node,-1])
        A_input_ = self.nonlin_map2(A_input_)
        A_input_ = tr.reshape(A_input_, [bs, tlen,num_node,-1])

        # print('A_input size is ', A_input_.size())

        ## positional encoding before mapping starting
        X_ = tr.reshape(A_input_, [bs,tlen,num_node, -1])
        X_ = tr.transpose(X_,1,2)
        X_ = tr.reshape(X_,[bs*num_node, tlen, -1])
        X_ = self.positional_encoding(X_)
        X_ = tr.reshape(X_,[bs,num_node, tlen, -1])
        X_ = tr.transpose(X_,1,2)
        A_input_ = X_

        ## positional encoding before mapping ending

        MPNN_output1 = self.MPNN1(A_input_)
        MPNN_output2 = self.MPNN2(A_input_)


        features1 = tr.reshape(MPNN_output1, [bs, -1])
        features2 = tr.reshape(MPNN_output2, [bs, -1])

        features = tr.cat([features1,features2],-1)

        features = self.fc(features)

        return features








#### best for NBA
class FC_STGNN_NBA(nn.Module):
    def __init__(self, 
        indim_fea,   
        Conv_out,
        lstmhidden_dim, 
        lstmout_dim, 
        conv_kernel,
        hidden_dim, 
        time_length, 
        num_node,   #结点数
        num_windows,   #窗口数
        moving_window,
        stride,     #窗口移动步长
        decay,      #随时间衰减
        pooling_choice,   #池化选择
        num_heads):
        super(FC_STGNN_NBA, self).__init__()
        # graph_construction_type = args.graph_construction_type
        
        #用于特征提取模块
        self.nonlin_map = Feature_extractor_1DCNN_NBA(indim_fea, lstmhidden_dim, lstmout_dim,kernel_size=conv_kernel)
        
        #包括线性层和批归一化层模块，进一步处理特征
        self.nonlin_map2 = nn.Sequential(
            nn.Linear(lstmout_dim*Conv_out, 2*hidden_dim),
            nn.BatchNorm1d(2*hidden_dim)
        )
        
        #位置编码模块
        self.positional_encoding = PositionalEncoding(2*hidden_dim,0.1,max_len=5000)

        
        #两个并行图卷积和池化模块   使用不同的移动窗口大小和步长
        
        self.MPNN1 = GraphConvpoolMPNN_block_v6(
            2*hidden_dim, 
            2*hidden_dim, 
            num_node, 
            time_length, 
            time_window_size=moving_window[0], 
            stride=stride[0], 
            decay = decay, 
            pool_choice=pooling_choice)
        
        self.MPNN2 = GraphConvpoolMPNN_block_v6(
            2*hidden_dim, 
            2*hidden_dim, 
            num_node, 
            time_length, 
            time_window_size=moving_window[1], 
            stride=stride[1], 
            decay = decay, 
            pool_choice=pooling_choice)

        
        # 得到个体结点在时间上的的嵌入表示后，将其展平，然后通过全连接层映射到最终的分类输出。
        self.fc = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(2*hidden_dim * num_windows, 2*hidden_dim)),
            ('relu1', nn.ReLU(inplace=True)),
            # ("dropout1",nn.Dropout(p=0.1)),   #加入dropout防止过拟合
            ('fc2', nn.Linear(2*hidden_dim, hidden_dim)),
            ('relu2', nn.ReLU(inplace=True)),
            # ("dropout1",nn.Dropout(p=0.1))
        ]))


        # 添加 Stacked Attention 计算邻接矩阵
        # self.attention = StackedAttention(embed_dim=hidden_dim, num_heads=num_heads)
        self.edge_fc = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2)  # 输出两个类的logits
            )



    def forward(self, X):
        # print(X.size())  #torch.Size([100, 5, 10, 3,8])
        # breakpoint()
        
        
        bs, tlen, num_node, dimension,feature_dim= X.size()  #batch_size    time_denpen_len是划分出的patchs数量

        # Graph Generation
        A_input = tr.reshape(X, [bs*tlen*num_node, dimension, feature_dim])   
        # print(A_input.shape)  #torch.Size([5000, 3, 8])
        
       
        
        A_input_ = self.nonlin_map(A_input)
        # print(A_input_.shape)  #torch.Size([5000, 18, 2])
        
       
        A_input_ = tr.reshape(A_input_, [bs*tlen*num_node,-1])  

        A_input_ = self.nonlin_map2(A_input_)

        #重塑
        # positional encoding before mapping starting
        X_ = tr.reshape(A_input_, [bs,tlen,num_node, -1])

        X_ = tr.transpose(X_,1,2)  #将1和2两个维度进行转换   tlen和num_node
        # print(X_.size())
        X_ = tr.reshape(X_,[bs*num_node, tlen, -1])
        # print(X_.size())
        X_ = self.positional_encoding(X_)
        # print(X_.size())
        X_ = tr.reshape(X_,[bs,num_node, tlen, -1])
        X_ = tr.transpose(X_,1,2)   #再换回来
        A_input_ = X_

        # print('A_input_.shape:',A_input_.shape)

        ## positional encoding before mapping ending

        MPNN_output1 = self.MPNN1(A_input_)
        MPNN_output2 = self.MPNN2(A_input_)
        #torch.Size([100, 4, 10, 16])
        #torch.Size([100, 4, 10, 16])
        
  
       
        # 重排列维度为 [100, 10, 4, 16]
        MPNN_output1 = MPNN_output1.permute(0, 2, 1, 3)
        MPNN_output2 = MPNN_output2.permute(0, 2, 1, 3)
      
        # 动态获取批量大小
        batch_size = MPNN_output1.size(0)
        # 合并最后两个维度  [batch_size, 10, 64]
        MPNN_output1 = MPNN_output1.reshape(batch_size, num_node, -1)
        MPNN_output2 = MPNN_output2.reshape(batch_size, num_node, -1)

        # print("MPNN_output1.shape",MPNN_output1.shape)
        # print("MPNN_output2.shape",MPNN_output2.shape)
        # breakpoint()

    
        # 保留每个节点的特征
        node_features = tr.cat([MPNN_output1, MPNN_output2], dim=-1)  # [bs, num_node, 96]
        # print(node_features.shape)  #torch.Size([100, 10, 96])
        
     

        # 再套一层FC层，将节点特征映射到最终的特征输出
        node_embeddings = self.fc(node_features)  # [bs, num_node, hidden_dim]


        # 计算邻接矩阵
        # adjacency_matrix = tr.bmm(node_embeddings, node_embeddings.transpose(1, 2))  # 计算余弦相似度
        # 计算堆叠注意力邻接矩阵
        # adjacency_matrix = self.attention(node_embeddings)


        bs, num_node, hidden_dim = node_embeddings.size()

        hi = node_embeddings.unsqueeze(2).expand(bs, num_node, num_node, -1)  # [bs, N, N, hidden_dim]
        hj = node_embeddings.unsqueeze(1).expand(bs, num_node, num_node, -1)  # [bs, N, N, hidden_dim]
       
      
        diff = hi - hj  # [bs, N, N, hidden_dim]
        elemwise_product = hi * hj  # [bs, N, N, hidden_dim]

        edge_input = tr.cat([diff, elemwise_product], dim=-1)  # [bs, N, N, 2*hidden_dim]
        logits = self.edge_fc(edge_input)  # [bs, N, N, 2]
        edge_prob = F.softmax(logits, dim=-1)
        adjacency_matrix = edge_prob[..., 1]  # [bs, N, N]

        
        # 创建布尔掩码，去掉对角线
        batch_size, num_node, _ = adjacency_matrix.size()
        mask = ~tr.eye(num_node, dtype=tr.bool, device=adjacency_matrix.device)  # [num_node, num_node]
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, num_node, num_node]

        # 仅保留非对角线元素并展开为向量,与标签一致没有对角线
        prediction = edge_prob[mask].view(batch_size, -1)  # [batch_size, num_node * (num_node - 1),2]

        return adjacency_matrix,prediction,node_embeddings



#### best for Pedestrians Dataset
class FC_STGNN_Pedestrians_(nn.Module):
    def __init__(self, 
                 indim_fea,   
                 Conv_out,
                 lstmhidden_dim, 
                 lstmout_dim, 
                 conv_kernel,
                 hidden_dim, 
                 time_length, 
                 num_windows,   # 滑动窗口的个数
                 moving_window,  #滑动窗口的大小
                 stride,     # 窗口移动步长
                 decay,      # 随时间衰减
                 pooling_choice,   # 池化选择
                 num_heads,
                ):
        super(FC_STGNN_Pedestrians, self).__init__()
        
        # 特征提取模块
        self.nonlin_map = Feature_extractor_1DCNN_Pedestrians(indim_fea, lstmhidden_dim, lstmout_dim, kernel_size=conv_kernel)
        
        # 包括线性层和批归一化层模块
        self.nonlin_map2 = nn.Sequential(
            nn.Linear(lstmout_dim * Conv_out, 2 * hidden_dim),
            nn.Dropout(p=0.3),   #加入dropout防止过拟合
            # nn.BatchNorm1d(2 * hidden_dim)
            # nn.LayerNorm(2 * hidden_dim)
        )
        
        # 位置编码模块
        self.positional_encoding = PositionalEncoding(2 * hidden_dim, 0.1, max_len=5000)
        
        # 图卷积和池化模块1
        self.MPNN1 = GraphConvpoolMPNN_block_v6_Ped(
            input_dim=2 * hidden_dim,
            output_dim=2*hidden_dim,
            time_length=time_length,
            time_window_size=moving_window[0],  
            stride=stride[0],                  
            decay=decay,
            pool_choice=pooling_choice
        )

        # 图卷积和池化模块2
        self.MPNN2 = GraphConvpoolMPNN_block_v6_Ped(
            input_dim=2 * hidden_dim,
            output_dim=2*hidden_dim,
            time_length=time_length,
            time_window_size=moving_window[1],  
            stride=stride[1],                   
            decay=decay,
            pool_choice=pooling_choice
        )
        
        
        # 全连接层
        self.fc = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(2*hidden_dim * num_windows,2*hidden_dim)),
            ('relu1', nn.LeakyReLU(0.01, inplace=True)),
            # ('relu1', nn.ReLU(inplace=True)),
            ("dropout1",nn.Dropout(p=0.1)),   #加入dropout防止过拟合
            ('fc2', nn.Linear(2*hidden_dim, hidden_dim)),
            ('relu2', nn.LeakyReLU(0.01, inplace=True)),
            # ('relu2', nn.ReLU(inplace=True)),
            ("dropout1",nn.Dropout(p=0.1))
        ]))


        # 添加 Stacked Attention 计算邻接矩阵
        self.attention = StackedAttention(embed_dim=hidden_dim, num_heads=num_heads)


    def forward(self, X):

        # print(X.size())  torch.Size([1, 2, 15, 2])
        

        bs, tlen, num_node, dimension, feature_dim = X.size()  # 动态获取批次和节点数

        # 特征提取
        A_input = tr.reshape(X, [bs * tlen * num_node, dimension, feature_dim])
        # print(A_input.shape) 
        A_input_ = self.nonlin_map(A_input)  
        # print(A_input_.shape)  
        
        A_input_ = tr.reshape(A_input_, [bs * tlen * num_node, -1])
          
        
        A_input_ = self.nonlin_map2(A_input_)
        
        # 位置编码
        X_ = tr.reshape(A_input_, [bs, tlen, num_node, -1])
        X_ = tr.transpose(X_, 1, 2)  # 转换 tlen 和 num_node 的维度
        X_ = tr.reshape(X_, [bs * num_node, tlen, -1])
        X_ = self.positional_encoding(X_)
        X_ = tr.reshape(X_, [bs, num_node, tlen, -1])
        X_ = tr.transpose(X_, 1, 2)  # 转换回原始顺序
        A_input_ = X_

        # print(A_input_.shape)  
        # breakpoint()

        # 图卷积和池化模块
        MPNN_output1 = self.MPNN1(A_input_)
        MPNN_output2 = self.MPNN2(A_input_)
    

        # 调整维度为 [batch_size, num_node, features]
        MPNN_output1 = MPNN_output1.permute(0, 2, 1, 3).reshape(bs, num_node, -1)
        MPNN_output2 = MPNN_output2.permute(0, 2, 1, 3).reshape(bs, num_node, -1)

        # print(MPNN_output1.shape)   
        # print(MPNN_output2.shape)
        # breakpoint()  
      
 
        # 合并两个并行的图卷积和池化模块的输出
        node_features = tr.cat([MPNN_output1, MPNN_output2], dim=-1)

        # 全连接层映射
        node_embeddings = self.fc(node_features)
       
        # 计算堆叠注意力邻接矩阵
        adjacency_matrix = self.attention(node_embeddings)
        


        # 邻接矩阵对称化
        adjacency_matrix = 0.5*(adjacency_matrix + adjacency_matrix.transpose(-1, -2)) 
       
       
        # 创建布尔掩码，去掉对角线
        mask = ~tr.eye(num_node, dtype=tr.bool, device=adjacency_matrix.device)
        mask = mask.unsqueeze(0).expand(bs, -1, -1)

        # 仅保留非对角线元素
        prediction = adjacency_matrix[mask].view(bs, -1)

        return adjacency_matrix, prediction,node_embeddings







#### best for Pedestrians Dataset
class FC_STGNN_Pedestrians(nn.Module):
    def __init__(self, 
                 indim_fea,   
                 Conv_out,
                 lstmhidden_dim, 
                 lstmout_dim, 
                 conv_kernel,
                 hidden_dim, 
                 time_length, 
                 num_windows,   # 滑动窗口的个数
                 moving_window,  #滑动窗口的大小
                 stride,     # 窗口移动步长
                 decay,      # 随时间衰减
                 pooling_choice,   # 池化选择
                 num_heads,
                ):
        super(FC_STGNN_Pedestrians, self).__init__()
        
        # 特征提取模块
        self.nonlin_map = Feature_extractor_1DCNN_Pedestrians(indim_fea, lstmhidden_dim, lstmout_dim, kernel_size=conv_kernel)
        
        # 包括线性层和批归一化层模块
        self.nonlin_map2 = nn.Sequential(
            nn.Linear(lstmout_dim * Conv_out, 2 * hidden_dim),
            nn.Dropout(p=0.3),   #加入dropout防止过拟合
            # nn.BatchNorm1d(2 * hidden_dim)
            # nn.LayerNorm(2 * hidden_dim)
        )
        
        # 位置编码模块
        self.positional_encoding = PositionalEncoding(2 * hidden_dim, 0.1, max_len=5000)
        
        # 图卷积和池化模块1
        self.MPNN1 = GraphConvpoolMPNN_block_v6_Ped(
            input_dim=2 * hidden_dim,
            output_dim=2*hidden_dim,
            time_length=time_length,
            time_window_size=moving_window[0],  
            stride=stride[0],                  
            decay=decay,
            pool_choice=pooling_choice
        )

        # 图卷积和池化模块2
        self.MPNN2 = GraphConvpoolMPNN_block_v6_Ped(
            input_dim=2 * hidden_dim,
            output_dim=2*hidden_dim,
            time_length=time_length,
            time_window_size=moving_window[1],  
            stride=stride[1],                   
            decay=decay,
            pool_choice=pooling_choice
        )
        
        
        # 全连接层
        self.fc = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(2*hidden_dim * num_windows,2*hidden_dim)),
            ('relu1', nn.LeakyReLU(0.01, inplace=True)),
            # ('relu1', nn.ReLU(inplace=True)),
            ("dropout1",nn.Dropout(p=0.1)),   #加入dropout防止过拟合
            ('fc2', nn.Linear(2*hidden_dim, hidden_dim)),
            ('relu2', nn.LeakyReLU(0.01, inplace=True)),
            # ('relu2', nn.ReLU(inplace=True)),
            ("dropout1",nn.Dropout(p=0.1))
        ]))


        # 添加 Stacked Attention 计算邻接矩阵
        self.attention = StackedAttention(embed_dim=hidden_dim, num_heads=num_heads)


    def forward(self, X):

        # print(X.size())  torch.Size([1, 2, 15, 2])
        

        bs, tlen, num_node, dimension, feature_dim = X.size()  # 动态获取批次和节点数

        # 特征提取
        A_input = tr.reshape(X, [bs * tlen * num_node, dimension, feature_dim])
        # print(A_input.shape) 
        A_input_ = self.nonlin_map(A_input)  
        # print(A_input_.shape)  
        
        A_input_ = tr.reshape(A_input_, [bs * tlen * num_node, -1])
          
        
        A_input_ = self.nonlin_map2(A_input_)
        
        # 位置编码
        X_ = tr.reshape(A_input_, [bs, tlen, num_node, -1])
        X_ = tr.transpose(X_, 1, 2)  # 转换 tlen 和 num_node 的维度
        X_ = tr.reshape(X_, [bs * num_node, tlen, -1])
        X_ = self.positional_encoding(X_)
        X_ = tr.reshape(X_, [bs, num_node, tlen, -1])
        X_ = tr.transpose(X_, 1, 2)  # 转换回原始顺序
        A_input_ = X_

        # print(A_input_.shape)  
        # breakpoint()

        # 图卷积和池化模块
        MPNN_output1 = self.MPNN1(A_input_)
        MPNN_output2 = self.MPNN2(A_input_)
    

        # 调整维度为 [batch_size, num_node, features]
        MPNN_output1 = MPNN_output1.permute(0, 2, 1, 3).reshape(bs, num_node, -1)
        MPNN_output2 = MPNN_output2.permute(0, 2, 1, 3).reshape(bs, num_node, -1)

        # print(MPNN_output1.shape)   
        # print(MPNN_output2.shape)
        # breakpoint()  
      
 
        # 合并两个并行的图卷积和池化模块的输出
        node_features = tr.cat([MPNN_output1, MPNN_output2], dim=-1)

        # 全连接层映射
        node_embeddings = self.fc(node_features)
       
        # 计算堆叠注意力邻接矩阵
        adjacency_matrix = self.attention(node_embeddings)
        


        # 邻接矩阵对称化
        adjacency_matrix = 0.5*(adjacency_matrix + adjacency_matrix.transpose(-1, -2)) 
       
       
        # 创建布尔掩码，去掉对角线
        mask = ~tr.eye(num_node, dtype=tr.bool, device=adjacency_matrix.device)
        mask = mask.unsqueeze(0).expand(bs, -1, -1)

        # 仅保留非对角线元素
        prediction = adjacency_matrix[mask].view(bs, -1)

        return adjacency_matrix, prediction,node_embeddings




#### FC概率得到的邻接矩阵
class FC_STGNN_Pedestrians_FC(nn.Module):
    def __init__(self, 
                 indim_fea,   
                 Conv_out,
                 lstmhidden_dim, 
                 lstmout_dim, 
                 conv_kernel,
                 hidden_dim, 
                 time_length, 
                 num_windows,   # 滑动窗口的个数
                 moving_window,  #滑动窗口的大小
                 stride,     # 窗口移动步长
                 decay,      # 随时间衰减
                 pooling_choice,   # 池化选择
                 num_heads,
                ):
        super(FC_STGNN_Pedestrians_FC, self).__init__()
        
        # 特征提取模块
        self.nonlin_map = Feature_extractor_1DCNN_Pedestrians(indim_fea, lstmhidden_dim, lstmout_dim, kernel_size=conv_kernel)
        
        # 包括线性层和批归一化层模块
        self.nonlin_map2 = nn.Sequential(
            nn.Linear(lstmout_dim * Conv_out, 2 * hidden_dim),
            nn.Dropout(p=0.3),   #加入dropout防止过拟合
            # nn.BatchNorm1d(2 * hidden_dim)
            # nn.LayerNorm(2 * hidden_dim)
        )
        
        # 位置编码模块
        self.positional_encoding = PositionalEncoding(2 * hidden_dim, 0.1, max_len=5000)
        
        # 图卷积和池化模块1
        self.MPNN1 = GraphConvpoolMPNN_block_v6_Ped(
            input_dim=2 * hidden_dim,
            output_dim=2*hidden_dim,
            time_length=time_length,
            time_window_size=moving_window[0],  
            stride=stride[0],                  
            decay=decay,
            pool_choice=pooling_choice
        )

        # 图卷积和池化模块2
        self.MPNN2 = GraphConvpoolMPNN_block_v6_Ped(
            input_dim=2 * hidden_dim,
            output_dim=2*hidden_dim,
            time_length=time_length,
            time_window_size=moving_window[1],  
            stride=stride[1],                   
            decay=decay,
            pool_choice=pooling_choice
        )
        
        
        # 全连接层
        self.fc = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(2*hidden_dim * num_windows,2*hidden_dim)),
            ('relu1', nn.LeakyReLU(0.01, inplace=True)),
            # ('relu1', nn.ReLU(inplace=True)),
            ("dropout1",nn.Dropout(p=0.1)),   #加入dropout防止过拟合
            ('fc2', nn.Linear(2*hidden_dim, hidden_dim)),
            ('relu2', nn.LeakyReLU(0.01, inplace=True)),
            # ('relu2', nn.ReLU(inplace=True)),
            ("dropout1",nn.Dropout(p=0.1))
        ]))


        self.edge_fc = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2)  # 输出两个类的logits
            )

    def forward(self, X):

        # print(X.size())  torch.Size([1, 2, 15, 2])
        

        bs, tlen, num_node, dimension, feature_dim = X.size()  # 动态获取批次和节点数

        # 特征提取
        A_input = tr.reshape(X, [bs * tlen * num_node, dimension, feature_dim])
        # print(A_input.shape) 
        A_input_ = self.nonlin_map(A_input)  
        # print(A_input_.shape)  
        
        A_input_ = tr.reshape(A_input_, [bs * tlen * num_node, -1])
          
        
        A_input_ = self.nonlin_map2(A_input_)
        
        # 位置编码
        X_ = tr.reshape(A_input_, [bs, tlen, num_node, -1])
        X_ = tr.transpose(X_, 1, 2)  # 转换 tlen 和 num_node 的维度
        X_ = tr.reshape(X_, [bs * num_node, tlen, -1])
        X_ = self.positional_encoding(X_)
        X_ = tr.reshape(X_, [bs, num_node, tlen, -1])
        X_ = tr.transpose(X_, 1, 2)  # 转换回原始顺序
        A_input_ = X_

        # print(A_input_.shape)  
        # breakpoint()

        # 图卷积和池化模块
        MPNN_output1 = self.MPNN1(A_input_)
        MPNN_output2 = self.MPNN2(A_input_)
    

        # 调整维度为 [batch_size, num_node, features]
        MPNN_output1 = MPNN_output1.permute(0, 2, 1, 3).reshape(bs, num_node, -1)
        MPNN_output2 = MPNN_output2.permute(0, 2, 1, 3).reshape(bs, num_node, -1)

        # print(MPNN_output1.shape)   
        # print(MPNN_output2.shape)
        # breakpoint()  
      
 
        # 合并两个并行的图卷积和池化模块的输出
        node_features = tr.cat([MPNN_output1, MPNN_output2], dim=-1)

        # 全连接层映射
        node_embeddings = self.fc(node_features)


        bs, num_nodes, hidden_dim = node_embeddings.size()

        hi = node_embeddings.unsqueeze(2).expand(bs, num_node, num_node, -1)  # [bs, N, N, hidden_dim]
        hj = node_embeddings.unsqueeze(1).expand(bs, num_node, num_node, -1)  # [bs, N, N, hidden_dim]
       
      
        diff = hi - hj  # [bs, N, N, hidden_dim]
        elemwise_product = hi * hj  # [bs, N, N, hidden_dim]

        edge_input = tr.cat([diff, elemwise_product], dim=-1)  # [bs, N, N, 2*hidden_dim]
        logits = self.edge_fc(edge_input)  # [bs, N, N, 2]
      
        edge_prob = F.softmax(logits, dim=-1)
        adjacency_matrix = edge_prob[..., 1]  # [bs, N, N]


        # 邻接矩阵对称化
        # adjacency_matrix = 0.5*(adjacency_matrix + adjacency_matrix.transpose(-1, -2)) 
       
       
        # 创建布尔掩码，去掉对角线,返回的预测是两个边的概率
        mask = ~tr.eye(num_node, dtype=tr.bool, device=adjacency_matrix.device)
        mask = mask.unsqueeze(0).expand(bs, -1, -1)
        prediction = edge_prob[mask].view(bs, -1)

        return adjacency_matrix, prediction,node_embeddings

