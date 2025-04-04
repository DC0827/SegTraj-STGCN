import torch as tr
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
import math
from collections import OrderedDict



class Feature_extractor_1DCNN_RUL(nn.Module):
    def __init__(self, input_channels, num_hidden, out_dim, kernel_size = 8, stride = 1, dropout = 0):
        super(Feature_extractor_1DCNN_RUL, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(input_channels, num_hidden, kernel_size=kernel_size,
                      stride=stride, bias=False, padding=(kernel_size//2)),
            nn.BatchNorm1d(num_hidden),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(num_hidden, out_dim, kernel_size=kernel_size, stride=1, bias=False, padding=1),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )


    def forward(self, x_in):
        # print('input size is {}'.format(x_in.size()))
        ### input dim is (bs, tlen, feature_dim)
        x = tr.transpose(x_in, -1,-2)

        x = self.conv_block1(x)
        x = self.conv_block2(x)

        return x


#用于对pacth特征提取的一维卷积神经网络
class Feature_extractor_1DCNN_NBA(nn.Module):
    def __init__(self, 
        input_channels,     #输入特征通道数      
        num_hidden,       
        embedding_dimension,   #
        kernel_size,   #卷积核的大小
        stride = 1, 
        dropout = 0):
        super(Feature_extractor_1DCNN_NBA, self).__init__()


        self.conv_block1 = nn.Sequential(
            nn.Conv1d(input_channels, num_hidden, kernel_size=kernel_size,
                      stride=stride, bias=False, padding=0),
            nn.BatchNorm1d(num_hidden),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            # nn.Dropout(dropout)
        )




    def forward(self, x_in):
        # print('input size is {}'.format(x_in.size()))
        
        x = tr.transpose(x_in, -1,-2)

        x = self.conv_block1(x)
        # print(x.size()) 
        # breakpoint()

        # x = self.conv_block2(x)
        # print(x.size())  
        # x = self.conv_block3(x)
        # print(x.size())   
        
        return x
    


#用于对pacth特征提取的一维卷积神经网络
class Feature_extractor_1DCNN_Pedestrians(nn.Module):
    def __init__(self, 
        input_channels,     #输入特征通道数      
        num_hidden,       
        embedding_dimension,   #
        kernel_size,   #卷积核的大小
        stride = 1, 
        dropout = 0):
        super(Feature_extractor_1DCNN_Pedestrians, self).__init__()


        self.conv_block1 = nn.Sequential(
            nn.Conv1d(input_channels, num_hidden, kernel_size=kernel_size,
                      stride=stride, bias=False, padding=0),
            # nn.LayerNorm([num_hidden,1]),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            # nn.Dropout(dropout)
        )

        # self.conv_block2 = nn.Sequential(
        #     nn.Conv1d(num_hidden, num_hidden*2, kernel_size=kernel_size, stride=1, bias=False, padding=2),
        #     nn.BatchNorm1d(num_hidden*2),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        # )

        # self.conv_block3 = nn.Sequential(
        #     nn.Conv1d(num_hidden*2, embedding_dimension, kernel_size=kernel_size, stride=1, bias=False, padding=3),
        #     nn.BatchNorm1d(embedding_dimension),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        # )


    def forward(self, x_in):
        # print('input size is {}'.format(x_in.size()))
        
        x = tr.transpose(x_in, -1,-2)

        # print(x.size())  # torch.Size([50, 8, 3])
        # breakpoint()

        x = self.conv_block1(x)
      

        # x = self.conv_block2(x)
        # print(x.size())  
        # x = self.conv_block3(x)
        # print(x.size())   
        
        return x



#用于对pacth特征提取的一维卷积神经网络
class Feature_extractor_1DCNN_HAR_SSC(nn.Module):
    def __init__(self, 
        input_channels,           #1
        num_hidden,       #48
        embedding_dimension,   #嵌入特征维度 18
        kernel_size = 3,   #卷积核
        stride = 1, 
        dropout = 0):
        super(Feature_extractor_1DCNN_HAR_SSC, self).__init__()


        self.conv_block1 = nn.Sequential(
            nn.Conv1d(input_channels, num_hidden, kernel_size=kernel_size,
                      stride=stride, bias=False, padding=(kernel_size//2)),
            nn.BatchNorm1d(num_hidden),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(num_hidden, num_hidden*2, kernel_size=kernel_size, stride=1, bias=False, padding=2),
            nn.BatchNorm1d(num_hidden*2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(num_hidden*2, embedding_dimension, kernel_size=kernel_size, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(embedding_dimension),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )


    def forward(self, x_in):
        # print('input size is {}'.format(x_in.size()))
        
        x = tr.transpose(x_in, -1,-2)

        x = self.conv_block1(x)
        # print(x.size())  torch.Size([1800, 48, 33])
        x = self.conv_block2(x)
        # print(x.size())  torch.Size([1800, 96, 17])
        x = self.conv_block3(x)
        # print(x.size())   torch.Size([1800, 18, 10])
        
        return x



#目前版本
#根据节点特征间的点积构建图的邻接矩阵，四种版本的区别在于是否引入线性映射层（参与训练）
class Dot_Graph_Construction_weights(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        #线性映射层，将输入特征映射到相同的维度
        self.mapping = nn.Linear(input_dim, input_dim)

    def forward(self, node_features):
        node_features = self.mapping(node_features)
        # node_features = F.leaky_relu(node_features)
        bs, N, dimen = node_features.size()

        node_features_1 = tr.transpose(node_features, 1, 2)  #在第1维和第2维进行转置

        Adj = tr.bmm(node_features, node_features_1)    #实现特征之间点积得到邻接矩阵
        

        #规整邻接矩阵
        eyes_like = tr.eye(N).repeat(bs, 1, 1).cuda()
        eyes_like_inf = eyes_like * 1e8
        Adj = F.leaky_relu(Adj - eyes_like_inf)
        Adj = F.softmax(Adj, dim=-1)
        # print(Adj[0])
        Adj = Adj + eyes_like
        # print(Adj[0])
        # if prior:
        return Adj

#线性映射层输入和输出维度不一致，尝试不同的特征映射
class Dot_Graph_Construction_weights_v2(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.mapping = nn.Linear(input_dim, hidden_dim)

    def forward(self, node_features):
        node_features = self.mapping(node_features)
        # node_features = F.leaky_relu(node_features)
        bs, N, dimen = node_features.size()

        node_features_1 = tr.transpose(node_features, 1, 2)

        Adj = tr.bmm(node_features, node_features_1)

        eyes_like = tr.eye(N).repeat(bs, 1, 1).cuda()
        eyes_like_inf = eyes_like * 1e8
        Adj = F.leaky_relu(Adj - eyes_like_inf)
        Adj = F.softmax(Adj, dim=-1)
        # print(Adj[0])
        Adj = Adj + eyes_like
        # print(Adj[0])
        # if prior:

        return Adj


class Dot_Graph_Construction(nn.Module):
    def __init__(self,input_dim):
        super().__init__()
    def forward(self,node_features):
        ## node features size is (bs, N, dimension)
        ## output size is (bs, N, N)
        bs, N, dimen = node_features.size()

        node_features_1 = tr.transpose(node_features, 1, 2)

        Adj = tr.bmm(node_features, node_features_1)

        eyes_like = tr.eye(N).repeat(bs, 1, 1).cuda()
        eyes_like_inf = eyes_like*1e8
        Adj = F.leaky_relu(Adj-eyes_like_inf)
        Adj = F.softmax(Adj, dim = -1)
        # print(Adj[0])
        Adj = Adj+eyes_like
        # print(Adj[0])
        # if prior:
        return Adj






class MPNN_mk(nn.Module):
    def __init__(self, input_dimension, outpuut_dinmension, k):
        ### In GCN, k means the size of receptive field. Different receptive fields can be concatnated or summed
        ### k=1 means the traditional GCN
        super(MPNN_mk, self).__init__()
        self.way_multi_field = 'sum' ## two choices 'cat' (concatnate) or 'sum' (sum up)
        self.k = k
        theta = []
        for kk in range(self.k):
            theta.append(nn.Linear(input_dimension, outpuut_dinmension))
        self.theta = nn.ModuleList(theta)

    def forward(self, X, A):
        ## size of X is (bs, N, A)
        ## size of A is (bs, N, N)
        GCN_output_ = []
        for kk in range(self.k):
            if kk == 0:
                A_ = A
            else:
                A_ = tr.bmm(A_,A)
            out_k = self.theta[kk](tr.bmm(A_,X))
            GCN_output_.append(out_k)

        if self.way_multi_field == 'cat':
            GCN_output_ = tr.cat(GCN_output_, -1)

        elif self.way_multi_field == 'sum':
            GCN_output_ = sum(GCN_output_)

        return F.leaky_relu(GCN_output_)


class MPNN_mk_v2(nn.Module):
    def __init__(self, input_dimension, outpuut_dinmension, k):
        ### In GCN, k means the size of receptive field. Different receptive fields can be concatnated or summed
        ### k=1 means the traditional GCN   这里k=1就是传统的图卷积
        super(MPNN_mk_v2, self).__init__()
        
        self.way_multi_field = 'sum' ## two choices 'cat' (concatnate) or 'sum' (sum up)
        self.k = k
        theta = []
        for kk in range(self.k):
            theta.append(nn.Linear(input_dimension, outpuut_dinmension))
        self.theta = nn.ModuleList(theta)    #多个线性层组成的列表
        # self.bn1 = nn.BatchNorm1d(outpuut_dinmension)

    def forward(self, X, A):
        ## size of X is (bs, N, A)    特征矩阵
        ## size of A is (bs, N, N)    邻接矩阵
        GCN_output_ = []          
        for kk in range(self.k):
            if kk == 0:
                A_ = A
            else:
                A_ = tr.bmm(A_,A)
            out_k = self.theta[kk](tr.bmm(A_,X))   #特征矩阵与邻接矩阵相乘
            GCN_output_.append(out_k)

        if self.way_multi_field == 'cat':
            GCN_output_ = tr.cat(GCN_output_, -1)

        elif self.way_multi_field == 'sum':
            GCN_output_ = sum(GCN_output_)

        GCN_output_ = tr.transpose(GCN_output_, -1, -2)
        # GCN_output_ = self.bn1(GCN_output_)
        GCN_output_ = tr.transpose(GCN_output_, -1, -2)

        # print(GCN_output_.size())
        # print(X.size())
        # breakpoint()

        # 原始输入和GCN卷积输出进行残差连接
        GCN_output_ = GCN_output_ +  X

        # print(GCN_output_.size())
        # breakpoint()

        return F.leaky_relu(GCN_output_)



# 定义图卷积层损失函数
def Graph_regularization_loss(X, Adj, gamma):
    ### X size is (bs, N, dimension)
    ### Adj size is (bs, N, N)
    X_0 = X.unsqueeze(-3)
    X_1 = X.unsqueeze(-2)

    X_distance = tr.sum((X_0 - X_1)**2, -1)

    Loss_GL_0 = X_distance*Adj
    Loss_GL_0 = tr.mean(Loss_GL_0)

    Loss_GL_1 = tr.sqrt(tr.mean(Adj**2))
    # print('Loss GL 0 is {}'.format(Loss_GL_0))
    # print('Loss GL 1 is {}'.format(Loss_GL_1))


    Loss_GL = Loss_GL_0 + gamma*Loss_GL_1


    return Loss_GL


#Implement the PE function.
class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(p=dropout)  #在前向传播是随机丢弃

        # Compute the positional encodings once in log space.
        pe = tr.zeros(max_len, d_model).cuda()
        position = tr.arange(0, max_len).unsqueeze(1)
        div_term = tr.exp(tr.arange(0, d_model, 2) *-(math.log(100.0) / d_model))
        pe[:, 0::2] = tr.sin(position * div_term)
        pe[:, 1::2] = tr.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):

        # x = x + torch.Tensor(self.pe[:, :x.size(1)],
        #                  requires_grad=False)
        # print(self.pe[0, :x.size(1),2:5])
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
        # return x

   
class StackedAttention_(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(StackedAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.scale = math.sqrt(embed_dim)  # 预计算缩放因子

        # 两组线性映射：一组用于生成 theta，一组用于生成 phi
        self.W_theta = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),  # 映射到 θ
                nn.ReLU(inplace=True),
                nn.Dropout(0.3)
            )
            for _ in range(num_heads)
        ])
        self.W_phi = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),  # 映射到 φ
                nn.ReLU(inplace=True),
                nn.Dropout(0.3)
            )
            for _ in range(num_heads)
        ])

    def forward(self, embeddings):
        # embeddings: [batch_size, num_nodes, embed_dim]
        
        batch_size, num_nodes, _ = embeddings.shape
        
        # 分别计算每个 head 的 θ 和 φ
        theta = tr.stack([W(embeddings) for W in self.W_theta], dim=1)  # [B, num_heads, N, D]
        phi = tr.stack([W(embeddings) for W in self.W_phi], dim=1)      # [B, num_heads, N, D]
        
        # 使用 einsum 批量计算每个 head 的注意力矩阵
        # 对每个 head 计算 θ 与 φ 的转置相乘，得到 [B, num_heads, N, N]
        attn_scores = tr.einsum('bhnd,bhmd->bhnm', theta, phi)
        
        # 缩放
        attn_scores = attn_scores / self.scale
        
        # 使用 sigmoid 将注意力分数映射到 [0,1]
        attn_scores = tr.sigmoid(attn_scores)
        
        # 对所有 head 的注意力矩阵做平均池化，得到最终的邻接矩阵 [B, N, N]
        adj_matrix = tr.mean(attn_scores, dim=1)
        
        return adj_matrix



# 堆叠注意力机制
class StackedAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(StackedAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.W = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(num_heads)])
        self.b = nn.ParameterList([nn.Parameter(tr.zeros(embed_dim)) for _ in range(num_heads)])

    def forward(self, embeddings):
        batch_size, num_nodes, _ = embeddings.shape
        attention_matrices = []

    
        for i in range(self.num_heads):
            theta = self.W[i](embeddings) + self.b[i]  # 线性变换
            attn = tr.bmm(theta, theta.transpose(1, 2))  # 计算注意力分数
            attn = attn / tr.sqrt(tr.tensor(self.embed_dim, dtype=tr.float32))  # 归一化
            # attn = tr.softmax(attn, dim=-1)  # 归一化到概率分布
            attn = tr.sigmoid(attn)  #归一化到0-1之间的概率
            attention_matrices.append(attn)

        # 最终邻接矩阵：多头注意力的平均池化
        adj_matrix = tr.mean(tr.stack(attention_matrices, dim=0), dim=0)  # [batch_size, num_nodes, num_nodes]
        
        return adj_matrix



def Conv_GraphST(input, time_window_size, stride):
    ## input size is (bs, time_length, num_sensors, feature_dim)
    ## output size is (bs, num_windows, num_sensors, time_window_size, feature_dim)
    bs, time_length, num_sensors, feature_dim = input.size()
    x_ = tr.transpose(input, 1, 3)

    y_ = F.unfold(x_, (num_sensors, time_window_size), stride=stride)

    y_ = tr.reshape(y_, [bs, feature_dim, num_sensors, time_window_size, -1])
    y_ = tr.transpose(y_, 1,-1)

    return y_


def Conv_GraphST_pad(input, time_window_size, stride, padding):
    ## input size is (bs, time_length, num_sensors, feature_dim)
    ## output size is (bs, num_windows, num_sensors, time_window_size, feature_dim)
    bs, time_length, num_sensors, feature_dim = input.size()
    x_ = tr.transpose(input, 1, 3)

    y_ = F.unfold(x_, (num_sensors, time_window_size), stride=stride, padding=[0,padding])

    y_ = tr.reshape(y_, [bs, feature_dim, num_sensors, time_window_size, -1])
    y_ = tr.transpose(y_, 1,-1)

    return y_




#生成图节点的衰减邻接矩阵，decay_rate是随时间的衰减率
def Mask_Matrix(num_node, time_length, decay_rate):
    Adj = tr.ones(num_node * time_length, num_node * time_length).cuda()
    # print(Adj.size())   18*18
    for i in range(time_length):
        v = 0
        for r_i in range(i,time_length):
            idx_s_row = i * num_node
            idx_e_row = (i + 1) * num_node
            idx_s_col = (r_i) * num_node
            idx_e_col = (r_i + 1) * num_node
            Adj[idx_s_row:idx_e_row, idx_s_col:idx_e_col] = Adj[idx_s_row:idx_e_row, idx_s_col:idx_e_col] * (decay_rate ** (v))
            v = v+1
        v=0
        for r_i in range(i+1):
            idx_s_row = i * num_node
            idx_e_row = (i + 1) * num_node
            idx_s_col = (i-r_i) * num_node
            idx_e_col = (i-r_i + 1) * num_node
            Adj[idx_s_row:idx_e_row,idx_s_col:idx_e_col] = Adj[idx_s_row:idx_e_row,idx_s_col:idx_e_col] * (decay_rate ** (v))
            v = v+1
    # print(Adj.size())    #18*18
    return Adj



#图卷积以及池化操作
class GraphConvpoolMPNN_block_v6(nn.Module):
    def __init__(self, input_dim, output_dim, num_sensors, time_length, time_window_size, stride, decay, pool_choice):
        super(GraphConvpoolMPNN_block_v6, self).__init__()
        self.time_window_size = time_window_size
        self.stride = stride
        self.output_dim = output_dim
        
        #选择图权重构建类
        self.graph_construction = Dot_Graph_Construction_weights(input_dim)

        #一维批标准化层
        # self.BN = nn.BatchNorm1d(input_dim)  
        
        #MPNN网络
        self.MPNN = MPNN_mk_v2(input_dim, output_dim, k=1)

        #预处理图结点关系矩阵
        self.pre_relation = Mask_Matrix(num_sensors,time_window_size,decay)

        self.pool_choice = pool_choice

    def forward(self, input):
        ## input size (bs, time_length, num_nodes, input_dim)
        ## output size (bs, output_node_t, output_node_s, output_dim)
        #时空图卷积处理
        input_con = Conv_GraphST(input, self.time_window_size, self.stride)
        ## input_con size (bs, num_windows, num_sensors, time_window_size, feature_dim)
        bs, num_windows, num_sensors, time_window_size, feature_dim = input_con.size()
        input_con_ = tr.transpose(input_con, 2,3)  #置换第2和第3维度
        input_con_ = tr.reshape(input_con_, [bs*num_windows, time_window_size*num_sensors, feature_dim])
        #生成图的邻接矩阵，先根据结点特征建图，接着乘上衰减矩阵
        A_input = self.graph_construction(input_con_)
        # print(A_input.size())    torch.Size([100, 18, 18])
        # print(self.pre_relation.size())  torch.Size([18, 18])
        A_input = A_input*self.pre_relation
        #批标准化   MPNN（k=1相当于GCN）
        input_con_ = tr.transpose(input_con_, -1, -2)
        # input_con_ = self.BN(input_con_)
        input_con_ = tr.transpose(input_con_, -1, -2)
        X_output = self.MPNN(input_con_, A_input)
        X_output = tr.reshape(X_output, [bs, num_windows, time_window_size,num_sensors, self.output_dim])
        # print(X_output.size())
        if self.pool_choice == 'mean':
            X_output = tr.mean(X_output, 2)
        elif self.pool_choice == 'max':
            X_output, ind = tr.max(X_output, 2)
        else:
            print('input choice for pooling cannot be read')
        # X_output = tr.reshape(X_output, [bs, num_windows*time_window_size,num_sensors, self.output_dim])
        # print(X_output.size())
        return X_output


class GraphConvpoolMPNN_block_v6_Ped(nn.Module):
    def __init__(self, input_dim, output_dim, time_length, time_window_size, stride, decay, pool_choice):
        super(GraphConvpoolMPNN_block_v6_Ped, self).__init__()
        self.inputdim = input_dim
        self.time_window_size = time_window_size
        self.stride = stride
        self.output_dim = output_dim
        self.decay = decay
        self.pool_choice = pool_choice

        # 图权重构建类
        self.graph_construction = Dot_Graph_Construction(input_dim)


        # MPNN 网络
        self.MPNN = MPNN_mk_v2(input_dim, output_dim, k=1)

        self.pool_choice = pool_choice

    def forward(self, input):
        """
        Args:
            input: 输入张量，形状为 (batch_size, time_length, num_nodes, input_dim)
        Returns:
            X_output: 输出张量，形状为 (batch_size, num_windows, num_nodes, output_dim)
        """
       

        # 时空图卷积处理
        input_con = Conv_GraphST(input, self.time_window_size, self.stride)
        # input_con 形状: (bs, num_windows, num_nodes, time_window_size, feature_dim)
        bs, num_windows, num_nodes, time_window_size, feature_dim = input_con.size()

        # 重排列数据以适配图卷积输入格式
        input_con_ = tr.transpose(input_con, 2, 3)  # 交换 num_nodes 和 time_window_size
        input_con_ = tr.reshape(input_con_, [bs * num_windows, time_window_size * num_nodes, feature_dim])

        # 动态生成图的邻接矩阵并修正
        A_input = self.graph_construction(input_con_)
        pre_relation = Mask_Matrix(num_nodes, time_window_size, self.decay).to(input.device)
        A_input = A_input * pre_relation

        # 图卷积
        X_output = self.MPNN(input_con_, A_input)

        # 恢复输出形状
        X_output = tr.reshape(X_output, [bs, num_windows, time_window_size, num_nodes, self.output_dim])

        # 池化操作
        if self.pool_choice == 'mean':
            X_output = tr.mean(X_output, dim=2)  # 平均池化
        elif self.pool_choice == 'max':
            X_output, _ = tr.max(X_output, dim=2)  # 最大池化
        else:
            raise ValueError("Invalid pool_choice. Use 'mean' or 'max'.")

        return X_output





class MPNN_block_seperate(nn.Module):
    def __init__(self, input_dim, output_dim, num_sensors, time_conv, time_window_size, stride, decay, pool_choice):
        super(MPNN_block_seperate, self).__init__()
        self.time_window_size = time_window_size
        self.stride = stride
        self.output_dim = output_dim

        self.graph_construction = Dot_Graph_Construction_weights(input_dim*2)
        self.BN = nn.BatchNorm1d(input_dim)

        self.Temporal = Feature_extractor_1DCNN_RUL(input_dim, input_dim*2, input_dim*2,kernel_size=3)
        self.time_conv = time_conv
        self.Spatial = MPNN_mk_v2(2*input_dim, output_dim, k=1)

        self.pre_relation = Mask_Matrix(num_sensors,time_window_size,decay)

        self.pool_choice = pool_choice
    def forward(self, input):
        ## input size (bs, time_length, num_nodes, input_dim)
        bs, time_length, num_nodes, input_dim = input.size()

        # input_con = Conv_GraphST(input, self.time_window_size, self.stride)
        # ## input_con size (bs, num_windows, num_sensors, time_window_size, feature_dim)
        # bs, num_windows, num_sensors, time_window_size, feature_dim = input_con.size()
        # input_con_ = tr.transpose(input_con, 2,3)
        # input_con_ = tr.reshape(input_con_, [bs*num_windows, time_window_size*num_sensors, feature_dim])

        tem_input = tr.transpose(input, 1,2)
        tem_input = tr.reshape(tem_input, [bs*num_nodes, time_length, input_dim])

        tem_output = self.Temporal(tem_input)
        # print(tem_output.size())
        tem_output = tr.reshape(tem_output, [bs, num_nodes, self.time_conv, 2*input_dim])
        spa_input = tr.transpose(tem_output, 1,2)

        spa_input = tr.reshape(spa_input, [bs*self.time_conv, num_nodes, 2*input_dim])
        A_input = self.graph_construction(spa_input)

        spa_output = self.Spatial(spa_input, A_input)



        return spa_output


class GraphMPNNConv_block(nn.Module):
    def __init__(self, input_dim, output_dim, num_sensors, time_window_size, stride, decay):
        super(GraphMPNNConv_block, self).__init__()
        self.time_window_size = time_window_size
        self.stride = stride
        self.output_dim = output_dim

        self.graph_construction = Dot_Graph_Construction_weights(input_dim)
        self.MPNN = MPNN_mk(input_dim, output_dim, k=1)
        self.pre_relation = Mask_Matrix(num_sensors, time_window_size, decay)


    def forward(self, input):
        ## input size (bs, time_length, num_nodes, input_dim)
        ## output size (bs, output_node_t, output_node_s, output_dim)

        input_con = Conv_GraphST(input, self.time_window_size, self.stride)
        ## input_con size (bs, num_windows, num_sensors, time_window_size, feature_dim)
        bs, num_windows, num_sensors, time_window_size, feature_dim = input_con.size()
        input_con_ = tr.transpose(input_con, 2, 3)
        input_con_ = tr.reshape(input_con_, [bs * num_windows, time_window_size * num_sensors, feature_dim])

        A_input = self.graph_construction(input_con_)

        A_input = A_input * self.pre_relation

        X_output = self.MPNN(input_con_, A_input)

        X_output = tr.reshape(X_output, [bs, num_windows, time_window_size, num_sensors, self.output_dim])

        X_output = tr.reshape(X_output, [bs, num_windows*time_window_size, num_sensors, self.output_dim])

        return X_output


class GraphMPNN_block(nn.Module):
    def __init__(self, input_dim, output_dim, num_sensors, time_length, decay):
        super(GraphMPNN_block, self).__init__()

        self.graph_construction = Dot_Graph_Construction_weights(input_dim)
        self.MPNN = MPNN_mk(input_dim, output_dim, k=1)
        self.pre_relation = Mask_Matrix(num_sensors,time_length,decay)

    def forward(self, input):
        bs, tlen, num_sensors, feature_dim = input.size()
        input_con_ = tr.reshape(input, [bs, tlen*num_sensors, feature_dim])

        A_input = self.graph_construction(input_con_)

        A_input = A_input*self.pre_relation

        X_output = self.MPNN(input_con_, A_input)

        X_output = tr.reshape(X_output, [bs, tlen, num_sensors, -1])

        return X_output

