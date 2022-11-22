import torch.nn as nn
import torch.nn.functional as F
import torch

class Net(torch.nn.Module):
    """
    This model is implemented based on the references:
        Paper: https://ojs.aaai.org/index.php/AAAI/article/view/3881
        Code: https://github.com/wanhuaiyu/ASTGCN-r-pytorch
    """

    def __init__(self, device, in_channels, nb_chev_filter, num_time_filter, T, P, num_nodes, time_strides, _1stChebNet,
                 batch_size, D, embedding_size):
        super(Net, self).__init__()
        self.device = device
        self.D = torch.Tensor(D).to(device)
        self.first_conv = nn.Conv2d(in_channels, num_time_filter, kernel_size=[1, 1], stride=[1, 1])
        self.BlockList1 = GAT_block(device=device, in_channels=64, nb_chev_filter=nb_chev_filter,
                                    num_time_filter=num_time_filter, T=T, num_nodes=num_nodes,
                                    time_strides=time_strides, _1stChebNet=_1stChebNet, batch_size=batch_size)
        self.BlockList2 = GAT_block(device=device, in_channels=64, nb_chev_filter=nb_chev_filter,
                                    num_time_filter=num_time_filter, T=T, num_nodes=num_nodes,
                                    time_strides=time_strides, _1stChebNet=_1stChebNet, batch_size=batch_size)
        self.final_conv = nn.Conv2d(T, T, kernel_size=(1, num_time_filter))

        self.weight_con = nn.Parameter(torch.FloatTensor(T, num_nodes).to(device))
        self.bias_con = nn.Parameter(torch.FloatTensor(num_nodes).to(device))
        self.weight_speed = nn.Parameter(torch.FloatTensor(T, num_nodes).to(device))
        self.bias_speed = nn.Parameter(torch.FloatTensor(num_nodes).to(device))

        self.time_embedding = nn.Embedding(embedding_size, 64)
        self.time_linear = nn.Linear(64, 1)

        self.lstm = nn.LSTM(T, 64, 2, batch_first=True)
        self.linear = nn.Linear(64, P)

    def feature_aggregation(self, x):
        wz_list = torch.split(x.permute(0, 3, 1, 2), 1, dim=1)
        sp, wz, vol, history, diff, time, timeofday, mask = wz_list[0], wz_list[1], wz_list[2].squeeze(1), wz_list[
            3].squeeze(1), wz_list[4], wz_list[5].squeeze(1), wz_list[6].squeeze(1), wz_list[7].squeeze(1)

        wz = wz * diff * self.weight_con + self.bias_con
        sp = sp * self.weight_speed + self.bias_speed
        sp = self.first_conv(torch.add(sp, wz))

        t = self.time_embedding(time.type(torch.IntTensor).permute(0,2,1).to(self.device)).permute(0, 3, 2, 1)
        sp_w = torch.add(sp, t)

        return sp_w

    def forward(self, x):
        x = self.feature_aggregation(x)

        x = self.BlockList1(x.permute(0, 2, 3, 1))
        x = self.BlockList2(x)  # [B, N, F, T] -> [B, T, N, F]
        x = self.final_conv(x)[:, :, :, -1]
        x, _ = self.lstm(x.permute(2, 0, 1))
        x = self.linear(x).permute(1, 2, 0)

        return x


class GAT_block(nn.Module):
    def __init__(self, device, in_channels, nb_chev_filter, num_time_filter, T, num_nodes, time_strides, _1stChebNet,
                 batch_size):
        super(GAT_block, self).__init__()
        self.TAL = Temporal_Attention_Layer(in_channels, batch_size)
        self.SAL = Spatial_Attention_Layer(in_channels, batch_size)
        self.cheb_conv_SAt = cheb_conv_withSAt(device, _1stChebNet, in_channels, nb_chev_filter)
        self.time_conv = nn.Conv2d(nb_chev_filter, num_time_filter, kernel_size=(1, 3), stride=(1, time_strides),
                                   padding=(0, 1))
        self.residual_conv = nn.Conv2d(in_channels, num_time_filter, kernel_size=(1, 1), stride=(1, time_strides))
        self.ln = nn.LayerNorm(num_time_filter)

    def forward(self, x):
        # temporal attention
        # The x should be [B, T, N, F_in]
        temporal_attention = self.TAL(x)  # [Batch_size,F_in, T, N]

        X, spatial_attention = self.SAL(temporal_attention)

        # cheb gcn
        spatial_gcn = self.cheb_conv_SAt(X.permute(0, 3, 2, 1), spatial_attention)  # (b, N, F_out, T)
        # convolution along the time axis

        time_conv_output = self.time_conv(spatial_gcn.permute(0, 2, 1, 3))  # (b,N,F,T)->(b,F,N,T)

        # residual shortcut
        x_residual = self.residual_conv(x.permute(0, 3, 2, 1))  # (b,N,F,T)->(b,F,N,T)
        x_residual = self.ln(F.relu(x_residual + time_conv_output).permute(0, 3, 2, 1)).permute(0, 2, 3, 1)
        # (b,F,N,T)->(b,T,N,F) -ln-> (b,T,N,F)->(b,N,F,T)

        return x_residual.permute(0, 3, 1, 2)


class Temporal_Attention_Layer(nn.Module):
    def __init__(self, in_channels, batch_size):
        super(Temporal_Attention_Layer, self).__init__()
        num_heads = 8
        self.batch_size = batch_size
        self.dim_per_head = in_channels // num_heads
        self.num_heads = num_heads
        self.query = nn.Conv2d(in_channels, self.dim_per_head * num_heads, kernel_size=[1, 1], stride=[1, 1])
        self.key = nn.Conv2d(in_channels, self.dim_per_head * num_heads, kernel_size=[1, 1], stride=[1, 1])
        self.value = nn.Conv2d(in_channels, self.dim_per_head * num_heads, kernel_size=[1, 1], stride=[1, 1])
        self.Droupout = nn.Dropout(p=0.3)

    def forward(self, x):
        query = self.query(x.permute(0, 3, 1, 2))
        key = self.key(x.permute(0, 3, 1, 2))
        value = self.value(x.permute(0, 3, 1, 2))

        query = torch.cat(torch.split(query, self.num_heads, dim=1), dim=0)
        key = torch.cat(torch.split(key, self.num_heads, dim=1), dim=0)
        value = torch.cat(torch.split(value, self.num_heads, dim=1), dim=0)

        attention = torch.matmul(query, key.permute(0, 1, 3, 2))
        attention /= (self.dim_per_head ** 0.5)
        attention = F.softmax(attention, dim=-1)

        X = torch.matmul(attention, value)
        X = torch.cat(torch.split(X, self.batch_size, dim=0), dim=1)
        X = self.Droupout(X)
        return X


class Spatial_Attention_Layer(nn.Module):
    def __init__(self, in_channels, batch_size):
        super(Spatial_Attention_Layer, self).__init__()
        num_heads = 8
        self.batch_size = batch_size
        self.in_channel = nn.Parameter(torch.FloatTensor(in_channels))
        self.dim_per_head = in_channels // num_heads
        self.num_heads = num_heads
        self.query = nn.Conv2d(in_channels, self.dim_per_head * num_heads, kernel_size=[1, 1], stride=[1, 1])
        self.key = nn.Conv2d(in_channels, self.dim_per_head * num_heads, kernel_size=[1, 1], stride=[1, 1])
        self.value = nn.Conv2d(in_channels, self.dim_per_head * num_heads, kernel_size=[1, 1], stride=[1, 1])
        self.Droupout = nn.Dropout(p=0.3)

    def forward(self, x):

        query = self.query(x.permute(0, 1, 3, 2))
        key = self.key(x.permute(0, 1, 3, 2))
        value = self.value(x.permute(0, 1, 3, 2))

        query = torch.cat(torch.split(query, self.num_heads, dim=1), dim=0)
        key = torch.cat(torch.split(key, self.num_heads, dim=1), dim=0)
        value = torch.cat(torch.split(value, self.num_heads, dim=1), dim=0)

        attention = torch.matmul(query, key.permute(0, 1, 3, 2))
        attention /= (self.dim_per_head ** 0.5)
        attention = F.softmax(attention, dim=-1)
        X = torch.matmul(attention, value)
        attention = torch.cat(torch.split(attention, self.batch_size, dim=0), dim=1)
        attention = torch.matmul(attention.permute(0, 2, 3, 1), self.in_channel)
        X = torch.cat(torch.split(X, self.batch_size, dim=0), dim=1)
        X = self.Droupout(X)

        return X, attention


class cheb_conv_withSAt(nn.Module):
    def __init__(self, device, _1stChebNet, in_channels, out_channels):
        super(cheb_conv_withSAt, self).__init__()
        self._1stChebNet = _1stChebNet
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = device
        self.Theta = nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE))

    def forward(self, x, spatial_attention):
        x = x.permute(0, 2, 3, 1)
        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape
        T_k_at = self._1stChebNet.mul(spatial_attention)
        outputs = []

        for time_step in range(num_of_timesteps):
            graph_signal = x[:, :, :, time_step]
            theta = self.Theta

            rhs = T_k_at.matmul(graph_signal)

            output = rhs.matmul(theta)

            outputs.append(output.unsqueeze(-1))

        return F.relu(torch.cat(outputs, dim=-1))


from utils import hypergraph_utils as hgut
import numpy as np

def GCNRWZ_model(args, dis, D):

    H = []
    n_knn = [5]
    for k in n_knn:
        H_tmp = hgut.H_KNN_distance(np.mat(dis), k)
        H.append(H_tmp)

    if args.expt_name == "richmond":
        embedding_size = 7 * 24 * 4 + 1
    else:
        embedding_size = 7 * 24 * 12 + 1

    # Hypergraph A_wave
    G = hgut.G_from_H(H, variable_weight=False)
    G = torch.from_numpy(G).type(torch.FloatTensor).to(args.device)

    nb_chev_filter = 64
    model = Net(args.device, args.in_channels, nb_chev_filter, 64, args.T, args.P, args.num_nodes, 1, G,
                args.batch_size, D, embedding_size)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)

    return model