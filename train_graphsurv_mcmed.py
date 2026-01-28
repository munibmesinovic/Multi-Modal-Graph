import numpy as np
import numba
import argparse
import time
import gc
import random
from math import ceil

import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import numba
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer

from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import pickle
import scipy.integrate
import warnings

from torch import Tensor

import seaborn as sn
sn.set_theme(style="white", palette="rocket_r")

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from torch import Tensor
from torch.nn import init
from torch.nn.parameter import Parameter

from torch.utils.data import TensorDataset

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")


from pycox.models import DeepHit
from pycox.evaluation import EvalSurv
from pycox.models import utils


durations_test = np.load('durations_test_MCMED.npy')
events_test = np.load('events_test_MCMED.npy')
x_train = np.load('x_train_MCMED.npy')
x_val = np.load('x_val_MCMED.npy')
x_test = np.load('x_test_MCMED.npy')
out_features = int(np.load('out_features_MCMED.npy'))
cuts = np.load('cuts_MCMED.npy')

with open('y_train_surv_MCMED.p', 'rb') as file:
    y_train_surv = pickle.load(file)

with open('y_val_surv_MCMED.p', 'rb') as file:
    y_val_surv = pickle.load(file)


train = tt.tuplefy(x_train, (y_train_surv, x_train))
val = (x_val, y_val_surv)

# init [num_variables, seq_length, num_classes]
num_nodes = x_val.shape[2]
seq_length = x_val.shape[1]

class multi_shallow_embedding(nn.Module):

    def __init__(self, num_nodes, k_neighs, num_graphs):
        super().__init__()

        self.num_nodes = num_nodes
        self.k = k_neighs
        self.num_graphs = num_graphs

        self.emb_s = Parameter(Tensor(num_graphs, num_nodes, 1))
        self.emb_t = Parameter(Tensor(num_graphs, 1, num_nodes))

    def reset_parameters(self):
        init.xavier_uniform_(self.emb_s)
        init.xavier_uniform_(self.emb_t)


    def forward(self, device):

        # adj: [G, N, N]
        adj = torch.matmul(self.emb_s, self.emb_t).to(device)

        # remove self-loop
        adj = adj.clone()
        idx = torch.arange(self.num_nodes, dtype=torch.long, device=device)
        adj[:, idx, idx] = float('-inf')

        # top-k-edge adj
        adj_flat = adj.reshape(self.num_graphs, -1)
        indices = adj_flat.topk(k=self.k)[1].reshape(-1)

        idx = torch.tensor([ i//self.k for i in range(indices.size(0)) ], device=device)

        adj_flat = torch.zeros_like(adj_flat).clone()
        adj_flat[idx, indices] = 1.
        adj = adj_flat.reshape_as(adj)

        return adj

class multi_shallow_embedding_with_static(nn.Module):
    def __init__(self, num_nodes, k_neighs, num_graphs, num_static_nodes, num_icd_codes, num_reports_features):
        super().__init__()

        self.num_nodes = num_nodes  # Number of nodes for dynamic features
        self.k = k_neighs  # Top-k edges for dynamic adjacency
        self.num_graphs = num_graphs  # Number of graphs (groups)
        self.num_static_nodes = num_static_nodes  # Number of nodes for static features
        self.num_icd_codes = num_icd_codes  # Number of ICD codes
        self.num_reports_features = num_reports_features  # Number of reports features

        # Learnable embeddings for dynamic features
        self.emb_s_dynamic = Parameter(Tensor(num_graphs, num_nodes, 1))
        self.emb_t_dynamic = Parameter(Tensor(num_graphs, 1, num_nodes))

        # Learnable embeddings for static features
        self.emb_s_static = Parameter(Tensor(1, num_static_nodes, 1))
        self.emb_t_static = Parameter(Tensor(1, 1, num_static_nodes))

        # Learnable embeddings for ICD features
        self.emb_s_icd = Parameter(Tensor(1, num_icd_codes, 1))
        self.emb_t_icd = Parameter(Tensor(1, 1, num_icd_codes))

        # Learnable embeddings for reports features
        self.emb_s_reports = Parameter(Tensor(1, num_reports_features, 1))
        self.emb_t_reports = Parameter(Tensor(1, 1, num_reports_features))

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize dynamic embeddings
        init.xavier_uniform_(self.emb_s_dynamic)
        init.xavier_uniform_(self.emb_t_dynamic)

        # Initialize static embeddings
        init.xavier_uniform_(self.emb_s_static)
        init.xavier_uniform_(self.emb_t_static)

        # Initialize ICD embeddings
        init.xavier_uniform_(self.emb_s_icd)
        init.xavier_uniform_(self.emb_t_icd)

        # Initialize reports embeddings
        init.xavier_uniform_(self.emb_s_reports)
        init.xavier_uniform_(self.emb_t_reports)

    def forward(self, device):

        # Dynamic adjacency matrix
        adj_dynamic = torch.matmul(self.emb_s_dynamic, self.emb_t_dynamic).to(device)

        # Static adjacency matrix
        adj_static = torch.matmul(self.emb_s_static, self.emb_t_static).to(device)

        # ICD adjacency matrix
        adj_icd = torch.matmul(self.emb_s_icd, self.emb_t_icd).to(device)

        # Reports adjacency matrix
        adj_reports = torch.matmul(self.emb_s_reports, self.emb_t_reports).to(device)

        # Remove self-loops for dynamic adjacency
        adj_dynamic = adj_dynamic.clone()
        idx_dynamic = torch.arange(self.num_nodes, dtype=torch.long, device=device)
        adj_dynamic[:, idx_dynamic, idx_dynamic] = float('-inf')

        # Remove self-loops for static adjacency
        adj_static = adj_static.clone()
        idx_static = torch.arange(self.num_static_nodes, dtype=torch.long, device=device)
        adj_static[:, idx_static, idx_static] = float('-inf')

        # Remove self-loops for ICD adjacency
        adj_icd = adj_icd.clone()
        idx_icd = torch.arange(self.num_icd_codes, dtype=torch.long, device=device)
        adj_icd[:, idx_icd, idx_icd] = float('-inf')

        # Remove self-loops for reports adjacency
        adj_reports = adj_reports.clone()
        idx_reports = torch.arange(self.num_reports_features, dtype=torch.long, device=device)
        adj_reports[:, idx_reports, idx_reports] = float('-inf')

        # Top-k-edge adj for dynamic features
        adj_dynamic_flat = adj_dynamic.reshape(self.num_graphs, -1)
        indices_dynamic = adj_dynamic_flat.topk(k=self.k)[1].reshape(-1)
        idx_dynamic = torch.tensor([i // self.k for i in range(indices_dynamic.size(0))], device=device)
        adj_dynamic_flat = torch.zeros_like(adj_dynamic_flat).clone()
        adj_dynamic_flat[idx_dynamic, indices_dynamic] = 1.
        adj_dynamic = adj_dynamic_flat.reshape_as(adj_dynamic)

        # Top-k-edge adj for static features (if needed)
        adj_static_flat = adj_static.reshape(1, -1)
        indices_static = adj_static_flat.topk(k=self.k)[1].reshape(-1)
        idx_static = torch.tensor([i // self.k for i in range(indices_static.size(0))], device=device)
        adj_static_flat = torch.zeros_like(adj_static_flat).clone()
        adj_static_flat[idx_static, indices_static] = 1.
        adj_static = adj_static_flat.reshape_as(adj_static)

        # Top-k-edge adj for ICD features (if needed)
        adj_icd_flat = adj_icd.reshape(1, -1)
        indices_icd = adj_icd_flat.topk(k=self.k)[1].reshape(-1)
        idx_icd = torch.tensor([i // self.k for i in range(indices_icd.size(0))], device=device)
        adj_icd_flat = torch.zeros_like(adj_icd_flat).clone()
        adj_icd_flat[idx_icd, indices_icd] = 1.
        adj_icd = adj_icd_flat.reshape_as(adj_icd)

        # Top-k-edge adj for reports features (if needed)
        adj_reports_flat = adj_reports.reshape(1, -1)
        indices_reports = adj_reports_flat.topk(k=self.k)[1].reshape(-1)
        idx_reports = torch.tensor([i // self.k for i in range(indices_reports.size(0))], device=device)
        adj_reports_flat = torch.zeros_like(adj_reports_flat).clone()
        adj_reports_flat[idx_reports, indices_reports] = 1.
        adj_reports = adj_reports_flat.reshape_as(adj_reports)

        return adj_dynamic, adj_static, adj_icd, adj_reports

class Group_Linear(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, bias=False):
        super().__init__()

        self.out_channels = out_channels
        self.groups = groups

        self.group_mlp = nn.Conv2d(in_channels * groups, out_channels * groups, kernel_size=(1, 1), groups=groups, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.group_mlp.reset_parameters()


    def forward(self, x: Tensor, is_reshape: False):
        """
        Args:
            x (Tensor): [B, C, N, F] (if not is_reshape), [B, C, G, N, F//G] (if is_reshape)
        """
        B = x.size(0)
        C = x.size(1)
        N = x.size(-2)
        G = self.groups

        if not is_reshape:
            # x: [B, C_in, G, N, F//G]
            x = x.reshape(B, C, N, G, -1).transpose(2, 3)
        # x: [B, G*C_in, N, F//G]
        x = x.transpose(1, 2).reshape(B, G*C, N, -1)

        out = self.group_mlp(x)
        out = out.reshape(B, G, self.out_channels, N, -1).transpose(1, 2)

        # out: [B, C_out, G, N, F//G]
        return out


class DenseGCNConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, bias=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin = Group_Linear(in_channels, out_channels, groups, bias=False)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        init.zeros_(self.bias)

    def norm(self, adj: Tensor, add_loop):
        if add_loop:
            adj = adj.clone()
            idx = torch.arange(adj.size(-1), dtype=torch.long, device=adj.device)
            adj[:, idx, idx] += 1

        deg_inv_sqrt = adj.sum(-1).clamp(min=1).pow(-0.5)

        adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)

        return adj


    def forward(self, x: Tensor, adj: Tensor, add_loop=True):
        """
        Args:
            x (Tensor): [B, C, N, F]
            adj (Tensor): [B, G, N, N]
        """
        adj = self.norm(adj, add_loop).unsqueeze(1)

        # x: [B, C, G, N, F//G]
        x = self.lin(x, False)

        adj = adj.repeat(1, x.shape[1], 1, 1, 1) # adj: [B, C, G, N, N]

        out = torch.matmul(adj, x)

        # out: [B, C, N, F]
        B, C, _, N, _ = out.size()
        out = out.transpose(2, 3).reshape(B, C, N, -1)

        if self.bias is not None:
            out = out.transpose(1, -1) + self.bias
            out = out.transpose(1, -1)

        return out


class DenseGINConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, eps=0, train_eps=True):
        super().__init__()

        self.mlp = Group_Linear(in_channels, out_channels, groups, bias=False)

        # Encoder part
        self.encoder_mean = Group_Linear(in_channels, out_channels, groups, bias=False)
        self.encoder_logvar = Group_Linear(in_channels, out_channels, groups, bias=False)

        # Decoder part (similar to the original DenseGINConv2d)
        self.mlp = Group_Linear(out_channels, in_channels, groups, bias=False)  # Adjust output channels

        self.init_eps = eps
        if train_eps:
            self.eps = Parameter(Tensor([eps]))
        else:
            self.register_buffer('eps', Tensor([eps]))

        self.reset_parameters()

    def reset_parameters(self):
        self.mlp.reset_parameters()
        self.eps.data.fill_(self.init_eps)

    def norm(self, adj: Tensor, add_loop):
        if add_loop:
            adj = adj.clone()
            idx = torch.arange(adj.size(-1), dtype=torch.long, device=adj.device)
            adj[..., idx, idx] += 1

        deg_inv_sqrt = adj.sum(-1).clamp(min=1).pow(-0.5)

        adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)

        return adj
    def reparameterize(self, mean, logvar):
        # Add an epsilon to prevent very large values
        epsilon = 1e-7
        std = torch.exp(0.5 * logvar) + epsilon
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x: Tensor, adj: Tensor, add_loop=True):
        """
        Args:
            x (Tensor): [B, C, N, F]
            adj (Tensor): [G, N, N]
        """
        B, C, N, _ = x.size()
        G = adj.size(0)

        # adj-norm
        adj = self.norm(adj, add_loop=False)

        # x: [B, C, G, N, F//G]
        x = x.reshape(B, C, N, G, -1).transpose(2, 3)

        out = torch.matmul(adj, x)

        # DYNAMIC
        x_pre = x[:, :, :-1, ...]

        # out = x[:, :, 1:, ...] + x_pre
        out[:, :, 1:, ...] = out[:, :, 1:, ...] + x_pre
        # out = torch.cat( [x[:, :, 0, ...].unsqueeze(2), out], dim=2 )

        if add_loop:
            out = (1 + self.eps) * x + out

        # out: [B, C, G, N, F//G]
        out = self.mlp(out, True)

        # out: [B, C, N, F]
        C = out.size(1)
        out2 = out.transpose(2, 3).reshape(B, C, N, -1)

        return out2

class Dense_TimeDiffPool2d(nn.Module):

    def __init__(self, pre_nodes, pooled_nodes, kern_size, padding):
        super().__init__()

        # TODO: add Normalization
        self.time_conv = nn.Conv2d(pre_nodes, pooled_nodes, (1, kern_size), padding=(0, padding))

        self.re_param = Parameter(Tensor(kern_size, 1))

    def reset_parameters(self):
        self.time_conv.reset_parameters()
        init.kaiming_uniform_(self.re_param, nonlinearity='relu')


    def forward(self, x: Tensor, adj: Tensor):
        """
        Args:
            x (Tensor): [B, C, N, F]
            adj (Tensor): [G, N, N]
        """
        x = x.transpose(1, 2)
        out = self.time_conv(x)
        out = out.transpose(1, 2)

        # s: [ N^(l+1), N^l, 1, K ]
        s = torch.matmul(self.time_conv.weight, self.re_param).view(out.size(-2), -1)

        out_adj = torch.matmul(torch.matmul(s, adj), s.transpose(0, 1))

        return out, out_adj


class DiffPoolLayer(nn.Module):
    def __init__(self, in_nodes, out_nodes, in_dim, hidden_dim):
        super().__init__()
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes

        # Pooling network (learns soft assignment)
        self.pool_net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_nodes)
        )

    def forward(self, x, adj):

        batch_size = x.size(0)

        # Expand adj to match batch size
        adj = adj.expand(batch_size, -1, -1)  # Shape: [batch_size, in_nodes, in_nodes]

        x = x.squeeze(1)  # Ensure shape is [128, 500]

        if x.dim() == 2:  # If shape is [batch_size, 500], reshape it
            x = x.unsqueeze(-1)  # [batch_size, 500, 1]

        S = self.pool_net(x)  # Ensure shape is [128, 500, 50]

        S = F.softmax(S, dim=-1)

        pooled_x = torch.matmul(S.transpose(1, 2), x)  # Fix should now work

        # Pool adjacency matrix
        S_transposed = S.transpose(1, 2)  # Shape: [batch_size, out_nodes, in_nodes]

        intermediate = torch.matmul(S_transposed, adj)  # Shape: [batch_size, out_nodes, in_nodes]

        pooled_adj = torch.matmul(intermediate, S)  # Shape: [batch_size, out_nodes, out_nodes]

        return pooled_x, pooled_adj

class GNNStack(nn.Module):

    def __init__(self, num_risks, gnn_model_type, num_layers, groups, pool_ratio, kern_size,
                 in_dim, hidden_dim, out_dim,
                 seq_len, num_nodes_total, num_nodes_dynamic, num_nodes_static, num_icd_codes, num_reports_features,
                 num_nodes_indiv, num_classes, batch_norm=True, dropout=0.7, activation=nn.ReLU()):

        super().__init__()


        self.attention_matrix = nn.Parameter(torch.randn(groups, num_nodes_total, num_nodes_total))
        nn.init.xavier_uniform_(self.attention_matrix)

        # Hierarchical Attention Matrices for Each Modality
        self.att_dynamic = nn.Parameter(torch.randn(groups, num_nodes_dynamic, num_nodes_dynamic))
        self.att_static = nn.Parameter(torch.randn(groups, num_nodes_static, num_nodes_static))
        self.att_icd = nn.Parameter(torch.randn(groups, 50, 50))
        self.att_reports = nn.Parameter(torch.randn(groups, 50, 50))

        # Initialize attention matrices
        nn.init.xavier_uniform_(self.att_dynamic)
        nn.init.xavier_uniform_(self.att_static)
        nn.init.xavier_uniform_(self.att_icd)
        nn.init.xavier_uniform_(self.att_reports)

        # Top-Level Attention for Combined Graph
        self.att_top_level = nn.Parameter(torch.randn(groups, num_nodes_total, num_nodes_total))
        nn.init.xavier_uniform_(self.att_top_level)

        k_neighs = self.num_nodes_dynamic = num_nodes_dynamic

        self.num_graphs = groups

        self.num_feats = seq_len
        if seq_len % groups:
            self.num_feats += ( groups - seq_len % groups )

        self.g_constr = multi_shallow_embedding_with_static(num_nodes_dynamic, k_neighs, self.num_graphs, num_nodes_static, num_icd_codes, num_reports_features)

        gnn_model, heads = self.build_gnn_model(gnn_model_type)

        assert num_layers >= 1, 'Error: Number of layers is invalid.'
        assert num_layers == len(kern_size), 'Error: Number of kernel_size should equal to number of layers.'
        paddings = [ (k - 1) // 2 for k in kern_size ]

        self.gconvs = nn.ModuleList(
            [gnn_model(in_dim, heads * in_dim, groups)] +
            [gnn_model(hidden_dim, heads * hidden_dim, groups) for _ in range(num_layers - 2)] +
            [gnn_model(out_dim, heads * out_dim, groups)]
        )

        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(heads * in_dim)] +
            [nn.BatchNorm2d(heads * hidden_dim) for _ in range(num_layers - 2)] +
            [nn.BatchNorm2d(heads * out_dim)]
        )

        self.left_num_nodes = []
        for layer in range(num_layers + 1):
            left_node = round( num_nodes_total * (1 - (pool_ratio*layer)) )
            if left_node > 0:
                self.left_num_nodes.append(left_node)
            else:
                self.left_num_nodes.append(1)
        self.diffpool = nn.ModuleList(
            [Dense_TimeDiffPool2d(self.left_num_nodes[layer], self.left_num_nodes[layer+1], kern_size[layer], paddings[layer]) for layer in range(num_layers - 1)] +
            [Dense_TimeDiffPool2d(self.left_num_nodes[-2], self.left_num_nodes[-1], kern_size[-1], paddings[-1])]
        )

        self.icdpool = DiffPoolLayer(num_icd_codes, 50, num_icd_codes, hidden_dim)
        self.reportspool = DiffPoolLayer(num_reports_features, 50, num_reports_features, hidden_dim)

        self.num_layers = num_layers
        self.dropout = dropout
        self.activation = activation

        self.softmax = nn.Softmax(dim=-1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.linear = nn.Linear(heads * out_dim, out_dim)

        self.risk_nets = torch.nn.ModuleList()
        for _ in range(num_risks):
            net = tt.practical.MLPVanilla(
                out_dim, num_nodes_indiv, num_classes,
                batch_norm, dropout,
            )
            self.risk_nets.append(net)

        self.reset_parameters()


    def reset_parameters(self):
        for gconv, bn, pool in zip(self.gconvs, self.bns, self.diffpool):
            gconv.reset_parameters()
            bn.reset_parameters()
            pool.reset_parameters()

        self.linear.reset_parameters()


    def build_gnn_model(self, model_type):
        if model_type == 'dyGCN2d':
            return DenseGCNConv2d, 1
        if model_type == 'dyGIN2d':
            return DenseGINConv2d, 1


    def forward(self, inputs: Tensor):

        inputs = torch.swapaxes(inputs, 1, 2)
        inputs = inputs[:, None, :, :]
        assert inputs.shape == (inputs.size(dim=0), 1, 1329, 24)

        # Separate static and dynamic features
        static_features = inputs[:, :, 38:61, 0]
        dynamic_features = inputs[:, :, :38, :]
        report_features = inputs[:, :, 61:829, 0]
        icd_features = inputs[:, :, 829:, 0]

        if inputs.size(-1) % self.num_graphs:
            pad_size = (self.num_graphs - inputs.size(-1) % self.num_graphs) / 2
            x = F.pad(inputs, (int(pad_size), ceil(pad_size)), mode='constant', value=0.0)
        else:
            x = inputs

        # Generate adjacency matrices
        adj_dynamic, adj_static, adj_icd, adj_reports = self.g_constr(x.device)

        adj_static = adj_static.repeat(6, 1, 1)

        # Apply ICDPool
        pooled_x, pooled_adj = self.icdpool(icd_features, adj_icd)

        pooled_x = pooled_x.repeat(1, 1, 24)
        pooled_x = pooled_x[:, None, :, :]

        pooled_adj = pooled_adj[0, :, :]
        pooled_adj = pooled_adj.repeat(6, 1, 1)

        # Apply pool for Reports
        pooled_x_reports, pooled_adj_reports = self.reportspool(report_features, adj_reports)

        pooled_x_reports = pooled_x_reports.repeat(1, 1, 24)
        pooled_x_reports = pooled_x_reports[:, None, :, :]

        pooled_adj_reports = pooled_adj_reports[0, :, :]
        pooled_adj_reports = pooled_adj_reports.repeat(6, 1, 1)

        # Apply modality-specific attention separately
        att_dynamic = self.softmax(self.att_dynamic)
        att_static = self.softmax(self.att_static)
        att_icd = self.softmax(self.att_icd)
        att_reports = self.softmax(self.att_reports)

        adj_dynamic = adj_dynamic * att_dynamic
        adj_static = adj_static * att_static
        pooled_adj = pooled_adj * att_icd
        pooled_adj_reports = pooled_adj_reports * att_reports

        # Initialize combined adjacency matrix (shape: [6, 111, 111])
        adj = torch.zeros(6, 161, 161)

        # Fill the top-left block with the static adjacency matrix
        adj[:, 38:61, 38:61] = adj_static

        # Fill the bottom-right block with the dynamic adjacency matrices
        adj[:, :38, :38] = adj_dynamic

        # Fill the bottom-right block with the ICD matrices
        adj[:, 61:111, 61:111] = pooled_adj

        # Fill the block with the reports adjacency matrices
        adj[:, 111:, 111:] = pooled_adj_reports

        adj = adj.to(x.device)

        # Extract first 61 columns from x
        x_preserved = x[:, :, :61, :]  # Shape: [128, 1, 61, 24]

        # Concatenate with pooled_x along the second dimension (feature axis)
        x = torch.cat((x_preserved, pooled_x, pooled_x_reports), dim=2)  # Shape: [128, 1, 161, 24]

        # Attention layer
        attention_scores = F.softmax(self.attention_matrix, dim=-1)

        adj = adj * attention_scores

        for gconv, bn, pool in zip(self.gconvs, self.bns, self.diffpool):
            # print(x.shape) torch.Size([32, 1, 80, 24])
            # x1 = tconv(x)  # Assuming x is the output from previous layer
            # # Apply GNN layer and other operations
            # print(x1.shape)

            s=x.shape[1]
            if s==1:
               x1=x.repeat(1, 128, 1,1)
            else:
               x1=x.repeat(1, 2, 1,1)

            temp = gconv(x1, adj)

            x, adj = pool(temp, adj)

            del temp, x1, adj  # Delete intermediate tensors that are no longer needed

            gc.collect()     # Explicitly trigger garbage collection
            torch.cuda.empty_cache()  # Empty the CUDA cache

            x = self.activation(bn(x))
            x = F.dropout(x, p=self.dropout, training=self.training)

        out = self.global_pool(x)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = [net(out) for net in self.risk_nets]
        out = torch.stack(out, dim=1)

        return out, attention_scores, att_dynamic, att_static, pooled_adj, pooled_adj_reports

    def predict(self, inputs: Tensor):

        inputs = torch.swapaxes(inputs, 1, 2)
        inputs = inputs[:, None, :, :]
        assert inputs.shape == (inputs.size(dim=0), 1, 1329, 24)

        # Separate static and dynamic features
        static_features = inputs[:, :, 38:61, 0]
        dynamic_features = inputs[:, :, :38, :]
        report_features = inputs[:, :, 61:829, 0]
        icd_features = inputs[:, :, 829:, 0]

        if inputs.size(-1) % self.num_graphs:
            pad_size = (self.num_graphs - inputs.size(-1) % self.num_graphs) / 2
            x = F.pad(inputs, (int(pad_size), ceil(pad_size)), mode='constant', value=0.0)
        else:
            x = inputs

        # Generate adjacency matrices
        adj_dynamic, adj_static, adj_icd, adj_reports = self.g_constr(x.device)

        adj_static = adj_static.repeat(6, 1, 1)

        # Apply ICDPool
        pooled_x, pooled_adj = self.icdpool(icd_features, adj_icd)

        pooled_x = pooled_x.repeat(1, 1, 24)
        pooled_x = pooled_x[:, None, :, :]

        pooled_adj = pooled_adj[0, :, :]
        pooled_adj = pooled_adj.repeat(6, 1, 1)

        # Apply pool for Reports
        pooled_x_reports, pooled_adj_reports = self.reportspool(report_features, adj_reports)

        pooled_x_reports = pooled_x_reports.repeat(1, 1, 24)
        pooled_x_reports = pooled_x_reports[:, None, :, :]

        pooled_adj_reports = pooled_adj_reports[0, :, :]
        pooled_adj_reports = pooled_adj_reports.repeat(6, 1, 1)

        # Apply **modality-specific attention** separately
        att_dynamic = self.softmax(self.att_dynamic)
        att_static = self.softmax(self.att_static)
        att_icd = self.softmax(self.att_icd)
        att_reports = self.softmax(self.att_reports)

        adj_dynamic = adj_dynamic * att_dynamic
        adj_static = adj_static * att_static
        pooled_adj = pooled_adj * att_icd
        pooled_adj_reports = pooled_adj_reports * att_reports

        # Initialise combined adjacency matrix (shape: [6, 111, 111])
        adj = torch.zeros(6, 161, 161)

        # Fill the top-left block with the static adjacency matrix
        adj[:, 38:61, 38:61] = adj_static

        # Fill the bottom-right block with the dynamic adjacency matrices
        adj[:, :38, :38] = adj_dynamic

        # Fill the bottom-right block with the ICD matrices
        adj[:, 61:111, 61:111] = pooled_adj

        # Fill the block with the reports adjacency matrices
        adj[:, 111:, 111:] = pooled_adj_reports

        adj = adj.to(x.device)

        # Extract first 61 columns from x
        x_preserved = x[:, :, :61, :]  # Shape: [128, 1, 61, 24]

        # Concatenate with pooled_x along the second dimension (feature axis)
        x = torch.cat((x_preserved, pooled_x, pooled_x_reports), dim=2)  # Shape: [128, 1, 161, 24]

        # Attention layer
        attention_scores = F.softmax(self.attention_matrix, dim=-1)

        adj = adj * attention_scores

        for gconv, bn, pool in zip(self.gconvs, self.bns, self.diffpool):
            # print(x.shape) torch.Size([32, 1, 80, 24])
            # x1 = tconv(x)  # Assuming x is the output from previous layer
            # # Apply GNN layer and other operations
            # print(x1.shape)

            s=x.shape[1]
            if s==1:
               x1=x.repeat(1, 128, 1,1)
            else:
               x1=x.repeat(1, 2, 1,1)

            temp = gconv(x1, adj)

            x, adj = pool(temp, adj)

            del temp, x1, adj  # Delete intermediate tensors that are no longer needed

            gc.collect()     # Explicitly trigger garbage collection
            torch.cuda.empty_cache()  # Empty the CUDA cache

            x = self.activation(bn(x))
            x = F.dropout(x, p=self.dropout, training=self.training)

        out = self.global_pool(x)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = [net(out) for net in self.risk_nets]
        out = torch.stack(out, dim=1)

        return out

def pad_col(input, val=0, where='end'):
    """Addes a column of `val` at the start of end of `input`."""
    if len(input.shape) != 2:
        raise ValueError(f"Only works for `phi` tensor that is 2-D.")
    pad = torch.zeros_like(input[:, :1])
    if val != 0:
        pad = pad + val
    if where == 'end':
        return torch.cat([input, pad], dim=1)
    elif where == 'start':
        return torch.cat([pad, input], dim=1)
    raise ValueError(f"Need `where` to be 'start' or 'end', got {where}")

def array_or_tensor(tensor, numpy, input):
    warnings.warn('Use `torchtuples.utils.array_or_tensor` instead', DeprecationWarning)
    return tt.utils.array_or_tensor(tensor, numpy, input)

def make_subgrid(grid, sub=1):
    subgrid = tt.TupleTree(np.linspace(start, end, num=sub+1)[:-1]
                        for start, end in zip(grid[:-1], grid[1:]))
    subgrid = subgrid.apply(lambda x: tt.TupleTree(x)).flatten() + (grid[-1],)
    return subgrid

def log_softplus(input, threshold=-15.):
    """Equivalent to 'F.softplus(input).log()', but for 'input < threshold',
    we return 'input', as this is approximately the same.
    """
    output = input.clone()
    above = input >= threshold
    output[above] = F.softplus(input[above]).log()
    return output

def cumsum_reverse(input: torch.Tensor, dim: int = 1) -> torch.Tensor:
    if dim != 1:
        raise NotImplementedError
    input = input.sum(1, keepdim=True) - pad_col(input, where='start').cumsum(1)
    return input[:, :-1]

class _Loss(torch.nn.Module):
    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__()
        self.reduction = reduction

def _reduction(Loss: Tensor, reduction: str = 'mean') -> Tensor:
    if reduction == 'none':
        return Loss
    elif reduction == 'mean':
        return Loss.mean()
    elif reduction == 'sum':
        return Loss.sum()
    raise ValueError(f"`reduction` = {reduction} is not valid. Use 'none', 'mean' or 'sum'.")

def nll_logistic_hazard(phi: Tensor, idx_durations: Tensor, events: Tensor,
                        reduction: str = 'mean') -> Tensor:
    if phi.shape[1] <= idx_durations.max():
        raise ValueError(f"Network output `phi` is too small for `idx_durations`."+
                         f" Need at least `phi.shape[1] = {idx_durations.max().item()+1}`,"+
                         f" but got `phi.shape[1] = {phi.shape[1]}`")
    if events.dtype is torch.bool:
        events = events.float()
    phi = phi.float()
    events = events.float()
    events = events.view(-1, 1)
    idx_durations = idx_durations.view(-1, 1)
    y_bce = torch.zeros_like(phi).scatter(1, idx_durations, events)
    bce = F.binary_cross_entropy_with_logits(phi, y_bce, reduction='none')
    Loss = bce.cumsum(1).gather(1, idx_durations).view(-1)
    return _reduction(Loss, reduction)

def nll_pmf_cr(phi: Tensor, idx_durations: Tensor, events: Tensor, reduction: str = 'mean',
               epsilon: float = 1e-7) -> Tensor:
    """Negative log-likelihood for PMF parameterizations. `phi` is the ''logit''.
    """
    # Should improve numerical stability by, e.g., log-sum-exp trick.
    # if events.dtype is torch.bool:
    #     events = events.float()
    events = events.view(-1) - 1
    events = events.int()
    # phi = phi.float()
    # events = events.float()
    event_01 = (events != -1).float()
    idx_durations = idx_durations.view(-1).int()
    batch_size = phi.size(0)
    sm = pad_col(phi.view(batch_size, -1)).softmax(1)[:, :-1].view(phi.shape)
    index = torch.arange(batch_size)
    part1 = sm[index, events, idx_durations].relu().add(epsilon).log().mul(event_01)
    part2 = (1 - sm.cumsum(2)[index, :, idx_durations].sum(1)).relu().add(epsilon).log().mul(1 - event_01)
    Loss = - part1.add(part2)
    return _reduction(Loss, reduction)

def _diff_cdf_at_time_i(pmf: Tensor, y: Tensor) -> Tensor:
    """R is the matrix giving the difference in CDF between individual
    i and j, at the event time of j.
    """
    n = pmf.shape[0]
    ones = torch.ones((n, 1), device=pmf.device)
    r = pmf.cumsum(1).matmul(y.transpose(0, 1))
    diag_r = r.diag().view(1, -1)
    r = ones.matmul(diag_r) - r
    return r.transpose(0, 1)

def rank_Loss(pmf: Tensor, y: Tensor, rank_mat: Tensor, sigma: float,
                       reduction: str = 'mean') -> Tensor:
    r = _diff_cdf_at_time_i(pmf, y)
    Loss = rank_mat * torch.exp(-r/sigma)
    Loss = Loss.mean(1, keepdim=True)
    return _reduction(Loss, reduction)

def rank_Loss_cr(phi: Tensor, idx_durations: Tensor, events: Tensor, rank_mat: Tensor,
                         sigma: float, reduction: str = 'mean') -> Tensor:
    if events.dtype is torch.bool:
        events = events.float()
    phi = phi.float()
    events = events.float()

    idx_durations = idx_durations.view(-1)
    events = events.view(-1) - 1
    event_01 = (events == -1).float()

    batch_size, n_risks = phi.shape[:2]
    pmf = pad_col(phi.view(batch_size, -1)).softmax(1)
    pmf = pmf[:, :-1].view(phi.shape)
    y = torch.zeros_like(pmf)
    y[torch.arange(batch_size), :, idx_durations] = 1.

    Loss = []
    for i in range(n_risks):
        rank_Loss_i = rank_Loss(pmf[:, i, :], y[:, i, :], rank_mat, sigma, 'none')
        Loss.append(rank_Loss_i.view(-1) * (events == i).float())

    if reduction == 'none':
        return sum(Loss)
    elif reduction == 'mean':
        return sum([lo.mean() for lo in Loss])
    elif reduction == 'sum':
        return sum([lo.sum() for lo in Loss])
    return _reduction(Loss, reduction)

class NLLLogistiHazardLoss(_Loss):
    def forward(self, phi: Tensor, idx_durations: Tensor, events: Tensor) -> Tensor:
        return nll_logistic_hazard(phi, idx_durations, events, self.reduction)

class _Loss(_Loss):
    def __init__(self, alpha: float, sigma: float, reduction: str = 'mean') -> None:
        super().__init__(reduction)
        self.alpha = alpha
        self.sigma = sigma

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float) -> None:
        if (alpha < 0) or (alpha > 1):
            raise ValueError(f"Need `alpha` to be in [0, 1]. Got {alpha}.")
        self._alpha = alpha

    @property
    def sigma(self) -> float:
        return self._sigma

    @sigma.setter
    def sigma(self, sigma: float) -> None:
        if sigma <= 0:
            raise ValueError(f"Need `sigma` to be positive. Got {sigma}.")
        self._sigma = sigma

class Loss(_Loss):
    def forward(self, phi: Tensor, idx_durations: Tensor, events: Tensor, rank_mat: Tensor) -> Tensor:
        nll =  nll_pmf_cr(phi, idx_durations, events, self.reduction)
        rank_Loss = rank_Loss_cr(phi, idx_durations, events, rank_mat, self.sigma, self.reduction)
        return self.alpha * nll + (1. - self.alpha) * rank_Loss

def _pair_rank_mat(mat, idx_durations, events, dtype='float32'):
    n = len(idx_durations)
    for i in range(n):
        dur_i = idx_durations[i]
        ev_i = events[i]
        if ev_i == 0:
            continue
        for j in range(n):
            dur_j = idx_durations[j]
            ev_j = events[j]
            if (dur_i < dur_j) or ((dur_i == dur_j) and (ev_j == 0)):
                mat[i, j] = 1
    return mat

def pair_rank_mat(idx_durations, events, dtype='float32'):
    idx_durations = idx_durations.reshape(-1)
    events = events.reshape(-1)
    n = len(idx_durations)
    mat = np.zeros((n, n), dtype=dtype)
    mat = _pair_rank_mat(mat, idx_durations, events, dtype)
    return mat

class Loss(nn.Module):
    def __init__(self, alpha: list, sigma: float = 0.1):
        super().__init__()
        self.alpha = alpha
        # self.Loss_surv = NLLLogistiHazardLoss()
        self.Loss_surv = Loss(alpha=alpha[0], sigma=sigma)

    def forward(self, phi, attention, att_dynamic, att_static, pooled_adj, pooled_adj_reports, idx_durations, events, rank_mat):
        rank_mat = pair_rank_mat(idx_durations.cpu().numpy(), events.cpu().numpy())
        rank_mat = torch.tensor(rank_mat, dtype=phi.dtype, device=phi.device)
        Loss_surv = self.Loss_surv(phi, idx_durations, events, rank_mat)
        return Loss_surv

class Dataset(tt.data.DatasetTuple):
    def __getitem__(self, index):
        input, target =  super().__getitem__(index)
        target = target.to_numpy()
        rank_mat = pair_rank_mat(*target)
        target = tt.tuplefy(*target, rank_mat).to_tensor()
        return tt.tuplefy(input, target)

class Survival_Model(tt.Model):
    def __init__(self, net, optimizer=None, device=None, alpha=0.2, sigma=0.1, duration_index=None, Loss=None):
        self.duration_index = duration_index
        if Loss is None:
            Loss = models.Loss.Loss(alpha, sigma)
        super().__init__(net, Loss, optimizer, device)

    @property
    def duration_index(self):
        return self._duration_index

    @duration_index.setter
    def duration_index(self, val):
        self._duration_index = val

    def make_dataloader(self, data, batch_size, shuffle, num_workers=0):
        dataloader = super().make_dataloader(data, batch_size, shuffle, num_workers,
                                             make_dataset=Dataset)
        return dataloader

    def make_dataloader_predict(self, input, batch_size, shuffle=False, num_workers=0):
        dataloader = super().make_dataloader(input, batch_size, shuffle, num_workers)
        return dataloader

    def predict_surv_df(self, input, batch_size=8224, eval_=True, num_workers=0):
        surv = self.predict_surv(input, batch_size, True, eval_, True, num_workers)
        return pd.DataFrame(surv, self.duration_index)

    def predict_surv(self, input, batch_size=8224, numpy=None, eval_=True,
                     to_cpu=False, num_workers=0):
        cif = self.predict_cif(input, batch_size, False, eval_, to_cpu, num_workers)
        surv = 1. - cif.sum(0)
        return tt.utils.array_or_tensor(surv, numpy, input)

    def predict_cif(self, input, batch_size=8224, numpy=None, eval_=True,
                     to_cpu=False, num_workers=0):
        pmf = self.predict_pmf(input, batch_size, False, eval_, to_cpu, num_workers)
        cif = pmf.cumsum(1)
        return tt.utils.array_or_tensor(cif, numpy, input)

    def predict_pmf(self, input, batch_size=8224, numpy=None, eval_=True,
                     to_cpu=False, num_workers=0):
        preds = self.predict(input, batch_size, False, eval_, False, to_cpu, num_workers)
        pmf = pad_col(preds.view(preds.size(0), -1)).softmax(1)[:, :-1]
        pmf = pmf.view(preds.shape).transpose(0, 1).transpose(1, 2)
        return tt.utils.array_or_tensor(pmf, numpy, input)

Loss = Loss(alpha=[0.2], sigma=0.1)

# Extract time-to-event and event label
num_durations = 10

net = GNNStack(num_risks=4, gnn_model_type='dyGIN2d', num_layers=1,
                     groups=6, pool_ratio=0.3, kern_size=[11],
                     in_dim=128, hidden_dim=64, out_dim=128,
                     seq_len=seq_length, num_nodes_total = 161, num_nodes_dynamic=38, num_nodes_static = 23, num_icd_codes = 500, num_reports_features = 768, num_nodes_indiv = 32, num_classes=out_features)

in_features = x_train.shape[2]-1

torch.cuda.set_device(DEVICE)

net = net.cuda(DEVICE)

# collect cache
# gc.collect()
# torch.cuda.empty_cache()

model = Survival_Model(net, tt.optim.Adam(0.0001), duration_index=cuts, Loss=Loss) # wrapper

metrics = dict(
    Loss_surv = Loss(alpha=[0.2], sigma=0.1),
)
callbacks = [tt.cb.EarlyStopping()]

batch_size = 32
epochs = 10
log = model.fit(x_train, y_train_surv, batch_size = batch_size, epochs = epochs, callbacks = callbacks, verbose = True, val_data=val, val_batch_size=32, metrics=metrics)

res = model.log.to_pandas()

res.head()

_ = res[['train_Loss', 'val_Loss']].plot()

_ = res[['train_Loss_surv', 'val_Loss_surv']].plot()

def idx_at_times(index_surv, times, steps='pre', assert_sorted=True):
    if assert_sorted:
        assert pd.Series(index_surv).is_monotonic_increasing, "Need 'index_surv' to be monotonic increasing"
    if steps == 'pre':
        idx = np.searchsorted(index_surv, times)
    elif steps == 'post':
        idx = np.searchsorted(index_surv, times, side='right') - 1
    return idx.clip(0, len(index_surv)-1)

@numba.njit
def _group_loop(n, surv_idx, durations, events, di, ni):
    idx = 0
    for i in range(n):
        idx += durations[i] != surv_idx[idx]
        di[idx] += events[i]
        ni[idx] += 1
    return di, ni

def kaplan_meier(durations, events, start_duration=0):
    n = len(durations)
    assert n == len(events)
    if start_duration > durations.min():
        warnings.warn(f"start_duration {start_duration} is larger than minimum duration {durations.min()}. "
            "If intentional, consider changing start_duration when calling kaplan_meier.")
    order = np.argsort(durations)
    durations = durations[order]
    events = events[order]
    surv_idx = np.unique(durations)
    ni = np.zeros(len(surv_idx), dtype='int')
    di = np.zeros_like(ni)
    di, ni = _group_loop(n, surv_idx, durations, events, di, ni)
    ni = n - ni.cumsum()
    ni[1:] = ni[:-1]
    ni[0] = n
    survive = 1 - di / ni
    zero_survive = survive == 0
    if zero_survive.any():
        i = np.argmax(zero_survive)
        surv = np.zeros_like(survive)
        surv[:i] = np.exp(np.log(survive[:i]).cumsum())
        # surv[i:] = surv[i-1]
        surv[i:] = 0.
    else:
        surv = np.exp(np.log(1 - di / ni).cumsum())
    if start_duration < surv_idx.min():
        tmp = np.ones(len(surv)+ 1, dtype=surv.dtype)
        tmp[1:] = surv
        surv = tmp
        tmp = np.zeros(len(surv_idx)+ 1, dtype=surv_idx.dtype)
        tmp[1:] = surv_idx
        surv_idx = tmp
    surv = pd.Series(surv, surv_idx)
    return surv

def administrative_scores(func):
    if not func.__class__.__module__.startswith('numba'):
        raise ValueError("Need to provide numba compiled function")
    def metric(time_grid, durations, durations_c, events, surv, index_surv, reduce=True, steps_surv='post'):
        if not hasattr(time_grid, '__iter__'):
            time_grid = np.array([time_grid])
        assert (type(time_grid) is type(durations) is type(events) is type(surv) is
                type(index_surv) is type(durations_c) is np.ndarray), 'Need all input to be np.ndarrays'
        assert (durations[events == 0] == durations_c[events == 0]).all(), 'Censored observations need same `durations` and `durations_c`'
        assert (durations[events == 1] <= durations_c[events == 1]).all(), '`durations` cannot be larger than `durations_c`'
        idx_ts_surv = idx_at_times(index_surv, time_grid, steps_surv, assert_sorted=True)
        scores, norm = _admin_scores(func, time_grid, durations, durations_c, events, surv, idx_ts_surv)
        if reduce is True:
            return scores.sum(axis=1) / norm
        return scores, norm.reshape(-1, 1)
    return metric

@numba.njit(parallel=True)
def _admin_scores(func, time_grid, durations, durations_c, events, surv, idx_ts_surv):
    def _single(func, ts, durations, durations_c, events, surv, idx_ts_surv_i,
                scores, n_indiv):
        for i in range(n_indiv):
            tt = durations[i]
            tc = durations_c[i]
            d = events[i]
            s = surv[idx_ts_surv_i, i]
            scores[i] = func(ts, tt, tc, d, s)

    n_times = len(time_grid)
    n_indiv = len(durations)
    scores = np.empty((n_times, n_indiv))
    scores.fill(np.nan)
    normalizer = np.empty(n_times)
    normalizer.fill(np.nan)
    for i in numba.prange(n_times):
        ts = time_grid[i]
        idx_ts_surv_i = idx_ts_surv[i]
        scores_i = scores[i]
        normalizer[i] = (durations_c >= ts).sum()
        _single(func, ts, durations, durations_c, events, surv, idx_ts_surv_i, scores_i, n_indiv)
    return scores, normalizer

@numba.njit
def _brier_score(ts, tt, tc, d, s):
    if (tt <= ts) and (d == 1) and (tc >= ts):
        return np.power(s, 2)
    if tt >= ts:
        return np.power(1 - s, 2)
    return 0.

@numba.njit
def _binomial_log_likelihood(ts, tt, tc, d, s, eps=1e-7):
    if s < eps:
        s = eps
    elif s > (1 - eps):
        s = 1 - eps
    if (tt <= ts) and (d == 1) and (tc >= ts):
        return np.log(1 - s)
    if tt >= ts:
        return np.log(s)
    return 0.

brier_score = administrative_scores(_brier_score)
binomial_log_likelihood = administrative_scores(_binomial_log_likelihood)


def _integrated_admin_metric(func):
    def metric(time_grid, durations, durations_c, events, surv, index_surv, steps_surv='post'):
        scores = func(time_grid, durations, durations_c, events, surv, index_surv, True, steps_surv)
        integral = scipy.integrate.simpson(y=scores, x=time_grid)
        return integral / (time_grid[-1] - time_grid[0])
    return metric

integrated_brier_score = _integrated_admin_metric(brier_score)
integrated_binomial_log_likelihood = _integrated_admin_metric(binomial_log_likelihood)

@numba.jit(nopython=True)
def _is_comparable(t_i, t_j, d_i, d_j):
    return ((t_i < t_j) & d_i) | ((t_i == t_j) & (d_i | d_j))

@numba.jit(nopython=True)
def _is_comparable_antolini(t_i, t_j, d_i, d_j):
    return ((t_i < t_j) & d_i) | ((t_i == t_j) & d_i & (d_j == 0))

@numba.jit(nopython=True)
def _is_concordant(s_i, s_j, t_i, t_j, d_i, d_j):
    conc = 0.
    if t_i < t_j:
        conc = (s_i < s_j) + (s_i == s_j) * 0.5
    elif t_i == t_j:
        if d_i & d_j:
            conc = 1. - (s_i != s_j) * 0.5
        elif d_i:
            conc = (s_i < s_j) + (s_i == s_j) * 0.5  # different from RSF paper.
        elif d_j:
            conc = (s_i > s_j) + (s_i == s_j) * 0.5  # different from RSF paper.
    return conc * _is_comparable(t_i, t_j, d_i, d_j)

@numba.jit(nopython=True)
def _is_concordant_antolini(s_i, s_j, t_i, t_j, d_i, d_j):
    return (s_i < s_j) & _is_comparable_antolini(t_i, t_j, d_i, d_j)

@numba.jit(nopython=True, parallel=True)
def _sum_comparable(t, d, is_comparable_func):
    n = t.shape[0]
    count = 0.
    for i in numba.prange(n):
        for j in range(n):
            if j != i:
                count += is_comparable_func(t[i], t[j], d[i], d[j])
    return count

@numba.jit(nopython=True, parallel=True)
def _sum_concordant(s, t, d):
    n = len(t)
    count = 0.
    for i in numba.prange(n):
        for j in range(n):
            if j != i:
                count += _is_concordant(s[i, i], s[i, j], t[i], t[j], d[i], d[j])
    return count

@numba.jit(nopython=True, parallel=True)
def _sum_concordant_disc(s, t, d, s_idx, is_concordant_func):
    n = len(t)
    count = 0
    for i in numba.prange(n):
        idx = s_idx[i]
        for j in range(n):
            if j != i:
                count += is_concordant_func(s[idx, i], s[idx, j], t[i], t[j], d[i], d[j])
    return count

def concordance_td(durations, events, surv, surv_idx, method='adj_antolini'):
    if np.isfortran(surv):
        surv = np.array(surv, order='C')
    assert durations.shape[0] == surv.shape[1] == surv_idx.shape[0] == events.shape[0]
    assert type(durations) is type(events) is type(surv) is type(surv_idx) is np.ndarray
    if events.dtype in ('float', 'float32'):
        events = events.astype('int32')
    if method == 'adj_antolini':
        is_concordant = _is_concordant
        is_comparable = _is_comparable
        return (_sum_concordant_disc(surv, durations, events, surv_idx, is_concordant) /
                _sum_comparable(durations, events, is_comparable))
    elif method == 'antolini':
        is_concordant = _is_concordant_antolini
        is_comparable = _is_comparable_antolini
        return (_sum_concordant_disc(surv, durations, events, surv_idx, is_concordant) /
                _sum_comparable(durations, events, is_comparable))
    return ValueError(f"Need 'method' to be e.g. 'antolini', got '{method}'.")

@numba.njit(parallel=True)
def _inv_cens_scores(func, time_grid, durations, events, surv, censor_surv, idx_ts_surv, idx_ts_censor,
                     idx_tt_censor, scores, weights, n_times, n_indiv, max_weight):
    def _inv_cens_score_single(func, ts, durations, events, surv, censor_surv, idx_ts_surv_i,
                               idx_ts_censor_i, idx_tt_censor, scores, weights, n_indiv, max_weight):
        min_g = 1./max_weight
        for i in range(n_indiv):
            tt = durations[i]
            d = events[i]
            s = surv[idx_ts_surv_i, i]
            g_ts = censor_surv[idx_ts_censor_i, i]
            g_tt = censor_surv[idx_tt_censor[i], i]
            g_ts = max(g_ts, min_g)
            g_tt = max(g_tt, min_g)
            score, w = func(ts, tt, s, g_ts, g_tt, d)
            #w = min(w, max_weight)
            scores[i] = score * w
            weights[i] = w

    for i in numba.prange(n_times):
        ts = time_grid[i]
        idx_ts_surv_i = idx_ts_surv[i]
        idx_ts_censor_i = idx_ts_censor[i]
        scores_i = scores[i]
        weights_i = weights[i]
        _inv_cens_score_single(func, ts, durations, events, surv, censor_surv, idx_ts_surv_i,
                               idx_ts_censor_i, idx_tt_censor, scores_i, weights_i, n_indiv, max_weight)

def _inverse_censoring_weighted_metric(func):
    if not func.__class__.__module__.startswith('numba'):
        raise ValueError("Need to provide numba compiled function")
    def metric(time_grid, durations, events, surv, censor_surv, index_surv, index_censor, max_weight=np.inf,
               reduce=True, steps_surv='post', steps_censor='post'):
        if not hasattr(time_grid, '__iter__'):
            time_grid = np.array([time_grid])
        assert (type(time_grid) is type(durations) is type(events) is type(surv) is type(censor_surv) is
                type(index_surv) is type(index_censor) is np.ndarray), 'Need all input to be np.ndarrays'
        n_times = len(time_grid)
        n_indiv = len(durations)
        scores = np.zeros((n_times, n_indiv))
        weights = np.zeros((n_times, n_indiv))
        idx_ts_surv = idx_at_times(index_surv, time_grid, steps_surv, assert_sorted=True)
        idx_ts_censor = idx_at_times(index_censor, time_grid, steps_censor, assert_sorted=True)
        idx_tt_censor = idx_at_times(index_censor, durations, 'pre', assert_sorted=True)
        if steps_censor == 'post':
            idx_tt_censor  = (idx_tt_censor - 1).clip(0)
            #  This ensures that we get G(tt-)
        _inv_cens_scores(func, time_grid, durations, events, surv, censor_surv, idx_ts_surv, idx_ts_censor,
                         idx_tt_censor, scores, weights, n_times, n_indiv, max_weight)
        if reduce is True:
            return np.sum(scores, axis=1) / np.sum(weights, axis=1)
        return scores, weights
    return metric

@numba.njit()
def _brier_score(ts, tt, s, g_ts, g_tt, d):
    if (tt <= ts) and d == 1:
        return np.power(s, 2), 1./g_tt
    if tt > ts:
        return np.power(1 - s, 2), 1./g_ts
    return 0., 0.

@numba.njit()
def _binomial_log_likelihood(ts, tt, s, g_ts, g_tt, d, eps=1e-7):
    s = eps if s < eps else s
    s = (1-eps) if s > (1 - eps) else s
    if (tt <= ts) and d == 1:
        return np.log(1 - s), 1./g_tt
    if tt > ts:
        return np.log(s), 1./g_ts
    return 0., 0.

brier_score = _inverse_censoring_weighted_metric(_brier_score)
binomial_log_likelihood = _inverse_censoring_weighted_metric(_binomial_log_likelihood)

def _integrated_inverce_censoring_weighed_metric(func):
    def metric(time_grid, durations, events, surv, censor_surv, index_surv, index_censor,
               max_weight=np.inf, steps_surv='post', steps_censor='post'):
        scores = func(time_grid, durations, events, surv, censor_surv, index_surv, index_censor,
                      max_weight, True, steps_surv, steps_censor)
        integral = scipy.integrate.simpson(y=scores, x=time_grid)
        return integral / (time_grid[-1] - time_grid[0])
    return metric

integrated_brier_score = _integrated_inverce_censoring_weighed_metric(brier_score)
integrated_binomial_log_likelihood = _integrated_inverce_censoring_weighed_metric(binomial_log_likelihood)

class EvalSurv:
    def __init__(self, surv, durations, events, censor_surv=None, censor_durations=None, steps='post'):
        assert (type(durations) == type(events) == np.ndarray), 'Need `durations` and `events` to be arrays'
        self.surv = surv
        self.durations = durations
        self.events = events
        self.censor_surv = censor_surv
        self.censor_durations = censor_durations
        self.steps = steps
        assert pd.Series(self.index_surv).is_monotonic_increasing

    @property
    def censor_surv(self):
        return self._censor_surv

    @censor_surv.setter
    def censor_surv(self, censor_surv):
        if isinstance(censor_surv, EvalSurv):
            self._censor_surv = censor_surv
        elif type(censor_surv) is str:
            if censor_surv == 'km':
                self.add_km_censor()
            else:
                raise ValueError(f"censor_surv cannot be {censor_surv}. Use e.g. 'km'")
        elif censor_surv is not None:
            self.add_censor_est(censor_surv)
        else:
            self._censor_surv = None

    @property
    def index_surv(self):
        return self.surv.index.values

    @property
    def steps(self):
        return self._steps

    @steps.setter
    def steps(self, steps):
        vals = ['post', 'pre']
        if steps not in vals:
            raise ValueError(f"`steps` needs to be {vals}, got {steps}")
        self._steps = steps

    def add_censor_est(self, censor_surv, steps='post'):
        if not isinstance(censor_surv, EvalSurv):
            censor_surv = self._constructor(censor_surv, self.durations, 1-self.events, None,
                                            steps=steps)
        self.censor_surv = censor_surv
        return self

    def add_km_censor(self, steps='post'):
        km = kaplan_meier(self.durations, 1-self.events)
        surv = pd.DataFrame(np.repeat(km.values.reshape(-1, 1), len(self.durations), axis=1),
                            index=km.index)
        return self.add_censor_est(surv, steps)

    @property
    def censor_durations(self):
        """Administrative censoring times."""
        return self._censor_durations

    @censor_durations.setter
    def censor_durations(self, val):
        if val is not None:
            assert (self.durations[self.events == 0] == val[self.events == 0]).all(),\
                'Censored observations need same `durations` and `censor_durations`'
            assert (self.durations[self.events == 1] <= val[self.events == 1]).all(),\
                '`durations` cannot be larger than `censor_durations`'
            if (self.durations == val).all():
                warnings.warn("`censor_durations` are equal to `durations`." +
                              " `censor_durations` are likely wrong!")
            self._censor_durations = val
        else:
            self._censor_durations = val

    @property
    def _constructor(self):
        return EvalSurv

    def __getitem__(self, index):
        if not (hasattr(index, '__iter__') or type(index) is slice) :
            index = [index]
        surv = self.surv.iloc[:, index]
        durations = self.durations[index]
        events = self.events[index]
        new = self._constructor(surv, durations, events, None, steps=self.steps)
        if self.censor_surv is not None:
            new.censor_surv = self.censor_surv[index]
        return new

    def plot_surv(self, **kwargs):
        if len(self.durations) > 50:
            raise RuntimeError("We don't allow to plot more than 50 lines. Use e.g. `ev[1:5].plot()`")
        if 'drawstyle' in kwargs:
            raise RuntimeError(f"`drawstyle` is set by `self.steps`. Remove from **kwargs")
        return self.surv.plot(drawstyle=f"steps-{self.steps}", **kwargs)

    def idx_at_times(self, times):
        return idx_at_times(self.index_surv, times, self.steps)

    def _duration_idx(self):
        return self.idx_at_times(self.durations)

    def surv_at_times(self, times):
        idx = self.idx_at_times(times)
        return self.surv.iloc[idx]

    def concordance_td(self, method='adj_antolini'):
        return concordance_td(self.durations, self.events, self.surv.values,
                              self._duration_idx(), method)

    def brier_score(self, time_grid, max_weight=np.inf):
        if self.censor_surv is None:
            raise ValueError("""Need to add censor_surv to compute Brier score. Use 'add_censor_est'
            or 'add_km_censor' for Kaplan-Meier""")
        bs = brier_score(time_grid, self.durations, self.events, self.surv.values,
                              self.censor_surv.surv.values, self.index_surv,
                              self.censor_surv.index_surv, max_weight, True, self.steps,
                              self.censor_surv.steps)
        return pd.Series(bs, index=time_grid).rename('brier_score')

    def nbll(self, time_grid, max_weight=np.inf):
        if self.censor_surv is None:
            raise ValueError("""Need to add censor_surv to compute the score. Use 'add_censor_est'
            or 'add_km_censor' for Kaplan-Meier""")
        bll = binomial_log_likelihood(time_grid, self.durations, self.events, self.surv.values,
                                           self.censor_surv.surv.values, self.index_surv,
                                           self.censor_surv.index_surv, max_weight, True, self.steps,
                                           self.censor_surv.steps)
        return pd.Series(-bll, index=time_grid).rename('nbll')

    def integrated_brier_score(self, time_grid, max_weight=np.inf):
        if self.censor_surv is None:
            raise ValueError("Need to add censor_surv to compute briser score. Use 'add_censor_est'")
        return integrated_brier_score(time_grid, self.durations, self.events, self.surv.values,
                                           self.censor_surv.surv.values, self.index_surv,
                                           self.censor_surv.index_surv, max_weight, self.steps,
                                           self.censor_surv.steps)

    def integrated_nbll(self, time_grid, max_weight=np.inf):
        if self.censor_surv is None:
            raise ValueError("Need to add censor_surv to compute the score. Use 'add_censor_est'")
        ibll = integrated_binomial_log_likelihood(time_grid, self.durations, self.events, self.surv.values,
                                                       self.censor_surv.surv.values, self.index_surv,
                                                       self.censor_surv.index_surv, max_weight, self.steps,
                                                       self.censor_surv.steps)
        return -ibll


cif = model.predict_cif(x_test)
cif1 = pd.DataFrame(cif[0], model.duration_index)
cif2 = pd.DataFrame(cif[1], model.duration_index)
cif3 = pd.DataFrame(cif[2], model.duration_index)
cif4 = pd.DataFrame(cif[3], model.duration_index)


ev1 = EvalSurv(1-cif1, durations_test, events_test == 0, censor_surv='km')
ev2 = EvalSurv(1-cif2, durations_test, events_test == 1, censor_surv='km')
ev3 = EvalSurv(1-cif3, durations_test, events_test == 2, censor_surv='km')
ev4 = EvalSurv(1-cif4, durations_test, events_test == 3, censor_surv='km')

ev1.concordance_td(), ev2.concordance_td(), ev3.concordance_td(), ev4.concordance_td()

last_row_values = cif2.iloc[-1]
column_with_lowest_last_row = last_row_values.idxmin()
print(f"The column with the lowest last row value in cif1 is: {column_with_lowest_last_row}")




