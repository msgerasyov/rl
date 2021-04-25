from __future__ import division
from __future__ import print_function

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import numpy as np
import scipy.sparse as sp

import time

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass=2, dropout=False):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

def update_graph(model, optimizer, features, adj, rew_states, loss,
                gcn_epochs=1, gcn_lambda=10.):
    if adj.shape[0] >1:
        labels = torch.zeros((len(features)))
        idx_train = torch.LongTensor([0])
        for r_s in rew_states:
            labels[r_s[0]] = torch.tensor([1.]) if r_s[1] > 0. else torch.tensor([0.])
            idx_train=torch.cat((idx_train, torch.LongTensor([r_s[0]]) ), 0)
        labels= labels.type(torch.LongTensor)
    else:
        labels = torch.zeros((len(features))).type(torch.LongTensor)
        idx_train = torch.LongTensor([0])

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    deg = np.diag(adj.toarray().sum(axis=1))
    laplacian = torch.from_numpy((deg - adj.toarray()).astype(np.float32))
    adj = normalize(sp.csr_matrix(adj) + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    t_total = time.time()
    for epoch in range(gcn_epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        soft_out= torch.unsqueeze(torch.nn.functional.softmax(output,dim=1)[:,1],1)
        loss_reg  = torch.mm(torch.mm(soft_out.T,laplacian),soft_out)
        loss_train +=  gcn_lambda * loss_reg.squeeze()
        loss_train.backward()
        optimizer.step()


def compute_graph_loss(model, features, adj, rew_states, loss, gcn_lambda=10.):
    if adj.shape[0] >1:
        labels = torch.zeros((len(features)))
        idx_train = torch.LongTensor([0])
        for r_s in rew_states:
            labels[r_s[0]] = torch.tensor([1.]) if r_s[1] > 0. else torch.tensor([0.])
            idx_train=torch.cat((idx_train, torch.LongTensor([r_s[0]]) ), 0)
        labels= labels.type(torch.LongTensor)
    else:
        labels = torch.zeros((len(features))).type(torch.LongTensor)
        idx_train = torch.LongTensor([0])

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    deg = np.diag(adj.toarray().sum(axis=1))
    laplacian = torch.from_numpy((deg - adj.toarray()).astype(np.float32))
    adj = normalize(sp.csr_matrix(adj) + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    model.train()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    soft_out= torch.unsqueeze(torch.nn.functional.softmax(output,dim=1)[:,1],1)
    loss_reg  = torch.mm(torch.mm(soft_out.T,laplacian),soft_out)
    loss_train +=  gcn_lambda * loss_reg.squeeze()

    return loss_train
