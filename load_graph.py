import gzip
import dgl
import pickle
import torch as th
import torch
import random
import numpy as np
import torch.nn as nn
import pandas as pd
import scipy.sparse as sp
import torch.nn as nn
import dgl.nn.pytorch as dglnn
import time
import concurrent.futures
import multiprocessing
from dgl.data.utils import save_graphs
from dgl.data.utils import load_graphs
from io import StringIO
from functools import partial
import torch.nn.functional as F
from sklearn.metrics.pairwise import euclidean_distances

manager = multiprocessing.Manager()
device = th.device("cuda" if th.cuda.is_available() else "cpu")
print(device, 'is available')

data = load_graphs("./g.bin")
g = data[0][0]
# calculate the number of nodes
n_genes = g.num_nodes('gene')
n_ref_cell = g.num_nodes('ref_cell')
n_que_cell = g.num_nodes('que_cell')

# calculate the number of edges
n_ref_cell_2_gene = g.number_of_edges('ref_cell_2_gene')
n_gene_2_ref_cell = g.number_of_edges('gene_2_ref_cell')
n_que_cell_2_gene = g.number_of_edges('que_cell_2_gene')
n_gene_2_que_cell = g.number_of_edges('gene_2_que_cell')

input_feature = 50
out_feature = 25

# randomly generate node embeddings
g.nodes['gene'].data['feature'] = torch.randn(n_genes, input_feature)
g.nodes['ref_cell'].data['feature'] = torch.randn(n_ref_cell, input_feature)
g.nodes['que_cell'].data['feature'] = torch.randn(n_que_cell, input_feature)

# randomly generate edge embeddings
g.edges['ref_cell_2_gene'].data['feature'] = torch.randn(n_ref_cell_2_gene, input_feature)
g.edges['gene_2_ref_cell'].data['feature'] = torch.randn(n_gene_2_ref_cell, input_feature)
g.edges['que_cell_2_gene'].data['feature'] = torch.randn(n_que_cell_2_gene, input_feature)
g.edges['gene_2_que_cell'].data['feature'] = torch.randn(n_gene_2_que_cell, input_feature)

# randomly generate training masks on user nodes and click edges
g.nodes['gene'].data['train_mask'] = torch.zeros(n_genes, dtype=torch.bool).bernoulli(0.6)
g.nodes['ref_cell'].data['train_mask'] = torch.zeros(n_ref_cell, dtype=torch.bool).bernoulli(0.6)
g.nodes['que_cell'].data['train_mask'] = torch.zeros(n_que_cell, dtype=torch.bool).bernoulli(0.6)

g.edges['ref_cell_2_gene'].data['train_mask'] = torch.zeros(n_ref_cell_2_gene, dtype=torch.bool).bernoulli(0.6)
g.edges['gene_2_ref_cell'].data['train_mask'] = torch.zeros(n_gene_2_ref_cell, dtype=torch.bool).bernoulli(0.6)
g.edges['que_cell_2_gene'].data['train_mask'] = torch.zeros(n_que_cell_2_gene, dtype=torch.bool).bernoulli(0.6)
g.edges['gene_2_que_cell'].data['train_mask'] = torch.zeros(n_gene_2_que_cell, dtype=torch.bool).bernoulli(0.6)

#############################
# with gzip.open('Fetal-Brain3_dge.txt.gz', 'rb') as f:
#     file_content = f.read()
#     data = StringIO(str(file_content, 'utf-8'))
# Brain_3_X = pd.read_csv(data)
# print(f'Brain 3 has {Brain_3_X.shape[0]} genes and {Brain_3_X.shape[1]-1} cells')



# Define a Heterograph Conv model
class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()

        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        return h
    
model = RGCN(input_feature, 25, out_feature, g.etypes)

gene_feats = g.nodes['gene'].data['feature']
ref_cell_feats = g.nodes['ref_cell'].data['feature']
que_cell_feats = g.nodes['que_cell'].data['feature']

ref_cell_2_gene_feats = g.edges['ref_cell_2_gene'].data['feature']
gene_2_ref_cell_feats = g.edges['gene_2_ref_cell'].data['feature']
que_cell_2_gene_feats = g.edges['que_cell_2_gene'].data['feature']
gene_2_que_cell_feats = g.edges['gene_2_que_cell'].data['feature']

ref_cell_mask = g.nodes['ref_cell'].data['train_mask']
que_cell_mask = g.nodes['que_cell'].data['train_mask']
# how about validation mask?

# use all the node and edge features
node_edge_features = {'gene': gene_feats, 'ref_cell': ref_cell_feats, 'que_cell': que_cell_feats, 'ref_cell_2_gene': ref_cell_2_gene_feats,
                 'gene_2_ref_cell': gene_2_ref_cell_feats, 'que_cell_2_gene': que_cell_2_gene_feats, 'gene_2_que_cell': gene_2_que_cell_feats}

embedding = model(g, node_edge_features)
               
gene_embedding = embedding['gene']
ref_cell_embedding = embedding['ref_cell']
que_cell_embedding = embedding['que_cell']
# print('length of embedding:', len(embedding))
# print('embedding:', embedding)

# use only node features
# node_features = {'gene': gene_feats, 'ref_cell': ref_cell_feats, 'que_cell': que_cell_feats}

# embedding = model(g, node_features)
               
# gene_embedding = embedding['gene']
# ref_cell_embedding = embedding['ref_cell']
# que_cell_embedding = embedding['que_cell']
# print('length of embedding:', len(embedding))
# print('embedding:', embedding)

ref_cell_feats = ref_cell_feats[ref_cell_mask]
que_cell_feats = que_cell_feats[que_cell_mask]

true_ref_cell_dis_matrix = euclidean_distances(ref_cell_feats, ref_cell_feats)
true_que_cell_dis_matrix = euclidean_distances(que_cell_feats, que_cell_feats)

true_ref_cell_dis_matrix = torch.from_numpy(true_ref_cell_dis_matrix)
true_que_cell_dis_matrix = torch.from_numpy(true_que_cell_dis_matrix)

# true_ref_cell_dis_matrix = true_ref_cell_dis_matrix.to(device)
# true_que_cell_dis_matrix = true_que_cell_dis_matrix.to(device)

# print(true_ref_cell_dis_matrix.shape)
# print(true_que_cell_dis_matrix.shape)

# need to change the evaluation to something else
# how to calculat the similarity matrix between cells in high dimension?
def evaluate(model, graph, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)
    

opt = torch.optim.Adam(model.parameters(), lr=0.1)
loss_func = nn.MSELoss()

print('Traininig...')
for epoch in range(10000):
    model.train()
    pred = model(g, node_edge_features)
    # compute loss
    pred_ref_cell_feats = pred['ref_cell']#.cpu().detach().numpy()
    # print('pred ref cell feature shape:', pred_ref_cell_feats.shape)
    pred_que_cell_feats = pred['que_cell']#.cpu().detach().numpy()
    # print('pred que cell feature shape:', pred_que_cell_feats.shape)
    
    train_pred_ref_cell_feats = pred_ref_cell_feats[ref_cell_mask]
    train_pred_que_cell_feats = pred_que_cell_feats[que_cell_mask]
    
#     val_pred_ref_cell_feats = pred_ref_cell_feats[1-ref_cell_mask]
#     val_pred_que_cell_feats = pred_que_cell_feats[1-que_cell_mask]
    # print('train_pred_ref_cell_feats:', train_pred_ref_cell_feats.shape)

    # print('train_pred_ref_cell_feats:', type(train_pred_ref_cell_feats), 'true_ref_cell_dis_matrix', type(true_ref_cell_dis_matrix))

    pred_ref_cell_dis_matrix = torch.cdist(train_pred_ref_cell_feats, train_pred_ref_cell_feats, p=2)
    pred_que_cell_dis_matrix = torch.cdist(train_pred_que_cell_feats, train_pred_que_cell_feats, p=2)
    
    # print('pred_ref_cell_dis_matrix:', pred_ref_cell_dis_matrix.shape, 'true_ref_cell_dis_matrix:', true_ref_cell_dis_matrix.shape)

    # pred_ref_cell_dis_matrix = euclidean_distances(pred_ref_cell_feats, pred_ref_cell_feats)
    # pred_que_cell_dis_matrix = euclidean_distances(pred_que_cell_feats, pred_que_cell_feats)
    
    # print(pred_ref_cell_dis_matrix.shape, true_ref_cell_dis_matrix.shape)
    # print(pred_que_cell_dis_matrix.shape, true_que_cell_dis_matrix.shape)
    # loss_1 = loss_func(pred_ref_cell_dis_matrix, true_ref_cell_dis_matrix)
    # print(loss_1)
    loss = loss_func(pred_ref_cell_dis_matrix, true_ref_cell_dis_matrix) + loss_func(pred_que_cell_dis_matrix, true_que_cell_dis_matrix)

    loss.backward()
    opt.step()
    opt.zero_grad()

    if epoch % 100 == 0:
        print('epoch: ', epoch, 'train_loss:', loss.item())

    # Save model if necessary.  Omitted in the example.