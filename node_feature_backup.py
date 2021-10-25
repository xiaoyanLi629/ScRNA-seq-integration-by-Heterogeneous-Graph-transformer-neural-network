# activate Dgl_07
# Bipartite graph
# Two datasets integration scenario
# cells are in one group and genes in another group
# Data graph construction method:
	# 1. put all data from two datasets together (selected)
		# Two bipartites, one is the union of genes and the other is all cells from two datasets (selected)
		# Three bipartites, one is gene, the other is cells from reference dataset, the other is cells from query dataset
	# 2. Train model on one referenc graph and test on another graph
# each cell has its own features
# each gene has its own features
# link weights represnt gene expressions
# message passing:
	# from cells to genes
	# from genes to cells
	# from cells to genes
	# from genes to cells
	# from cells to genes
	# from genes to cells
# loss function, minimize the cell-cell distance based on learned features.
# data is scDeepSort >> Data >> HCL >> Brain >> Fetal_Brain3, Fetal_Brain4


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

def set_random(random_seed = 1):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    dgl.seed(random_seed)
    if device == 'cuda':
        torch.cuda.manual_seed(random_seed)
set_random()

Brain_3_annot = pd.read_csv('Fetal-Brain3_Anno.csv', index_col=0)  # (gene, cell)

with gzip.open('Fetal-Brain3_dge.txt.gz', 'rb') as f:
    file_content = f.read()
    data = StringIO(str(file_content, 'utf-8'))
Brain_3_X = pd.read_csv(data)
print(f'Brain 3 has {Brain_3_X.shape[0]} genes and {Brain_3_X.shape[1]-1} cells')

Brain_3_X.rename( columns={'Unnamed: 0':'Gene_name'}, inplace=True )
# Brain_3_X[:5]

Brain_3_set = set(Brain_3_X['Gene_name'])

Brain_4_annot = pd.read_csv('Fetal-Brain4_Anno.csv', index_col=0)  # (gene, cell)
with gzip.open('Fetal-Brain4_dge.txt.gz', 'rb') as f:
    file_content = f.read()
    data = StringIO(str(file_content, 'utf-8'))
Brain_4_X = pd.read_csv(data)
print(f'Brain 4 has {Brain_4_X.shape[0]} genes and {Brain_4_X.shape[1]-1} cells')

Brain_4_X.rename( columns={'Unnamed: 0':'Gene_name'}, inplace=True )
# Brain_4_X[:5]

Brain_4_set = set(Brain_4_X['Gene_name'])

inter_gene_set = Brain_3_set & Brain_4_set

print('length of Brain_3_set:', len(Brain_3_set))
print('length of Brain_4_set:', len(Brain_4_set))
print('length of inter_gene_set:', len(inter_gene_set))

start = time.perf_counter()
uncommon_gene_row = set()
for i in range(len(Brain_3_X)):
    gene = Brain_3_X.iloc[i][0]
    if gene not in inter_gene_set:
        uncommon_gene_row.add(i)
Brain_3_X_drop = Brain_3_X.drop(list(uncommon_gene_row))
Brain_3_X_drop = Brain_3_X_drop.sort_values(by = 'Gene_name')


uncommon_gene_row = set()
for i in range(len(Brain_4_X)):
    gene = Brain_4_X.iloc[i][0]
    if gene not in inter_gene_set:
        uncommon_gene_row.add(i)
Brain_4_X_drop = Brain_4_X.drop(list(uncommon_gene_row))
Brain_4_X_drop = Brain_4_X_drop.sort_values(by = 'Gene_name')

# print('Brain_3_X_drop shape:', Brain_3_X_drop.shape)
# print('Brain_4_X_drop shape:', Brain_4_X_drop.shape)
print(f'Brain drop 3 has {Brain_3_X_drop.shape[0]} genes and {Brain_3_X_drop.shape[1]-1} cells')
print(f'Brain drop 4 has {Brain_4_X_drop.shape[0]} genes and {Brain_4_X_drop.shape[1]-1} cells')

end = time.perf_counter()
print(f'Reading data finished in {end-start} seconds')

start = time.perf_counter()

def cell_graph(cell_id, dataset):
    if cell_id % 500 == 0:
        print(f'professing cell {cell_id} ...')
    if cell_id == dataset.shape[1]-1:
        print(f'professing last cell {cell_id} ...')
    cell = dataset.shape[0] * [cell_id-1]
    gene = list(range(dataset.shape[0]))
    weight = list(dataset.iloc[:, cell_id])
    
    return [cell_id, cell, gene, weight]


pool = multiprocessing.Pool(11)


temp_func = partial(cell_graph, dataset = Brain_3_X_drop)
blocks = pool.map(func=temp_func, iterable=list(range(1, Brain_3_X_drop.shape[1])), chunksize=1000)

pool.close()
pool.join()
    
end = time.perf_counter()
print(f'Reading data Brain 3 finished in {end-start} seconds')

# blocks = result_list
Brain_3_edges = blocks
Brain_3_edges.sort()

# (cell_id-1, cell, gene, weight)
# [ref_cell_total, ref_gene_total, que_cell_total, que_gene_total]
start = time.perf_counter()

ref_cell_total = []
ref_gene_total = []
ref_weight_total = []
for i in range(len(Brain_3_edges)):
    ref_cell_total = ref_cell_total + Brain_3_edges[i][1]
    ref_gene_total = ref_gene_total + Brain_3_edges[i][2]
    ref_weight_total = ref_weight_total + Brain_3_edges[i][3]
    if i % 500 == 0:
        print(f'Constructing Brain 3 of cell {i} of total {len(Brain_3_edges)} cells')

end = time.perf_counter()
print(f'Constructing Brain 3 graph edge data finished in {end-start} seconds')

start = time.perf_counter()

def cell_graph(cell_id, dataset):
    if cell_id % 500 == 0:
        print(f'professing cell {cell_id} ...')
    if cell_id == dataset.shape[1]-1:
        print(f'professing last cell {cell_id} ...')
    cell = dataset.shape[0] * [cell_id-1]
    gene = list(range(dataset.shape[0]))
    weight = list(dataset.iloc[:, cell_id])
    
    return (cell_id, cell, gene, weight)


pool = multiprocessing.Pool(11)
temp_func = partial(cell_graph, dataset = Brain_4_X_drop)
blocks = pool.map(func=temp_func, iterable=list(range(1, Brain_4_X_drop.shape[1])), chunksize=1000)

pool.close()
pool.join()
    
end = time.perf_counter()
print(f'Reading data Brain 4 finished in {end-start} seconds')

# blocks = result_list
Brain_4_edges = blocks
Brain_4_edges.sort()

# (cell_id-1, cell, gene, weight)
# [ref_cell_total, ref_gene_total, que_cell_total, que_gene_total]
start = time.perf_counter()

que_cell_total = []
que_gene_total = []
que_weight_total = []
for i in range(len(Brain_4_edges)):
    que_cell_total = que_cell_total + Brain_4_edges[i][1]
    que_gene_total = que_gene_total + Brain_4_edges[i][2]
    que_weight_total = que_weight_total + Brain_4_edges[i][3]
    if i % 500 == 0:
        print(f'Constructing Brain 4 of cell {i} of total {len(Brain_4_edges)} cells')

end = time.perf_counter()
print(f'Constructing Brain 4 graph edge data finished in {end-start} seconds')

# save graph edge tensor
import pickle

# save
with open('edges.pickle', 'wb') as handle:
    pickle.dump([ref_cell_total, ref_gene_total, ref_weight_total, que_cell_total, que_gene_total, que_weight_total], handle)
    
# open
start = time.perf_counter()

with open('edges.pickle', 'rb') as handle:
    data = pickle.load(handle)
end = time.perf_counter()

print(f'Saving edge data finished in {end-start} seconds')

# hetero_graph = dgl.heterograph({
#     ('user', 'follow', 'user'): (follow_src, follow_dst),
#     ('user', 'followed-by', 'user'): (follow_dst, follow_src),
#     ('user', 'click', 'item'): (click_src, click_dst),
#     ('item', 'clicked-by', 'user'): (click_dst, click_src),
#     ('user', 'dislike', 'item'): (dislike_src, dislike_dst),
#     ('item', 'disliked-by', 'user'): (dislike_dst, dislike_src),
#     ('user', 'fake_follow', 'fake_user'): (follow_src, follow_dst),
# })

# get gene ids

# Brain_3_X_drop

g = dgl.heterograph({
    ('ref_cell', 'ref_cell_2_gene', 'gene'): (torch.tensor(ref_cell_total), torch.tensor(ref_gene_total)),
    ('gene', 'gene_2_ref_cell', 'ref_cell'): (torch.tensor(ref_gene_total), torch.tensor(ref_cell_total)),
    ('que_cell', 'que_cell_2_gene', 'gene'): (torch.tensor(que_cell_total), torch.tensor(que_gene_total)),
    ('gene', 'gene_2_que_cell', 'que_cell'): (torch.tensor(que_gene_total), torch.tensor(que_cell_total)),
    # ('gene', 'interaction', 'gene'): ()
})


g.edges['ref_cell_2_gene'].data['expression'] = th.tensor(ref_weight_total)
g.edges['gene_2_ref_cell'].data['expression'] = th.tensor(ref_weight_total)
g.edges['que_cell_2_gene'].data['expression'] = th.tensor(que_weight_total)
g.edges['gene_2_que_cell'].data['expression'] = th.tensor(que_weight_total)

# save the graph
start = time.perf_counter()
save_graphs("./g.bin", g)
end = time.perf_counter()
print(f'Saving graph finished in {end-start} seconds')

g = load_graphs("./g.bin")

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
print(len(embedding))
print(embedding)

# use only node features
node_features = {'gene': gene_feats, 'ref_cell': ref_cell_feats, 'que_cell': que_cell_feats}

embedding = model(g, node_features)
               
gene_embedding = embedding['gene']
ref_cell_embedding = embedding['ref_cell']
que_cell_embedding = embedding['que_cell']
print(len(embedding))
print(embedding)

true_ref_cell_dis_matrix = euclidean_distances(ref_cell_feats, ref_cell_feats)
true_que_cell_dis_matrix = euclidean_distances(ref_cell_feats, ref_cell_feats)
print(true_ref_cell_dis_matrix.shape)
print(true_que_cell_dis_matrix.shape)

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
    

opt = torch.optim.Adam(model.parameters())
loss_func = nn.MSELoss()


for epoch in range(5):
    model.train()
    pred = model(g, node_features)
    # compute loss
    pred_ref_cell_feats = pred['ref_cell']
    print('pred ref cell feature shape:', pred_ref_cell_feats.shape)
    pred_que_cell_feats = pred['que_cell']
    print('pred que cell feature shape:', pred_que_cell_feats.shape)
    
    train_pred_ref_cell_feats = pred_ref_cell_feats[ref_cell_mask]
    train_pred_que_cell_feats = pred_que_cell_feats[que_cell_mask]
    
#     val_pred_ref_cell_feats = pred_ref_cell_feats[1-ref_cell_mask]
#     val_pred_que_cell_feats = pred_que_cell_feats[1-que_cell_mask]
    
    pred_ref_cell_dis_matrix = euclidean_distances(pred_ref_cell_feats, pred_ref_cell_feats)
    pred_que_cell_dis_matrix = euclidean_distances(pred_que_cell_feats, pred_que_cell_feats)
    
    loss = loss_func(pred_ref_cell_dis_matrix, true_ref_cell_dis_matrix) + loss_func(pred_que_cell_dis_matrix, true_que_cell_dis_matrix)

    opt.zero_grad()
    loss.backward()
    opt.step()
    print(loss.item())

    # Save model if necessary.  Omitted in the example.
