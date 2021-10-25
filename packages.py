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
from io import StringIO
from functools import partial
import torch.nn.functional as F
from sklearn.metrics.pairwise import euclidean_distances