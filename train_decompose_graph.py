"""
This is to decompse a graph data in to flows.
How to run the code:
python train_decompose_graph.py --dataset=acm-gtn

It will produce a file to store the flows. It is named 'decomposed_paths_central_rectangle_acm-gtn'.
"""


import argparse
import numpy as np
import time
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from gat import GAT
from utils import EarlyStopping
import fun_decompose_graph_central_rectangle
import argparse
import pickle
from pathlib import Path
import dgl
import numpy as np
import torch as th
import pdb
from experiment.node_classification import node_classification_minibatch, node_classification_fullbatch
from experiment.link_prediction import link_prediction_minibatch, link_prediction_fullbatch
from model.MECCH import MECCH, khopMECCH
from model.baselines.RGCN import RGCN
from model.baselines.HGT import HGT
from model.baselines.HAN import HAN, HAN_lp
from model.modules import LinkPrediction_minibatch, LinkPrediction_fullbatch
from utils import metapath2str, get_metapath_g, get_khop_g, load_data_nc, load_data_lp, \
    get_save_path, load_base_config, load_model_config

    
def accuracy(logits, labels):
    _, indices = th.max(logits, dim=1)
    correct = th.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


def evaluate(model, features, labels, mask):
    model.eval()
    with th.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        return accuracy(logits, labels)


def main(args):
    # load and preprocess dataset
    dir_path_list = []
    # load data
    
    g, in_dim_dict, out_dim, train_nid_dict, val_nid_dict, test_nid_dict = load_data_nc(args.dataset)
    print("Loaded data from dataset: {}".format(args.dataset))

    target_ntype = list(g.ndata["y"].keys())[0]
    # check cuda
    use_cuda = args.gpu >= 0 and th.cuda.is_available()
    if use_cuda:
        args.device = th.device('cuda', args.gpu)
    else:
        args.device = th.device('cpu')
    dataset = args.dataset

    g = g.to(args.device)
    print("Loaded data from dataset: {}".format(args.dataset))
    keysList = list(train_nid_dict.keys())
    To_be_projected = th.cat((train_nid_dict[keysList[0]],val_nid_dict[keysList[0]],test_nid_dict[keysList[0]]))
    # all_nid_dict={keysList[0]:To_be_projected}
    # all_nid_dict = {k: v.to(args.device) for k, v in all_nid_dict.items()}

    G = g
    fun_decompose_graph_central_rectangle.main_of_decompose(G, args.dataset, To_be_projected)
    print('done')
    exit(0)

    


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GAT')
    register_data_args(parser)
    parser.add_argument("--gpu", type=int, default=-1,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--num-heads", type=int, default=8,
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=1,
                        help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=8,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=.6,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=.6,
                        help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument('--fastmode', action="store_true", default=False,
                        help="skip re-evaluate the validation set")
    args = parser.parse_args()
    print(args)

    main(args)
