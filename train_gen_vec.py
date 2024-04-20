"""
This is to produce the node vectors using a graph neural network.
How to run the code:
python train_gen_vec.py --dataset=acm-gtn

It will produce a file to store node vector. It is named 'logits_of_acm-gtn'
"""

#

import argparse
import numpy as np
import time
import tqdm
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from gat import GAT
import fun_decompose_graph_central_rectangle
import argparse
import pickle
from pathlib import Path
import dgl
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
    get_save_path, load_base_config, load_model_config, EarlyStopping


def accuracy(logits, labels):
    _, indices = th.max(logits, dim=1)
    correct = th.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)

def projection(model, g, all_nid_dict, dir_path, args):
    # feature projection
    dataset = args.dataset
    model.to(args.device)
    g = g.to(args.device)
    all_nid_dict = {k: v.to(args.device) for k, v in all_nid_dict.items()}

    assert len(g.ndata["y"].keys()) == 1
    target_ntype = list(g.ndata["y"].keys())[0]

    # Use GPU-based neighborhood sampling if possible
    num_workers = 4 if args.device.type == "CPU" else 0
    if args.n_neighbor_samples <= 0:
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args.n_layers)
    else:
        sampler = dgl.dataloading.MultiLayerNeighborSampler([{
            etype: args.n_neighbor_samples for etype in g.canonical_etypes}] * args.n_layers)
    pro_dataloader = dgl.dataloading.DataLoader(
        g,
        all_nid_dict,
        sampler,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        device=args.device,
    )
    model.load_state_dict(th.load(str(dir_path / "checkpoint.pt")))
    model.eval()
    with tqdm.tqdm(pro_dataloader) as tq, th.no_grad():
        projections_list = []
        # y_true_list = []
        for iteration, (input_nodes, output_nodes, blocks) in enumerate(tq):
            input_features = blocks[0].srcdata["x"]
            output_labels = blocks[-1].dstdata["y"]

            projections_dict = model(blocks, input_features)

            projections_list.append(projections_dict[target_ntype].cpu().numpy())
            # y_true_list.append(output_labels[target_ntype].cpu().numpy())
        projections = np.concatenate(projections_list, axis=0)
        dump_dict = {'projections':projections}
        pickle.dump(dump_dict, open("projections_of_"+dataset, "wb"))
    # projection of the features

    
def evaluate(model, features, labels, mask):
    model.eval()
    with th.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        return accuracy(logits, labels)

def evaluate_and_save_vec(model, features, labels, mask, dataset):
    model.eval()
    with th.no_grad():
        logits = model(features)
        dump_dict = {'logits':logits}
        pickle.dump(dump_dict, open("logits_of_"+dataset, "wb"))

        logits = logits[mask]
        labels = labels[mask]
        return accuracy(logits, labels)


def main(args):
    t1= time.time()
    test_macro_f1_list = []
    test_micro_f1_list = []
    dir_path_list = []
    for _ in range(args.repeat):
        dir_path_list.append(get_save_path(args))
    # load and preprocess dataset

    g, in_dim_dict, out_dim, train_nid_dict, val_nid_dict, test_nid_dict = load_data_nc(args.dataset)
    print("Loaded data from dataset: {}".format(args.dataset))
    # get the id of the nodes that needs to be projected.
    keysList = list(train_nid_dict.keys())
    To_be_projected = th.cat((train_nid_dict[keysList[0]],val_nid_dict[keysList[0]],test_nid_dict[keysList[0]]))
    all_nid_dict={keysList[0]:To_be_projected}
    if args.gpu < 0:
        cuda = False
        args.device = th.device('cpu')
    else:
        cuda = True
        th.cuda.set_device(args.gpu)
        args.device = th.device('cuda', args.gpu)
    #     features = features.cuda()
    #     labels = labels.cuda()
    #     train_mask = train_mask.cuda()
    #     val_mask = val_mask.cuda()
    #     test_mask = test_mask.cuda()

    # g = data.graph
    # # add self loop
    # g.remove_edges_from(nx.selfloop_edges(g))
    # g = DGLGraph(g)
    # g.add_edges(g.nodes(), g.nodes())
    # n_edges = g.number_of_edges()
    # create model + model-specific data preprocessing
    # heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
    # model = GAT(g,
    #             args.num_layers,
    #             num_feats,
    #             args.num_hidden,
    #             n_classes,
    #             heads,
    #             F.elu,
    #             args.in_drop,
    #             args.attn_drop,
    #             args.negative_slope,
    #             args.residual)
    # print(model)
    # stopper = EarlyStopping(patience=100)
    # if cuda:
    #     model.cuda()
    # loss_fcn = th.nn.CrossEntropyLoss()
    
    g, selected_metapaths = get_metapath_g(g, args)
    n_heads_list = [args.n_heads] * args.n_layers
    model = MECCH(
        g,
        selected_metapaths,
        in_dim_dict,
        args.hidden_dim,
        out_dim,
        args.n_layers,
        n_heads_list,
        dropout=args.dropout,
        context_encoder=args.context_encoder,
        use_v=args.use_v,
        metapath_fusion=args.metapath_fusion,
        residual=args.residual,
        layer_norm=args.layer_norm
    )
    minibatch_flag = True
    # stopper = EarlyStopping(patience=100)
    # # use optimizer
    # optimizer = th.optim.Adam(
    #     model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # # initialize graph
    # dur = []
    # for epoch in range(args.epochs):
    #     model.train()
    #     if epoch >= 3:
    #         t0 = time.time()
    #     # forward
    #     logits = model(features)
    #     loss = loss_fcn(logits[train_mask], labels[train_mask])

    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()

    #     if epoch >= 3:
    #         dur.append(time.time() - t0)

    #     train_acc = accuracy(logits[train_mask], labels[train_mask])

        # if args.fastmode:
        #     val_acc = accuracy(logits[val_mask], labels[val_mask])
        # else:
        #     val_acc = evaluate(model, features, labels, val_mask)
        #     if stopper.step(val_acc, model):   
        #         break
    if minibatch_flag:
        test_macro_f1, test_micro_f1 = node_classification_minibatch(model, g, train_nid_dict, val_nid_dict,
                                                                        test_nid_dict, dir_path_list[0], args)
    else:
        test_macro_f1, test_micro_f1 = node_classification_fullbatch(model, g, train_nid_dict, val_nid_dict,
                                                                        test_nid_dict, dir_path_list[0], args)
    test_macro_f1_list.append(test_macro_f1)
    test_micro_f1_list.append(test_micro_f1)

    # print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | TrainAcc {:.4f} |"
    #         " ValAcc {:.4f} | ETputs(KTEPS) {:.2f}".
    #         format(epoch, np.mean(dur), loss.item(), train_acc,
    #                 val_acc, n_edges / np.mean(dur) / 1000))

    # print()
    model.load_state_dict(th.load('es_checkpoint.pt'))
    #acc = evaluate(model, features, labels, test_mask)
    # acc = evaluate_and_save_vec(model, features, labels, test_mask, args.dataset)
    #print("Test Accuracy {:.4f}".format(acc))
    projection(model, g, all_nid_dict, dir_path_list[0], args)
    time_proj= time.time()-t1
    print("Features projected in {:.4f} second(s)".format(time_proj))


# if __name__ == '__main__':

    # parser = argparse.ArgumentParser(description='GAT')
    # register_data_args(parser)
    # parser.add_argument("--gpu", type=int, default=-1,
    #                     help="which GPU to use. Set -1 to use CPU.")
    # parser.add_argument("--epochs", type=int, default=200,
    #                     help="number of training epochs")
    # parser.add_argument("--num-heads", type=int, default=8,
    #                     help="number of hidden attention heads")
    # parser.add_argument("--num-out-heads", type=int, default=1,
    #                     help="number of output attention heads")
    # parser.add_argument("--num-layers", type=int, default=1,
    #                     help="number of hidden layers")
    # parser.add_argument("--num-hidden", type=int, default=8,
    #                     help="number of hidden units")
    # parser.add_argument("--residual", action="store_true", default=False,
    #                     help="use residual connection")
    # parser.add_argument("--in-drop", type=float, default=.6,
    #                     help="input feature dropout")
    # parser.add_argument("--attn-drop", type=float, default=.6,
    #                     help="attention dropout")
    # parser.add_argument("--lr", type=float, default=0.005,
    #                     help="learning rate")
    # parser.add_argument('--weight-decay', type=float, default=5e-4,
    #                     help="weight decay")
    # parser.add_argument('--negative-slope', type=float, default=0.002,
    #                     help="the negative slope of leaky relu")
    # parser.add_argument('--fastmode', action="store_true", default=False,
    #                     help="skip re-evaluate the validation set")
    # args = parser.parse_args()
    # print(args)
if __name__ == "__main__":
    parser = argparse.ArgumentParser("My HGNNs")
    parser.add_argument('--model', '-m', type=str, required=True, help='name of model')
    parser.add_argument('--dataset', '-d', type=str, required=True, help='name of dataset')
    parser.add_argument('--task', '-t', type=str, default='node_classification', help='type of task')
    parser.add_argument("--gpu", '-g', type=int, default=-1, help="which gpu to use, specify -1 to use CPU")
    parser.add_argument('--config', '-c', type=str, help='config file for model hyperparameters')
    parser.add_argument('--repeat', '-r', type=int, default=1, help='repeat the training and testing for N times')

    args = parser.parse_args()
    if args.config is None:
        args.config = "./configs/{}.json".format(args.model)

    configs = load_base_config()
    configs.update(load_model_config(args.config, args.dataset))
    configs.update(vars(args))
    args = argparse.Namespace(**configs)
    print(args)
    main(args)
