"""
load data
"""

import os
import torch as th
import torch.utils.data as data
#from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import utils
import pickle
from dgl.data import register_data_args, load_data
from utils import metapath2str, get_metapath_g, get_khop_g, load_data_nc, get_save_path, load_base_config, load_model_config
import pdb
import dgl

#from utils_fixed_lable import load_data, accuracy

class my_data_set(data.Dataset):
    def __init__(self, args, transform = None, target_transform=None):

        print('loading dataset ', args.dataset)
        args.n_neighbor_samples = 0
        load_dict = pickle.load(open("projections_of_"+args.dataset, "rb"))
        features = load_dict['projections']

        # features = features.data.numpy()
        features = np.squeeze(features)
        #features = torch.Tensor(features)  let features's type be numpy data
        # print('features shape', features.shape)

        self.gat_vec = np.squeeze(features)
        
        load_dict = pickle.load(open("decomposed_paths_central_rectangle_"+args.dataset, "rb"))
        paths = load_dict['decomposed_paths']
        # load and preprocess dataset

        g, in_dim_dict, out_dim, train_nid_dict, val_nid_dict, test_nid_dict = load_data_nc(args.dataset)
        print("Loaded data from dataset: {}".format(args.dataset))

        target_ntype = list(g.ndata["y"].keys())[0]

        dataset = args.dataset
        g = g.to(args.device)
        print("Loaded data from dataset: {}".format(args.dataset))
        keysList = list(train_nid_dict.keys())
        To_be_projected = th.cat((train_nid_dict[keysList[0]],val_nid_dict[keysList[0]],test_nid_dict[keysList[0]]))
        all_nid_dict={keysList[0]:To_be_projected}
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
        all_dataloader = dgl.dataloading.DataLoader(
            g,
            all_nid_dict,
            sampler,
            batch_size=len(all_nid_dict[target_ntype]),
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            device=args.device,
        )

        for input_nodes, output_nodes, blocks in all_dataloader:
            input_features = blocks[0].srcdata["x"]
            output_labels = blocks[-1].dstdata["y"]

        #features = torch.FloatTensor(data.features)
        labels = th.cuda.LongTensor(output_labels[keysList[0]])
        train_mask = train_nid_dict[keysList[0]]
        val_mask = val_nid_dict[keysList[0]]
        test_mask = test_nid_dict[keysList[0]]
        # num_feats = features.shape[1]
        n_classes = out_dim
        n_edges = g.number_of_edges()
        print("""----Data statistics------'
          #Edges %d
          #Classes %d 
          #Train samples %d
          #Val samples %d
          #Test samples %d""" %
              (n_edges, n_classes,
               (train_mask>-1).sum().item(),
               (val_mask>-1).sum().item(),
               (test_mask>-1).sum().item()))

        #idx_train = th.nonzero(train_mask).reshape(-1)
        #idx_val = th.nonzero(val_mask).reshape(-1)
        #idx_test = th.nonzero(test_mask).reshape(-1)
        idx_train = train_nid_dict[keysList[0]]
        idx_val = val_nid_dict[keysList[0]]
        idx_test = test_nid_dict[keysList[0]]
        # # Load data
        # adj, features_no_use, labels, idx_train, idx_val, idx_test = load_data(dataset='citeseer')

        info_dict = {}
        info_dict['Projected'] = set(To_be_projected.tolist())
        info_dict['features'] = features
        info_dict['labels'] = labels
        info_dict['n_classes'] = n_classes
        info_dict['train_idx_set']  = set(idx_train.data.tolist())
        info_dict['val_idx_set'] = set(idx_val.data.tolist())
        info_dict['test_idx_set'] = set(idx_test.data.tolist())
        print('len train_idx_set ', len(info_dict['train_idx_set']))
        print('len val_idx_set ', len(info_dict['val_idx_set']))
        print('len test_idx_set ', len(info_dict['test_idx_set']))
        label_set = set()
        for i in range(labels.shape[0]):
            #print(labels[i])
            label_set.add(labels[i].data.tolist())
        print('labels len', len(label_set))
        
        # for key in ['train', 'val', 'test']:
        #     for key2 in ['datas', 'labels', 'masks']:
        #         info_dict[key+'_'+key2] = []
        # train_datas = [], val_datas = [], test_datas = []
        # train_labels = [], val_labels = [], test_labels = []
        # train_masks = [], val_masks = [], test_masks = []
        
        info_dict = self._seperate_train_val_test_path(paths, info_dict)

        """
        for path in paths:
            for key in ['train', 'val', 'test']:
                idx_set = info_dict[key+'_idx_set']
                contains_idx_node, data_of_a_path, label_of_a_path, mask_of_a_path =\
                 self._prepare_data(path, idx_set, features, labels)
                if contains_idx_node:
                    info_dict[key+'_datas'].append(data_of_a_path)
                    info_dict[key+'_labels'].append(label_of_a_path)
                    info_dict[key+'_masks'].append(mask_of_a_path) 

        for key in ['train', 'val', 'test']:
            info_dict[key+'_datas'] = np.array(info_dict[key+'_datas'])
            info_dict[key+'_labels'] = np.array(info_dict[key+'_labels'])
            info_dict[key+'_masks'] = np.array(info_dict[key+'_masks'])
            print(key+'_datas'+ ' shape',  info_dict[key+'_datas'].shape)
            print(key+'_labels'+ ' shape',  info_dict[key+'_labels'].shape)
            print(key+'_masks'+ ' shape',  info_dict[key+'_masks'].shape)
            print(info_dict[key+'_datas'][0])
        """

        #self.root = root
        self.info_dict = info_dict
        self.transform = transform
        self.target_transform = target_transform

    def _seperate_train_val_test_path(self, paths, info_dict):
        for key in ['train', 'val', 'test','others']:
            info_dict[key+'_paths'] = []

        for path in paths:
            find_a_set = False
            for key in ['train', 'val', 'test']:
                idx_set = info_dict[key+'_idx_set']
                if path[0][2] in idx_set:
                    find_a_set = True
                    info_dict[key+'_paths'].append(path)
                    break
            if not find_a_set:
                info_dict['others'+'_paths'].append(path)

        for key in ['train', 'val', 'test','others']:
            print(key+'_paths len', len(info_dict[key+'_paths']))
        return info_dict


    def __getitem__(self, index):
        data = self.datas[index]
        label = self.labels[index]
        return th.Tensor(data), th.Tensor(label)

    def __len__(self):
        return len(self.info_dict['train_paths'])

    def get_data_len(self, key): # key = train val test
        return len(self.info_dict[key+'_paths'])

    def get_a_path_data(self, key, index):
        paths = self.info_dict[key+'_paths'][index]
        #print(path)
        
        data_of_paths = []
        label_of_paths = []

        # option 1 directly use the mean of the neighbor features 
        # option 2 0 paddings
        # option 3 directly substitue
        opt=3
        for path in paths:
            data_of_a_path = []

               
            for node in path:
                if node not in self.info_dict['Projected']:
                    if opt == 3:
                        node = path[2]  
                data_of_a_path.append(self.info_dict['features'][node])
            #data_of_a_path = torch.stack(data_of_a_path, 0)
            #print('data_of_a_path 1', data_of_a_path)

            data_of_paths.append(data_of_a_path)

        label_of_paths.append(self.info_dict['labels'][paths[0][2]])
        node_idx = paths[0][2]
            
        # print('get data')
        # exit(0)

        data_of_a_path = np.array(data_of_paths)
        #print('data_of_a_path 2', data_of_a_path.shape)
        gat_vec = self.gat_vec[paths[0][2]]
        
        return th.unsqueeze(th.Tensor(data_of_paths), 0)  , \
        th.unsqueeze(th.Tensor(label_of_paths), 0), \
        th.unsqueeze(th.Tensor(gat_vec), 0), \
        node_idx

    # def get_a_batch_path_data(self, key, index, batch_size):
    #     paths = self.info_dict[key+'_paths'][index:index+batch_size]
    #     #print(path)
    #     idx_set = self.info_dict[key+'_idx_set']
    #     data_of_a_path = []
    #     label_of_a_path = []
    #     mask_of_a_path = []
    #     for path in paths:
    #         for node in path:
    #             data_of_a_path.append(self.info_dict['features'][node])
    #             label_of_a_path.append(self.info_dict['labels'][node])
    #             if node in idx_set:
    #                 mask_of_a_path.append(1)
    #             else:
    #                 mask_of_a_path.append(0)
    #     # print('get data')
    #     # exit(0)
    #     data_of_a_path = np.array(data_of_a_path)
    #     # add a batch size dimmension
    #     return torch.unsqueeze(torch.Tensor(data_of_a_path), 0)  , \
    #     torch.unsqueeze(torch.Tensor(label_of_a_path), 0), \
    #     torch.unsqueeze(torch.Tensor(mask_of_a_path), 0)

    


def test_my_data_set():

    dataloader = my_data_set(root = "./test_data")
    print('data_len train ', dataloader.get_data_len('train'))
    data_of_paths, label_of_paths = dataloader.get_a_path_data('others', 0)
    print(data_of_paths.shape) # torch.Size([1, 10, 5, 1433])
    print(label_of_paths)
    

    # print('len dataloader:', len(dataloader))

    # for index , (data, label) in enumerate(dataloader):
    #     print('data:',data)
    #     print ('label:',label)
    

    
if __name__ == "__main__":

    test_my_data_set()