import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import argparse
import random
import numpy as np
from dgl.data import register_data_args, load_data
import my_data_loader_dgl_part2
import models_attention_along_path4 #change model number here 1
from experiment.node_classification import node_classification_minibatch, node_classification_fullbatch
from experiment.link_prediction import link_prediction_minibatch, link_prediction_fullbatch
from model.MECCH import MECCH, khopMECCH
from model.baselines.RGCN import RGCN
from model.baselines.HGT import HGT
from model.baselines.HAN import HAN, HAN_lp
from model.modules import LinkPrediction_minibatch, LinkPrediction_fullbatch
from utils import metapath2str, get_metapath_g, get_khop_g, load_data_nc, load_data_lp, get_save_path, load_base_config, load_model_config
import pdb

# python train_HetaFlow_along_path5.py --dataset=acm-gtn

# 1 layer but big model, firt decomposed paths , multi-pool, LeakyReLU
model_num = 'path_att_1'  #change model number here 2
num_labels = 7 # 

# parser = argparse.ArgumentParser("My HGNNs")
parser = argparse.ArgumentParser(description='GAT')
# employing attention mechanism for adjustment part
register_data_args(parser)

parser.add_argument('--ds', type=str, default=-1, help='name of dataset')

parser.add_argument("--gpu", type=int, default=0,
                    help="which GPU to use. Set -1 to use CPU.")
parser.add_argument("--no_cuda", type=bool, default=0,
                    help="whether has GPU to use. Set 1 to use CPU.")
parser.add_argument("--epochs", type=int, default=30,
                    help="number of training epochs")
parser.add_argument("--num-heads", type=int, default=1,
                    help="number of hidden attention heads")
parser.add_argument("--num-out-heads", type=int, default=1,
                    help="number of output attention heads")
parser.add_argument("--num-layers", type=int, default=1,
                    help="number of hidden layers")
parser.add_argument("--num_hidden", type=int, default=num_labels,
                    help="number of hidden units")
parser.add_argument("--seed", type=int, default=88,
                    help="number of seed")
parser.add_argument("--residual", action="store_true", default=False,
                    help="use residual connection")
parser.add_argument("--in-drop", type=float, default=.75,
                    help="input feature dropout")
parser.add_argument("--dropout", type=float, default=.75,
                    help="attention dropout")
parser.add_argument("--learning_rate", type=float, default=0.1,
                    help="learning rate")
parser.add_argument('--weight-decay', type=float, default= 5e-4,
                    help="weight decay")
parser.add_argument('--negative-slope', type=float, default=0.002,
                    help="the negative slope of leaky relu")
parser.add_argument('--fastmode', action="store_true", default=False,
                    help="skip re-evaluate the validation set")
parser.add_argument('--alpha', type=float, default=0.002, help='Alpha for the leaky_relu.')

parser.add_argument('--model', '-m',default='GAT', type=str, required=True, help='name of model')
 
parser.add_argument('--task', '-t', type=str, default='node_classification', help='type of task')

parser.add_argument('--config', '-c', type=str, help='config file for model hyperparameters')

parser.add_argument('--repeat', '-r', type=int, default=1, help='repeat the training and testing for N times')
args = parser.parse_args()
args.n_neighbor_samples = 0
if args.config is None:
    args.config = "./configs/{}.json".format(args.model)

configs = load_base_config()
configs.update(load_model_config(args.config, args.dataset))
# 
configs.update(vars(args))
args = argparse.Namespace(**configs)
print(args)



args.cuda = not args.no_cuda and th.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
th.manual_seed(args.seed)
if args.cuda:
    th.cuda.manual_seed(args.seed)


# Device configuration
use_cuda = args.gpu >= 0 and th.cuda.is_available()
device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
if use_cuda:
    args.device = th.device('cuda', args.gpu)

else:
    args.device = th.device('cpu')
# Hyper parameters
num_epochs = args.epochs
learning_rate = args.learning_rate

# Data loader
# args.dataset='cora'
print(args)

dataloader = my_data_loader_dgl_part2.my_data_set(args) # data loader use multi process, which will cause error at windows 


# Convolutional neural network (two convolutional layers)
# The convolution layer keep the conveluted sequence length the same as input.
# input tensor size (batch_size, channels, 1)



model = models_attention_along_path4.GAT(
                nfeat=dataloader.info_dict['features'].shape[1], 
                nhid=args.num_hidden, 
                nclass=dataloader.info_dict['n_classes'], 
                dropout=args.dropout, 
                nheads=args.num_heads, 
                alpha=args.alpha)  #change model number here 3
model = model.to(device)

# Loss and optimizer
#criterion = nn.CrossEntropyLoss()
#criterion = F.nll_loss()
optimizer = th.optim.Adam(model.parameters(), lr=args.learning_rate)

# Train the model
train_steps = dataloader.get_data_len('train')
val_steps = dataloader.get_data_len('val')
test_steps = dataloader.get_data_len('test')
print('train_steps', train_steps)
print('val_steps', val_steps)
print('test_steps', val_steps)

max_val_acc = 0
max_val_epoch = 0
set_patient = 400
patient = 0

for epoch in range(num_epochs):
    #print('train at epoch ', epoch)
    epoch_loss = 0
    for i in range(train_steps): #train_steps
        # if i%10000 == 0:
        #     print('train at step', i)
        datas, labels, gat_vec, node_idx = dataloader.get_a_path_data('train', i)
        datas = datas.to(device)
        #datas = torch.squeeze(datas)
        labels = labels.to(device)
        #print('labels shape ', labels.shape)
        labels = labels.reshape(-1)
        #print('labels reshaped ', labels.shape)
        labels = labels.long()
        gat_vec = gat_vec.to(device)
        # pdb.set_trace()
        # Forward pass
        outputs = model(datas)
        

        # outputs = torch.masked_select(outputs, masks.type(torch.ByteTensor))
        # labels = torch.masked_select(labels, masks.type(torch.ByteTensor))
        #print(outputs.shape)
        #print(labels.shape)
        loss = F.nll_loss(outputs, labels)

        epoch_loss = epoch_loss + loss.item()
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if 1: #i%80000==0 and i>0:
        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
           .format(epoch+1, num_epochs, i+1, train_steps, loss.item()), end='')
        print(' epoch_loss', epoch_loss, end='')
        epoch_loss = 0

        # Test the model
        model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        with th.no_grad():
            correct = 0
            total = 0
            for i in range(val_steps): # val_steps
                datas, labels, gat_vec, node_idx = dataloader.get_a_path_data('val', i)
                images = datas.to(device)
                labels = labels.to(device)
                labels = labels.reshape(-1)
                labels = labels.long()
                gat_vec = gat_vec.to(device)
                outputs = model(images, is_training=False)
                _, predicted = th.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print('  Accuracy on val data: {} %'.format(100 * correct / total))
            val_acc = correct / total
            if epoch > 20 :
                if val_acc > max_val_acc:
                    max_val_acc = val_acc
                    max_val_epoch = epoch
                    patient = 0
                    # Save the model checkpoint
                    th.save(model.state_dict(), 'saved_'+str(model_num)+'_'+str(epoch)+'model.ckpt')
                else:
                    patient += 1
                    if patient > set_patient:
                        break

        model.train()


# Final test accuray
model.load_state_dict(th.load('saved_'+str(model_num)+'_'+str(max_val_epoch)+'model.ckpt'))
model.eval()
with th.no_grad():
    correct = 0
    total = 0
    for i in range(test_steps): # val_steps
        datas, labels, gat_vec, node_idx = dataloader.get_a_path_data('test', i)
        images = datas.to(device)
        labels = labels.to(device)
        labels = labels.reshape(-1)
        labels = labels.long()
        gat_vec = gat_vec.to(device)
        outputs = model(images, is_training=False)
        _, predicted = th.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Accuracy on test data: {} %'.format(100 * correct / total))


        

    