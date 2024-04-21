# HetaFlow



## Requriment
To run the code of this repository, the following requriments are needed.
- python          3
- pytorch         2.2.1+cu121
- dgl             2.1.0+cu121
- torchvision     0.17.1+cu121
- tqdm            4.66.2
-  Intall this lib first: [dgl](https://github.com/dmlc/dgl)

## Simple Run
Dowload the repository, intall the requriments. Excute the following in a CMD or shell or terminal:
`python train_HetaFlow_along_path5.py --dataset=dblp-gtn  --model GAT --epochs 150 --learning_rate 5e-5`



This is the output:
```
ubuntu@VM-0-8-ubuntu:~/backup/GFCN$ python3 train_HetaFlow_along_path5.py --dataset=dblp-gtn --model GAT
Base configs loaded.
Model configs loaded.
dblp-gtn dataset configs for this model loaded, override defaults.
...
Epoch [24/30], Step [800/800], Loss: 8.5199 epoch_loss 364.7407269842561  Accuracy on val data: 95.25 %
Epoch [25/30], Step [800/800], Loss: 8.5164 epoch_loss 364.731803859032  Accuracy on val data: 95.25 %
Epoch [26/30], Step [800/800], Loss: 8.5135 epoch_loss 364.72594831377097  Accuracy on val data: 95.25 %
Epoch [27/30], Step [800/800], Loss: 8.5110 epoch_loss 364.72216447466377  Accuracy on val data: 95.25 %
Epoch [28/30], Step [800/800], Loss: 8.5090 epoch_loss 364.7197684542894  Accuracy on val data: 95.25 %
Epoch [29/30], Step [800/800], Loss: 8.5073 epoch_loss 364.7182649160564  Accuracy on val data: 95.25 %
Epoch [30/30], Step [800/800], Loss: 8.5058 epoch_loss 364.7173888884259  Accuracy on val data: 95.25 %
Accuracy on test data: 94.32971648582429 %
```

## Detail Run
- Step1
Decompse a graph data (DBLP) in to flows.
`python train_decompose_graph.py --dataset=dblp-gtn`
It will produce a file to store the flows. It is named **decomposed_paths_central_rectangle_DBLP**.

- Step2
Produce the node vectors using a graph neural network.
`python train_gen_vec.py -t node_classification -d dblp-gtn -g 0`
It will produce a file to store node vector. It is named **logits_of_DBLP**

- Step3
Train a GAT model whose attention is along flows or paths (to do the feature adjustments) and test its accuracy.
`python train_HetaFlow_along_path5.py --dataset=dblp-gtn  --model GAT --epochs 150 --learning_rate 5e-5`

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL, -m MODEL
                        name of model (can also use other models like GCN, HAN to do the adjustments/ projections)
  --dataset DATASET, -d DATASET
                        name of dataset
  --task TASK, -t TASK  type of task
  --gpu GPU, -g GPU     which gpu to use, specify -1 to use CPU
  --config CONFIG, -c CONFIG
                        config file for model hyperparameters
  --repeat REPEAT, -r REPEAT
                        repeat the training and testing for N times
## Datasets
All datasets are preprocessed by GTN. Details of how to change/ update the datasets are listed in the folders.
IMDB
ACM
DBLP
