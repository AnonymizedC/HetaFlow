#import utils
import numpy as np
import pickle
import networkx as nx
import pdb
import dgl
import time
import torch as th

def gen_two_idx(used_path_idx_set, start, lens):
    # generat i j. start <= i < j < len , not in the set. if not find, one of ij will be -1 
    i = -1
    j = -1
    for idx in range(start, lens):
        if idx not in used_path_idx_set:
            i = idx
    for idx in range(i+1, lens):
        if idx not in used_path_idx_set:
            j = idx
    return i, j


def mirror_padding_paths(path_list):
    # 012  --> 21012
    ret_paths = []
    if len(path_list) == 0:
        return ret_paths
    for path in path_list:
        half_path = path[1::]
        #print('half_path', half_path)
        half_path.reverse()
        #print('half_path', half_path)
        #print('reverse path', path, path[1::], half_path)
        pad_path = half_path + path
        ret_paths.append(pad_path)
    return ret_paths

def add_paths(path_list):
    # path 3 2 5    paht 3 7 8 ---> 5 2 3 7 8
    used_path_idx_set = set()
    ret_paths = []
    for start in range(len(path_list)):
        idx1, idx2 = gen_two_idx(used_path_idx_set, start, len(path_list))
        if idx1 == -1 or idx2 == -1:
            break  #
        while idx2 < len(path_list):
            path1 = path_list[idx1]
            path2 = path_list[idx2]
            set1 = set(path1[1::])
            set2 = set(path2[1::])
            if len(set1 & set2) == 0:
                half_path = path1[1::]
                half_path.reverse()
                merge_path = half_path + path2
                used_path_idx_set.add(idx1)
                used_path_idx_set.add(idx2)
                ret_paths.append(merge_path)
                idx2 = len(path_list)
            idx2 += 1
    return ret_paths

def type_embeding(paths_, G2, num_types, type_nodelist, name_typelist):
    path_type_embed          = []

    for path in paths_:
        noded_type_embeding      = []
        for i in path:
            noded_type_embeding.append(name_typelist.index(G2.nodes[i]['ntype']))
        path_type_embed.append(noded_type_embeding)

            

            
        
    return path_type_embed

def gen_paths(G, source_node=0):
    # pdb.set_trace()
    # generate some path in the graph
    # G = dgl.to_networkx(G)

    edges_list = list(nx.dfs_edges(G, source=source_node, depth_limit=2))
    #print(edges_list)

    if len(edges_list) == 0:
        print(' this source_node find no neighbor ', source_node)
        print(edges_list)
    
    # creat a small graph
    G_small = nx.Graph()
    for edge in edges_list:
        G_small.add_edge(edge[0], edge[1])
    node_set = set()
    for node1, node2 in edges_list:
        node_set.add(node1)
        node_set.add(node2)
    if source_node in node_set:
        node_set.remove(source_node)
    # pdb.set_trace()
    #print(node_set)
    paths_len3 = []
    paths_len2 = []
    for node in node_set:
        paths = list(nx.shortest_simple_paths(G_small, source_node, node))
        for path in paths:
            if len(path) == 3:
                paths_len3.append(path)
            elif len(path) == 2:
                #print('len 2 path', path)
                paths_len2.append(path + [path[-1]])  #padding path len 2 to 3

    #print('paths_len3', paths_len3)
    #print('paths_len2', paths_len2)
    if len(paths_len3)==0 and len(paths_len2)==0:
        print('no edge of node:', source_node)
        print(edges_list)
        paths_len3 = [[source_node, source_node, source_node]]
    merge_paths = add_paths(paths_len3)

    # print('paths_len3', merge_paths)
    # print(len(merge_paths))
    if len(merge_paths) < 10:
        merge_paths = merge_paths + mirror_padding_paths(paths_len3)
        # print('paths_len3 mirror', merge_paths)
        # print(len(merge_paths))
    if len(merge_paths) < 10:

        merge_paths = merge_paths + add_paths(paths_len2)
        # print('paths_len2', merge_paths)
        # print(len(merge_paths))
    if len(merge_paths) < 10:
        merge_paths = merge_paths + mirror_padding_paths(paths_len2)
    while len(merge_paths) < 10:
        merge_paths = merge_paths + merge_paths
    # print(len(merge_paths))
    merge_paths = merge_paths[0:10]
    # exit(0)
    return merge_paths

def main_of_decompose(G, dataset_str, To_be_projected):
    # input a graph
    print('graph node:', G.number_of_nodes())
    print('graph edge:', G.number_of_edges())

    short_path_nodes = []
    decomposed_paths = []
    Type_encodings   = []
    T1 = time.time()
    G2 = dgl.to_networkx(G)
    type_nodelist    = th.tensor
    num_types = len(G.ntypes)
    type_nodelist = G.nodes(G.ntypes[0])
    name_typelist = []
    maxnum_node = 0
    for i in range(num_types):
        if G.num_nodes(G.ntypes[i]) >   maxnum_node:
            maxnum_node = G.num_nodes(G.ntypes[i])
    type_nodelist=th.ones(1,maxnum_node)*-1
    #type_nodetemp   =th.ones(5)*-13
    # pdb.set_trace()
    for i in range(num_types):
        type_nodetemp   = th.ones(1,maxnum_node)*-1
        type_nodetemp[0,:len(G.nodes(G.ntypes[i]))] = G.nodes(G.ntypes[i])
        type_nodelist   = th.cat([type_nodelist,type_nodetemp])
        name_typelist.append(G.ntypes[i])


    for i in To_be_projected:  #G.number_of_nodes()
        print('decomposing node ', i)
        
        ret_paths = gen_paths(G2, source_node=i.item())
        decomposed_paths.append(ret_paths)
        Type_encoding = type_embeding(ret_paths, G2, num_types, type_nodelist, name_typelist)
        #Type_encodings(type_embeding)
 
        Type_encodings.append(Type_encoding)


        if len(ret_paths) == 0:
            short_path_nodes.append(i)
        print('Decomposing_time for this node:',  (time.time()-T1))
        T1 = time.time()
    print('decomposed_paths len ', len(decomposed_paths))
    print('short_path_nodes ', short_path_nodes )

    
    dump_file = 'decomposed_paths_central_rectangle_'+dataset_str
    dump_dict = {'decomposed_paths':decomposed_paths,
                'Type_encodings':Type_encodings}
    pickle.dump(dump_dict, open(dump_file, "wb"))
    load_dict = pickle.load(open(dump_file, "rb"))
    print('decomposed_paths len ', len(load_dict['decomposed_paths'] ))
    #print('short_paths ', load_dict['short_paths']  )



