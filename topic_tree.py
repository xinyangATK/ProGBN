
import numpy as np
from graphviz import Digraph
import heapq
import pickle

#
def node(graph, voc, phi, layer=2, idx=0, num=10):
    max_index=heapq.nlargest(num, range(len(phi[:,idx])), phi[:,idx].take)
    label = ''
    for i in range(len(max_index)):
        label += str(voc[max_index[i]])+'\n'
    graph.node(str(layer)+'_'+str(idx),str(idx) + ' ' + label)

#
def weight_load(file_name):
    with open(file_name,"rb") as f:
        weight=pickle.load(f)
    return weight

# 
def voc_load(file_name):
    voc = []
    with open(file_name,'r') as f:
        lines = f.readlines()
        for idx,line in enumerate(lines):
            voc.append(line.strip())
    return voc


def plot_tree(weight, voc, topic_id=0,threshold=0.05,num=20):
    graph = Digraph()
    idx_2 = id 
    temp = threshold #
    num_layer = len(weight)
    idx_top = [topic_id]
    phi = []
    phi_0 = 1.0
    for idx, each_phi in enumerate(weight):
        if idx == 0:
            phi.append(each_phi)
        else:
            phi.append(np.dot(phi[-1], each_phi))
    for each in range(num_layer-1, 0, -1):
        weight_top = weight[each]
        weight_down = weight[each-1]
        phi_top = phi[each]
        phi_down = phi[each-1]
        flag = True
        for each_top in idx_top:
            node(graph, voc, phi_top, layer=each, idx=each_top, num=num)
            idx = np.where(weight_top[:, each_top] > temp)
            for i in idx[0]:
                node(graph, voc, phi_down, layer=each-1, idx=i,num=num)
                graph.attr('edge', penwidth=str(weight_top[i][topic_id]*10))
                graph.edge('{}_{}'.format(each, each_top), '{}_{}'.format(each-1, i))
            if flag:
                idx_top = idx[0]
                flag = False
            else:
                idx_top = np.append(idx_top, idx[0], 0)

    return graph
    #
    # weight_0 = weight[0]
    # weight_1 = weight[1]
    # weight_2 = weight[2]
    # phi_2 = weight_0.dot(weight_1).dot(weight_2)
    # phi_1 = weight_0.dot(weight_1)
    # phi_0 = weight[0]
    # #
    # node(graph,2,idx_2,num,voc,phi_2)
    # idx_1 = np.where(weight_2[:,idx_2]>temp)
    # for i in idx_1[0]:
        # node(graph,1,i,10,voc,phi_1)
        # graph.attr('edge',penwidth=str(weight_2[i][idx_2]*10))
        # graph.edge('2_'+str(idx_2),'1_'+str(i))
        # idx_0 = np.where(weight_1[:,i]>temp)
        # for j in idx_0[0]:
            # node(graph,0,j,10,voc,phi_0)
            # graph.attr('edge',penwidth=str(weight_1[j][i]*10))
            # graph.edge('1_'+str(i),'0_'+str(j))
    # #graph.view()
    # return graph

