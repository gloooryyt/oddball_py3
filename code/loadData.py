'''
Data load and pre process for oddball

@author:
Tao Yu (gloooryyt@gmail.com)

'''

import numpy as np
import networkx as nx

#load data, a weighted undirected graph
def load_data(path):
    data = np.loadtxt(path).astype('int32')
    G = nx.Graph()
    for ite in data:
        G.add_edge(ite[0], ite[1], weight=ite[2])
    return G


def get_feature(G):
    #feature dictionary which format is {node i's id:Ni, Ei, Wi, λw,i}
    featureDict = {}
    nodelist = list(G.nodes)
    for ite in nodelist:
        featureDict[ite] = []
        #the number of node i's neighbor
        Ni = G.degree(ite)
        featureDict[ite].append(Ni)
        #the set of node i's neighbor
        iNeighbor = list(G.neighbors(ite))
        #the number of edges in egonet i
        Ei = 0
        #sum of weights in egonet i
        Wi = 0
        #the principal eigenvalue(the maximum eigenvalue with abs) of egonet i's weighted adjacency matrix
        Lambda_w_i = 0
        Ei += Ni
        egonet = nx.Graph()
        for nei in iNeighbor:
            Wi += G[nei][ite]['weight']
            egonet.add_edge(ite, nei, weight=G[nei][ite]['weight'])
        iNeighborLen = len(iNeighbor)
        for it1 in range(iNeighborLen):
            for it2 in range(it1+1, iNeighborLen):
                #if it1 in it2's neighbor list
                if iNeighbor[it1] in list(G.neighbors(iNeighbor[it2])):
                    Ei += 1
                    Wi += G[iNeighbor[it1]][iNeighbor[it2]]['weight']
                    egonet.add_edge(iNeighbor[it1], iNeighbor[it2], weight=G[iNeighbor[it1]][iNeighbor[it2]]['weight'])
        egonet_adjacency_matrix = nx.adjacency_matrix(egonet).todense()
        eigenvalue, eigenvector = np.linalg.eig(egonet_adjacency_matrix)
        eigenvalue.sort()
        Lambda_w_i = max(abs(eigenvalue[0]), abs(eigenvalue[-1]))
        featureDict[ite].append(Ei)
        featureDict[ite].append(Wi)
        featureDict[ite].append(Lambda_w_i)
    return featureDict