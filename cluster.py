import numpy as np
import networkx as nx
from sklearn.cluster import KMeans


def cluster(g, T=None, nc=None):
    if T is None:
        T = int(np.floor(g.number_of_nodes() / g.number_of_edges() * np.log2(g.number_of_nodes()) + 2))
        T += T % 2
    vectors = []
    for n in g.nodes:
        s = np.zeros(g.number_of_nodes())
        s[n] = 1
        for t in range(T):
            st = np.copy(s)
            for i in range(len(s)):
                for b in g.neighbors(i):
                    st[b] += s[i]
            s = np.copy(st)
        vectors.append(s / np.sum(s))
    if nc is not None:
        return KMeans(n_clusters=nc).fit(vectors).labels_
    nc = 2
    prev_score = -1
    d = dict(nx.shortest_path_length(g))
    while True:
        d_in = 0
        n_in = 0
        d_out = 0
        n_out = 0
        labels = KMeans(n_clusters=nc, n_init=16).fit(vectors).labels_
        for i in range(g.number_of_nodes()):
            for j in range(i + 1, g.number_of_nodes()):
                if labels[i] == labels[j]:
                    d_in += d[i][j]
                    n_in += 1
                else:
                    d_out += d[i][j]
                    n_out += 1
        current_score = d_out / n_out / d_in * n_in
        if current_score <= prev_score * 1.035:
            return prev_labels
        prev_score = current_score
        prev_labels = labels
        nc += 1
