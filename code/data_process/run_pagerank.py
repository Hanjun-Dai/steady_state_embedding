import cPickle as cp
import numpy as np
import networkx as nx
import sys

if __name__ == '__main__':
    dataset = sys.argv[1]
    damp = float(sys.argv[2])

    with open('../../dropbox/gcn_data/%s/ind.%s.graph' % (dataset, dataset), 'rb') as f:
        d = cp.load(f)
        g = nx.from_dict_of_lists(d)
        print 'connected?', nx.is_connected(g)
        scores = nx.pagerank(g)

        with open('../../dropbox/gcn_data/%s/pr-%s.txt' % (dataset, sys.argv[2]), 'w') as f:
            for i in range(len(g)):
                f.write('%.12f\n' % (scores[i] * len(g)))