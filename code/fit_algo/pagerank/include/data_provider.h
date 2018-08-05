#ifndef DATA_PROVIDER_H
#define DATA_PROVIDER_H

#include <map>
#include <string>
#include <vector>
#include "config.h"
#include "tensor/tensor.h"
#include "nn/nn_all.h"

using namespace gnn;

class DataProvider
{
public:

    DataProvider(std::vector<int>& idxes, unsigned _batch_size);
    DataProvider(int num_nodes, unsigned _batch_size);

    std::map< std::string, void* > GetNextBatch();

    void SampleNodes(std::vector<int>& nodes);
    void GetNeighbors(const std::vector<int>& nodes, std::vector< std::vector< int > >& neighbors);

    void Init(std::vector<int>& idxes, unsigned _batch_size);

    std::map<int, int> node_dict;
    std::vector<int> sample_idxes, batch_nodes;
    std::vector< std::vector< int > > batch_neighbors;
    unsigned batch_size;
    bool is_ready;

    DTensor<CPU, int> node_maps;
    SpTensor<CPU, Dtype> mat_select, mat_neighbor;
    DTensor<CPU, Dtype> batch_label;
    unsigned pos;
};

#endif