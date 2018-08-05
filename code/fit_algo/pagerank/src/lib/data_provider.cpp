#include "data_provider.h"
#include "global.h"
#include <algorithm>

void DataProvider::Init(std::vector<int>& idxes, unsigned _batch_size)
{
    sample_idxes = idxes;
    batch_size = _batch_size;
    is_ready = false;

    if (batch_size < sample_idxes.size())
        std::random_shuffle(sample_idxes.begin(), sample_idxes.end());
    else 
        batch_size = sample_idxes.size();

    pos = 0;
}

DataProvider::DataProvider(std::vector<int>& idxes, unsigned _batch_size)
{
    Init(idxes, _batch_size);
}

DataProvider::DataProvider(int num_nodes, unsigned _batch_size)
{
    std::vector<int> idxes;
    for (int i = 0; i < num_nodes; ++i)
        idxes.push_back(i);
    Init(idxes, _batch_size);
}

void DataProvider::SampleNodes(std::vector<int>& nodes)
{
    nodes.resize(batch_size);
    
    if (pos + batch_size > sample_idxes.size())
    {
        std::random_shuffle(sample_idxes.begin(), sample_idxes.end());
        pos = 0;
    }

    for (size_t i = pos; i < pos + batch_size; ++i)
        nodes[i - pos] = sample_idxes[i];
    pos += batch_size;
}

void DataProvider::GetNeighbors(const std::vector<int>& nodes, std::vector< std::vector< int > >& neighbors)
{
    neighbors.resize(nodes.size());
    for (size_t i = 0; i < nodes.size(); ++i)
    {
        neighbors[i].clear();
        auto* cur_node = node_list[nodes[i]];
        for (auto* node : cur_node->adj_list)
        {
            neighbors[i].push_back(node->idx);
        }
    }
}

std::map< std::string, void* > DataProvider::GetNextBatch()
{
    std::map< std::string, void* > inputs;
    if (!is_ready || batch_size != sample_idxes.size())
    {
        SampleNodes(batch_nodes);
        GetNeighbors(batch_nodes, batch_neighbors);
        
        node_dict.clear();
        auto& nodes = batch_nodes;
        auto& neighbors = batch_neighbors;

        for (size_t i = 0; i < nodes.size(); ++i)
        {
            int cur_node = nodes[i];
            if (!node_dict.count(cur_node))
            {
                int cur_idx = node_dict.size();
                node_dict[cur_node] = cur_idx;
            }
            for (size_t j = 0; j < neighbors[i].size(); ++j)
            {
                int adj = neighbors[i][j];
                if (!node_dict.count(adj))
                {
                    int cur_idx = node_dict.size();
                    node_dict[adj] = cur_idx;
                }
            }
        }

        node_maps.Reshape({node_dict.size()});
        for (auto& p : node_dict)
        {
            node_maps.data->ptr[p.second] = p.first;
        }

        int nnz_node = nodes.size();
        int nnz_neighbor = 0;
        for (size_t i = 0; i < neighbors.size(); ++i)
            nnz_neighbor += neighbors[i].size();
    
        mat_select.Reshape({nodes.size(), node_dict.size()});    
        mat_neighbor.Reshape({nodes.size(), node_dict.size()});
        mat_select.ResizeSp(nnz_node, nodes.size() + 1);
        mat_neighbor.ResizeSp(nnz_neighbor, nodes.size() + 1);

        nnz_neighbor = 0;
        nnz_node = 0;
        for (size_t i = 0; i < neighbors.size(); ++i)
        {
            mat_select.data->row_ptr[i] = nnz_node;
            mat_select.data->col_idx[i] = node_dict[nodes[i]];
            mat_select.data->val[i] = 1.0;
            nnz_node++;

            mat_neighbor.data->row_ptr[i] = nnz_neighbor;
            for (size_t j = 0; j < neighbors[i].size(); ++j)
            {
                mat_neighbor.data->col_idx[nnz_neighbor] = node_dict[neighbors[i][j]];
                if (cfg::avg)
                    mat_neighbor.data->val[nnz_neighbor] = 1.0 / neighbors[i].size();
                else
                    mat_neighbor.data->val[nnz_neighbor] = 1.0 / node_list[neighbors[i][j]]->adj_list.size();
                nnz_neighbor++;
            }
        }

        assert(nnz_node == mat_select.data->nnz);
        assert(nnz_neighbor == mat_neighbor.data->nnz);
        mat_select.data->row_ptr[nodes.size()] = nnz_node;
        mat_neighbor.data->row_ptr[nodes.size()] = nnz_neighbor;

        if (cfg::is_regression)
        {
            batch_label.Reshape({nodes.size(), (size_t)1});
            for (size_t i = 0; i < nodes.size(); ++i)
                batch_label.data->ptr[i] = scores[nodes[i]];            
        }
    }
    inputs["node_select"] = &mat_select;
    inputs["neighbor_gather"] = &mat_neighbor;
    inputs["node_maps"] = &node_maps;
    inputs["label"] = &batch_label;

    is_ready = true;
    return inputs;
}