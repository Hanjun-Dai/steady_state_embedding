#include "util.h"
#include "config.h"
#include "global.h"
#include <iostream>
#include <sstream>

using namespace gnn;

void Normalize(DTensor<mode, Dtype>& embed_mat, int row_idx)
{
    if (!cfg::norm_embed)
        return;
    auto vec = embed_mat.GetRowRef(row_idx, 1);
    auto norm = vec.Norm2() + 1e-10;
    vec.Scale(1.0 / norm);
}

void Normalize(DTensor<mode, Dtype>& embed_mat)
{
    if (!cfg::norm_embed)
        return;
    for (size_t i = 0; i < embed_mat.rows(); ++i)
        Normalize(embed_mat, i);
}

void LoadIdxes(const char* fname, std::vector<int>& idx_list)
{
    std::ifstream fin(fname);
    idx_list.clear();
    int idx;
    while (fin >> idx)
    {
        idx_list.push_back(idx);
    }
}

int get_dim(const char* fname)
{
    std::ifstream fin(fname);
    std::string st;
    std::getline(fin, st);
    std::stringstream ss(st);
    int l = 0;
    while (ss >> st)
        l++;
    return l;
}

void LoadFeat(const char* fname)
{
    FILE* fid = fopen(fname, "r");
    dense_node_feat.Reshape({(size_t)cfg::num_nodes, (size_t)cfg::dim_feat});
    for (int node = 0; node < cfg::num_nodes; ++node)
        for (int j = 0; j < cfg::dim_feat; ++j)
            fscanf(fid, "%f", dense_node_feat.data->ptr + node * cfg::dim_feat + j);
    dense_m_node_feat.CopyFrom(dense_node_feat);
}