#include "util.h"
#include "config.h"
#include "global.h"
#include <iostream>
#include <sstream>

using namespace gnn;

void Normalize(DTensor<mode, Dtype>& embed_mat, int row_idx)
{
    return;
    auto vec = embed_mat.GetRowRef(row_idx, 1);
    auto norm = vec.Norm2() + 1e-10;
    vec.Scale(1.0 / norm);
}

void Normalize(DTensor<mode, Dtype>& embed_mat)
{
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
    if (!cfg::sparse_feat)
    {
        FILE* fid = fopen(fname, "r");
        dense_node_feat.Reshape({(size_t)cfg::num_nodes, (size_t)cfg::dim_feat});
        for (int node = 0; node < cfg::num_nodes; ++node)
            for (int j = 0; j < cfg::dim_feat; ++j)
                fscanf(fid, "%f", dense_node_feat.data->ptr + node * cfg::dim_feat + j);
        dense_m_node_feat.CopyFrom(dense_node_feat);
        return;
    }
    if (cfg::has_feat)
    {
        FILE* fid = fopen(fname, "r");
        std::vector< std::vector< std::pair<int, Dtype> > > sp_feat_list;
        sp_feat_list.resize(cfg::num_nodes);
        std::vector< Dtype > row_sum(cfg::num_nodes);
        for (int node = 0; node < cfg::num_nodes; ++node)
        {
            int num;
            fscanf(fid, "%d", &num);
            auto& cur_feat_list = sp_feat_list[node];
            cur_feat_list.resize(num);
            row_sum[node] = 0;
            for (int i = 0; i < num; ++i)
            {
                int idx;
                float val;
                fscanf(fid, "%d:%f", &idx, &val);
                cur_feat_list[i] = std::make_pair(idx, (Dtype)val);
                row_sum[node] += val;
            }
            if (strcmp(cfg::data_name, "pubmed"))
                row_sum[node] = 1.0;
        }
        fclose(fid);
        int nnz = 0;
        for (int i = 0; i < cfg::num_nodes; ++i)
            nnz += sp_feat_list[i].size();
        
        sp_node_feat.Reshape({(size_t)cfg::num_nodes, (size_t)cfg::dim_feat});
        sp_node_feat.ResizeSp(nnz, cfg::num_nodes + 1);
        nnz = 0;
        for (int i = 0; i < cfg::num_nodes; ++i)
        {
            sp_node_feat.data->row_ptr[i] = nnz;
            auto& cur_feat_list = sp_feat_list[i];
            for (size_t j = 0; j < cur_feat_list.size(); ++j)
            {
                sp_node_feat.data->col_idx[nnz] = cur_feat_list[j].first;
                sp_node_feat.data->val[nnz] = cur_feat_list[j].second / row_sum[i];
                nnz += 1;
            }
        }
        assert(nnz == sp_node_feat.data->nnz);
        sp_node_feat.data->row_ptr[cfg::num_nodes] = nnz;
    } else {
        return;
        sp_node_feat.Reshape({(size_t)cfg::num_nodes, (size_t)cfg::dim_feat});
        sp_node_feat.ResizeSp(cfg::num_nodes, cfg::num_nodes + 1);
        assert(cfg::dim_feat == cfg::num_nodes);
        for (int i = 0; i < cfg::num_nodes; ++i)
        {
            sp_node_feat.data->row_ptr[i] = i;
            sp_node_feat.data->col_idx[i] = i;
            sp_node_feat.data->val[i] = 1.0;
        }
        sp_node_feat.data->row_ptr[cfg::num_nodes] = cfg::num_nodes;
    }
}
