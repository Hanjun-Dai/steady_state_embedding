#include <cstdio>
#include "config.h"
#include "graph.h"
#include <random>
#include <algorithm>
#include <iostream>
#include <fstream>
#include "tensor/tensor_all.h"

using namespace gnn;

void LoadGraph()
{
    FILE* f_meta = fopen(fmt::sprintf("%s/meta.txt", cfg::out_folder).c_str(), "r");
    int num_nodes;
    fscanf(f_meta, "%d", &num_nodes);
    fclose(f_meta);

    FILE* f_adj = fopen(fmt::sprintf("%s/adj_list.txt", cfg::out_folder).c_str(), "r");

    node_list.resize(num_nodes);
    for (int i = 0; i < num_nodes; ++i)
        node_list[i] = new Node(i);

    cfg::num_dir_edges = 0;
    for (int i = 0; i < num_nodes; ++i)
    {
        int m;
        fscanf(f_adj, "%d", &m);
        for (int j = 0; j < m; ++j)
        {
            int v;
            fscanf(f_adj, "%d", &v);
            assert(v >= 0 && v < num_nodes);
            node_list[i]->AddNeighbor(node_list[v]);
        }
        cfg::num_dir_edges += m;
    }
    fclose(f_adj);
    
    cfg::num_nodes = num_nodes;
}

SpTensor<CPU, Dtype> w;
DTensor<CPU, Dtype> x_last, x, x_true;

void BuildSpMat()
{
    w.Reshape({(size_t)cfg::num_nodes, (size_t)cfg::num_nodes});
    w.ResizeSp(cfg::num_dir_edges, cfg::num_nodes + 1);

    int nnz = 0;
    for (int i = 0; i < cfg::num_nodes; ++i)
    {
        w.data->row_ptr[i] = nnz;

        for (auto* p : node_list[i]->adj_list)
        {
            w.data->col_idx[nnz] = p->idx;
            w.data->val[nnz] = (double)1.0 / (double)node_list[p->idx]->adj_list.size();
            nnz++;
        }
    }
    
    assert(nnz == cfg::num_dir_edges);
    w.data->row_ptr[cfg::num_nodes] = nnz;
}

void PageRank()
{    
    x.Reshape({(size_t)cfg::num_nodes, (size_t)1});
    x.Fill((double)1.0 / (double)cfg::num_nodes);

    if (cfg::is_test && cfg::score_file)
    {
        x_true.Reshape({(size_t)cfg::num_nodes, (size_t)1});
        FILE* fid = fopen(cfg::score_file, "r");
        for (int i = 0; i < cfg::num_nodes; ++i)
            fscanf(fid, "%lf", x_true.data->ptr + i);
        fclose(fid);
    }

    bool finished = false;
    for (int i = 0; i < cfg::max_iter; ++i)
    {
        std::cerr << "iter " << i << std::endl;
        x_last.CopyFrom(x);

        x.MM(w, x_last, Trans::N, Trans::N, cfg::alpha, 0.0);

        x.Add((1.0 - cfg::alpha) / (double)cfg::num_nodes);

        x_last.Axpy(-1.0, x);
        double err = x_last.ASum();

        if (cfg::is_test)
        {
            x_last.CopyFrom(x_true);
            x_last.Axpy(-cfg::num_nodes, x);
            double err = x_last.ASum();
            std::cerr << "mae: " << err / cfg::num_nodes << std::endl;
        }
        // if (err < cfg::tol * (double)cfg::num_nodes)
        if (err < cfg::tol)
        {
            finished = true;
            break;
        }            

        auto s = x.ASum();
        x.Scale(1.0 / s);                
    }

    if (!finished)
        std::cerr << "warning! not finished within " << cfg::max_iter << " iterations" << std::endl;
}

void PrintPR()
{
    FILE* fout = fopen(fmt::sprintf("%s/pr-%.2f.txt", cfg::out_folder, cfg::alpha).c_str(), "w");

    for (int i = 0; i < cfg::num_nodes; ++i)
        fprintf(fout, "%.12f\n", x.data->ptr[i] * cfg::num_nodes);

    fclose(fout);

    std::vector<int> idxes(cfg::num_nodes);

    for (int i = 0; i < cfg::num_nodes; ++i)
        idxes[i] = i;
    std::random_shuffle(idxes.begin(), idxes.end());

    int t = cfg::num_nodes * 0.9;

    FILE* f_test = fopen(fmt::sprintf("%s/test_idx.txt", cfg::out_folder).c_str(), "w");
    for (int i = t; i < cfg::num_nodes; ++i)
        fprintf(f_test, "%d\n", idxes[i]);
    fclose(f_test);

    FILE* f_all = fopen(fmt::sprintf("%s/all_idx.txt", cfg::out_folder).c_str(), "w");
    for (int i = 0; i < cfg::num_nodes; ++i)
        fprintf(f_all, "%d\n", i);
    fclose(f_all);

    for (int p = 1; p <= 9; ++p)
    {
        double frac = p / 10.0;
        int num_train = cfg::num_nodes * frac;

        FILE* f_idx = fopen(fmt::sprintf("%s/train_idx-%.1f.txt", cfg::out_folder, frac).c_str(), "w");
        for (int i = 0; i < num_train; ++i)
            fprintf(f_idx, "%d\n", idxes[i]);
        fclose(f_idx);

        frac /= 10.0;
        num_train = cfg::num_nodes * frac;
        f_idx = fopen(fmt::sprintf("%s/train_idx-%.2f.txt", cfg::out_folder, frac).c_str(), "w");
        for (int i = 0; i < num_train; ++i)
            fprintf(f_idx, "%d\n", idxes[i]);
        fclose(f_idx);
    }
}

int main(const int argc, const char** argv)
{    
    cfg::LoadParams(argc, argv);
    GpuHandle::Init(0, 1);

    LoadGraph();

    BuildSpMat();

    PageRank();

    if (!cfg::is_test)
        PrintPR();

    GpuHandle::Destroy();
}
