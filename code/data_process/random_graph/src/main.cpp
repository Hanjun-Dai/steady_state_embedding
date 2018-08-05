#include <cstdio>
#include "config.h"
#include "graph.h"
#include <random>
#include <chrono>

void BuildBA()
{
    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_real_distribution<Dtype> distribution(0.0, 1.0);

    std::vector<int> flatten_nodes;
    assert(cfg::n > cfg::m && cfg::m);
    node_list.resize(cfg::n);
    for (int i = 0; i < cfg::n; ++i)
        node_list[i] = new Node(i);

    for (int i = 0; i < cfg::m; ++i)
    {
        node_list[cfg::m]->AddNeighbor(node_list[i]);
        node_list[i]->AddNeighbor(node_list[cfg::m]);

        flatten_nodes.push_back(cfg::m);
        flatten_nodes.push_back(i);
    }

    for (int i = cfg::m; i < cfg::n; ++i)
    {
        std::set<int> sampled;
        while ((int)sampled.size() < cfg::m)
        {
            int k = distribution(generator) * flatten_nodes.size();
            k = flatten_nodes[k];
            if (!sampled.count(k))
            {
                sampled.insert(k);
                node_list[i]->AddNeighbor(node_list[k]);
                node_list[k]->AddNeighbor(node_list[i]);

                flatten_nodes.push_back(k);
                flatten_nodes.push_back(i);
            }
        }
    }
}

void BuildConn()
{
    node_list.resize(cfg::n);
    for (int i = 0; i < cfg::n; ++i)
        node_list[i] = new Node(i);

    std::default_random_engine generator;
    std::uniform_real_distribution<Dtype> distribution(0.0, 1.0);    

    for (int i = 1; i < cfg::n; ++i)
    {
        int k = distribution(generator) * i;
        node_list[i]->AddNeighbor(node_list[k]);
        node_list[k]->AddNeighbor(node_list[i]);
    }

    
}

void PrintGraph()
{
    FILE* f_adj = fopen(fmt::sprintf("%s/adj_list.txt", cfg::out_folder).c_str(), "w");
    FILE* f_edge = fopen(fmt::sprintf("%s/edge_list.txt", cfg::out_folder).c_str(), "w");
    for (int i = 0; i < cfg::n; ++i)
    {
        int num_adj = node_list[i]->adj_list.size();
        fprintf(f_adj, "%d", num_adj);
        for (auto p : node_list[i]->adj_list)
        {
            fprintf(f_adj, " %d", p->idx);
            fprintf(f_edge, "%d %d\n", i, p->idx);
        }            
        fprintf(f_adj, "\n");
    }
    fclose(f_adj);
    fclose(f_edge);
}

int main(const int argc, const char** argv)
{
    cfg::LoadParams(argc, argv);

    FILE* f_meta = fopen(fmt::sprintf("%s/meta.txt", cfg::out_folder).c_str(), "w");
    fprintf(f_meta, "%d 1\n", cfg::n);
    fclose(f_meta);

    if (!strcmp("ba", cfg::g_type))
        BuildBA();
    else if (!strcmp("conn", cfg::g_type))
    {

    }

    PrintGraph();
}