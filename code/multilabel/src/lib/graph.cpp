#include "graph.h"
#include "config.h"
#include "util.h"
#include <string>
#include <vector>
#include <queue>
#include <iostream>
#include <sstream>
#include "global.h"

Node::Node(int _idx) : idx(_idx)
{
    adj_list.clear();    
}

void Node::AddNeighbor(Node* y)
{
    adj_list.push_back(y);
}

std::default_random_engine Node::generator;
std::uniform_real_distribution<double> Node::distribution(0.0, 1.0);

void LoadGraph(const char* data_root)
{
    FILE* f_meta = fopen(fmt::sprintf("%s/meta.txt", data_root).c_str(), "r");
    int num_nodes, num_labels, dim_feat;
    fscanf(f_meta, "%d %d", &num_nodes, &num_labels); 
    if (cfg::has_feat)
    {
        fscanf(f_meta, "%d", &dim_feat);
        std::ifstream fin(cfg::f_feat);
        std::string st;
        std::getline(fin, st);
        cfg::sparse_feat = false;
        for (auto c : st)
            if (c == ':')
            {
                cfg::sparse_feat = true;
                break;
            }
        if (!cfg::sparse_feat)
            dim_feat = get_dim(cfg::f_feat);
    } else {
        cfg::sparse_feat = true;
        dim_feat = num_nodes;
    }
        
    fclose(f_meta);

    if (cfg::f_label)
    {
        assert(!cfg::f_score);
        num_labels = get_dim(cfg::f_label);
    } else {
        assert(cfg::f_score);
        num_labels = get_dim(cfg::f_score);
    }

    FILE* f_adj = fopen(fmt::sprintf("%s/adj_list.txt", data_root).c_str(), "r");
    node_list.resize(num_nodes);
    for (int i = 0; i < num_nodes; ++i)
        node_list[i] = new Node(i);
    
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
    }
    fclose(f_adj);

    if (cfg::is_regression)
    {
        FILE* f_label = fopen(cfg::f_score, "r");
        scores.resize(num_nodes);
        for (int i = 0; i < num_nodes; ++i)
        {
            fscanf(f_label, "%f", &scores[i]);
        }
            
        fclose(f_label);
    } else {
        FILE* f_label = fopen(cfg::f_label, "r");
        labels.resize(num_nodes);
        topk_list.resize(num_nodes);
        for (int i = 0; i < num_nodes; ++i)
        {
            labels[i].resize(num_labels);
            int num = 0;
            for (int j = 0; j < num_labels; ++j)
            {
                fscanf(f_label, "%d", &labels[i][j]);
                if (labels[i][j])
                    num++;
            }
            topk_list[i] = num;
        }
        fclose(f_label);
    }
    cfg::num_nodes = num_nodes;
    cfg::num_labels = num_labels;
    cfg::dim_feat = dim_feat;

    std::cerr << "num_nodes: " << cfg::num_nodes << std::endl;
    std::cerr << "num_labels: " << cfg::num_labels << std::endl;
    std::cerr << "dim_feat: " << cfg::dim_feat << std::endl; 
    std::cerr << "is sparse feat: " << cfg::sparse_feat << std::endl; 
}