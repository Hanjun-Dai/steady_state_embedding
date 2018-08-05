#ifndef GRAPH_H
#define GRAPH_H

#include <map>
#include <string>
#include <vector> 
#include <set>
#include <random>

class Node
{
public:    

    Node(int _idx);

    void AddNeighbor(Node* y);

    int idx;
    std::vector< Node* > adj_list;

    static std::default_random_engine generator;
    static std::uniform_real_distribution<double> distribution;
};

void LoadGraph(const char* data_root);

#endif