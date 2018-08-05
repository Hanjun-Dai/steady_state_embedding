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
};

extern std::vector<Node*> node_list;
#endif