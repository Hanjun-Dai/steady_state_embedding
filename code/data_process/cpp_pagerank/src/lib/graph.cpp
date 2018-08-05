#include "graph.h"
#include "config.h"
#include <string>
#include <vector>
#include <queue>

std::vector<Node*> node_list;

Node::Node(int _idx) : idx(_idx)
{
    adj_list.clear();    
}

void Node::AddNeighbor(Node* y)
{
    adj_list.push_back(y);
}
