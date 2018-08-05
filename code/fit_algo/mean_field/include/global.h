#ifndef GLOBAL_H
#define GLOBAL_H

#include "config.h"
#include "nn/factor_graph.h"
#include "nn/param_set.h"
#include "tensor/tensor_all.h"
#include "graph.h"
#include <map>
#include <set>

using namespace gnn;

extern std::vector<Node*> node_list;
extern ParamSet<mode, Dtype> model;
extern std::vector<int> train_idxes, test_idxes;
extern std::set<int> train_set, test_set;
extern std::vector< std::vector<int> > labels;
extern std::vector< float > scores;
extern std::vector<int> topk_list;
extern SpTensor<CPU, Dtype> sp_node_feat;
extern SpTensor<mode, Dtype> sp_m_node_feat;
extern DTensor<CPU, Dtype> dense_node_feat;
extern DTensor<mode, Dtype> dense_m_node_feat;

#endif