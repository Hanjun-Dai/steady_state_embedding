#include "global.h"

std::vector<Node*> node_list;
ParamSet<mode, Dtype> model;
std::vector<int> train_idxes, test_idxes;
std::set<int> train_set, test_set;
std::vector< std::vector<int> > labels;
std::vector< float > scores;
std::vector<int> topk_list;
SpTensor<CPU, Dtype> sp_node_feat;
SpTensor<mode, Dtype> sp_m_node_feat;
DTensor<CPU, Dtype> dense_node_feat;
DTensor<mode, Dtype> dense_m_node_feat;
