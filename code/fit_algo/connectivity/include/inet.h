#ifndef I_NET_H
#define I_NET_H

#include <map>
#include <string>
#include <vector>
#include "config.h"
#include "tensor/tensor.h"
#include "nn/nn_all.h"

using namespace gnn;

class INet
{
public:
    INet();
    std::shared_ptr< DTensorVar<mode, Dtype> > GetPred();

    virtual void BuildNet() = 0;
    virtual void BuildBatchGraph(std::vector<int>& nodes, 
                                std::vector< std::vector<int> >& neighbors) = 0;
    void BuildCommon(std::vector<int>& nodes, 
                    std::vector< std::vector<int> >& neighbors);

    SpTensor<CPU, Dtype> mat_select, mat_neighbor;
    SpTensor<mode, Dtype> m_mat_select, m_mat_neighbor;
    DTensor<CPU, Dtype> batch_label;
    DTensor<mode, Dtype> m_batch_label;
    
    std::map< std::string, void* > inputs;
    std::shared_ptr< DTensorVar<mode, Dtype> > loss, tilde_v, pred;
    // std::shared_ptr< DTensorVar<CPU, int> > sampled_y_idx, hitk;
    FactorGraph fg;
};

#endif