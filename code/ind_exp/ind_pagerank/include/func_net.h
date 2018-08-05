#ifndef FUNC_NET_H
#define FUNC_NET_H

#include "inet.h"

using namespace gnn;

class FuncNet : public INet
{
public:
    FuncNet();

    virtual void BuildNet() override;

    SpTensor<CPU, Dtype> sp_batch_label;
    SpTensor<mode, Dtype> m_sp_batch_label;
    DTensor<mode, Dtype> vec_one;
};

#endif