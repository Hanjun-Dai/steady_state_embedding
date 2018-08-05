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

    std::shared_ptr< DTensorVar<mode, Dtype> > loss, tilde_v, pred;
    FactorGraph fg;
};

#endif