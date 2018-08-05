#ifndef REG_NET_H
#define REG_NET_H

#include "inet.h"

using namespace gnn;

class RegNet : public INet
{
public:
    RegNet();

    virtual void BuildNet() override;
    
    std::shared_ptr< DTensorVar<mode, Dtype> > loss_mae;
};

#endif