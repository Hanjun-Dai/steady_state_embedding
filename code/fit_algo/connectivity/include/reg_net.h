#ifndef REG_NET_H
#define REG_NET_H

#include "inet.h"

using namespace gnn;

class RegNet : public INet
{
public:
    RegNet();

    virtual void BuildNet() override;
    virtual void BuildBatchGraph(std::vector<int>& nodes, 
                                std::vector< std::vector<int> >& neighbors) override;
    std::shared_ptr< DTensorVar<mode, Dtype> > loss_mae;
};

#endif