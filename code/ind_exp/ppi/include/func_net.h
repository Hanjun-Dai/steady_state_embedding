#ifndef FUNC_NET_H
#define FUNC_NET_H

#include "inet.h"

using namespace gnn;

class FuncNet : public INet
{
public:
    FuncNet();

    virtual void BuildNet() override;

};

#endif