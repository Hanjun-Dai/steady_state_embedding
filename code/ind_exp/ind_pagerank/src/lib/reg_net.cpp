#include "reg_net.h"
#include "graph.h"
#include "util/graph_struct.h"
#include <algorithm>
#include "global.h"

RegNet::RegNet() : INet()
{
}

void RegNet::BuildNet()
{
    pred = GetPred();
    auto label = add_const< DTensorVar<mode, Dtype> >(fg, "label", true);
    
    loss = af< SquareError >(fg, {pred, label});
    loss = af< ReduceMean >(fg, {loss});

    loss_mae = af< AbsError >(fg, {pred, label}, PropErr::N);
    loss_mae = af< ReduceMean >(fg, {loss_mae});
}
