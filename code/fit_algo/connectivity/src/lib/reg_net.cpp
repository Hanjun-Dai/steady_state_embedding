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

void RegNet::BuildBatchGraph(std::vector<int>& nodes, std::vector< std::vector<int> >& neighbors)
{
    BuildCommon(nodes, neighbors);

    batch_label.Reshape({nodes.size(), (size_t)1});
    for (size_t i = 0; i < nodes.size(); ++i)
        batch_label.data->ptr[i] = scores[nodes[i]];
    m_batch_label.CopyFrom(batch_label);

    inputs["label"] = &m_batch_label;
}
