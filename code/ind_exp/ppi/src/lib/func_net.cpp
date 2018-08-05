#include "func_net.h"
#include "graph.h"
#include "util/graph_struct.h"
#include <algorithm>
#include "global.h"

FuncNet::FuncNet() : INet()
{
}

void FuncNet::BuildNet()
{
    auto pred_linear = GetPred();
    auto label = add_const< DTensorVar<mode, Dtype> >(fg, "label", true);

    auto test_pred = pred_linear;// af< MatMul >(fg, {v_i, classifier});
    assert(cfg::multi_label);

    pred = af< Sigmoid >(fg, {test_pred});
    auto vec_one = add_const< DTensorVar<mode, Dtype> >(fg, "vec_one", true);
    auto loss_all = af< BinaryLogLoss >(fg, {pred_linear, label}, true);
    loss = af< MatMul >(fg, {loss_all, vec_one});

    loss = af< ReduceMean >(fg, {loss});
}