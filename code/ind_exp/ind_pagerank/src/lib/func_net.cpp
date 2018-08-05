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
    auto sp_label = add_const< SpTensorVar<mode, Dtype> >(fg, "sp_label", true);

    auto test_pred = pred_linear;// af< MatMul >(fg, {v_i, classifier});
    if (cfg::multi_label)
    {
        pred = af< Sigmoid >(fg, {test_pred});
        auto vec_one = add_const< DTensorVar<mode, Dtype> >(fg, "vec_one", true);
        auto loss_all = af< BinaryLogLoss >(fg, {pred_linear, label}, true);
        loss = af< MatMul >(fg, {loss_all, vec_one}); 
    } else {
        loss = af< CrossEntropy >(fg, {pred_linear, sp_label}, true);
        pred = af< Softmax >(fg, {test_pred});
    }

    loss = af< ReduceMean >(fg, {loss});
}
