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

// void FuncNet::BuildBatchGraph(std::vector<int>& nodes, std::vector< std::vector<int> >& neighbors)
// {
    // BuildCommon(nodes, neighbors);
    
    // batch_label.Reshape({nodes.size(), (size_t)cfg::num_labels});
    // sp_batch_label.Reshape({nodes.size(), (size_t)cfg::num_labels});
    // sp_batch_label.ResizeSp(nodes.size(), nodes.size() + 1);

    // for (size_t i = 0; i < neighbors.size(); ++i)
    // {
    //     sp_batch_label.data->row_ptr[i] = i; 
    //     sp_batch_label.data->val[i] = 1.0;
    //     for (int j = 0; j < cfg::num_labels; ++j)
    //     {
    //         batch_label.data->ptr[i * batch_label.cols() + j] = labels[nodes[i]][j];
    //         if (labels[nodes[i]][j])
    //             sp_batch_label.data->col_idx[i] = j;
    //     }
    // }
    // sp_batch_label.data->row_ptr[nodes.size()] = nodes.size();
    // m_sp_batch_label.CopyFrom(sp_batch_label);
    // m_batch_label.CopyFrom(batch_label);

    // vec_one.Reshape({(size_t)cfg::num_labels, (size_t)1});
    // vec_one.Fill(1.0);
    // if (cfg::multi_label)
    //     inputs["vec_one"] = &vec_one;
    // inputs["label"] = &m_batch_label;
    // inputs["sp_label"] = &m_sp_batch_label;
// }
