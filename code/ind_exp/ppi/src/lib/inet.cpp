#include "inet.h"
#include "graph.h"
#include "util/graph_struct.h"
#include <algorithm>
#include "global.h"

INet::INet()
{
}

std::shared_ptr< DTensorVar<mode, Dtype> > INet::GetPred()
{
    auto full_node_embed = model.nondiff_params["node_embed"];
    auto init_w = model.params["init_w"];
    auto W = model.params["W"];
    auto V = model.params["V"];
    auto classifier = model.params["classifier"];    
    fg.AddParam(full_node_embed);
    fg.AddParam(init_w);
    fg.AddParam(W);
    fg.AddParam(V);
    fg.AddParam(classifier);

    auto var_nodemaps = add_const< DTensorVar<mode, int> >(fg, "node_maps", true);

    auto all_input = add_const< DTensorVar<mode, Dtype> >(fg, "node_input", true);
    auto node_input = af< RowSelection<mode, Dtype> >(fg, std::make_pair(all_input, var_nodemaps)); 
    
    auto node_embed = af< RowSelection<mode, Dtype> >(fg, std::make_pair(full_node_embed, var_nodemaps));    
    auto node_feat = af< FullyConnected >(fg, {node_input, init_w});

    auto node_select = add_const< SpTensorVar<mode, Dtype> >(fg, "node_select", true);
    auto v_i = af< MatMul >(fg, {node_select, node_embed});
    auto cur_feat = af< MatMul >(fg, {node_select, node_feat});    

    auto neighbor_gather = add_const< SpTensorVar<mode, Dtype> >(fg, "neighbor_gather", true);

    auto neighbor_embed = af< MatMul >(fg, {neighbor_gather, node_embed});
    auto neighbor_feat = af< MatMul >(fg, {neighbor_gather, node_feat});

    auto concat = af< ConcatCols >(fg, {v_i, cur_feat, neighbor_feat, neighbor_embed});
    concat = af<ReLU>(fg, {concat});
    auto h1 = af< FullyConnected >(fg, {concat, W});
    h1 = af< ReLU >(fg, {h1});

    tilde_v = h1;

    auto pred_linear = af< FullyConnected >(fg, {h1, classifier});

    return pred_linear;
}
