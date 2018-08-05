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
    inputs.clear();
    inputs["node_input"] = &sp_m_node_feat;

    auto init_w = model.params["init_w"];
    auto node_embed = model.params["node_embed"];
    auto W = model.params["W"];
    auto V = model.params["V"];
    auto classifier = model.params["classifier"];
    fg.AddParam(init_w);
    fg.AddParam(node_embed);
    fg.AddParam(W);
    fg.AddParam(V);
    fg.AddParam(classifier);

    auto node_input = add_const< SpTensorVar<mode, Dtype> >(fg, "node_input", true);
    auto node_feat = af< MatMul >(fg, {node_input, init_w});

    auto node_select = add_const< SpTensorVar<mode, Dtype> >(fg, "node_select", true);
    auto v_i = af< MatMul >(fg, {node_select, node_embed});

    auto neighbor_gather = add_const< SpTensorVar<mode, Dtype> >(fg, "neighbor_gather", true);

    auto neighbor_embed = af< MatMul >(fg, {neighbor_gather, node_embed});
    auto neighbor_feat = af< MatMul >(fg, {neighbor_gather, node_feat});

    auto h1 = neighbor_embed;
    h1 = af< Tanh >(fg, {h1});
    
    tilde_v = h1;
    tilde_v = af< L2ColNorm >(fg, {h1}, 1);

    auto pred_linear = af< FullyConnected >(fg, {tilde_v, classifier});

    return pred_linear;
}

void INet::BuildCommon(std::vector<int>& nodes, std::vector< std::vector<int> >& neighbors)
{
    int nnz_node = nodes.size();
    int nnz_neighbor = 0;
    for (size_t i = 0; i < neighbors.size(); ++i)
        nnz_neighbor += neighbors[i].size();

    mat_select.Reshape({nodes.size(), (size_t)cfg::num_nodes});
    mat_neighbor.Reshape({nodes.size(), (size_t)cfg::num_nodes});
    mat_select.ResizeSp(nnz_node, nodes.size() + 1);
    mat_neighbor.ResizeSp(nnz_neighbor, nodes.size() + 1);

    nnz_neighbor = 0;
    nnz_node = 0;
    for (size_t i = 0; i < neighbors.size(); ++i)
    {
        mat_select.data->row_ptr[i] = nnz_node;
        mat_select.data->col_idx[i] = nodes[i];
        mat_select.data->val[i] = 1.0;
        nnz_node++;

        mat_neighbor.data->row_ptr[i] = nnz_neighbor;
        for (size_t j = 0; j < neighbors[i].size(); ++j)
        {
            mat_neighbor.data->col_idx[nnz_neighbor] = neighbors[i][j];
            if (cfg::avg)
                mat_neighbor.data->val[nnz_neighbor] = 1.0 / neighbors[i].size();
            else
                mat_neighbor.data->val[nnz_neighbor] = 1.0 / node_list[neighbors[i][j]]->adj_list.size();
            nnz_neighbor++;
        }
    }

    assert(nnz_node == mat_select.data->nnz);
    assert(nnz_neighbor == mat_neighbor.data->nnz);
    mat_select.data->row_ptr[nodes.size()] = nnz_node;
    mat_neighbor.data->row_ptr[nodes.size()] = nnz_neighbor;
    m_mat_select.CopyFrom(mat_select);
    m_mat_neighbor.CopyFrom(mat_neighbor);

    inputs["node_select"] = &m_mat_select;
    inputs["neighbor_gather"] = &m_mat_neighbor;
}
