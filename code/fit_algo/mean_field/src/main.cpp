#include "config.h"
#include "nn/nn_all.h"
#include "tensor/tensor_all.h"
#include <algorithm>
#include "global.h"
#include "func_net.h"
#include "reg_net.h"
#include "util.h"

INet* net;

using namespace gnn;

std::vector<int> node_idxes;
void SampleNodes(std::vector<int>& nodes, std::set<int>* region = nullptr)
{
    if (region == nullptr || cfg::batch_size <= region->size())
        nodes.resize(cfg::batch_size);
    else
        nodes.resize(region->size());
    if (node_idxes.size() != node_list.size())
    {
        node_idxes.resize(node_list.size());
        for (int i = 0; i < cfg::num_nodes; ++i)
            node_idxes[i] = i;
    }
    std::random_shuffle(node_idxes.begin(), node_idxes.end());
    int t = 0;
    for (size_t i = 0; i < node_idxes.size(); ++i)
    {
        if (!region || region->count(node_idxes[i]))
        {
            nodes[t] = node_idxes[i];
            t++;
        }
        if (t == (int)nodes.size())
            break;
    }
}

void GetNeighbors(const std::vector<int>& nodes, std::vector< std::vector< int > >& neighbors)
{
    neighbors.resize(nodes.size());
    for (size_t i = 0; i < nodes.size(); ++i)
    {
        neighbors[i].clear();
        auto* cur_node = node_list[nodes[i]];
        for (auto* node : cur_node->adj_list)
        {
            neighbors[i].push_back(node->idx);
        }
    }
}

void UpdateEmbedding(int node_idx, DTensor<mode, Dtype>& new_embed)
{
    auto& node_embed = model.params["node_embed"]->value;
    auto old_embed = node_embed.GetRowRef(node_idx, 1);
    old_embed.Axpby(0.9, new_embed, 0.1);
    // old_embed.CopyFrom(new_embed);
    Normalize(node_embed, node_idx);
}

void GetTopK(Dtype* pred, int k, std::vector<int>& a)
{
    a.resize(cfg::num_labels);
    for (int i = 0; i < cfg::num_labels; ++i)
        a[i] = i;
    for (int i = 0; i < k; ++i)
        for (int j = i + 1; j < cfg::num_labels; ++j)
            if (pred[ a[i] ] < pred[ a[j] ])
            {
                int tmp = a[i];
                a[i] = a[j];
                a[j] = tmp;
            }
}

void Evaluate(DTensor<CPU, int>& pred, std::vector<int>& cur_test_idxes)
{
    std::vector<float> num_correct(cfg::num_labels), num_true(cfg::num_labels), num_pred(cfg::num_labels);
    for (int i = 0; i < cfg::num_labels; ++i)
    {
        num_correct[i] = num_true[i] = num_pred[i] = 0;
    }

    for (size_t j = 0; j < cur_test_idxes.size(); ++j)
    {
        int idx = cur_test_idxes[j];
        assert(topk_list[idx]);
        if (topk_list[idx] == 0)
            continue;
        for (int i = 0; i < cfg::num_labels; ++i)
        {
            int p = pred.data->ptr[j * cfg::num_labels + i];
            int t = labels[idx][i];
            num_true[i] += t;
            num_pred[i] += p;
            num_correct[i] += (t == p && t);
        }
    }
    float macro_f1 = 0.0;
    float nc = 0.0, np = 0.0, nt = 0.0;
    for (int i = 0; i < cfg::num_labels; ++i)
    {
        nc += num_correct[i];
        np += num_pred[i];
        nt += num_true[i];
        if (num_correct[i] <= 0)
            continue;
        float prec = num_correct[i] / num_pred[i];
        float recall = num_correct[i] / num_true[i];
        float f1 = 2 * prec * recall / (prec + recall);
        macro_f1 += f1;
    }
    macro_f1 /= cfg::num_labels;
    float prec = nc / np, recall = nc / nt;
    float micro_f1 = 2 * prec * recall / (prec + recall);
    std::cerr << "macro: " << macro_f1 << " micro: " << micro_f1 << " accuracy: " << nc / cur_test_idxes.size();
}

DTensor<CPU, int> binary_pred;
void TestLoop(std::vector<int>& cur_test_idxes)
{
    std::vector< std::vector< int > > neighbors;
    GetNeighbors(cur_test_idxes, neighbors);

    net->BuildBatchGraph(cur_test_idxes, neighbors);
    if (cfg::is_regression)
    {
        auto* rnet = dynamic_cast<RegNet*>(net);
        rnet->fg.FeedForward({rnet->loss_mae}, rnet->inputs, Phase::TEST);
        std::cerr << "mae: " << rnet->loss_mae->value.AsScalar() << std::endl;
        return;
    }
    net->fg.FeedForward({net->pred}, net->inputs, Phase::TEST);

    std::vector< int > test_pred, test_label;

    auto& mat_pred = net->pred->value;
    binary_pred.Reshape(mat_pred.shape.dims);
    binary_pred.Zeros();
    std::vector<int> topk;
    for (size_t j = 0; j < cur_test_idxes.size(); ++j)
    {
        int idx = cur_test_idxes[j];
        GetTopK(mat_pred.data->ptr + j * cfg::num_labels, topk_list[idx], topk);
        for (int k = 0; k < topk_list[idx]; ++k)
            binary_pred.data->ptr[j * cfg::num_labels + topk[k]] = 1;
    }
    Evaluate(binary_pred, cur_test_idxes);
    std::cerr << std::endl;
}

void MainLoop()
{
    std::cerr << "main loop" << std::endl;
	//MomentumSGDOptimizer<mode, Dtype> learner(&model, cfg::lr, cfg::momentum, cfg::l2_penalty);
	AdamOptimizer<mode, Dtype> learner(&model, cfg::lr, cfg::l2_penalty);

	int max_iter = (long long)cfg::max_iter;
	int init_iter = cfg::iter;
	if (init_iter > 0)
	{
		std::cerr << fmt::sprintf("loading model for iter=%d", init_iter) << std::endl;
		model.Load(fmt::sprintf("%s/iter_%d.model", cfg::save_dir, init_iter));
	}
	
    std::vector<int> nodes;
    std::vector< std::vector< int > > neighbors;
	for (; cfg::iter <= max_iter; ++cfg::iter)
	{
		if (cfg::iter % cfg::test_interval == 0)
		{	
            TestLoop(test_idxes);
		}
		if (cfg::iter % cfg::save_interval == 0 && cfg::iter != init_iter)
		{			
			printf("saving model for iter=%d\n", cfg::iter);
			model.Save(fmt::sprintf("%s/iter_%d.model", cfg::save_dir, cfg::iter));
		}

        Dtype f_loss = 0.0;
        for (unsigned fi = 0; fi < cfg::f_iter; ++fi)
        {
            SampleNodes(nodes, &train_set);
            for (auto p : nodes)
                assert(train_set.count(p));
            GetNeighbors(nodes, neighbors);
            net->BuildBatchGraph(nodes, neighbors);
            net->fg.FeedForward({net->loss}, net->inputs, Phase::TRAIN);
            net->fg.BackPropagate({net->loss});
            learner.Update();
            f_loss += net->loss->value.AsScalar();
        }

        for (unsigned vi = 0; vi < cfg::v_iter; ++vi)
        {
            DTensor<mode, Dtype> est;
            SampleNodes(nodes);

            for (int i = 0; i < cfg::n_sample; ++i)
            {
                GetNeighbors(nodes, neighbors);
                net->BuildBatchGraph(nodes, neighbors);
                net->fg.FeedForward({net->tilde_v}, net->inputs, Phase::TEST);
                auto& cur_v = net->tilde_v->value;
                if (i)
                    est.Axpy(1.0, cur_v);
                else
                    est.CopyFrom(cur_v);
            }
            est.Scale(1.0 / cfg::n_sample);
            for (size_t i = 0; i < nodes.size(); ++i)
            {
                auto new_embed = est.GetRowRef(i, 1);
                UpdateEmbedding(nodes[i], new_embed);
            }
        }

    	if (cfg::iter % cfg::report_interval == 0)
		{	
            std::cerr << "iter: " << cfg::iter;
            std::cerr << "\tloss: " << f_loss / cfg::f_iter << std::endl;
		}
	}
}

std::shared_ptr< DTensorVar<mode, Dtype> > GetVar(std::string name, int x, int y)
{
    auto param = add_diff<DTensorVar>(model, name, {(size_t)x, (size_t)y});
    auto ss = sqrt(6.0 / (x + y));
    param->value.SetRandU(-ss, ss);
    return param;
}

void InitParams()
{
    GetVar("node_embed", cfg::num_nodes, cfg::n_embed);
    GetVar("init_w", cfg::dim_feat, cfg::n_embed);
    GetVar("W", cfg::n_embed * 2 + 1, cfg::n_embed);
    GetVar("V", cfg::n_embed + 1, cfg::n_embed);
    GetVar("classifier", cfg::n_embed, cfg::is_regression ? 1 : cfg::num_labels);
}

int main(const int argc, const char** argv)
{
	srand(1);
	cfg::LoadParams(argc, argv);
	GpuHandle::Init(cfg::dev_id, 1);

    std::cerr << "loading graph" << std::endl;
    LoadGraph(cfg::data_root);

    std::cerr << "loading idxes" << std::endl;
    LoadIdxes(cfg::f_train_idx, train_idxes, train_set);
    LoadIdxes(cfg::f_test_idx, test_idxes, test_set);

    std::cerr << "loading features" << std::endl;
    LoadFeat(cfg::f_feat);
    std::cerr << "everyting loaded" << std::endl;

    InitParams();

    if (cfg::is_regression)
    {
        net = new RegNet();
    } else
        net = new FuncNet();
    net->BuildNet();
    MainLoop();

    auto& node_embed = model.nondiff_params["node_embed"]->value;
    for (int i = 0; i < 10; ++i)
    {
        for (int j = 0; j < 10; ++j)
            std::cerr << node_embed.data->ptr[i * node_embed.cols() + j] << " ";
        std::cerr << std::endl;
    }
	GpuHandle::Destroy();
	return 0;	
}
