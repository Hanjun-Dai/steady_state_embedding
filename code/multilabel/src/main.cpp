#include "config.h"
#include "nn/nn_all.h"
#include "tensor/tensor_all.h"
#include <algorithm>
#include "global.h"
#include "func_net.h"
#include "data_provider.h"
#include "reg_net.h"
#include "util.h"
#include <ctime>
#include <chrono>
INet* net;

using namespace gnn;

DataProvider *f_train_stream, *v_train_stream, *test_stream;

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
        if (topk_list[idx] == 0)
            std::cerr << idx << std::endl;
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

void UpdateEmbedding(int node_idx, DTensor<mode, Dtype>& new_embed)
{
    auto& node_embed = model.params["node_embed"]->value;
    auto old_embed = node_embed.GetRowRef(node_idx, 1);
    old_embed.Axpby(0.9, new_embed, 0.1);
    // old_embed.CopyFrom(new_embed);
    Normalize(node_embed, node_idx);
}

DTensor<CPU, int> binary_pred;
void TestLoop(std::vector<int>& cur_test_idxes)
{    
    auto inputs = test_stream->GetNextBatch();
    net->fg.FeedForward({net->pred}, inputs, Phase::TEST);

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
	// MomentumSGDOptimizer<mode, Dtype> learner(&model, cfg::lr, cfg::momentum, cfg::l2_penalty);
	AdamOptimizer<mode, Dtype> learner(&model, cfg::lr, cfg::l2_penalty);

	int max_iter = (long long)cfg::max_iter;
	int init_iter = cfg::iter;
	if (init_iter > 0)
	{
		std::cerr << fmt::sprintf("loading model for iter=%d", init_iter) << std::endl;
		model.Load(fmt::sprintf("%s/iter_%d.model", cfg::save_dir, init_iter));
	}
	
    double t_fi = 0, t_vi = 0;
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
        auto t_fi_start = std::chrono::high_resolution_clock::now();
        for (unsigned fi = 0; fi < cfg::f_iter; ++fi)
        {            
            auto inputs = f_train_stream->GetNextBatch();                        
            net->fg.FeedForward({net->loss}, inputs, Phase::TRAIN);                        
            net->fg.BackPropagate({net->loss});                        
            learner.Update();            
            f_loss += net->loss->value.AsScalar();
        }
        auto t_fi_end = std::chrono::high_resolution_clock::now();
	    t_fi += std::chrono::duration_cast<std::chrono::milliseconds>(t_fi_end-t_fi_start).count();

        auto t_vi_start = std::chrono::high_resolution_clock::now();
        for (unsigned vi = 0; vi < cfg::v_iter; ++vi)
        {
            DTensor<mode, Dtype> est;
            auto inputs = v_train_stream->GetNextBatch();

            for (int i = 0; i < cfg::n_sample; ++i)
            {                
                net->fg.FeedForward({net->tilde_v}, inputs, Phase::TEST);
                auto& cur_v = net->tilde_v->value;
                if (i)
                    est.Axpy(1.0, cur_v);
                else
                    est.CopyFrom(cur_v);                
            }
            est.Scale(1.0 / cfg::n_sample);

            for (size_t i = 0; i < v_train_stream->batch_nodes.size(); ++i)
            {
                auto new_embed = est.GetRowRef(i, 1);
                UpdateEmbedding(v_train_stream->batch_nodes[i], new_embed);
            }
        }
        auto t_vi_end = std::chrono::high_resolution_clock::now();
	    t_vi += std::chrono::duration_cast<std::chrono::milliseconds>(t_vi_end-t_vi_start).count();

    	if (cfg::iter % cfg::report_interval == 0)
		{	
            std::cerr << "iter: " << cfg::iter;
            std::cerr << "\tfi: " << t_fi << "\tvi: " << t_vi;
            std::cerr << "\tloss: " << f_loss / cfg::f_iter << std::endl;
            t_fi = t_vi = 0;
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
    GetVar("init_w", cfg::num_nodes, cfg::n_embed);
    GetVar("W", cfg::n_embed * 1 + 1, cfg::n_embed);
    //GetVar("V", cfg::n_embed + 1, cfg::n_embed);
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
    LoadIdxes(cfg::f_train_idx, train_idxes);
    std::cerr << "num_train: " << train_idxes.size() << std::endl;
    LoadIdxes(cfg::f_test_idx, test_idxes);
    std::cerr << "num_test: " << test_idxes.size() << std::endl;

    std::cerr << "loading features" << std::endl;
    LoadFeat(cfg::f_feat);
    std::cerr << "everyting loaded" << std::endl;

    InitParams();

    f_train_stream = new DataProvider("f_train", train_idxes, cfg::batch_size);
    v_train_stream = new DataProvider("v_train", cfg::num_nodes, cfg::batch_size);
    test_stream = new DataProvider("test", test_idxes, test_idxes.size());

    if (cfg::is_regression)
    {
        net = new RegNet();
    } else
        net = new FuncNet();
    net->BuildNet();
    MainLoop();
	GpuHandle::Destroy();
	return 0;	
}
