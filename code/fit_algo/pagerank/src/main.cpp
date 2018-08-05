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

void UpdateEmbedding(int node_idx, DTensor<mode, Dtype>& new_embed)
{
    auto& node_embed = model.params["node_embed"]->value;
    auto old_embed = node_embed.GetRowRef(node_idx, 1);
    old_embed.Axpby(0.9, new_embed, 0.1);
    // old_embed.CopyFrom(new_embed);
    Normalize(node_embed, node_idx);
}

void TestLoop(std::vector<int>& cur_test_idxes)
{    
    assert(cfg::is_regression);
    auto* rnet = dynamic_cast<RegNet*>(net);
    auto inputs = test_stream->GetNextBatch();
    rnet->fg.FeedForward({rnet->loss_mae, rnet->loss}, inputs, Phase::TEST);
    std::cerr << "mae: " << rnet->loss_mae->value.AsScalar();
    std::cerr << "\tmse: " << rnet->loss->value.AsScalar() << std::endl;
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
    //GetVar("init_w", cfg::dim_feat, cfg::n_embed);
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

    f_train_stream = new DataProvider(train_idxes, cfg::batch_size);
    v_train_stream = new DataProvider(cfg::num_nodes, cfg::batch_size);
    test_stream = new DataProvider(test_idxes, test_idxes.size());

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
