#ifndef cfg_H
#define cfg_H

#include <iostream>
#include <cstring>
#include <fstream>
#include <set>
#include <map>

#include "util/fmt.h"
#include "util/gnn_macros.h"
typedef float Dtype;
typedef gnn::CPU mode;

struct cfg
{
    static bool is_regression, avg, sparse_feat;
    static int iter;
    static int num_nodes, num_labels, dim_feat;;
    static int n_hidden;
    static int n_sample;
    static unsigned batch_size, dev_id;
    static unsigned n_embed;
    static unsigned max_iter, f_iter, v_iter; 
    static unsigned test_interval; 
    static unsigned report_interval; 
    static unsigned save_interval;
    static bool has_feat, multi_label;
    static Dtype lr;
    static Dtype l2_penalty; 
    static Dtype momentum;
    static Dtype w_scale;
    static const char* data_name;
    static const char *save_dir, *data_root, *f_feat, *f_idx, *f_score, *f_label;
    static char *f_train_idx, *f_test_idx;

    static void LoadParams(const int argc, const char** argv)
    {
        for (int i = 1; i < argc; i += 2)
        {
            if (strcmp(argv[i], "-score") == 0)
		        f_score = argv[i + 1];
            if (strcmp(argv[i], "-label") == 0)
		        f_label = argv[i + 1];
            if (strcmp(argv[i], "-reg") == 0)
		        is_regression = atoi(argv[i + 1]);
            if (strcmp(argv[i], "-avg") == 0)
		        avg = atoi(argv[i + 1]);
            if (strcmp(argv[i], "-feat") == 0)
		        f_feat = argv[i + 1];
            if (strcmp(argv[i], "-data_root") == 0)
		        data_root = argv[i + 1];
            if (strcmp(argv[i], "-data_name") == 0)
		        data_name = argv[i + 1];
            if (strcmp(argv[i], "-f_idx") == 0)
		        f_idx = argv[i + 1];
		    if (strcmp(argv[i], "-lr") == 0)
		        lr = atof(argv[i + 1]);
            if (strcmp(argv[i], "-n_hidden") == 0)
                n_hidden = atoi(argv[i + 1]); 
            if (strcmp(argv[i], "-n_sample") == 0)
                n_sample = atoi(argv[i + 1]); 
            if (strcmp(argv[i], "-dev_id") == 0)
                dev_id = atoi(argv[i + 1]);                         
            if (strcmp(argv[i], "-cur_iter") == 0)
                iter = atoi(argv[i + 1]);      
		    if (strcmp(argv[i], "-embed") == 0)
			    n_embed = atoi(argv[i + 1]);
    		if (strcmp(argv[i], "-max_iter") == 0)
	       		max_iter = atoi(argv[i + 1]);
    		if (strcmp(argv[i], "-f_iter") == 0)
	       		f_iter = atoi(argv[i + 1]);
    		if (strcmp(argv[i], "-v_iter") == 0)
	       		v_iter = atoi(argv[i + 1]);
    		if (strcmp(argv[i], "-batch_size") == 0)
	       		batch_size = atoi(argv[i + 1]);                   
		    if (strcmp(argv[i], "-int_test") == 0)
    			test_interval = atoi(argv[i + 1]);
    	   	if (strcmp(argv[i], "-int_report") == 0)
    			report_interval = atoi(argv[i + 1]);
    		if (strcmp(argv[i], "-int_save") == 0)
    			save_interval = atoi(argv[i + 1]);
    		if (strcmp(argv[i], "-l2") == 0)
    			l2_penalty = atof(argv[i + 1]);            
            if (strcmp(argv[i], "-w_scale") == 0)
                w_scale = atof(argv[i + 1]);
    		if (strcmp(argv[i], "-m") == 0)
    			momentum = atof(argv[i + 1]);	
    		if (strcmp(argv[i], "-svdir") == 0)
    			save_dir = argv[i + 1];
        }
        
        f_train_idx = new char[1000];
        f_test_idx = new char[1000];
        sprintf(f_train_idx, "%s/%s.txt", data_root, f_idx);
        sprintf(f_test_idx, "%s/test%s.txt", data_root, f_idx + 5);

        has_feat = false;
        multi_label = false;
        if (!strcmp(data_name, "citeseer") 
            || !strcmp(data_name, "pubmed")
            || !strcmp(data_name, "cora"))
            has_feat = true;
        if (!strcmp(data_name, "citeseer") 
            || !strcmp(data_name, "pubmed")
            || !strcmp(data_name, "cora")
            || !strcmp(data_name, "nell"))
            multi_label = false;
        if (is_regression)
        {
            multi_label = false;
        }
            

        std::cerr << "multi_label = " << multi_label << std::endl;
        std::cerr << "has_feat = " << has_feat << std::endl;
        std::cerr << "avg = " << avg << std::endl;
        std::cerr << "is_regression = " << is_regression << std::endl;
        std::cerr << "n_sample = " << n_sample << std::endl;
        std::cerr << "v_iter = " << v_iter << std::endl;
        std::cerr << "f_iter = " << f_iter << std::endl;
        std::cerr << "n_hidden = " << n_hidden << std::endl;
        std::cerr << "dev_id = " << dev_id << std::endl;
        std::cerr << "batch_size = " << batch_size << std::endl;
        std::cerr << "n_embed = " << n_embed << std::endl;
        std::cerr << "max_iter = " << max_iter << std::endl;
    	std::cerr << "test_interval = " << test_interval << std::endl;
    	std::cerr << "report_interval = " << report_interval << std::endl;
    	std::cerr << "save_interval = " << save_interval << std::endl;
    	std::cerr << "lr = " << lr << std::endl;
        std::cerr << "w_scale = " << w_scale << std::endl;
    	std::cerr << "l2_penalty = " << l2_penalty << std::endl;
    	std::cerr << "momentum = " << momentum << std::endl;
    	std::cerr << "init iter = " << iter << std::endl;	
    }    
};

#endif
