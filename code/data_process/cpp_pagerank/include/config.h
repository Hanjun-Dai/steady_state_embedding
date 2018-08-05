#ifndef cfg_H
#define cfg_H

#include <iostream>
#include <cstring>
#include <fstream>
#include <set>
#include <map>

#include "util/fmt.h"
#include "util/gnn_macros.h"
typedef double Dtype;
typedef gnn::CPU mode;

struct cfg
{
    static int max_iter, num_nodes, num_dir_edges;
    static Dtype tol, alpha;
    static const char *out_folder, *score_file;
    static bool is_test;

    static void LoadParams(const int argc, const char** argv)
    {
        for (int i = 1; i < argc; i += 2)
        {
            if (strcmp(argv[i], "-out") == 0)
		        out_folder = argv[i + 1];         
            if (strcmp(argv[i], "-is_test") == 0)
		        is_test = atoi(argv[i + 1]);
            if (strcmp(argv[i], "-score_file") == 0)
		        score_file = argv[i + 1];
            if (strcmp(argv[i], "-alpha") == 0)
		        alpha = atof(argv[i + 1]);
            if (strcmp(argv[i], "-tol") == 0)
		        tol = atof(argv[i + 1]);
            if (strcmp(argv[i], "-max_iter") == 0)
		        max_iter = atof(argv[i + 1]);                
        }
        
        std::cerr << "is_test = " << is_test << std::endl;
        std::cerr << "tol = " << tol << std::endl;
        std::cerr << "alpha = " << alpha << std::endl;
    	std::cerr << "max_iter = " << max_iter << std::endl;
    }    
};

#endif
