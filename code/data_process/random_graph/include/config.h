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
    static int n, m;
    static Dtype p;
    static const char* g_type, *out_folder;

    static void LoadParams(const int argc, const char** argv)
    {
        for (int i = 1; i < argc; i += 2)
        {
            if (strcmp(argv[i], "-g_type") == 0)
		        g_type = argv[i + 1];
            if (strcmp(argv[i], "-out") == 0)
		        out_folder = argv[i + 1];                
            if (strcmp(argv[i], "-n") == 0)
		        n = atoi(argv[i + 1]);
            if (strcmp(argv[i], "-m") == 0)
		        m = atoi(argv[i + 1]);
            if (strcmp(argv[i], "-p") == 0)
		        p = atof(argv[i + 1]);
        }
        
        std::cerr << "n = " << n << std::endl;
        std::cerr << "m = " << m << std::endl;
    	std::cerr << "p = " << p << std::endl;
    	std::cerr << "g_type = " << g_type << std::endl;
    }    
};

#endif
