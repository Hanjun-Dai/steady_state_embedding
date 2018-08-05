#include "config.h"

int cfg::max_iter = 100;
int cfg::num_nodes = 0;
int cfg::num_dir_edges = 0;
Dtype cfg::tol = 1e-6;
Dtype cfg::alpha = 0.85;
const char* cfg::score_file = nullptr;
const char* cfg::out_folder = "./";
bool cfg::is_test = false;