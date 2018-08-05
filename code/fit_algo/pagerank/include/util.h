#ifndef UTIL_H
#define UTIL_H

#include "config.h"
#include "tensor/tensor_all.h"

using namespace gnn;

void Normalize(DTensor<mode, Dtype>& embed_mat);

void Normalize(DTensor<mode, Dtype>& embed_mat, int row_idx);

void LoadIdxes(const char* fname, std::vector<int>& idx_list);

void LoadFeat(const char* fname);

int get_dim(const char* fname);

#endif