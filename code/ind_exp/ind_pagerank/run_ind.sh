#!/bin/bash
make

n=10000
m=1
dataset=ind-pr-ba-$n-$m
data_root=../../../data/pagerank_ba/$dataset

result_root=$HOME/scratch/results/fixpoint/pr-ba-$n-$m

test_iters=10000
batch_size=256
n_hidden=16
n_embed=16
learning_rate=0.001
max_iter=100000
cur_iter=0
w_scale=0.01
f_iter=1
n_sample=1
v_iter=5
avg=0
test_idx=$data_root/all_idx.txt
save_dir=$result_root/embed-$n_embed

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

./build/main \
    -avg $avg \
    -test $test_iters \
    -saved_model $save_dir/iter_best.model \
    -reg 1 \
    -score $data_root/pr-0.85.txt \
    -feat $data_root/features.txt \
    -train_idx $data_root/all_idx.txt \
    -test_idx $test_idx \
    -data_name $dataset \
    -f_iter $f_iter \
    -v_iter $v_iter \
    -n_sample $n_sample \
    -data_root $data_root \
    -n_hidden $n_hidden \
    -lr $learning_rate \
    -max_iter $max_iter \
    -svdir $save_dir \
    -embed $n_embed \
    -batch_size $batch_size \
    -m 0.9 \
    -l2 0.0001 \
    -w_scale $w_scale \
    -int_report 10 \
    -int_test 10 \
    -int_save 100000 \
    -cur_iter $cur_iter
