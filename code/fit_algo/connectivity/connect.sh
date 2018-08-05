#!/bin/bash
make

dataset=n-chains-connect
data_root=../../../data/algo_data/$dataset

result_root=$HOME/scratch/results/fixpoint/$dataset

batch_size=128
n_hidden=16
n_embed=16
learning_rate=0.01
max_iter=10000
cur_iter=0
w_scale=0.01
f_iter=1
n_sample=1
v_iter=50
f_idx=train_idx-0.1
save_dir=$result_root/embed-$n_embed-v-$v_iter

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

./build/main \
    -avg 1 \
    -label $data_root/label.txt \
    -f_idx $f_idx \
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
    -l2 0.0005 \
    -w_scale $w_scale \
    -int_report 10 \
    -int_test 1 \
    -int_save 10000 \
    -cur_iter $cur_iter \
    2>&1 | tee $save_dir/log-${f_idx}.txt
