#!/bin/bash
make

self=0
dataset=ppi-self-${self}
data_root=../../../dropbox/data/ind_ppi/$dataset

result_root=$HOME/scratch/results/fixpoint/$dataset

batch_size=128
n_hidden=256
n_embed=256
learning_rate=0.01
max_iter=1000000
cur_iter=0
w_scale=0.01
f_iter=1
n_sample=1
v_iter=1
save_dir=$result_root/embed-$n_embed-v-$v_iter

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

./build/main \
    -avg 0 \
    -feat $data_root/features.txt \
    -label $data_root/label.txt \
    -train_idx $data_root/train_idx.txt \
    -test_idx $data_root/test_idx.txt \
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
    -int_report 50 \
    -int_test 50 \
    -int_save 10000 \
    -cur_iter $cur_iter \
    2>&1 | tee $save_dir/log.txt
