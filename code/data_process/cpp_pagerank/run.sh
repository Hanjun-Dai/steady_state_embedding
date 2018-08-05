#!/bin/bash

n=10000000
m=4
p=0
g_type=ba

is_test=0
output_folder=pr-$g_type-$n-$m

./build/main \
    -out $output_folder \
    -tol 1e-6 \
    -alpha 0.85 \
    -is_test $is_test \
    -score_file $output_folder/pr-0.85.txt 
