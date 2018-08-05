#!/bin/bash

n=10000000
m=1
p=0
g_type=ba

output_folder=pr-$g_type-$n-$m

if [ ! -e $output_folder ];
then
    mkdir -p $output_folder
fi

./build/main \
    -n $n \
    -m $m \
    -p $p \
    -g_type $g_type \
    -out $output_folder
