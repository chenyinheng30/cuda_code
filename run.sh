#!/bin/bash

source=$1
source=${source:=he.cu}
arr=(${source//./ /})

clang $source -o ${arr[0]} -lcudart_static -lcublas 2>/tmp/run_cuda.log
if [ $? -eq 0 ]; then
    ./${arr[0]}
    rm ${arr[0]}
else
    cat /tmp/run_cuda.log 1>&2
fi