#!/bin/bash

prama=($@)
source=${prama[0]}
source=${source:=he.cu}
arr=(${source//./ /})

clang++ $source -o ${arr[0]} -lcudart_static ${prama[@]:1} 2>/tmp/run_cuda_error.log
if [ $? -eq 0 ]; then
    ./${arr[0]}
    rm ${arr[0]}
else
    cat /tmp/run_cuda_error.log 1>&2
    rm /tmp/run_cuda_error.log
fi