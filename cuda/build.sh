#! /bin/bash
nvcc --extended-lambda -std=c++17 -O3 test_vec.cu ../common/cuda_exception.cu ../common/cuda_constants.cu ../common/cuda_helpers.cu -I../common -l cuda -o test_vec
nvcc --extended-lambda -std=c++17 -O3 vec_ops.cu ../common/cuda_exception.cu ../common/cuda_constants.cu ../common/cuda_helpers.cu -I../common -l cuda -o vec_ops
