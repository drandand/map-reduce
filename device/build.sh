#! /bin/bash
nvcc device.cu ../common/cuda_exception.cu ../common/cuda_constants.cu ../common/cuda_helpers.cu -I../common -l cuda -o device