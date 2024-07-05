#ifndef CUDA_HELPERS_CUH
#define CUDA_HELPERS_CUH

#include <string>
#include <cuda.h>
#include <cudaProfiler.h>

/// @brief Verify the result code represents a success; if not raise an exception
/// @param result_code Result code to check
/// @param msg Message to include in the exception to help isolate the location of the error
extern void verify_code(CUresult result_code, std::string msg);

#endif