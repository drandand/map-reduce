#include "cuda_helpers.cuh"
#include "cuda_exception.cuh"

/// @brief Verify the result code represents a success; if not raise an exception
/// @param result_code Result code to check
/// @param msg Message to include in the exception to help isolate the location of the error
void verify_code(CUresult result_code, std::string msg)
{
    if (result_code != CUDA_SUCCESS)
    {
        throw cuda_exception(result_code, msg);
    }
}
