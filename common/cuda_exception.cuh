#ifndef CUDA_EXCEPTION_CUH
#define CUDA_EXCEPTION_CUH

#include <iostream>
#include <exception>
#include <string>
#include <cuda.h>
#include <cudaProfiler.h>

/// @brief Class to encapsulate an exception to wrap CUDA error
/// states which may arise.
class cuda_exception : public std::exception
{
private:
    /// @brief Result code associated with the exception
    const CUresult _result_code;

    /// @brief Additional information which can be incorporated
    /// into an error message
    const std::string _msg;

public:
    /// @brief Class constructor taking in a result code and a supplemental
    /// message
    /// @param result_code Result code associated with the exception
    /// @param msg Amplifying or contextual information about the result
    cuda_exception(CUresult result_code, const std::string &msg = "");

    /// @brief Convert the exception into a string value for display
    /// @return String representation of the exception
    virtual std::string to_string() const;

    /// @brief Retrieve the error code for the exception
    /// @return The error code for the exception
    virtual inline CUresult error_code() const { return this->_result_code; }
};

/// @brief Standard output stream operator to display the exception
/// @param os Output stream where the exception will be displayed
/// @param ex Exception to display
/// @return Output stream passed in so it can be used for next
/// link in the output stream chain.
std::ostream &operator<<(std::ostream &os, const cuda_exception &ex);

#endif