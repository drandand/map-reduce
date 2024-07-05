#include "cuda_exception.cuh"
#include "cuda_constants.cuh"

/// @brief Class constructor taking in a result code and a supplemental
/// message
/// @param result_code Result code associated with the exception
/// @param msg Amplifying or contextual information about the result
cuda_exception::cuda_exception(CUresult result_code, const std::string &msg)
    : _result_code(result_code), _msg(msg) {}

/// @brief Convert the exception into a string value for display
/// @return String representation of the exception
std::string cuda_exception::to_string() const
{
    std::string result_name = RESULT_NAMES.at(this->_result_code);

    return this->_msg + std::string(" : ") + result_name;
}

/// @brief Standard output stream operator to display the exception
/// @param os Output stream where the exception will be displayed
/// @param ex Exception to display
/// @return Output stream passed in so it can be used for next
/// link in the output stream chain.
std::ostream &operator<<(std::ostream &os, const cuda_exception &ex)
{
    os << ex.to_string();
    return os;
}
