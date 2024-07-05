#ifndef CUDA_CONSTANTS_CUH
#define CUDA_CONSTANTS_CUH

#include <map>
#include <string>
#include <vector>
#include <cuda.h>

/// @brief Map of CUDA Device Attribute enum values to the associated name
/// as a string.
extern const std::map<CUdevice_attribute, std::string> DEVICE_ATTRIBUTE_NAMES;

/// @brief Map of CUDA result enum values to the associated names as string values
extern const std::map<CUresult, std::string> RESULT_NAMES;

/// @brief List of CUDA device attribute values useful for iterating through them
extern const std::vector<CUdevice_attribute> DEVICE_ATTRIBUTE_VALUES;

/// @brief List of CUDA result enum values which is potentially useful, though
/// there's not a good use case for that which comes to mind.
extern const std::vector<CUresult> RESULT_VALUES;

#endif