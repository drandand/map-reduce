#include <map>
#include <iostream>
#include <vector>
#include <string>
#include <cuda.h>
#include <cudaProfiler.h>

#include "cuda_exception.cuh"
#include "cuda_helpers.cuh"
#include "cuda_constants.cuh"

/// @brief Show the information for all of the devices accessible to this host
void showCudaDeviceInfo()
{
    CUresult err_code = cuInit(0);
    verify_code(err_code, "cuInit");

    int deviceCount;

    err_code = cuDeviceGetCount(&deviceCount);
    verify_code(err_code, "cuDeviceGetCount");

    for (std::size_t dev = 0; dev < deviceCount; ++dev)
    {
        char devName[256];
        CUdevice device;

        err_code = cuDeviceGet(&device, dev);
        verify_code(err_code, "cuDeviceGet");

        err_code = cuDeviceGetName(devName, 256, dev);
        verify_code(err_code, "cuDeviceGetName");
        std::cout << "device 0: " << std::string(devName) << std::endl;

        for (auto attr : DEVICE_ATTRIBUTE_VALUES)
        {
            int pi;
            std::string attr_name = DEVICE_ATTRIBUTE_NAMES.at(attr);

            err_code = cuDeviceGetAttribute(&pi, attr, device);
            verify_code(err_code, std::string("cuDeviceGetAttribute ") + attr_name);

            std::cout << "device " << dev << " / " << attr_name << ": " << pi << std::endl;
        }
    }
}

int main(int argc, char *argv[])
{
    try
    {
        showCudaDeviceInfo();
    }
    catch (cuda_exception &ex)
    {
        std::cerr << ex << std::endl;
    }
}