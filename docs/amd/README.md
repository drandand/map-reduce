# amd
This folder contains performance time measurements for a system containing an [AMD Ryzen Threadripper 2990WX 32-Core processor running at 3.0 GHz](https://en.wikipedia.org/wiki/Threadripper#Colfax_(Threadripper_2000_series,_Zen+_based)) and an [NVidia GeForce RTX 4080Ti](https://www.techpowerup.com/gpu-specs/geforce-rtx-4080-ti.c3887) based graphics card.

Each of these files contains information about the performance for each of the approaches described.  These metrics measure the time required to perform each of the portions of generating random arrays of the given size and to perform the initialization, computation and verification of the angle computation.  In the case of CUDA implementations, there is also a field for time required to copy content from the kernel down to the host.  The metrics captured are:

* size - number of elements in the test
* init (ms) - Time in milliseconds to initialize the vectors
* comp (ms) - Number of milliseconds to compute the angle between the two randomly generated vecotrs.
* copy (ms) - Number of milliseconds to copy content from the kernel to the host (CUDA only)
* veri (ms) - Number of milliseconds to verify the answer from the code being tested
* theta 0 rad - Angle (radians) between the two angles generated as measured with the code being tested
* theta 0 deg - Angle (degrees) between the two angles generated as measured with the code being tested
* theta 1 rad - Angle (radians) between the two angles generated using the reference method for verifying the results of the code under test
* theta 1 deg - Angle (degrees) between the two angles generated using the reference method for verifying the results of the code under test
* theta diff rad - Difference in the angle (radians) between the angle measurements of both methods
* theta diff deg - Difference in the angle (degrees) between the angle measurements of both methods

## [cpp_gcc_rate.csv](cpp_gcc_rate.csv)
Provides performance data for running the test suite in cpp with the vec_ops application compiled using the gcc compiler running in WSL2 (Ubuntu 24.04 LTS).

## [cpp_msvc_rate.csv](cpp_msvc_rate.csv)
Provides performance data for running the test suite in cpp with the vec_ops.exe application compiled using the mscv compiler running on a Windows platform.

## [cuda_nvcc_ubuntu_rate.csv](cuda_nvcc_ubuntu_rate.csv)
Provides performance data for running the test suite in cuda with the vec_ops application compiled using the nvcc / gcc compiler running in WSL2 (Ubuntu 24.04 LTS).

## [cuda_nvcc_windows_rate.csv](cuda_nvcc_windows_rate.csv)
Provides performance data for running the test suite in cuda with the vec_ops.exe application compiled using the nvcc / mscv compiler running on a Windows platform.
