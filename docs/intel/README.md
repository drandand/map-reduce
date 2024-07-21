# intel
This folder contains performance time measurements for a system containing an [Intel Core i7 11850H 8 Core / 16 Thread 2.5 GHz](https://www.intel.com/content/www/us/en/products/sku/213799/intel-core-i711850h-processor-24m-cache-up-to-4-80-ghz/specifications.html) processor and an [NVIDIA RTX A3000 Laptop GPU](https://www.techpowerup.com/gpu-specs/rtx-a3000-mobile.c3806) based graphics card.

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

## [cpp_wsl2.csv](cpp_wsl2.csv)
Provides performance data for running the test suite in cpp with the vec_ops application compiled using the gcc compiler running in WSL2 (Ubuntu 24.04 LTS).

## [cpp_win11.csv](cpp_win11.csv)
Provides performance data for running the test suite in cpp with the vec_ops.exe application compiled using the mscv compiler running on a Windows platform.

## [cuda_wsl2.csv](cuda_wsl2.csv)
Provides performance data for running the test suite in cuda with the vec_ops application compiled using the nvcc / gcc compiler running in WSL2 (Ubuntu 24.04 LTS).

## [cuda_win11.csv](cuda_win11.csv)
Provides performance data for running the test suite in cuda with the vec_ops.exe application compiled using the nvcc / mscv compiler running on a Windows platform.
