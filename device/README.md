# device
This program queries and displays hardware attributes about each CUDA device installed and accessible on the host.  See the documentation for [cuDeviceGetAttribute](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g9c3e1414f0ad901d3278a4d6645fc266) for more information about the attributes queried.

On Linux, the executable will be named "device" and is invoked using:

```device```

On Windows, the executable will be named "device.exe" and is invoked using:

```device.exe```

## source code

### [device.cu](device.cu)
Source code contrasting approaches to compute the angle between two vectors using C++ and CUDA.

## bash scripts

### [build.sh](build.sh)
Script to build an executable on Linux (including WSL2) using nvcc and gcc.  As of this writing, nvcc 12.5 is the latest version available within the CUDA toolkit and is not compatible with gcc version 14 or later.  This will result in a single executable file named "device" which can be removed using the clean.sh script.

### [clean.sh](clean.sh)
Script to remove products generated when running build.sh.

## windows scripts

### [build.bat](build.bat)
Script to build an executable on a Windows host using nvcc and msvc (cl).  This will generate and executable file, "device.exe" and some intermediate files, all of which can be removed using clean.bat.

### [clean.bat](clean.bat)
Script to remove the products generated when running build.bat.
