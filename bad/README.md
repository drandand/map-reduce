# bad
The code in this folder seeks to intentially mis-use cudaMemcpy to cause destructive overwriting from the device onto the host.  To invoke this, the user provides four positive integer arguments.

On Linux, the executable will be named "bad" and is invoked using:

```bad <h_size> <d_size> <h_to_d_size> <d_to_h_size>```

On Windows, the executable will be named "bad.exe" and is invoked using:

```bad.exe <h_size> <d_size> <h_to_d_size> <d_to_h_size>```

The arguments passed to the executable are:

- h_size - number of ints to allocate on host.
- d_size - number of ints to allocate on device.
- h_to_d_size - number of ints to copy from host to device.
- d_to_h_size - number of ints to copy from device to host.

## source code

### [bad.cu](bad.cu)
Source code exploring how cudaMemcpy behaves when passed arguments that do not align to the allocated sizes of content on the device as well as in either the stack or heap on the host.

## bash scripts

### [build.sh](build.sh)
Script to build an executable on Linux (including WSL2) using nvcc and gcc.  As of this writing, nvcc 12.5 is the latest version available within the CUDA toolkit and is not compatible with gcc version 14 or later.  This will result in a single executable file named "bad" which can be removed using the clean.sh script.

### [clean.sh](clean.sh)
Script to remove products generated when running build.sh.

## windows scripts

### [build.bat](build.bat)
Script to build an executable on a Windows host using nvcc and msvc (cl).  This will generate and executable file, "bad.exe" and some intermediate files, all of which can be removed using clean.bat.

### [clean.bat](clean.bat)
Script to remove the products generated when running build.bat.
