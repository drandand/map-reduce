# basic
The code in this folder provides a basic example contrasting two approaches to computing the angle between two random vectors.  The first approach can be seen in the function ```cppAngle``` in [basic.cu](basic.cu).  The second approach, ```cudaAngle``` uses CUDA to perform the computation and invokes host function ```reduceSum``` and kernel functions ```mul``` and ```step```.

On Linux, the executable will be named "basic" and is invoked using:

```basic```

On Windows, the executable will be named "basic.exe" and is invoked using:

```basic.exe```

## source code

### [basic.cu](basic.cu)
Source code contrasting approaches to compute the angle between two vectors using C++ and CUDA.

## bash scripts

### [build.sh](build.sh)
Script to build an executable on Linux (including WSL2) using nvcc and gcc.  As of this writing, nvcc 12.5 is the latest version available within the CUDA toolkit and is not compatible with gcc version 14 or later.  This will result in a single executable file named "basic" which can be removed using the clean.sh script.

### [clean.sh](clean.sh)
Script to remove products generated when running build.sh.

## windows scripts

### [build.bat](build.bat)
Script to build an executable on a Windows host using nvcc and msvc (cl).  This will generate and executable file, "basic.exe" and some intermediate files, all of which can be removed using clean.bat.

### [clean.bat](clean.bat)
Script to remove the products generated when running build.bat.
