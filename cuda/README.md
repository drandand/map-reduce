# cuda
This folder contains a sample implementation of map-reduce operations on vectors using a combination of C++ and CUDA.  It leverages templates, operator overloading and lambdas on a class library called ```vec```.  Specifically, use of lambdas and templates are crafted to allow them to operate in conjunction with calls to kernel functions, and so are somewhat different than the traditional approaches used with modern C++.

The product of the build scripts is two executables.  The first performs unit testing on the ```vec``` class and associated functions.  The second coducts performance measurement of the ```vec``` class.

## unit testing

The unit tests provide a collection of templated functions which run through the operations in [vec.cuh](vec.cuh) to ensure they function properly. It takes no arguments and is not configurable. After using one of the build scripts, run the unit test by using one of the following methods:

**Windows**

```test_vec.exe```

**Linux (including WSL2)**

```./test_vec```

## Performance testing

To conduct performance testing, invoke the vec_ops executable and provide 0 or more positive integer values as command line arguments.  Each positive integer value passed specifies the number of elements in the vectors to use to perform the test.  For instance, passing ```vec_ops 3 4 5``` will perform 3 tests with vector length 3 in the first test, 4 in the second and 5 in the third.  If no numbers are given, it defaults to a single test with 65,536 vector elements.

Each run will perform the following measurements on each test case posed:

* Time to initialze the two vectors used in the test
* Time to compute the angle between the two vectors
* Time to copy the results from the device kernel to the host
* Time to perform verification that the angle is correct

It will also show the angle computed between the two vectors in both radians and degrees for both the computation and verification steps.

Usage is as follows:

**Windows**

```vec_ops.exe [size0 [size1 [size2 [...]]]]```

**Linux**

```vec_ops [size0 [size1 [size2 [...]]]]```

In both invocations, each size is a positive integer, and there can be any number of them, including none.  If none, then a single test with 65,536 will be performed.

## source code

### [vec.cuh](vec.hpp)

Contains a template class and several template functions using CUDA to perform map and reduce functions on vectors.

### [test_vec.cu](test_vec.cu)

Contains unit tests for the ```vec``` class to ensure the functions and methods in [vec.cuh](vec.cuh).

### [vec_ops.cu](vec_ops.cu)

Performs timing on the ```vec``` content by computing the angle between two vectors of sizes passed through the command line.

## bash scripts

## [build.sh](build.sh)
Script to build an executable on Linux (including WSL2) using nvcc and gcc.  As of this writing, nvcc 12.5 is the latest version available within the CUDA toolkit and is not compatible with gcc version 14 or later.  This will result in a two executables files, "test_vec" and "vec_ops" which can be removed using the clean.sh script.

## [clean.sh](clean.sh)
Script to remove products generated when running build.sh.

## [test.sh](test.sh)
Runs a collection of canned test cases with vector lengths containing between 10 and 100,000,000 elements, stepped by powers of 10 between each test case.  It will run the test case with 10 vector elements twice, the first time to account for some initialization of the CUDA run time; the second run does not suffer from the initialization penalty and better represents the actual performance of the salient operations.

## windows scripts

## [build.bat](build.bat)
Script to build an executable on a Windows host using nvcc and msvc (cl).  This will result in a two executables files, "test_vec.exe" and "vec_ops.exe" and some intermediate files, all of which can be removed using clean.bat.

## [clean.bat](clean.bat)
Script to remove the products generated when running build.bat.

## [test.bat](test.bat)
Runs a collection of canned test cases with vector lengths containing between 10 and 100,000,000 elements, stepped by powers of 10 between each test case.  It will run the test case with 10 vector elements twice, the first time to account for some initialization of the CUDA run time; the second run does not suffer from the initialization penalty and better represents the actual performance of the salient operations.
