# map-reduce
Demonstration of simple map / reduce functions comparing implementations in C++ and CUDA.
## bad
Contains an experiment to see what happens if cudaMemcpy is used to overwrite the bounds of memory allocated either on the stack or on the heap on the host or on the device.
## basic
Contains a single file comparison of how to compute the angle between two vectors in C++ and CUDA.
## common
Contains common modules used in the other CUDA applications here.
## cpp
Contains a C++ implementation of generic map-reduce operations on a purpose built vector class.
## cuda
Contains a CUDA implementation of generic map-reduce operations on a purpose built vector class.
## device
Contains code used to show hardware details about the installed devices.
## docs
Contains spreadsheets showing relative performance metrics for the C++ and CUDA implementations of map-reduce operations.
# context
The files in the subfolders contained here have been tested on Windows 11 and [Windows Subsystem for Linux Version 2 (WSL2)](https://learn.microsoft.com/en-us/windows/wsl/install) using [Ubuntu 24.04 LTS](https://apps.microsoft.com/detail/9nz3klhxdjp5?amp%3Bgl=US&hl=en-us&gl=US).

Tests on straight windows involves using a combination of the [Microsoft C Compiler](https://learn.microsoft.com/en-us/cpp/build/building-on-the-command-line?view=msvc-170) available from the Microsoft [Visual Studio Community Edition](https://visualstudio.microsoft.com/vs/community/)

Test on WSL2 / Ubuntu 24.04 use the [GCC compiler](https://gcc.gnu.org/).

Installation of the NVidia CUDA Toolkit can be found on the [CUDA Toolkit Installation](https://developer.nvidia.com/cuda-downloads) instructions.
# resources
Each of the folders having a code element with a main function has a number of scripts used to compile, clean and test.  These are intended to run from the command line.  For Windows, use either a CMD window or PowerShell.  For Linux or WSL, use bash.
## build
* Windows (CMD / PowerShell): build.bat
* Bash: build.sh

Builds the program(s) with the source code contained in the folder and the common folder.
## clean
* Windows (CMD / PowerShel): clean.bat
* Bash: clean.sh

Cleans any of the build products generated from running the build script in the same folder.  For Windows, these are files with extensions .exp .lib and .exe.  On Linux / WSL2, it is just the name of the executable(s) generated when running the build.sh bash script in the same folder.  There's a .gitignore for the Windows artifacts, but it doesn't delete any of the Linux build products... which is why these clean scripts are provided.
## test
* Windows (CMD / PowerShell): test.bat
* Bash: test.sh

This only exists in the cpp and cuda folders.

These scripts run a series of test cases against the compiled programs in the respective folders.
