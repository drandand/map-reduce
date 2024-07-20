# map-reduce
Demonstration of simple map / reduce functions comparing implementations in C++ and CUDA.
## [bad](bad)
Contains an experiment to see what happens if cudaMemcpy is used to overwrite the bounds of memory allocated either on the stack or on the heap on the host or on the device.
## [basic](basic)
Contains a single file comparison of how to compute the angle between two vectors in C++ and CUDA.
## [common](common)
Contains common modules used in the other CUDA applications here.
## [cpp](cpp)
Contains a C++ implementation of generic map-reduce operations on a purpose built vector class.
## [cuda](cuda)
Contains a CUDA implementation of generic map-reduce operations on a purpose built vector class.
## [device](device)
Contains code used to show hardware details about the installed devices.
## [docs](docs)
Contains spreadsheets showing relative performance metrics for the C++ and CUDA implementations of map-reduce operations.

# context
The files in the subfolders contained here have been tested on Windows 11 and [Windows Subsystem for Linux Version 2 (WSL2)](https://learn.microsoft.com/en-us/windows/wsl/install) using [Ubuntu 24.04 LTS](https://apps.microsoft.com/detail/9nz3klhxdjp5?amp%3Bgl=US&hl=en-us&gl=US).

Tests on straight Windows involves using a combination of the [Microsoft C Compiler](https://learn.microsoft.com/en-us/cpp/build/building-on-the-command-line?view=msvc-170) available from the Microsoft [Visual Studio Community Edition](https://visualstudio.microsoft.com/vs/community/) within the [x64 Native Command Prompt](https://learn.microsoft.com/en-us/cpp/build/how-to-enable-a-64-bit-visual-cpp-toolset-on-the-command-line?view=msvc-170).

Test on WSL2 / Ubuntu 24.04 use the [GCC compiler](https://gcc.gnu.org/).

Installation instructions for the NVidia CUDA Toolkit can be found at [CUDA Toolkit Installation](https://developer.nvidia.com/cuda-downloads).
# resources
Each of the folders either CUDA ot C++ files and various Windows and bash scripts.  The scripts are for building executables, cleaning products of the compilation process and for running tests using the compiled programs.  These are intended to run from the command line.  For Windows, use the x86 Native Command Prompt and for Linux or WSL, use bash.
## build
Builds the program(s) with the source code contained in the folder and the common folder.
* Windows (CMD / PowerShell): build.bat
* Bash: build.sh

## clean
Cleans any of the build products generated from running the build script in the same folder.  For Windows, these are files with extensions .exp .lib and .exe.  On Linux / WSL2, it is just the name of the executable(s) generated when running the build.sh bash script in the same folder.  There's a .gitignore for the Windows artifacts, but it doesn't delete any of the Linux build products... which is why these clean scripts are provided.
* Windows (CMD / PowerShel): clean.bat
* Bash: clean.sh

## test
This only exists in the cpp and cuda folders.

These scripts run a series of test cases against the compiled programs in the respective folders.
* Windows (CMD / PowerShell): test.bat
* Bash: test.sh

