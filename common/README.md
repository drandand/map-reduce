# common

This folder contains modules which containing common helper functions for CUDA programs elswehere in this project.

## constants

The constants module contains maps and vectors of used to resolve error messages and device features.

### [cuda_constants.cu](cuda_constants.cu)

File containing the constant definitions used elsewhere.

### [cuda_constants.cuh](cuda_constants.cuh)

File containing the constant header declarations referencing the definitions in [cuda_constants.cu](cuda_constants.cu).

## exception

The exception module contains a class declaration and definition for CUDA related error states.

### [cuda_exception.cu](cuda_exception.cu)

File containing the class definition for the cuda_excpetion class.

### [cuda_exception.cuh](cuda_exception.cuh)

File containing the header declations referenced in [cuda_exception.cu](cuda_exception.cu).

## helpers

File containing helper functions available to other CUDA modules.

### [cuda_helpers.cu](cuda_helpers.cu)

File containing the helper function definitions.

### [cuda_helpers.cuh](cuda_helpers.cuh)

FIle containing the header declarations referenced in [cuda_helpers.cu](cuda_helpers.cu).
