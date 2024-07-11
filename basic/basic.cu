#include <iostream>
#include <random>
#include <chrono>
#include <nvfunctional>
#include <cuda.h>
#include <cudaProfiler.h>

#include "cuda_exception.cuh"
#include "cuda_helpers.cuh"

// Macro for computing the ceiling of half of the given number
#define HALF_CEIL(N) (((N) / 2) + ((N) % 2))

// Macro for computing the ceiling of a number divided by another
#define DIV_CEIL(N, D) (((N) / (D)) + ((N) % (D) > 0))

/// @brief Threads per block
constexpr std::size_t THREADS_PER_BLOCK = 256;

/// @brief Size of the vectors to operate on
constexpr std::size_t SIZE = 1024;

/// @brief Number of blocks needed to accommodate the vector
constexpr std::size_t NUM_BLOCKS = DIV_CEIL(SIZE, THREADS_PER_BLOCK);

/// @brief Convenience to convert radians to degrees
constexpr double RAD_TO_DEG = 180.0 / 3.141592653589793238463;

/// @brief Return the nanosecond timestamp from the start of the POSIX epoch
/// @return Number of nanoseconds from the start of the POSIX epoch
unsigned long long time_ns()
{
    return std::chrono::time_point_cast<std::chrono::nanoseconds>(
               std::chrono::system_clock::now())
        .time_since_epoch()
        .count();
}

/// @brief Multiply each element of the two input arrays (d_x and d_y) and place the
/// products as elements of the output array (d_res)
/// @param size Number of elements in each of the vectors
/// @param d_res Device result array
/// @param d_x Vector on left side of the dot operator
/// @param d_y Vector on the right side of the dot operator
__global__ void mul(std::size_t size, double *d_res, const double *d_x, const double *d_y)
{
    std::size_t idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < size)
    {
        d_res[idx] = d_x[idx] * d_y[idx];
    }
}

/// @brief Perform a single step of a reduce poeration on the d_src array
/// @param size Number of elements in the source array
/// @param d_res Output array where the results will be held.  It will have size ceil(size / 2)
/// @param d_src Inpur array containing the input array to perform a partial sum upon.
__global__ void step(std::size_t size, double *d_res, const double *d_src)
{
    std::size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    std::size_t span = HALF_CEIL(size);

    if (idx < span)
    {
        d_res[idx] = ((idx + span) < size) ? d_src[idx] + d_src[idx + span] : d_src[idx];
    }
}

/// @brief Sum the elements of the vector
/// @param size Number of elements in the vector
/// @param du Array of values on the CUDA device
/// @return Sum of the elements in the array given
double reduceSum(std::size_t size, const double *du)
{
    std::size_t next_size = HALF_CEIL(size);
    double *dr; // Storage for intermediate results
    bool first = true;

    cudaMalloc(&dr, next_size * sizeof(double));

    while (size > 1)
    {

        const double *dt = first ? du : ((const double *)dr);
        std::size_t num_blocks = DIV_CEIL(next_size, THREADS_PER_BLOCK);

        step<<<num_blocks, THREADS_PER_BLOCK>>>(size, dr, dt);
        size = next_size;
        next_size = HALF_CEIL(size);
        first = false;
    }

    double s; // Place to put the sum

    cudaMemcpy(&s, dr, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(dr);

    return s;
}

/// @brief Compute the angle between the two vectors given.  This uses
/// traditional C to perform the computation.
/// @param size Number of elements in the two vectors
/// @param u First array on host to use in the angle computation
/// @param v Second array on host to use in the angle computation
/// @return Angle (radians) between the two vectors
double cppAngle(std::size_t size, const double *u, const double *v)
{
    // Places to hold the sum for the dot procuct and the
    // squared magnitude of each of the vectors
    double uv = 0.0;
    double u2 = 0.0;
    double v2 = 0.0;

    // Compute each of the sums
    for (auto pu = u, pv = v; pu != u + size; ++pu, ++pv)
    {
        uv += (*pu) * (*pv);
        u2 += (*pu) * (*pu);
        v2 += (*pv) * (*pv);
    }

    // Compute and return the angle between the two vectors
    return acos(uv / (sqrt(u2) * sqrt(v2)));
}

/// @brief Compute the angle between the two given vectors using the CUDA
/// device
/// @param size Number of elements in the vectors
/// @param u Array on host containing the first vector
/// @param v Array on host containing the second vector
/// @return Angle (radians) between the two vectors
double cudaAngle(std::size_t size, const double *u, const double *v)
{
    // Pointers to arrays on the device
    double *dp;
    double *du;
    double *dv;

    // Allocate space on the device for each of the vectors
    cudaMalloc(&dp, size * sizeof(double));
    cudaMalloc(&du, size * sizeof(double));
    cudaMalloc(&dv, size * sizeof(double));

    // Copy the values in the host arrays up to the device
    cudaMemcpy(du, u, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dv, v, size * sizeof(double), cudaMemcpyHostToDevice);

    // Compute the dot product of the two vectors
    mul<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(size, dp, du, dv);
    double uv = reduceSum(size, dp);

    // Compute the square of the magnitude of the first vector
    mul<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(size, dp, du, du);
    double u2 = reduceSum(size, dp);

    // Compute the square of the magnitude of the second vector
    mul<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(size, dp, dv, dv);
    double v2 = reduceSum(size, dp);

    // Deallocate space used for the arrays on the device
    cudaFree(dp);
    cudaFree(du);
    cudaFree(dv);

    // Compute and return the angle between the two vectors
    return acos(uv / (sqrt(u2) * sqrt(v2)));
}

/// @brief Generate an array of normal random values with the given mean
/// and standard deviation
/// @param size Number of elements to allocated in the arrays
/// @param mean Mean to use in the normal distribution
/// @param stdev Standard deviation to use in the normal distribution
/// @param gen Generator to use to generate draws from the normal distribution
/// @return An array containing normally distributed random values
double *genVec(std::size_t size, double mean, double stdev, std::mt19937 &gen)
{
    std::normal_distribution<double> dist(mean, stdev);

    double *vec = new double[size];
    for (auto ptr_vec = vec; ptr_vec != vec + size; ++ptr_vec)
    {
        *ptr_vec = dist(gen);
    }

    return vec;
}

int main(int argc, char *argv[])
{
    std::mt19937 gen;

    // Generate two vectors
    double *h_u = genVec(SIZE, 0.0, 1.0, gen);
    double *h_v = genVec(SIZE, 0.0, 1.0, gen);

    // Compute the angles between the two vectors using the
    // two different approaches
    double cuda_angle = cudaAngle(SIZE, h_u, h_v) * RAD_TO_DEG;
    double cpp_angle = cppAngle(SIZE, h_u, h_v) * RAD_TO_DEG;

    // Show the results
    std::cout << "cuda_angle: " << cuda_angle << " deg / "
              << "cpp_angle: " << cpp_angle << " deg / "
              << "diff: " << (cuda_angle - cpp_angle)
              << std::endl;

    // Clean up
    delete[] h_u;
    delete[] h_v;
}