#include <iostream>
#include <random>
#include <chrono>
#include <nvfunctional>
#include <cuda.h>
#include <cudaProfiler.h>

#include "cuda_exception.cuh"
#include "cuda_helpers.cuh"

constexpr std::size_t BLOCK_SIZE = 256;

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
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

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
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    std::size_t span = (size / 2) + (size % 2);

    if (idx < span)
    {
        d_res[idx] = ((idx + span) < size) ? d_src[idx] + d_src[idx + span] : d_src[idx];
    }
}

constexpr std::size_t div_ceil(std::size_t n, std::size_t d)
{
    return (n / d) + (n % d > 0);
}

constexpr std::size_t half_ceil(std::size_t n)
{
    return (n / 2) + (n % 2);
}

double reduceSum(std::size_t size, const double *d_src)
{
    std::size_t next_size = half_ceil(size);
    double *d_res;
    bool first = true;

    cudaMalloc(&d_res, next_size * sizeof(double));

    while (size > 1)
    {
        const double *d_tmp = first ? d_src : ((const double *)d_res);
        std::size_t num_blocks = div_ceil(size, BLOCK_SIZE);

        step<<<num_blocks, BLOCK_SIZE>>>(size, d_res, d_tmp);
        size = next_size;
        next_size = half_ceil(size);
        first = false;
    }

    double h_res;
    cudaMemcpy(&h_res, d_res, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_res);

    return h_res;
}

int main(int argc, char *argv[])
{
    CUresult err_code = cuInit(0);
    verify_code(err_code, "cuInit");

    constexpr std::size_t N = 1024;
    constexpr std::size_t num_blocks = div_ceil(N, BLOCK_SIZE);

    std::mt19937 gen;
    std::normal_distribution<double> dist;

    double *h_x = new double[N];
    double *h_y = new double[N];
    for (auto ptr_x = h_x, ptr_y = h_y; ptr_x != h_x + N; ++ptr_x, ++ptr_y)
    {
        *ptr_x = dist(gen);
        *ptr_y = dist(gen);
    }

    double *d_dest;
    double *d_x;
    double *d_y;

    cudaMalloc(&d_dest, N * sizeof(double));
    cudaMalloc(&d_x, N * sizeof(double));
    cudaMalloc(&d_y, N * sizeof(double));

    cudaMemcpy(d_x, h_x, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, N * sizeof(double), cudaMemcpyHostToDevice);

    mul<<<num_blocks, BLOCK_SIZE>>>(N, d_dest, d_x, d_y);
    double sum = reduceSum(N, d_dest);

    double *h_dest = new double[N];
    cudaMemcpy(h_dest, d_dest, N * sizeof(double), cudaMemcpyDeviceToHost);

    double t_sum = 0.0;
    for (double *ptr = h_dest; ptr != h_dest + N; ++ptr)
    {
        t_sum += *ptr;
    }

    std::cout << "sum: " << sum << " / t_sum: " << t_sum << " / diff: " << (sum - t_sum) << std::endl;

    cudaFree(d_dest);
    cudaFree(d_x);
    cudaFree(d_y);
    delete[] h_x;
    delete[] h_y;
    delete[] h_dest;
}