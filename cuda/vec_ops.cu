
#include <iostream>
#include <curand_kernel.h>
#include <chrono>
#include <unordered_map>
#include <cuda.h>
#include <cudaProfiler.h>

#include "vec.cuh"
#include "cuda_exception.cuh"
#include "cuda_helpers.cuh"
#include "cuda_constants.cuh"

#define INIT "initialize"
#define OPER "operation"
#define COPY "copy"
#define VERI "verify"
#define THETA0 "theta0"
#define THETA1 "theta1"

/// @brief Return the nanosecond timestamp from the start of the POSIX epoch
/// @return Number of nanoseconds from the start of the POSIX epoch
long long time_ns()
{
    return std::chrono::time_point_cast<std::chrono::nanoseconds>(
               std::chrono::system_clock::now())
        .time_since_epoch()
        .count();
}

/// @brief Retrieve normally distributed random values on the CUDA device kernel
/// @param mean Mean to feed to the distribution
/// @param stdev Standard deviation to feed to the distribution
/// @param seed Seed to initialize the generator
/// @param idx Index offset to adjust the seed for the generator
/// @return A draw from N(mean, stdev)
__device__ double normal(double mean, double stdev, unsigned long long seed, int idx)
{
    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, 0, &state);
    return curand_normal_double(&state);
}

/// @brief Run a test where two vectors are generated with random values and then the angle between them is computed
/// @param size Number of elements in each vector generated
/// @return Map containing the time for each phase of the test (initialize and compute)
std::unordered_map<std::string, double> run_test(std::size_t size)
{
    double mean = 0.0;
    double stdev = 1.0;

    unsigned long long seed = time_ns();
    auto norm0 = [seed, mean, stdev] __device__(std::size_t idx) -> double
    { return normal(mean, stdev, seed - 1, idx); };

    auto norm1 = [seed, mean, stdev] __device__(std::size_t idx) -> double
    { return normal(mean, stdev, seed + 1, idx); };

    double sum_dot = 0.0;
    double sum_x = 0.0;
    double sum_y = 0.0;

    auto t0_start = time_ns();
    vec<double> x(size, norm0);
    vec<double> y(size, norm1);
    auto t0_stop = time_ns();

    auto t1_start = time_ns();
    double dot_xy = dot(x, y);
    double mag_x = mag(x);
    double mag_y = mag(y);
    double theta0 = acos(dot(x, y) / (mag(x) * mag(y)));
    auto t1_stop = time_ns();

    auto t2_start = time_ns();
    x.to_host();
    y.to_host();
    auto t2_stop = time_ns();

    auto t3_start = time_ns();
    for (std::size_t idx = 0; idx < size; ++idx)
    {
        sum_dot += x[idx] * y[idx];
        sum_x += x[idx] * x[idx];
        sum_y += y[idx] * y[idx];
    }
    double theta1 = acos(sum_dot / (sqrt(sum_x) * sqrt(sum_y)));
    auto t3_stop = time_ns();

    std::unordered_map<std::string, double> ret = {
        {INIT, (t0_stop - t0_start) * 1e-6},
        {OPER, (t1_stop - t1_start) * 1e-6},
        {COPY, (t2_stop - t2_start) * 1e-6},
        {VERI, (t3_stop - t3_start) * 1e-6},
        {THETA0, theta0},
        {THETA1, theta1},
    };

    return ret;
}

/// @brief Main function called when program starts
/// @param argc Number of command line arguments provided
/// @param argv Array of C-style strings containing the command line
/// arguments provided when the program is invoked.
/// @return 0 if no error occurred; not 0 for any instance where an error occurred
int main(int argc, char *argv[])
{
    std::vector<std::string> args(argv + 1, argv + argc);
    std::vector<std::size_t> sizes;

    for (std::string arg : args)
    {
        sizes.push_back(std::atoi(arg.c_str()));
    }

    if (sizes.size() == 0)
    {
        sizes.push_back(65536);
    }

    constexpr double rad_to_deg = 180.0 / 3.141592653589793238463;

    CUresult err_code = cuInit(0);
    verify_code(err_code, "cuInit");

    for (auto size : sizes)
    {
        std::unordered_map<std::string, double> perf = run_test(size);
        std::cout << " count: " << size << " / "
                  << INIT << ": " << perf[INIT] << " ms / "
                  << OPER << ": " << perf[OPER] << " ms / "
                  << COPY << ": " << perf[COPY] << " ms / "
                  << VERI << ": " << perf[VERI] << " ms / "
                  << THETA0 << ": " << perf[THETA0] << " rad / " << perf[THETA0] * rad_to_deg << " deg / "
                  << THETA1 << ": " << perf[THETA1] << " rad / " << perf[THETA1] * rad_to_deg << " deg "
                  << std::endl;
    }

    return 0;
}