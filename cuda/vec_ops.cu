#include "vec.cuh"
#include "cuda_exception.cuh"
#include "cuda_helpers.cuh"
#include "cuda_constants.cuh"
#include <cuda.h>
#include <cudaProfiler.h>
#include <curand_kernel.h>

#include <iostream>
#include <chrono>
#include <unordered_map>
#include <string>
#include <vector>
#include <regex>
#include <exception>

const std::string INIT("initialize");
const std::string OPER("operation");
const std::string COPY("copy");
const std::string VERI("verify");
const std::string THETA0("theta0");
const std::string THETA1("theta1");

constexpr std::size_t DEFAULT_SIZE = 65536;
constexpr double RAD_TO_DEG = 5.72957795130823e1;

/// @brief Provide information about this program and its usage
/// @param cli CLI invokation for this run
void show_help(const std::string &cli)
{
    std::cout << "This conduct performance metrics on the CUDA implementation of the vector" << std::endl;
    std::cout << "operations." << std::endl
              << std::endl;
    std::cout << "This takes zero (0) or more positive integer arguments in the command line." << std::endl
              << std::endl;
    std::cout << "Each of the arguments represents a length of the vectors to use in the" << std::endl;
    std::cout << "benchmarking, where two vectors of the size are randomly generated and the" << std::endl;
    std::cout << "angle between the two is computed and the time taken to perform different" << std::endl;
    std::cout << "aspects of the operation timed." << std::endl
              << std::endl;
    std::cout << "The timing performed is for the initialization of the two arrays, computing" << std::endl;
    std::cout << "the angle betwee the vectors and verifying the results." << std::endl
              << std::endl;
    std::cout << "If no sizes are given, a single run with " << DEFAULT_SIZE << " vector" << std::endl;
    std::cout << "elements." << std::endl
              << std::endl;
    std::cout << "usage: " << cli << " [size0 [size1 [size2 [...]]]]" << std::endl
              << std::endl;
}

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
    unsigned long long seed = time_ns();
    auto norm0 = [seed] __device__(std::size_t idx) -> double
    { return normal(0.0, 1.0, seed - 1, idx); };

    auto norm1 = [seed] __device__(std::size_t idx) -> double
    { return normal(0.0, 1.0, seed + 1, idx); };

    double sum_dot = 0.0;
    double sum_x = 0.0;
    double sum_y = 0.0;

    auto t0_start = time_ns();
    vec<double> x(size, norm0);
    vec<double> y(size, norm1);
    auto t0_stop = time_ns();

    auto t1_start = time_ns();
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

/// @brief Parse input arguments to get the vector sizes for performing
/// the benchmarking
/// @param argc Number of command line arguments provided
/// @param argv Array of C-style strings containing the command line
/// arguments provided when the program is invoked.
/// @return Vector containing the vector sizes to use as benchmarks
std::vector<std::size_t> get_sizes(int argc, char *argv[])
{
    std::regex pattern("^\\d+$");

    std::vector<std::string> args(argv + 1, argv + argc);
    std::vector<std::size_t> vals;

    for (auto arg : args)
    {
        if (!std::regex_match(arg, pattern))
        {
            throw std::invalid_argument("Received non-integer argument");
        }
        vals.push_back(std::stoi(arg));
    }

    if (vals.size() == 0)
    {
        vals.push_back(DEFAULT_SIZE);
    }

    return vals;
}

/// @brief Main function called when program starts
/// @param argc Number of command line arguments provided
/// @param argv Array of C-style strings containing the command line
/// arguments provided when the program is invoked.
/// @return 0 if no error occurred; not 0 for any instance where an error occurred
int main(int argc, char *argv[])
{
    std::vector<std::size_t> sizes;

    try
    {
        sizes = get_sizes(argc, argv);
    }
    catch (std::invalid_argument e)
    {
        std::string cli(argv[0]);
        show_help(cli);
        return -1;
    }

    for (auto size : sizes)
    {
        std::unordered_map<std::string, double> perf = run_test(size);
        std::cout << " count: " << size << " / "
                  << INIT << ": " << perf[INIT] << " ms / "
                  << OPER << ": " << perf[OPER] << " ms / "
                  << COPY << ": " << perf[COPY] << " ms / "
                  << VERI << ": " << perf[VERI] << " ms / "
                  << THETA0 << ": " << perf[THETA0] << " rad / " << perf[THETA0] * RAD_TO_DEG << " deg / "
                  << THETA1 << ": " << perf[THETA1] << " rad / " << perf[THETA1] * RAD_TO_DEG << " deg "
                  << std::endl;
    }

    return 0;
}