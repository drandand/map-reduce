#include "vec.hpp"

#include <iostream>
#include <chrono>
#include <string>
#include <vector>
#include <unordered_map>
#include <random>
#include <cmath>
#include <numbers>

#define INIT "initialize"
#define OPER "operation"
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

/// @brief Run a test where two vectors are generated with random values and then the angle between them is computed
/// @param size Number of elements in each vector generated
/// @param gen Generator of random values to feed the distribution
/// @return Map containing the time for each phase of the test (initialize and compute)
std::unordered_map<std::string, double> run_test(std::size_t size, std::mt19937 &gen)
{
    std::normal_distribution<double> dist;
    double sum_dot = 0.0;
    double sum_x = 0.0;
    double sum_y = 0.0;

    auto t0 = time_ns();
    vec<double> x(size, [&gen, &dist](std::size_t idx)
                  { return dist(gen); });
    vec<double> y(size, [&gen, &dist](std::size_t idx)
                  { return dist(gen); });
    auto t1 = time_ns();
    double theta0 = acos(dot(x, y) / (mag(x) * mag(y)));
    auto t2 = time_ns();
    for (std::size_t idx = 0; idx < size; ++idx)
    {
        sum_dot += x[idx] * y[idx];
        sum_x += x[idx] * x[idx];
        sum_y += y[idx] * y[idx];
    }
    double theta1 = acos(sum_dot / (sqrt(sum_x) * sqrt(sum_y)));
    auto t3 = time_ns();

    std::unordered_map<std::string, double> ret = {
        {INIT, (t1 - t0) * 1e-6},
        {OPER, (t2 - t1) * 1e-6},
        {VERI, (t3 - t2) * 1e-6},
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
    std::mt19937 gen;

    for (std::string arg : args)
    {
        sizes.push_back(std::atoi(arg.c_str()));
    }

    if (sizes.size() == 0)
    {
        sizes.push_back(65536);
    }

    constexpr double rad_to_deg = 180.0 / 3.141592653589793238463;

    for (auto size : sizes)
    {
        std::unordered_map<std::string, double> perf = run_test(size, gen);
        std::cout << " count: " << size << " / "
                  << "  init: " << perf[INIT] << " ms /"
                  << "  oper: " << perf[OPER] << " ms / "
                  << "verify: " << perf[VERI] << " ms / "
                  << "theta0: " << perf[THETA0] << " rad / " << perf[THETA0] * rad_to_deg << " deg / "
                  << "theta1: " << perf[THETA1] << " rad / " << perf[THETA1] * rad_to_deg << " deg "
                  << std::endl;
    }

    return 0;
}