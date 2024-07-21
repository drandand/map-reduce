#include "vec.hpp"
#include <random>
#include <cmath>

#include <iostream>
#include <chrono>
#include <string>
#include <vector>
#include <regex>
#include <exception>
#include <unordered_map>
#include <functional>

const std::string SIZE("size");
const std::string INIT("init (ms)");
const std::string OPER("oper (ms)");
const std::string VERI("veri (ms)");
const std::string T0RD("theta 0 rad");
const std::string T0DG("theta 0 deg");
const std::string T1RD("theta 1 rad");
const std::string T1DG("theta 1 deg");
const std::string DRAD("theta diff rad");
const std::string DDEG("theta diff deg");

const std::vector<std::string> LABELS = {SIZE, INIT, OPER, VERI, T0RD, T0DG, T1RD, T1DG, DRAD, DDEG};

constexpr std::size_t DEFAULT_SIZE = 65536;
constexpr double RAD_TO_DEG = 5.72957795130823e1;

typedef std::unordered_map<std::string, double> performance;
typedef std::vector<performance> performance_list;

/// @brief Provide information about this program and its usage
/// @param cli CLI invokation for this run
void show_help(const std::string &cli)
{
    std::cout << "This conduct performance metrics on the C++ implementation of the vector" << std::endl;
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

/// @brief Call a function which takes no arguments but returns a value of type T and time how long
/// it takes to call that function.
/// @tparam T Return type of the function
/// @param fn Function to time
/// @return Pair containing the result of calling the function and the time it took to run the function
template <typename T>
std::pair<T, double> timer_ms(std::function<T()> fn)
{
    auto t0 = time_ns();
    T result = fn();
    auto t1 = time_ns();

    return std::make_pair(result, (t1 - t0) * 1e-6);
}

/// @brief Compute the angle between the two vectors given.  This uses
/// traditional C to perform the computation.
/// @param size Number of elements in the two vectors
/// @param u First array on host to use in the angle computation
/// @param v Second array on host to use in the angle computation
/// @return Angle (radians) between the two vectors
double cppAngle(const vec<double> &u, const vec<double> &v)
{
    assert(u.size() == v.size());

    // Places to hold the sum for the dot procuct and the
    // squared magnitude of each of the vectors
    double uv = 0.0;
    double u2 = 0.0;
    double v2 = 0.0;

    // Compute each of the sums
    for (std::size_t idx = 0; idx < u.size(); ++idx)
    {
        uv += u.get(idx) * v.get(idx);
        u2 += u.get(idx) * u.get(idx);
        v2 += v.get(idx) * v.get(idx);
    }

    // Compute and return the angle between the two vectors
    return acos(uv / (sqrt(u2) * sqrt(v2)));
}

/// @brief Build a pair of vectors of the given size
/// @param count Number of vectors to build
/// @param size Number of elements each vec will contain
/// @return A vector of lenght 'count' containing vec's of length 'size'
std::vector<vec<double>> build_vectors(std::size_t count, std::size_t size)
{
    std::mt19937 gen;
    std::normal_distribution<double> dist(0.0, 1.0);
    auto norm = [&gen, &dist](std::size_t idx)
    { return dist(gen); };

    std::vector<vec<double>> result;
    for (std::size_t idx = 0; idx < count; ++idx)
        result.push_back(vec<double>(size, norm));

    return result;
}

/// @brief Run a test where two vectors are generated with random values and then the angle between them is computed
/// @param size Number of elements in each vector generated
/// @return Vector containing outcomes for this test case
performance run_test(std::size_t size)
{
    auto init = timer_ms<std::vector<vec<double>>>([size]()
                                                   { return build_vectors(2, size); });

    vec<double> &u = init.first[0];
    vec<double> &v = init.first[1];

    auto oper = timer_ms<double>([&u, &v]() -> double
                                 { return acos(dot(u, v) / (mag(u) * mag(v))); });
    double theta0_rad = oper.first;
    double theta0_deg = theta0_rad * RAD_TO_DEG;

    auto veri = timer_ms<double>([&u, &v]() -> double
                                 { return cppAngle(u, v); });
    double theta1_rad = veri.first;
    double theta1_deg = theta1_rad * RAD_TO_DEG;

    double theta_diff_rad = abs(theta0_rad - theta1_rad);
    double theta_diff_deg = abs(theta0_deg - theta1_deg);

    performance results = {
        {SIZE, (double)size},
        {INIT, init.second},
        {OPER, oper.second},
        {VERI, veri.second},
        {T0RD, theta0_rad},
        {T0DG, theta0_deg},
        {T1RD, theta1_rad},
        {T1DG, theta1_deg},
        {DRAD, theta_diff_rad},
        {DDEG, theta_diff_deg}};

    return results;
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

/// @brief Display the results in a CSV compliant format
/// @param results Results to print
void report(const performance_list &results)
{
    bool first = true;
    for (auto label : LABELS)
    {
        if (!first)
            std::cout << ',';
        std::cout << "\"" << label << "\"";
        first = false;
    }
    std::cout << std::endl;

    for (auto result : results)
    {
        first = true;
        for (auto label : LABELS)
        {
            if (!first)
                std::cout << ',';
            std::cout << result[label];
            first = false;
        }
        std::cout << std::endl;
    }
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

    performance_list results;
    for (auto size : sizes)
        results.push_back(run_test(size));

    report(results);

    return 0;
}