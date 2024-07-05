#include "vec.hpp"

#include <cassert>
#include <string>
#include <iostream>
#include <cmath>

#define PASS "PASS"

/// @brief Gerenate a vector of the given size and type where
/// arr[idx] = idx
/// @tparam T Type of each element in the vector
/// @param size Number of elements to place into the vector
/// @return Vector generated of the specified size and type
template <typename T>
vec<T> gen_vec(std::size_t size)
{
    vec<T> ret_val(size, T(0));
    for (std::size_t idx = 0; idx < size; ++idx)
    {
        ret_val[idx] = T(idx);
    }

    return ret_val;
}
/// @brief Unit test to verify each of the public constructors work
/// properly.
/// @tparam T Type of elements for use in each avector
/// @return String containing "PASS" if all the tests pass
template <typename T>
std::string test_ini()
{
    std::size_t size = 10;

    vec<T> a(size, T(0));
    vec<T> b = {T(0), T(1), T(2), T(3), T(4), T(5), T(6), T(7), T(8), T(9)};
    vec<T> c(b);
    vec<T> d = gen_vec<T>(size);
    vec<T> e(size, [](std::size_t idx) -> T
             { return T(idx); });

    assert(a.size() == size);
    assert(b.size() == size);
    assert(c.size() == size);
    assert(d.size() == size);
    assert(e.size() == size);

    for (std::size_t idx = 0; idx < size; ++idx)
    {
        assert(a[idx] == T(0));
        assert(b[idx] == T(idx));
        assert(c[idx] == T(idx));
        assert(d[idx] == T(idx));
        assert(e[idx] == T(idx));

        assert(a.get(idx) == 0);
        assert(b.get(idx) == T(idx));
        assert(c.get(idx) == T(idx));
        assert(d.get(idx) == T(idx));
        assert(e.get(idx) == T(idx));
    }

    return PASS;
}

// @brief Test the map and op functions for the vec class to verify they work
/// @tparam T Type elements to use when testing the vector
/// @param size Number of elements to allocate in the vectors used to test
/// the map and op functionality
/// @return String containing "PASS" only if all tests pass
template <typename T>
std::string test_map(std::size_t size)
{
    const T TWO = T(2);
    const T THREE = T(3);
    const T FIVE = T(5);

    vec<T> src = gen_vec<T>(size);

    vec<T> a = vec<T>::template map<T>(src, [TWO](T x, std::size_t idx) -> T
                                       { return TWO * x; });
    vec<T> b = vec<T>::template map<T>(src, [THREE](T x, std::size_t idx) -> T
                                       { return THREE * x; });
    vec<T> c = vec<T>::template op<T, T>(a, b, [](T x, T y, std::size_t idx) -> T
                                         { return x + y; });

    assert(a.size() == size);
    assert(b.size() == size);
    assert(c.size() == size);

    for (std::size_t idx = 0; idx < size; ++idx)
    {
        T val = T(idx);
        assert(a[idx] == TWO * val);
        assert(b[idx] == THREE * val);
        assert(c[idx] == FIVE * val);

        assert(a.get(idx) == TWO * val);
        assert(b.get(idx) == THREE * val);
        assert(c.get(idx) == FIVE * val);
    }

    return PASS;
}

/// @brief Test the various vector addition operators
/// @tparam T  Type of elements in the vectors used to perform the test
/// @param size Number of elements to allocate in each vector used to perform
/// the test
/// @return String containing "PASS" only if all tests pass
template <typename T>
std::string test_add(std::size_t size)
{
    const T TWO = T(2);
    const T THREE = T(3);
    vec<T> a = gen_vec<T>(size);
    vec<T> b = gen_vec<T>(size);

    vec<T> c = a + b;
    vec<T> d = a + TWO;
    vec<T> e = THREE + b;

    assert(a.size() == size);
    assert(b.size() == size);
    assert(c.size() == size);
    assert(d.size() == size);
    assert(e.size() == size);

    for (std::size_t idx = 0; idx < size; ++idx)
    {
        T val = T(idx);
        assert(a[idx] == val);
        assert(b[idx] == val);
        assert(c[idx] == val + val);
        assert(d[idx] == val + TWO);
        assert(e[idx] == THREE + val);

        assert(a.get(idx) == val);
        assert(b.get(idx) == val);
        assert(c.get(idx) == val + val);
        assert(d.get(idx) == val + TWO);
        assert(e.get(idx) == THREE + val);
    }

    return PASS;
}

/// @brief Test the various vector subtraction operators
/// @tparam T  Type of elements in the vectors used to perform the test
/// @param size Number of elements to allocate in each vector used to perform
/// the test
/// @return String containing "PASS" only if all tests pass
template <typename T>
std::string test_sub(std::size_t size)
{
    const T TWO = T(2);
    const T THREE = T(3);
    vec<T> a = gen_vec<T>(size);
    vec<T> b = gen_vec<T>(size);

    vec<T> c = a - b;
    vec<T> d = a - TWO;
    vec<T> e = THREE - b;

    assert(a.size() == size);
    assert(b.size() == size);
    assert(c.size() == size);
    assert(d.size() == size);
    assert(e.size() == size);

    for (std::size_t idx = 0; idx < size; ++idx)
    {
        T val = T(idx);
        assert(a[idx] == val);
        assert(b[idx] == val);
        assert(c[idx] == val - val);
        assert(d[idx] == val - TWO);
        assert(e[idx] == THREE - val);

        assert(a.get(idx) == val);
        assert(b.get(idx) == val);
        assert(c.get(idx) == val - val);
        assert(d.get(idx) == val - TWO);
        assert(e.get(idx) == THREE - val);
    }

    return PASS;
}
/// @brief Test the various vector multiplication operators
/// @tparam T  Type of elements in the vectors used to perform the test
/// @param size Number of elements to allocate in each vector used to perform
/// the test
/// @return String containing "PASS" only if all tests pass
template <typename T>
std::string test_mul(std::size_t size)
{
    const T TWO = T(2);
    const T THREE = T(3);
    vec<T> a = gen_vec<T>(size);
    vec<T> b = gen_vec<T>(size);

    vec<T> c = a * b;
    vec<T> d = a * TWO;
    vec<T> e = THREE * b;

    assert(a.size() == size);
    assert(b.size() == size);
    assert(c.size() == size);
    assert(d.size() == size);
    assert(e.size() == size);

    for (std::size_t idx = 0; idx < size; ++idx)
    {
        T val = T(idx);
        assert(a[idx] == val);
        assert(b[idx] == val);
        assert(c[idx] == val * val);
        assert(d[idx] == val * TWO);
        assert(e[idx] == THREE * val);

        assert(a.get(idx) == val);
        assert(b.get(idx) == val);
        assert(c.get(idx) == val * val);
        assert(d.get(idx) == val * TWO);
        assert(e.get(idx) == THREE * val);
    }

    return PASS;
}

/// @brief Test the various vector division operators
/// @tparam T  Type of elements in the vectors used to perform the test
/// @param size Number of elements to allocate in each vector used to perform
/// the test
/// @return String containing "PASS" only if all tests pass
template <typename T>
std::string test_div(std::size_t size)
{
    const T TWO = T(2);
    const T THREE = T(3);
    const T TEN = T(10);
    vec<T> a = gen_vec<T>(size) + TEN;
    vec<T> b = gen_vec<T>(size) + TEN;

    vec<T> c = a / b;
    vec<T> d = a / TWO;
    vec<T> e = THREE / b;

    assert(a.size() == size);
    assert(b.size() == size);
    assert(c.size() == size);
    assert(d.size() == size);
    assert(e.size() == size);

    for (std::size_t idx = 0; idx < size; ++idx)
    {
        T val = T(idx) + TEN;

        assert(a[idx] == val);
        assert(b[idx] == val);
        assert(c[idx] == val / val);
        assert(d[idx] == val / TWO);
        assert(e[idx] == THREE / val);

        assert(a.get(idx) == val);
        assert(b.get(idx) == val);
        assert(c.get(idx) == val / val);
        assert(d.get(idx) == val / TWO);
        assert(e.get(idx) == THREE / val);
    }

    return PASS;
}

/// @brief Return the factorial of the value given
/// @tparam T Type of result to return
/// @param n Argument for the factorial operator
/// @return Factorial of n
template <typename T>
T fac(int n)
{
    T ret = T(1);

    for (int i = 2; i <= n; i++)
    {
        ret *= T(i);
    }

    return ret;
}

/// @brief Test some of the support functions used to compute
/// attributes of the vectors
/// @tparam T Type of elements to allocate in the test vectors
/// @param size Number of elements to allocate in the vectors to test
/// @param do_mag Flag to indicate whether to compute the vector magnitude
/// @return String containing "PASS" only if all tests pass
template <typename T>
std::string test_fun(std::size_t size, bool do_mag)
{
    const T ONE = T(1);
    const T TWO = T(2);
    const T SIX = T(6);
    const T SIZE = T(size);
    const T EXP_SUM = SIZE * (SIZE + ONE) / TWO;
    const T EXP_PROD = fac<T>(size);
    const T EXP_DOT = SIZE * (SIZE + 1) * (TWO * SIZE + ONE) / SIX;

    vec<T> a = gen_vec<T>(size) + ONE;

    assert(a.size() == size);
    assert(sum(a) == EXP_SUM);
    assert(prod(a) == EXP_PROD);
    assert(dot(a, a) == EXP_DOT);
    assert(mag2(a) == EXP_DOT);
    if (do_mag)
    {
        T EXP_MAG = sqrt(EXP_DOT);
        assert(mag(a) == EXP_MAG);
    }

    return PASS;
}

/// @brief Function to perform unit test on the == and != operators
/// @tparam T Type of elements in the vectors to test
/// @param size Number of elements to allocate in each vector to test
/// @return String containing "PASS" only if all tests pass
template <typename T>
std::string test_cmp(std::size_t size)
{
    T TWO = T(2);
    vec<T> a = gen_vec<T>(size);
    vec<T> b = a;
    vec<T> c = a * TWO;

    assert(a == b);
    assert(a != c);

    return PASS;
}

int main(int argc, char *argv[])
{
    const std::size_t size = 10;
    std::cout << "test_ini<int>: " << test_ini<int>() << std::endl;
    std::cout << "test_map<int>: " << test_map<int>(size) << std::endl;
    std::cout << "test_add<int>: " << test_add<int>(size) << std::endl;
    std::cout << "test_sub<int>: " << test_sub<int>(size) << std::endl;
    std::cout << "test_mul<int>: " << test_mul<int>(size) << std::endl;
    std::cout << "test_div<int>: " << test_div<int>(size) << std::endl;
    std::cout << "test_fun<int>: " << test_fun<int>(size, false) << std::endl;
    std::cout << "test_cmp<int>: " << test_cmp<int>(size) << std::endl;
    std::cout << std::endl;

    std::cout << "test_ini<unsigned int>: " << test_ini<unsigned int>() << std::endl;
    std::cout << "test_map<unsigned int>: " << test_map<unsigned int>(size) << std::endl;
    std::cout << "test_add<unsigned int>: " << test_add<unsigned int>(size) << std::endl;
    std::cout << "test_sub<unsigned int>: " << test_sub<unsigned int>(size) << std::endl;
    std::cout << "test_mul<unsigned int>: " << test_mul<unsigned int>(size) << std::endl;
    std::cout << "test_div<unsigned int>: " << test_div<unsigned int>(size) << std::endl;
    std::cout << "test_fun<unsigned int>: " << test_fun<unsigned int>(size, false) << std::endl;
    std::cout << "test_cmp<unsigned int>: " << test_cmp<unsigned int>(size) << std::endl;
    std::cout << std::endl;

    std::cout << "test_ini<float>: " << test_ini<float>() << std::endl;
    std::cout << "test_map<float>: " << test_map<float>(size) << std::endl;
    std::cout << "test_add<float>: " << test_add<float>(size) << std::endl;
    std::cout << "test_sub<float>: " << test_sub<float>(size) << std::endl;
    std::cout << "test_mul<float>: " << test_mul<float>(size) << std::endl;
    std::cout << "test_div<float>: " << test_div<float>(size) << std::endl;
    std::cout << "test_fun<float>: " << test_fun<float>(size, true) << std::endl;
    std::cout << "test_cmp<float>: " << test_cmp<float>(size) << std::endl;
    std::cout << std::endl;

    std::cout << "test_ini<double>: " << test_ini<double>() << std::endl;
    std::cout << "test_map<double>: " << test_map<double>(size) << std::endl;
    std::cout << "test_add<double>: " << test_add<double>(size) << std::endl;
    std::cout << "test_sub<double>: " << test_sub<double>(size) << std::endl;
    std::cout << "test_mul<double>: " << test_mul<double>(size) << std::endl;
    std::cout << "test_div<double>: " << test_div<double>(size) << std::endl;
    std::cout << "test_fun<double>: " << test_fun<double>(size, true) << std::endl;
    std::cout << "test_cmp<double>: " << test_cmp<double>(size) << std::endl;
    std::cout << std::endl;

    return 0;
}