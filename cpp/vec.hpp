#ifndef VEC_HPP
#define VEC_HPP

#include <iostream>
#include <numeric>
#include <cassert>
#include <functional>
#include <algorithm>
#include <initializer_list>
#include <sstream>
#include <cmath>

/// @brief Vector class used for map / reduct functions
/// @tparam T Type of element the vector will contain
template <typename T>
class vec
{
private:
    /// @brief Array containing the vector elements
    T *_vec;

    /// @brief Number of elements the vector contains
    const std::size_t _size;

    /// @brief Base constructor for most of the others which takes a size
    /// and initializes all of the elements to a preliminary value.
    /// @param size Size used to initialize the member variables of the
    /// class instance.
    vec(std::size_t size)
        : _size(size)
    {
        assert(this->_size > 0);
        this->_vec = new T[size];
    }

    /// @brief Private constuctor used to map the elements of another
    /// vector into the new one using the given function
    /// @tparam S Type of elements in the source vector
    /// @param src Source vector to transform to populuate this new one
    /// @param fn Function to map elements from source vector into this new one
    template <typename S>
    vec(const vec<S> &src, std::function<T(S, std::size_t)> fn)
        : vec(src._size)
    {
        const S *src_ptr = src._vec;
        T *dst_ptr = this->_vec;
        for (std::size_t i = 0; i < this->_size; ++i, ++src_ptr, ++dst_ptr)
        {
            *dst_ptr = fn(*src_ptr, i);
        }
    }

    /// @brief Private constructor used to perform an operation on two source vectors
    /// to produce a new vector of the same size as the two source vectors.
    /// @tparam L Type of element in the vector on the left side of the operand
    /// @tparam R Type of element in the vector on the right side of the operand
    /// @param l_src Left vector to combine with to produce the new vector
    /// @param r_src Right vector to combine with to produce the new vector
    /// @param fn Function to combine the left and right elements to compute each element of the new vector
    template <typename L, typename R>
    vec(const vec<L> &l_src, const vec<R> &r_src, std::function<T(L, R, std::size_t)> fn)
        : _size(l_src._size)
    {
        assert(this->_size > 0);
        assert(l_src._size == r_src._size);
        this->_vec = new T[this->_size];

        const L *l_ptr = l_src._vec;
        const R *r_ptr = r_src._vec;

        T *dst_ptr = this->_vec;
        for (std::size_t i = 0; i < this->_size; ++i, ++l_ptr, ++r_ptr, ++dst_ptr)
        {
            *dst_ptr = fn(*l_ptr, *r_ptr, i);
        }
    }

public:
    /// @brief Constructor which uses an initializer list to create the new vector
    /// @param src Source initializer list used to populate the new vector
    vec(const std::initializer_list<T> &src)
        : vec(src.size())
    {
        std::copy(src.begin(), src.end(), this->_vec);
    }

    /// @brief Copy constructor which copies the elements of the source vector into
    /// this new one
    /// @param src Source vector used to populate this new one
    vec(const vec<T> &src)
        : vec(src._size)
    {
        std::copy(src._vec, src._vec + src._size, this->_vec);
    }

    /// @brief Move constructor to improve performance
    /// @param src Source vector contrubuting the internal array
    vec(vec<T> &&src)
        : _size(src._size),
          _vec(src._vec)
    {
        src._vec = nullptr;
    }

    /// @brief Initialize a new vector to have length size with each element having
    /// the given value, val.  If omitted, the value will be the default value
    /// of type T... for numeric content it should be 0
    /// @param size Size of the new vector
    /// @param val Value to set each element of the vector
    vec(std::size_t size, const T &val)
        : vec(size, [val](std::size_t idx)
              { return val; }) {}

    /// @brief Create a new vector of the given size, initializing each element with
    /// the given function being fed the index of the vector to initialize
    /// @param size Number of elements to create in the new vector
    /// @param fn Function used to initialize each new element of the vector
    vec(std::size_t size, std::function<T(std::size_t)> fn)
        : vec(size)
    {
        T *ptr = this->_vec;
        for (std::size_t idx = 0; idx < this->_size; ++idx, ++ptr)
        {
            *ptr = fn(idx);
        }
    }

    /// @brief Class destructor which takes care of deallocating content associated
    /// with the vector class instance.  If this vector has been the source of a move
    /// it will have nullptr as its value, and should not deallocate the array.
    /// Otherwise, deallocate the array.
    virtual ~vec()
    {
        if (this->_vec != nullptr)
        {
            delete[] this->_vec;
        }
    }

    /// @brief Retrieve the number of elements in the vector
    /// @return The number of elements in the vector
    virtual inline std::size_t size() const { return this->_size; }

    /// @brief Retrieve a reference to the element in the array with the given index
    /// @param idx Index of the element reference to retrieve
    /// @return Reference to the element at the given index, idx.
    virtual inline T &operator[](std::size_t idx) { return this->_vec[idx % this->_size]; }

    /// @brief Retrieve a local copy of the element in the array with the given index
    /// @param idx Index of the element to copy and return to calling routine
    /// @return Copy of the value of the element at given index
    virtual inline T get(std::size_t idx) const { return this->_vec[idx % this->_size]; }

    /// @brief Perform a map from one vector to another bystransforming the
    /// source vector to the destination using the given function applied
    /// on each element of the source vector
    /// @tparam S Type of elements stored in the source vector
    /// @param src Source vector
    /// @param fn Function used to transform source vector into destination vector
    /// @return The source function transformed on an elementwise basis to produce
    /// the destination vector
    template <typename S>
    static vec<T> map(const vec<S> &src, std::function<T(S, std::size_t)> fn)
    {
        return vec<T>(src, fn);
    }

    /// @brief Perform a transform on two source vectors to produce a new one
    /// @tparam L Type of data contained in the left most of the source vectors
    /// @tparam R Type of data contained in the right most of the source vectors
    /// @param l_src Left vector to contribute to the new vector
    /// @param r_src Right vector to contribute to the new vector
    /// @param fn Function operating elementwise on each of the left and right
    /// vectors to produce each element of the resulting vector
    /// @return New vector which is the combining of the two source vectors when the
    /// goven function is applied to each.
    template <typename L, typename R>
    static vec<T> op(
        const vec<L> &l_src,
        const vec<R> &r_src,
        std::function<T(L, R, std::size_t)> fn)
    {
        return vec<T>(l_src, r_src, fn);
    }

    /// @brief Perform a reduce operation on the vector to collapse
    /// the entire vector into a single value of type T.
    /// @param fn Function used to operate on each element successively to accumulate
    /// the single value the reduce function will return
    /// @return Reduced value of the vector when the given function is applied
    virtual T reduce(std::function<T(T, T)> fn) const
    {
        T init = *(this->_vec);
        const T *start = this->_vec + 1;
        const T *end = this->_vec + this->_size;
        return this->_size > 1 ? std::accumulate(start, end, init, fn) : init;
    }

    /// @brief Produce and return a string representation of the vector
    /// @return A string representation of the vector
    virtual std::string to_string() const
    {
        std::stringstream ss;

        char delim = '[';
        for (T *ptr = this->_vec; ptr != this->_vec + this->_size; ++ptr)
        {
            ss << delim << *ptr;
            delim = ',';
        }

        ss << ']';

        return ss.str();
    }

    /// @brief This is used so as to allow vectors of other base types
    /// internal access into the private members and methods of this
    /// class
    /// @tparam S Type of elements the friend vector contains
    template <typename S>
    friend class vec;
};

/// @brief Compute and return the element-wise sum of two vectors
/// @tparam T Type of elements the two vectors contain
/// @param x Left operand of the vector addition operation
/// @param y Right operand of the vector addition operation
/// @return Element-wise sum of the two vectors
template <typename T>
vec<T> operator+(const vec<T> &x, const vec<T> &y)
{
    return vec<T>::template op<T, T>(x, y, [](const T &l, const T &r, std::size_t idx) -> T
                                     { return l + r; });
}

/// @brief Add a scalar value to each element of a vector
/// @tparam T Type of the scalar and each vector element
/// @param x Scalar value to serve as the left operand for each vector element
/// @param y Vector containing the right hand operands for the operation
/// @return The sum of the given scalar added to each vector element
template <typename T>
vec<T> operator+(const T &x, const vec<T> &y)
{
    return vec<T>::template map<T>(y, [&x](const T &r, std::size_t idx) -> T
                                   { return x + r; });
}

/// @brief Add a scalar value to each element of a vector
/// @tparam T Type of the scalar and each vector element
/// @param x Vector containing the left hand operands for the operation
/// @param y Scalar value to serve as the right operand for each vector element
/// @return The sum of the given scalar added to each vector element
template <typename T>
vec<T> operator+(const vec<T> &x, const T &y)
{
    return vec<T>::template map<T>(x, [&y](const T &l, std::size_t idx) -> T
                                   { return l + y; });
}

/// @brief Compute and return the element-wise difference of two vectors
/// @tparam T Type of elements the two vectors contain
/// @param x Left operand of the vector subtraction operation
/// @param y Right operand of the vector subtraction operation
/// @return Element-wise difference of the two vectors
template <typename T>
vec<T> operator-(const vec<T> &x, const vec<T> &y)
{
    return vec<T>::template op<T, T>(x, y, [](const T &l, const T &r, std::size_t idx) -> T
                                     { return l - r; });
}

/// @brief Subtract from a scalar value each element of a vector
/// @tparam T Type of the scalar and each vector element
/// @param x Scalar value to serve as the left operand for each vector element
/// @param y Vector containing the right hand operands for the operation
/// @return The difference each vector element subtracted from the scalar
template <typename T>
vec<T> operator-(const T &x, const vec<T> &y)
{
    return vec<T>::template map<T>(y, [&x](const T &r, std::size_t idx) -> T
                                   { return x - r; });
}

/// @brief Subtract a scalar value from each element of a vector
/// @tparam T Type of the scalar and each vector element
/// @param x Vector containing the left hand operands for the operation
/// @param y Scalar value to serve as the right operand for each vector element
/// @return The element-wise difference of the given scalar and each vector element
template <typename T>
vec<T> operator-(const vec<T> &x, const T &y)
{
    return vec<T>::template map<T>(x, [&y](const T &l, std::size_t idx) -> T
                                   { return l - y; });
}

/// @brief Compute and return the element-wise product of two vectors
/// @tparam T Type of elements the two vectors contain
/// @param x Left operand of the vector multiplication operation
/// @param y Right operand of the vector multiplication operation
/// @return Element-wise product of the two vectors
template <typename T>
vec<T> operator*(const vec<T> &x, const vec<T> &y)
{
    return vec<T>::template op<T, T>(x, y, [](const T &l, const T &r, std::size_t idx) -> T
                                     { return l * r; });
}

/// @brief Multiply a scalar value by each element of a vector
/// @tparam T Type of the scalar and each vector element
/// @param x Scalar value to serve as the left operand for each vector element
/// @param y Vector containing the right hand operands for the operation
/// @return The product of the scalar and each vector element
template <typename T>
vec<T> operator*(const T &x, const vec<T> &y)
{
    return vec<T>::template map<T>(y, [&x](const T &r, std::size_t idx) -> T
                                   { return x * r; });
}

/// @brief Multiply a scalar value by each element of a vector
/// @tparam T Type of the scalar and each vector element
/// @param x Vector containing the left hand operands for the operation
/// @param y Scalar value to serve as the right operand for each vector element
/// @return The element-wise product of the given scalar and each vector element
template <typename T>
vec<T> operator*(const vec<T> &x, const T &y)
{
    return vec<T>::template map<T>(x, [&y](const T &l, std::size_t idx) -> T
                                   { return l * y; });
}

/// @brief Compute and return the element-wise quotient of two vectors
/// @tparam T Type of elements the two vectors contain
/// @param x Left operand of the vector division operation
/// @param y Right operand of the vector division operation
/// @return Element-wise quotient of the two vectors
template <typename T>
vec<T> operator/(const vec<T> &x, const vec<T> &y)
{
    return vec<T>::template op<T, T>(x, y, [](const T &l, const T &r, std::size_t idx) -> T
                                     { return l / r; });
}

/// @brief Divide a scalar value by each element of a vector
/// @tparam T Type of the scalar and each vector element
/// @param x Scalar value to serve as the left operand for each vector element
/// @param y Vector containing the right hand operands for the operation
/// @return The quotent of the scalar and each vector element
template <typename T>
vec<T> operator/(const T &x, const vec<T> &y)
{
    return vec<T>::template map<T>(y, [&x](const T &r, std::size_t idx) -> T
                                   { return x / r; });
}

/// @brief Divide each element if a vector by a scalar value
/// @tparam T Type of the scalar and each vector element
/// @param x Vector containing the left hand operands for the operation
/// @param y Scalar value to serve as the right operand for each vector element
/// @return The the vector divided element-wise by a scalar value
template <typename T>
vec<T> operator/(const vec<T> &x, const T &y)
{
    return vec<T>::template map<T>(x, [&y](const T &l, std::size_t idx) -> T
                                   { return l / y; });
}

/// @brief Compare two vectors and return true if they are equal
/// @tparam T Type of each element of the two arrays
/// @param x Right vector to compare
/// @param y Left vector to compare
/// @return True if and only iff the two vectors are equal
template <typename T>
bool operator==(const vec<T> &x, const vec<T> &y)
{
    vec<bool> cmp = vec<bool>::template op<T, T>(x, y, [](const T &l, const T &r, std::size_t idx) -> bool
                                                 { return l == r; });

    return cmp.reduce([](bool l, bool r) -> bool
                      { return l && r; });
}

/// @brief Compare two vectors and return true if they are not equal
/// @tparam T Type of each element of the two arrays
/// @param x Right vector to compare
/// @param y Left vector to compare
/// @return True if and only iff the two vectors are not equal
template <typename T>
bool operator!=(const vec<T> &x, const vec<T> &y)
{
    return !(x == y);
}

/// @brief Return the sum of all the elements of the given array
/// @tparam T Type of elements in the given vector and sum
/// @param x Vector to sum
/// @return Sum of the vector elements
template <typename T>
T sum(const vec<T> &x)
{
    return x.reduce([](const T &l, const T &r) -> T
                    { return l + r; });
}

/// @brief Return the product of all the elements of the given array
/// @tparam T Type of elements in the given vector and multiply
/// @param x Vector to multiply
/// @return Product of the vector elements
template <typename T>
T prod(const vec<T> &x)
{
    return x.reduce([](const T &l, const T &r) -> T
                    { return l * r; });
}

/// @brief Return the inner (i.e. dot) product of the two vectors
/// @tparam T Type of elements in the vectors and the result
/// @param x First vector to use in computing the dot product
/// @param y Second vector to use in computing the dot product
/// @return Dot product of the two vectors given
template <typename T>
T dot(const vec<T> &x, const vec<T> &y)
{
    return sum(x * y);
}

/// @brief Return the square of the magnitude the given vector
/// @tparam T Type of each element in the vector and the result
/// @param x Vector to compute the square of the magnitude
/// @return Square of the magnitude of the given vector
template <typename T>
T mag2(const vec<T> &x)
{
    return dot(x, x);
}

/// @brief Return the magnitude of the given vector
/// @tparam T Type of the elements in the vector and the result
/// @param x Vector to compute the magnitude of
/// @return Magnitude of the given vector
template <typename T>
T mag(const vec<T> &x)
{
    return sqrt(mag2(x));
}

/// @brief Stream the string representation of the vector to the given output stream
/// @tparam T Type of each element in the vector
/// @param os Output stream where the vector will be sent
/// @param vec Vector to send the output stream
/// @return Output stream where the vector was sent
template <typename T>
std::ostream &operator<<(std::ostream &os, const vec<T> vec)
{
    os << vec.to_string();
    return os;
}

#endif