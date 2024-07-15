#ifndef VEC_CUH
#define VEC_CUH

#include <string>
#include <sstream>
#include <iostream>
#include <cassert>
#include <cmath>

/// @brief Return the ceiling of the unsigned integer division when the
/// numerator is divided by the denominator
/// @param n Numerator
/// @param d Divisor
/// @return Ceiling of the unsigned integer division n / d.
std::size_t div_ceil(std::size_t n, std::size_t d)
{
    return (n / d) + (n % d > 0);
}

/// @brief Similar to div_ceil, but optimized for the case where the
/// divisor is 2.
/// @param n Numerator
/// @return Ceiling of n / 2
std::size_t half_ceil(std::size_t n)
{
    return (n / 2) + (n % 2);
}

/// @brief Global function to use a function to initialize the elements of a
/// kernel array to a given value
/// @tparam T Type of values the resulting array will contain
/// @param size Number of elements in the array to initialize
/// @param d_res Array in the device kernel to be initialized
/// @param val Fixed value to assign to each element of the result array
template <typename T>
__global__ void cuda_val_init(std::size_t size, T *d_res, T val)
{
    std::size_t idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < size)
    {
        d_res[idx] = val;
    }
}

/// @brief Global function to use a function to initialize the elements of a
/// kernel array.
/// @tparam T Type of values the resulting array will contain
/// @tparam FUNC Function type transforming the index of the array element into
// a value for that element.
/// @param size Number of elements in the array to initialize
/// @param d_res Array in the device kernel to be initialized
/// @param fn Function to map the array index to a value to assign the
/// corresponding array element.
template <typename T, typename FUNC>
__global__ void cuda_fn_init(std::size_t size, T *d_res, FUNC fn)
{
    std::size_t idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < size)
    {
        d_res[idx] = fn(idx);
    }
}

/// @brief Global function to use a function to map the values of
/// one array onto another array
/// @tparam T Type of values the resulting array will contain
/// @tparam S Type of values the source array contains
/// @tparam FUNC Function type transforming the index of the array element into
// a value for that element.
/// @param size Number of elements in the array to initialize
/// @param d_res Array in the device kernel to be initialized
/// @param d_src Array containing elements to map to the destination array
/// @param fn Function to map the array index to a value to assign the
/// corresponding array element.
template <typename T, typename S, typename FUNC>
__global__ void cuda_map(std::size_t size, T *d_res, const S *d_src, FUNC fn)
{
    std::size_t idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < size)
    {
        d_res[idx] = fn(d_src[idx], idx);
    }
}

/// @brief Global function which performs an operation on two arrays in the
/// kernel and places the results into another array in the kernel.
/// @tparam T Type of the resulting values when the operation is performed on
/// the operands
/// @tparam L Type of the "left" side operand
/// @tparam R Type of the "right" side operand
/// @tparam FUNC function while returns an argument of type T and takes
/// arguments of type L, R, std::size_t in that order
/// @param size Number of elements in each of the arrays given
/// @param d_res Array in the device kernel where the results will be placed
/// @param d_x Array in the device kernel where the "left" operands are
/// located
/// @param d_y Array in the device kernel where the "right" operands are
/// located
/// @param fn Function to map values of types L and R into type T in
/// conjunction with the array index
template <typename T, typename L, typename R, typename FUNC>
__global__ void cuda_op(
    std::size_t size,
    T *d_res,
    const L *d_x,
    const R *d_y,
    FUNC fn)
{
    std::size_t idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < size)
    {
        d_res[idx] = fn(d_x[idx], d_y[idx], idx);
    }
}

/// @brief Global function which performs a single step in a reduce operation
/// on a function.  A single call to this function will perform the operation
/// on elements i and i+s where s is half the size given.  The result will be
/// stored in element i.
/// @tparam T Type of values in the array
/// @tparam FUNC Function type taking two arguments ot type T and returning a
/// single value of type T.  This function should be commutative.
/// @param size Number of elements in the array to reduce
/// @param d_res Array where the results of this step will be held
/// @param d_src Array where the source arguments are stored
/// @param fn Function to map two arguments of type T to a single argument
/// of type T. The function fn should be commutative
template <typename T, typename FUNC>
__global__ void cuda_step(std::size_t size, T *d_res, const T *d_src, FUNC fn)
{
    std::size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    std::size_t span = (size / 2) + (size % 2);

    if (idx < span)
    {
        d_res[idx] = ((idx + span) < size) ? fn(d_src[idx], d_src[idx + span]) : d_src[idx];
    }
}

/// @brief Vector class used for map / reduct functions
/// @tparam T Type of element the vector will contain
template <typename T>
class vec
{
private:
    /// @brief Size of the CUDA block to use for these operations
    const static std::size_t THREADS_PER_BLOCK = 256;

    /// @brief Number of elements the vector contains
    const std::size_t _size;

    /// @brief Number of blocks neede to process this vector
    const std::size_t BLOCK_COUNT;

    /// @brief Array dedicated to hold the value of the array on the host
    /// The authoritative value for the array is the one on the device
    T *_vec;

    /// @brief Array dedicated to hold the value of the array on the device
    /// The device vector is considered to hold the authoritative value of the
    /// vector
    T *_d_vec;

    /// @brief Base constructor for most of the others which takes a size
    /// and initializes all of the elements to a preliminary value.
    /// @param size Size used to initialize the member variables of the
    /// class instance.
    vec(std::size_t size)
        : _size(size),
          BLOCK_COUNT(div_ceil(size, THREADS_PER_BLOCK)),
          _vec(nullptr),
          _d_vec(nullptr)
    {
        assert(this->_size > 0);
        this->_vec = new T[size];
        cudaMalloc(&(this->_d_vec), size * sizeof(T));
    }

    /// @brief Private constuctor used to map the elements of another
    /// vector into the new one using the given function
    /// @tparam S Type of elements in the source vector
    /// @param src Source vector to transform to populuate this new one
    /// @param fn Function to map elements from source vector into this new one
    template <typename S, typename FUNC>
    vec(const vec<S> &src, FUNC fn)
        : vec(src._size)
    {
        cuda_map<<<this->BLOCK_COUNT, THREADS_PER_BLOCK>>>(this->_size, this->_d_vec, src._d_vec, fn);
    }

    /// @brief Private constructor used to perform an operation on two source vectors
    /// to produce a new vector of the same size as the two source vectors.
    /// @tparam L Type of element in the vector on the left side of the operand
    /// @tparam R Type of element in the vector on the right side of the operand
    /// @param l_src Left vector to combine with to produce the new vector
    /// @param r_src Right vector to combine with to produce the new vector
    /// @param fn Function to combine the left and right elements to compute each element of the new vector
    template <typename L, typename R, typename FUNC>
    vec(const vec<L> &l_src, const vec<R> &r_src, FUNC fn)
        : _size(l_src._size),
          BLOCK_COUNT(div_ceil(l_src._size, THREADS_PER_BLOCK)),
          _vec(nullptr),
          _d_vec(nullptr)
    {
        assert(l_src._size == r_src._size);
        this->_vec = new T[l_src._size];
        cudaMalloc(&(this->_d_vec), l_src._size * sizeof(T));

        cuda_op<<<this->BLOCK_COUNT, THREADS_PER_BLOCK>>>(this->_size, this->_d_vec, l_src._d_vec, r_src._d_vec, fn);
    }

public:
    /// @brief Initialize a new vector to have length size with each element having
    /// the given value, val.  If omitted, the value will be the default value
    /// of type T... for numeric content it should be 0
    /// @param size Size of the new vector
    /// @param val Value to set each element of the vector
    vec(std::size_t size, T val)
        : vec(size)
    {
        cuda_val_init<<<this->BLOCK_COUNT, THREADS_PER_BLOCK>>>(size, this->_d_vec, val);
    }

    /// @brief Constructor which uses an initializer list to create the new vector
    /// @param src Source initializer list used to populate the new vector
    vec(const std::initializer_list<T> &src)
        : vec(src.size())
    {
        std::copy(src.begin(), src.end(), this->_vec);
        this->to_device();
    }

    /// @brief Copy constructor which copies the elements of the source vector into
    /// this new one
    /// @param src Source vector used to populate this new one
    vec(const vec<T> &src)
        : vec(src._size)
    {
        cudaMemcpy(this->_d_vec, src._d_vec, this->_size * sizeof(T), cudaMemcpyDeviceToDevice);
    }

    /// @brief Move constructor to improve performance
    /// @param src Source vector contrubuting the internal array
    vec(vec<T> &&src)
        : _size(src._size),
          _vec(src._vec),
          _d_vec(src._d_vec),
          BLOCK_COUNT(src.BLOCK_COUNT)
    {
        src._vec = nullptr;
        src._d_vec = nullptr;
    }

    /// @brief Create a new vector of the given size, initializing each element with
    /// the given function being fed the index of the vector to initialize
    /// @tparam FUNC Function type used to initialize the vector
    /// @param size Number of elements to create in the new vector
    /// @param fn Function used to initialize each new element of the vector
    template <typename FUNC>
    vec(std::size_t size, FUNC fn)
        : vec(size)
    {
        cuda_fn_init<<<this->BLOCK_COUNT, THREADS_PER_BLOCK>>>(size, this->_d_vec, fn);
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
            this->_vec = nullptr;
        }

        if (this->_d_vec != nullptr)
        {
            cudaFree(this->_d_vec);
            this->_d_vec = nullptr;
        }
    }

    /// @brief Copy the content in the device kernel to the host
    virtual inline void to_host() { cudaMemcpy(this->_vec, this->_d_vec, this->_size * sizeof(T), cudaMemcpyDeviceToHost); }

    /// @brief Copy the content of the host to the device kernel
    virtual inline void to_device() { cudaMemcpy(this->_d_vec, this->_vec, this->_size * sizeof(T), cudaMemcpyHostToDevice); }

    /// @brief Retrieve the number of elements in the vector
    /// @return The number of elements in the vector
    virtual inline std::size_t size() const { return this->_size; }

    /// @brief Retrieve a reference to the element in the array with the given index
    /// @param idx Index of the element reference to retrieve
    /// @return Reference to the element at the given index, idx.
    virtual T &operator[](std::size_t idx) { return this->_vec[idx % this->_size]; }

    /// @brief Retrieve a local copy of the element in the array with the given index
    /// @param idx Index of the element to copy and return to calling routine
    /// @return Copy of the value of the element at given index
    virtual T get(std::size_t idx) const { return this->_vec[idx % this->_size]; }

    /// @brief Perform a map from one vector to another by transforming the
    /// source vector to the destination using the given function applied
    /// on each element of the source vector
    /// @tparam S Type of elements stored in the source vector
    /// @tparam FUNC Type of function used to map source vector onto the new vector
    /// @param src Source vector
    /// @param fn Function used to transform source vector into destination vector
    /// @return The source function transformed on an elementwise basis to produce
    /// the destination vector
    template <typename S, typename FUNC>
    static vec<T> map(const vec<S> &src, FUNC fn)
    {
        return vec<T>(src, fn);
    }

    /// @brief Perform a transform on two source vectors to produce a new one
    /// @tparam L Type of data contained in the left most of the source vectors
    /// @tparam R Type of data contained in the right most of the source vectors
    /// @tparam FUNC Type of function to operate on two vector elements to
    /// create a corresponding value in the resulting vector
    /// @param l_src Left vector to contribute to the new vector
    /// @param r_src Right vector to contribute to the new vector
    /// @param fn Function operating elementwise on each of the left and right
    /// vectors to produce each element of the resulting vector
    /// @return New vector which is the combining of the two source vectors when the
    /// goven function is applied to each.
    template <typename L, typename R, typename FUNC>
    static vec<T> op(
        const vec<L> &l_src,
        const vec<R> &r_src,
        FUNC fn)
    {
        return vec<T>(l_src, r_src, fn);
    }

    /// @brief Perform a reduce operation on the vector to collapse
    /// the entire vector into a single value of type T.
    /// @tparam FUNC Type of function used to perform the reduce.  It should
    /// take two arguments and return a single argument of type T and should
    /// be commutative
    /// @param fn Function used to operate on each element successively to
    /// reduce the values in the array
    /// @return Reduced value of the vector when the given function is applied
    template <typename FUNC>
    T reduce(FUNC fn) const
    {
        std::size_t size = this->_size;
        std::size_t next_size = half_ceil(size);
        T *d_res;
        bool first = true;

        cudaMalloc(&d_res, next_size * sizeof(T));

        while (size > 1)
        {
            T *d_tmp = first ? this->_d_vec : d_res;
            std::size_t num_blocks = div_ceil(size, THREADS_PER_BLOCK);

            cuda_step<<<this->BLOCK_COUNT, THREADS_PER_BLOCK>>>(size, d_res, d_tmp, fn);
            size = next_size;
            next_size = half_ceil(size);
            first = false;
        }

        T h_res;
        cudaMemcpy(&h_res, d_res, sizeof(T), cudaMemcpyDeviceToHost);
        cudaFree(d_res);

        return h_res;
    }

    /// @brief Produce and return a string representation of the vector
    /// @return A string representation of the vector
    virtual std::string to_string() const
    {
        std::stringstream ss;
        T *h_ptr = new T[this->_size];
        cudaMemcpy(h_ptr, this->_d_vec, this->_size * sizeof(T), cudaMemcpyDeviceToHost);

        char delim = '[';
        for (T *ptr = h_ptr; ptr != h_ptr + this->_size; ++ptr)
        {
            ss << delim << *ptr;
            delim = ',';
        }

        ss << ']';

        delete[] h_ptr;
        return ss.str();
    }

    /// @brief This is used so as to allow vectors of other base types
    /// internal access into the private members and methods of this
    /// class
    /// @tparam S Type of elements the friend vector contains
    template <typename S>
    friend class vec;
};

/// @brief Return the sum of all the elements of the given array
/// @tparam T Type of elements in the given vector and sum
/// @param x Vector to sum
/// @return Sum of the vector elements
template <typename T>
T sum(const vec<T> &x)
{
    return x.reduce([] __device__(const T &l, const T &r) -> T
                    { return l + r; });
}

/// @brief Return the product of all the elements of the given array
/// @tparam T Type of elements in the given vector and multiply
/// @param x Vector to multiply
/// @return Product of the vector elements
template <typename T>
T prod(const vec<T> &x)
{
    return x.reduce([] __device__(const T &l, const T &r) -> T
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

/// @brief Compute and return the element-wise sum of two vectors
/// @tparam T Type of elements the two vectors contain
/// @param x Left operand of the vector addition operation
/// @param y Right operand of the vector addition operation
/// @return Element-wise sum of the two vectors
template <typename T>
vec<T> operator+(const vec<T> &x, const vec<T> &y)
{
    return vec<T>::template op<T, T>(x, y, [] __device__(const T &l, const T &r, std::size_t idx) -> T
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
    return vec<T>::template map<T>(y, [x] __device__(const T &r, std::size_t idx) -> T
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
    return vec<T>::template map<T>(x, [y] __device__(const T &l, std::size_t idx) -> T
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
    return vec<T>::template op<T, T>(x, y, [] __device__(const T &l, const T &r, std::size_t idx) -> T
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
    return vec<T>::template map<T>(y, [x] __device__(const T &r, std::size_t idx) -> T
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
    return vec<T>::template map<T>(x, [y] __device__(const T &l, std::size_t idx) -> T
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
    return vec<T>::template op<T, T>(x, y, [] __device__(const T &l, const T &r, std::size_t idx) -> T
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
    return vec<T>::template map<T>(y, [x] __device__(const T &r, std::size_t idx) -> T
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
    return vec<T>::template map<T>(x, [y] __device__(const T &l, std::size_t idx) -> T
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
    return vec<T>::template op<T, T>(x, y, [] __device__(const T &l, const T &r, std::size_t idx) -> T
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
    return vec<T>::template map<T>(y, [x] __device__(const T &r, std::size_t idx) -> T
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
    return vec<T>::template map<T>(x, [y] __device__(const T &l, std::size_t idx) -> T
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
    vec<bool> cmp = vec<bool>::template op<T, T>(x, y, [] __device__(const T &l, const T &r, std::size_t idx) -> bool
                                                 { return l == r; });

    return cmp.reduce([] __device__(bool l, bool r) -> bool
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

/// @brief Stream the string representation of the vector to the given output stream
/// @tparam T Type of each element in the vector
/// @param os Output stream where the vector will be sent
/// @param vec Vector to send the output stream
/// @return Output stream where the vector was sent
template <typename T>
std::ostream &operator<<(std::ostream &os, const vec<T> &vec)
{
    os << vec.to_string();
    return os;
}

#endif