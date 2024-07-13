#include <string>
#include <vector>
#include <iostream>
#include <regex>

void show_help(const std::string &cli)
{
    std::cout << "This explores what happens when copying memory to or from a CUDA device." << std::endl;
    std::cout << "If the sizes aren't matched, it's likely bad things happens, but it's not" << std::endl;
    std::cout << "what those bad things are.  This explores the badness." << std::endl;
    std::cout << std::endl;
    std::cout << "usage: " << cli << " <h_size> <d_size> <h_to_d_size> <d_to_h_size>" << std::endl;
    std::cout << std::endl;
    std::cout << "    h_size - number of ints to allocate on host." << std::endl;
    std::cout << "    d_size - number of ints to allocate on device." << std::endl;
    std::cout << "    h_to_d_size - number of ints to copy from host to device." << std::endl;
    std::cout << "    d_to_h_size - number of ints to copy from device to host." << std::endl;
    std::cout << std::endl;
}

int main(int argc, char *argv[])
{
    std::string cli = argv[0];
    std::regex pattern("^\\d+$");

    if (argc != 5)
    {
        show_help(cli);
        return -1;
    }

    std::vector<std::string> args(argv + 1, argv + argc);
    std::vector<std::size_t> vals;

    for (auto arg : args)
    {
        if (!std::regex_match(arg, pattern))
        {
            show_help(cli);
            return -1;
        }
        vals.push_back(std::stoi(arg));
    }

    const std::size_t h_size = vals[0];
    const std::size_t d_size = vals[1];
    const std::size_t h_to_d_size = vals[2];
    const std::size_t d_to_h_size = vals[3];

    std::cout << "allocating host array of " << h_size << " ints..." << std::endl;
    int *h_vec = new int[h_size];
    std::cout << "success\ninitializing host array..." << std::endl;
    for (std::size_t i = 0; i < h_size; ++i)
    {
        h_vec[i] = 0;
    }

    int *d_vec;
    std::cout << "success\nallocating device array of " << d_size << " ints." << std::endl;
    cudaMalloc(&d_vec, d_size);

    std::cout << "success\ncopying " << h_to_d_size << " ints from host to device..." << std::endl;
    cudaMemcpy(d_vec, h_vec, h_to_d_size * sizeof(int), cudaMemcpyHostToDevice);

    std::cout << "success\ncopying " << d_to_h_size << " ints from device to host..." << std::endl;
    cudaMemcpy(h_vec, d_vec, d_to_h_size * sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "success" << std::endl;

    int h_fixed[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    std::cout << "success\ncopying " << h_to_d_size << " ints from host (stack size 10) to device..." << std::endl;
    cudaMemcpy(d_vec, h_fixed, h_to_d_size * sizeof(int), cudaMemcpyHostToDevice);

    std::cout << "success\ncopying " << d_to_h_size << " ints from device to host (stack size 10)..." << std::endl;
    cudaMemcpy(h_fixed, d_vec, d_to_h_size * sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "success" << std::endl;
}