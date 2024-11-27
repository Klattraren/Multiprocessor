#include <vector>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <random>
using namespace std;
// The odd-even sort algorithm
// Total number of odd phases + even phases = the number of elements to sort
__device__ void swap_numbers(int* a, int* b)
{
    int tmp = *a;
    *a = *b;
    *b = tmp;
}

__global__ void oddeven_sort_kernel(int* numbers, int s)
{
    // printf("ThreadIdx: %d\n", threadIdx.x);
    int odd_even;
    for (int i = 1; i <= s; i++) {
        odd_even = i %2;
        for (int j = threadIdx.x*2+odd_even; j < s-1; j = j + 2048) {
            if (numbers[j] > numbers[j + 1]) {
                swap_numbers(&numbers[j], &numbers[j + 1]);
            }
        }
        __syncthreads();
    }
}

void oddeven_sort(std::vector<int>& numbers)
{
    auto s = numbers.size();
    int* device_numbers;

    cudaMalloc(&device_numbers, s * sizeof(int));
    cudaMemcpy(device_numbers, numbers.data(), s * sizeof(int), cudaMemcpyHostToDevice);

    oddeven_sort_kernel<<<1, 2048>>>(device_numbers, s);

    cudaMemcpy(numbers.data(), device_numbers, s * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(device_numbers);
}

void print_sort_status(std::vector<int> numbers)
{
    std::cout << "The input is sorted?: " << (std::is_sorted(numbers.begin(), numbers.end()) == 0 ? "False" : "True") << std::endl;
}

void print_number(std::vector<int> numbers)
{
    for (auto number : numbers)
    {
        std::cout << number << " ";
    }
    std::cout << std::endl;
}

int main()
{
    constexpr unsigned int size = 100000; // Number of elements in the input
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> distrib(0, 100000);

    // Initialize a vector with integers of value 0
    std::vector<int> numbers(size);
    // Populate our vector with (pseudo)random numbers
    srand(time(0));
    std::generate(numbers.begin(), numbers.end(), rand);

    // for (int i = 0; i < size; i++)
    // {
    //     numbers[i] = distrib(gen);
    // }
    // print_number(numbers);
    print_sort_status(numbers);
    auto start = std::chrono::steady_clock::now();
    oddeven_sort(numbers);
    auto end = std::chrono::steady_clock::now();
    // print_number(numbers);
    print_sort_status(numbers);
    std::cout << "Elapsed time =  " << std::chrono::duration<double>(end - start).count() << " sec\n";
}