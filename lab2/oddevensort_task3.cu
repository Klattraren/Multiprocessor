#include <vector>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <random>
#define DEBUG false
#define THREADS 128
#define BLOCKS 4
#define THREADSPERBLOCK THREADS/BLOCKS
using namespace std;


// The odd-even sort algorithm
// Total number of odd phases + even phases = the number of elements to sort
__device__ void swap_numbers(int* a, int* b)
{
    int tmp = *a;
    *a = *b;
    *b = tmp;
}

__global__ void oddeven_sort_kernel(int* numbers, int s, int i)
{
    int odd_even;
    odd_even = i %2;
    for (int j = ((blockIdx.x+1)*threadIdx.x)*2+odd_even; j < s-1; j = j + THREADSPERBLOCK) {
        if (numbers[j] > numbers[j + 1]) {
            swap_numbers(&numbers[j], &numbers[j + 1]);
        }
    }
}

void oddeven_sort(std::vector<int>& numbers)
{
    auto s = numbers.size();
    int* device_numbers;

    cudaMalloc(&device_numbers, s * sizeof(int));
    cudaMemcpy(device_numbers, numbers.data(), s * sizeof(int), cudaMemcpyHostToDevice);

    for (int i = 1; i <= s; i++) {
        oddeven_sort_kernel<<<BLOCKS, THREADSPERBLOCK>>>(device_numbers, s, i);
    }

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
    // Initialize a vector with integers of value 0
    std::vector<int> numbers(size);
    // Populate our vector with (pseudo)random numbers
    srand(time(0));
    std::generate(numbers.begin(), numbers.end(), rand);

    //Debug mode for the code, setting random number from 0 to 100 for easier readability
    if (DEBUG){
        cout << "DEBUG MODE" << endl;


        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<> distrib(0, 100000);
        for (int i = 0; i < size; i++)
        {
            numbers[i] = distrib(gen);
        }
        print_number(numbers);
    }

    print_sort_status(numbers);
    auto start = std::chrono::steady_clock::now();
    oddeven_sort(numbers);
    auto end = std::chrono::steady_clock::now();

    if (DEBUG){
        print_number(numbers);
    }

    print_sort_status(numbers);
    std::cout << "Elapsed time =  " << std::chrono::duration<double>(end - start).count() << " sec\n";
}