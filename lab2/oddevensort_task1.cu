#include <vector>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <random>
#define DEBUG false
#define THREADS 1024
#define SIZE 100000
using namespace std;

// The odd-even sort algorithm
// Total number of odd phases + even phases = the number of elements to sort

//Swap function since we can't use std::swap in device code
__device__ void swap_numbers(int* a, int* b)
{
    int tmp = *a;
    *a = *b;
    *b = tmp;
}

__global__ void oddeven_sort_kernel(int* numbers, int s)
{

    int odd_even;
    int start_index = threadIdx.x*2;
    //Performing the sort looping trough all phases
    for (int i = 1; i <= s; i++) {

        //Calculating if the phase is odd or even
        odd_even = i %2;
        for (int j = start_index+odd_even; j < s-1; j = j + THREADS) {
            if (numbers[j] > numbers[j + 1]) {
                swap_numbers(&numbers[j], &numbers[j + 1]);
            }
        }
        //wait for all threads to finish
        __syncthreads();
    }
}

void oddeven_sort(std::vector<int>& numbers)
{
    auto s = numbers.size();
    int* device_numbers;

    //Allocating memory on the device
    cudaMalloc(&device_numbers, s * sizeof(int));
    cudaMemcpy(device_numbers, numbers.data(), s * sizeof(int), cudaMemcpyHostToDevice);

    //Starting the calculation in one block
    oddeven_sort_kernel<<<1, THREADS>>>(device_numbers, s);

    //Copying the sorted array back to the host and freeing memory on the device
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
    constexpr unsigned int size = SIZE; // Number of elements in the input
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