/***************************************************************************
 *
 * Paralell version of task 2, Odd-Even sort
 *
 * Author: Samuel Nyberg, sany21@student.bth.se
 * Author: Tobias Mattsson, tomt21@student.bth.se
 *
 ***************************************************************************/

#include <vector>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <random>
#define DEBUG false
#define SIZE 100000
#define THREADSPERBLOCK 1024
#define BLOCKS (SIZE+THREADSPERBLOCK-1)/(2*THREADSPERBLOCK)
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

__global__ void oddeven_sort_kernel(int* numbers, int s, int i)
{
    //Calculating if the phase is odd or even
    int odd_even = i % 2;

    //Calculating wich index the thread is at a global level
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    //Calculating the index of the next element to compare
    int j = 2 * index + odd_even;
    if (j < s - 1 && numbers[j] > numbers[j + 1]) {
        swap_numbers(&numbers[j], &numbers[j + 1]);
    }
}

void oddeven_sort(std::vector<int>& numbers)
{
    auto s = numbers.size();
    int* device_numbers;

    //Allocating memory on the device
    cudaMalloc(&device_numbers, s * sizeof(int));
    cudaMemcpy(device_numbers, numbers.data(), s * sizeof(int), cudaMemcpyHostToDevice);

    //Performing the sort looping trough all phases, doing this on the host side to sync the kernels
    for (int i = 1; i <= s; i++) {
        oddeven_sort_kernel<<<BLOCKS, THREADSPERBLOCK>>>(device_numbers, s, i);
    }

    //Copying the sorted array back to the host and freeing memory on the device
    cudaMemcpy(numbers.data(), device_numbers, s * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(device_numbers);
}

void print_sort_status(std::vector<int> numbers)
{
    std::cout << "The input is sorted?: " << (std::is_sorted(numbers.begin(), numbers.end()) == 0 ? "False" : "True") << std::endl;
}

void print_number(std::vector<int> numbers)
//printfunction used to debugg code
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