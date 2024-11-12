#include <stdio.h>
#include <math.h>

// int calculate_levels(int n_threads) {
//     return (int)ceil(log2(n_threads + 1));
// }
#define THREADS 10

int test()
{
    double num = 5.6, result;
    int x2 = 6;
    result = log2(THREADS+1);
    printf("log(%.1f) = %.2f", num, result);
}

int main()
{
    test(6);

    return 0;
}