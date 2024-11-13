#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdbool.h>

#define KILO (1024)
#define MEGA (1024*1024)
#define MAX_ITEMS (64*MEGA)
#define swap(v, a, b) {unsigned tmp; tmp=v[a]; v[a]=v[b]; v[b]=tmp;}
#define AMOUNT_THREADS 2
#define MAX_LEVELS ((int)ceil(log2(AMOUNT_THREADS + 1)) - 1)

static int *v;

// Thread argument struct
typedef struct ThreadArgs {
    int *v;
    unsigned int low;
    unsigned int high;
    unsigned int lvl;
} ThreadArgs;

pthread_t threads[AMOUNT_THREADS - 1];
int threads_left = AMOUNT_THREADS - 1;

static void init_array(void) {
    v = (int *)malloc(MAX_ITEMS * sizeof(int));
    for (int i = 0; i < MAX_ITEMS; i++)
        v[i] = rand();
}

static unsigned partition(int *v, unsigned low, unsigned high, unsigned pivot_index) {
    if (pivot_index != low) swap(v, low, pivot_index);
    pivot_index = low;
    low++;

    while (low <= high) {
        if (v[low] <= v[pivot_index])
            low++;
        else if (v[high] > v[pivot_index])
            high--;
        else
            swap(v, low, high);
    }

    if (high != pivot_index) swap(v, pivot_index, high);
    return high;
}

static void *quick_sort(void *arg) {
    ThreadArgs *args = (ThreadArgs *)arg;
    unsigned low = args->low, high = args->high;
    int *v = args->v;

    if (low >= high) return NULL;

    unsigned pivot_index = partition(v, low, high, (low + high) / 2);

    ThreadArgs argsleft = {v, low, pivot_index - 1, args->lvl + 1};
    ThreadArgs argsright = {v, pivot_index + 1, high, args->lvl + 1};

    if (threads_left > 0 && args->lvl < MAX_LEVELS) {
        threads_left--;
        pthread_create(&threads[threads_left], NULL, quick_sort, (void *)&argsleft);
        quick_sort((void *)&argsright);  // Continue sorting right part in the current thread
        pthread_join(threads[threads_left], NULL);  // Wait for the left part to finish
        threads_left++;
    } else {
        quick_sort((void *)&argsleft);
        quick_sort((void *)&argsright);
    }

    return NULL;
}

int main(int argc, char **argv) {
    init_array();
    printf("Max LEVELS: %d\n", MAX_LEVELS);

    ThreadArgs arg = {v, 0, MAX_ITEMS - 1, 0};
    quick_sort(&arg);

    free(v);
    return 0;
}
