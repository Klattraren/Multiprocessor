#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h> // pthread types and functions
#include <stdbool.h> // Added bools

#define KILO (1024)
#define MEGA (1024*1024)
#define MAX_ITEMS (64*MEGA)
// #define MAX_ITEMS 200
#define swap(v, a, b) {unsigned tmp; tmp=v[a]; v[a]=v[b]; v[b]=tmp;}
#define AMOUNT_THREADS 16
#define MAX_LEVELS (int)ceil(log2(AMOUNT_THREADS + 1))-1

static int *v;

// Creating a struct to pass arguments to the thread
typedef struct ThreadArgs{
    int *v;
    unsigned int low;
    unsigned int high;
    unsigned int t_nr;
    bool threaded;
    unsigned int lvl;
} ThreadArgs;

pthread_t threads[AMOUNT_THREADS-1];
pthread_t thread;

int threads_left = AMOUNT_THREADS-1;
int initiation_step = AMOUNT_THREADS-1;

// Initializing the mutex lock
pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

static void
print_array(void)
{
    int i;
    for (i = 0; i < MAX_ITEMS; i++)
        printf("%d ", v[i]);
    printf("\n");
}

static void
init_array(void)
{
    int i;
    v = (int *) malloc(MAX_ITEMS * sizeof(int));
    for (i = 0; i < MAX_ITEMS; i++)
        v[i] = rand();
}

static unsigned
partition(int *v, unsigned low, unsigned high, unsigned pivot_index)
{
    if (pivot_index != low)
        swap(v, low, pivot_index);

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

    if (high != pivot_index)
        swap(v, pivot_index, high);
    return high;
}

static void
quick_sort(ThreadArgs *arg)
{
    ThreadArgs *argsleft = (ThreadArgs *)malloc(sizeof(ThreadArgs));
    ThreadArgs *argsright = (ThreadArgs *)malloc(sizeof(ThreadArgs));
    unsigned low = arg->low;
    unsigned high = arg->high;
    int *v = arg->v;
    unsigned pivot_index;

    if (low >= high)
        return;

    pivot_index = (low + high) / 2;
    pivot_index = partition(v, low, high, pivot_index);

    int size = high - low;

    bool started_thread = false;
    int thread_started = -1;

    if (low < pivot_index) {
        pthread_mutex_lock(&lock); // Locking the mutex
        if (threads_left > 0 && arg->lvl < MAX_LEVELS) {
            threads_left--;
            pthread_mutex_unlock(&lock); // Unlocking after modifying threads_left
            started_thread = true;
            argsleft->v = v;
            argsleft->low = low;
            argsleft->high = pivot_index - 1;
            argsleft->threaded = true;
            thread_started = threads_left;
            argsleft->t_nr = threads_left;
            argsleft->lvl = arg->lvl + 1;
            printf("\033[0;37mThreads left: %d on level: \033[0;32m %d \033[0;37m with size: %d\n", threads_left, argsleft->lvl, argsleft->high - argsleft->low);
            pthread_create(&thread, NULL, (void *)quick_sort, (void *)argsleft);
        } else {
            pthread_mutex_unlock(&lock); // Unlocking if no thread is created
            argsleft->v = v;
            argsleft->low = low;
            argsleft->high = pivot_index - 1;
            argsleft->threaded = false;
            argsleft->lvl = arg->lvl + 1;
            quick_sort(argsleft);
        }
    }

    if (pivot_index < high) {
        argsright->v = v;
        argsright->low = pivot_index + 1;
        argsright->high = high;
        argsright->threaded = false;
        argsright->lvl = arg->lvl + 1;
        quick_sort(argsright);
    }

    if (arg->threaded == true) {
        pthread_join(threads[arg->t_nr], NULL);
        pthread_mutex_lock(&lock); // Locking mutex before modifying threads_left
        threads_left++;
        pthread_mutex_unlock(&lock); // Unlocking after modification
        printf("THREAD DONE nr: %d, now there are: %d available pthreads\n", arg->t_nr, threads_left);
    }
    free(argsleft);
    free(argsright);
}

int
main(int argc, char **argv)
{
    init_array();
    printf("Max LEVELS: %d\n", MAX_LEVELS);
    ThreadArgs arg;
    arg.v = v;
    arg.low = 0;
    arg.high = MAX_ITEMS - 1;
    arg.lvl = 0;

    quick_sort(&arg);
    //print_array();

    // Destroying the mutex lock after use
    pthread_mutex_destroy(&lock);
    return 0;
}
