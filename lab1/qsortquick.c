/***************************************************************************
 *
 * Sequential version of Quick sort
 *
 ***************************************************************************/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h> // pthread types and functions
#include <stdbool.h> //Added bools
#include<unistd.h>

#define KILO (1024)
#define MEGA (1024*1024)
#define MAX_ITEMS (64*MEGA)
#define swap(v, a, b) {unsigned tmp; tmp=v[a]; v[a]=v[b]; v[b]=tmp;}
#define AMOUNT_THREADS 32
#define MAX_LEVELS (int)ceil(log2(AMOUNT_THREADS + 1))

//adding mutex

static int *v;



//Creating a struct to pass arguments to the thread
typedef struct ThreadArgs{
    int *v;
    unsigned int low;
    unsigned int high;
    unsigned int t_nr;
    bool threaded;
    unsigned int lvl;
} ThreadArgs;

//Creating X threads
pthread_t threads[AMOUNT_THREADS-1];

//Creating a variable that knows how many threads are left
int threads_left = AMOUNT_THREADS-1;

typedef struct Stack {
    int data[AMOUNT_THREADS-1]; // Use struct ThreadArgs
    int top;
} Stack;

//Stack pointer
Stack *sp;


// Initialize the stack
void initStack(Stack* s) {
    s->top = -1;
}

// Check if the stack is empty
int isEmpty(Stack* s) {
    return s->top == -1;
}

// Check if the stack is full
int isFull(Stack* s) {
    return s->top == (AMOUNT_THREADS) - 1;
}

// Push an integer onto the stack
// Push a ThreadArgs onto the stack
void push(Stack *s, int value) {
    if (isFull(s)) {
        printf("Stack overflow\n");
        return;
    }
    s->data[++(s->top)] = value;
    printf("Pushed to stack\n");
}

// Pop a ThreadArgs from the stack
int pop(Stack *s) {
    if (isEmpty(s)) {
        printf("Stack underflow\n");
        return -1;
    }
    return s->data[(s->top)--];
}

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
    v = (int *) malloc(MAX_ITEMS*sizeof(int));
    for (i = 0; i < MAX_ITEMS; i++)
        v[i] = rand();
}

static unsigned
partition(int *v, unsigned low, unsigned high, unsigned pivot_index)
{
    /* move pivot to the bottom of the vector */
    if (pivot_index != low)
        swap(v, low, pivot_index);

    pivot_index = low;
    low++;

    /* invariant:
     * v[i] for i less than low are less than or equal to pivot
     * v[i] for i greater than high are greater than pivot
     */

    /* move elements into place */
    while (low <= high) {
        if (v[low] <= v[pivot_index])
            low++;
        else if (v[high] > v[pivot_index])
            high--;
        else
            swap(v, low, high);
    }

    /* put pivot back between two groups */
    if (high != pivot_index)
        swap(v, pivot_index, high);
    return high;
}



/*quick_sort(int *v, unsigned low, unsigned high)*/
static void
quick_sort(ThreadArgs *arg)
{
    ThreadArgs *argsleft = (ThreadArgs *)malloc(sizeof(ThreadArgs));
    ThreadArgs *argsright = (ThreadArgs *)malloc(sizeof(ThreadArgs));
    unsigned low = arg->low;
    unsigned high = arg->high;
    int *v = arg->v;

    unsigned pivot_index;

    /* no need to sort a vector of zero or one element */
    if (low >= high)
        return;

    /* select the pivot value */
    pivot_index = (low+high)/2;

    /* partition the vector */
    pivot_index = partition(v, low, high, pivot_index);

    /* sort the two sub arrays */
    if (low < pivot_index)
        if (threads_left > 0 && arg->lvl < MAX_LEVELS){
            // int thread_nr = pop(sp);
            // printf("Popped from stack with thread nr: %d\n", thread_nr);
            // pthread_mutex_unlock(&lock);
            argsleft->v = v;
            argsleft->low = low;
            argsleft->high = pivot_index-1;
            argsleft->threaded = true;
            // pthread_mutex_lock(&lock);
            threads_left--;
            argsleft->t_nr = threads_left;
            argsleft->lvl = arg->lvl + 1;
            printf("\033[0;37mThreads left: %d on level: \033[0;32m %d \033[0;37m with amount: %d\n", threads_left, argsleft->lvl,argsleft->high-argsleft->low);
            pthread_create(&threads[threads_left], NULL, quick_sort, (void *)argsleft);
            }
        else{
            argsleft->v = v;
            argsleft->low = low;
            argsleft->high = pivot_index-1;
            argsleft->threaded = false;
            argsleft->lvl = arg->lvl + 1;
            quick_sort(argsleft);
        }

    if (pivot_index < high){
        argsright->v = v;
        argsright->low = pivot_index+1;
        argsright->high = high;
        argsright->threaded = false;
        argsright->lvl = arg->lvl + 1;
        quick_sort(argsright);
        }

    // if (arg->threaded == true){
    //     printf("Thread done nr: %d\n", arg->t_nr);
    //     threads_done++;
    //     // pthread_exit(NULL);
    // }

    // if (argsleft->threaded == true && argsleft->threaded == true){
    //     for (int i = 0; i < AMOUNT_THREADS; i++){
    //         pthread_join(threads[i], NULL);
    //     }
    // }
    // if (argsleft->threaded == true){
    //     printf("Joining left thread from level: %d and nr: \033[0;31m %d \033[0;30m\n", argsleft->lvl, argsleft->t_nr);
    //     pthread_join(threads[argsleft->t_nr], NULL);
    // }
    // if (argsright->threaded == true){
    //     printf("Joining right thread from level: %d and nr: \033[0;31m %d \033[0;30m\n", argsright->lvl, argsright->t_nr);
    //     pthread_join(threads[argsright->t_nr], NULL);
    // }
    if (arg->threaded == true){
        // push(sp, arg->t_nr);
        printf("Joining thread from level: %d and nr: \033[0;31m %d \033[0;37m\n", arg->lvl, arg->t_nr);
    }

    free(argsleft);
    free(argsright);
}

int
main(int argc, char **argv)
{
    init_array();
    //print_array();
   // printf("Max LEVELS: %d\n", MAX_LEVELS);
    // Stack s;
    // sp = &s;
    // initStack(sp);
    // for (int i = 0; i < AMOUNT_THREADS-1; i++){
    //     push(sp, i);
    // }
    ThreadArgs arg;
    arg.v = v;
    arg.low = 0;
    arg.high = MAX_ITEMS-1;
    arg.lvl = 0;

    quick_sort(&arg);
    // while (threads_done != AMOUNT_THREADS-1){
    //     // printf("total Threads done: %d\n", threads_done);
    //     sleep(0.1);
    // }
    //print_array();
}