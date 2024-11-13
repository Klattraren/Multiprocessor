/***************************************************************************
 *
 * Sequential version of Quick sort
 *
 ***************************************************************************/
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h> // pthread types and functions
#include <stdbool.h> // Added bools
#define DEBUG false

#define KILO (1024)
#define MEGA (1024*1024)
#define MAX_ITEMS (64*MEGA)
//#define MAX_ITEMS 200
#define AMOUNT_THREADS 12
#define swap(v, a, b) {unsigned tmp; tmp=v[a]; v[a]=v[b]; v[b]=tmp;}
#define MAX_LEVELS (int)ceil(log2(AMOUNT_THREADS + 1))-1

static int *v;

int nr_workers_last_level = 0;

pthread_t threads[AMOUNT_THREADS-1];
int threads_left = AMOUNT_THREADS-1;
pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;


/////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////

typedef struct {
    int data[AMOUNT_THREADS-1];
    int top;
} Stack;

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
    return s->top == AMOUNT_THREADS - 2;
}

// Push an integer onto the stack
void push(Stack* s, int value) {
    if (isFull(s)) {
        printf("Stack overflow\n");
        return;
    }
    s->data[++(s->top)] = value;
}

// Pop an integer from the stack
int pop(Stack* s) {
    if (isEmpty(s)) {
        printf("Stack underflow\n");
        return -1;  // Return -1 if the stack is empty
    }
    return s->data[(s->top)--];
}

/////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////

typedef struct ThreadArgs{
    int *v;
    unsigned int low;
    unsigned int high;
    unsigned int t_nr;
    bool threaded;
    unsigned int lvl;
    Stack *s;
} ThreadArgs;


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
        if(!DEBUG)
            v[i] = rand();
        else
            v[i] = rand()%100;
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

static void
quick_sort(ThreadArgs *arg)
{
    unsigned pivot_index;

    ThreadArgs *argsleft = (ThreadArgs *)malloc(sizeof(ThreadArgs));
    ThreadArgs *argsright = (ThreadArgs *)malloc(sizeof(ThreadArgs));
    unsigned low = arg->low;
    unsigned high = arg->high;
    int *v = arg->v;

    /* no need to sort a vector of zero or one element */
    if (low >= high)
        return;

    /* select the pivot value */
    pivot_index = (low+high)/2;
    pivot_index = partition(v, low, high, pivot_index);

    //Setting up the left and right arguments
    argsleft->s = argsright->s = arg->s;
    argsleft->v = argsright->v = v;
    argsleft->lvl = argsright->lvl = arg->lvl + 1;

    argsright->high = high;
    argsright->low = pivot_index+1;
    argsright->v = v;

    argsleft->high = pivot_index-1;
    argsleft->low = low;
    argsleft->v = v;

    int initiated_thread = -1;
    int size = argsleft->high - argsleft->low;
    /* sort the two sub arrays */
    if (low < pivot_index){
        pthread_mutex_lock(&lock); // Locking the mutex
        if (!isEmpty(arg->s) && (arg->lvl < MAX_LEVELS || nr_workers_last_level > 3) && size > 100000){
            int thread_nr = pop(arg->s);
            initiated_thread = thread_nr;
            threads_left--;
            pthread_mutex_unlock(&lock); // Unlocking after modifying threads_left
            printf("\033[0;37mThreads left: %d on level: \033[0;32m %d \033[0;37m with size: %d\n", threads_left, arg->lvl, argsleft->high - argsleft->low);
            if (arg->lvl == MAX_LEVELS-1)
                nr_workers_last_level++;
            pthread_create(&threads[thread_nr], NULL, (void *)quick_sort, (void *)argsleft);
        }
        else{
            pthread_mutex_unlock(&lock);
            quick_sort(argsleft);
        }
    }
    if (pivot_index < high)
        quick_sort(argsright);

    if (initiated_thread != -1){
        printf("Joining thread nr: %d\n", initiated_thread);
        pthread_join(threads[initiated_thread], NULL);
        pthread_mutex_lock(&lock);
        push(arg->s, initiated_thread);
        threads_left++;
        pthread_mutex_unlock(&lock);
    }

    free(argsleft);
    free(argsright);
}

int
main(int argc, char **argv)
{

    init_array();
    Stack s;
    initStack(&s);
    for (int i = 0; i < AMOUNT_THREADS-1; i++) {
        push(&s, i);
    }
    //print_array();
    ThreadArgs arg;
    arg.v = v;
    arg.low = 0;
    arg.high = MAX_ITEMS - 1;
    arg.lvl = 0;
    arg.s = &s;

    quick_sort(&arg);
    if (DEBUG)
        print_array();
    //print_array();
}
