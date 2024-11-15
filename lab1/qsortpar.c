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
#define AMOUNT_THREADS 4
#define swap(v, a, b) {unsigned tmp; tmp=v[a]; v[a]=v[b]; v[b]=tmp;}
#define MAX_LEVELS (int)ceil(log2(AMOUNT_THREADS + 1))-1


static int *v;

int nr_workers_last_level = 0;

pthread_t threads[AMOUNT_THREADS-1];
int threads_left = AMOUNT_THREADS-1;
pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;


/////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////
//The stack should take quciksort pointers
typedef struct ThreadArgs {
    int *v;
    unsigned int low;
    unsigned int high;
    unsigned int t_nr;
    bool threaded;
    unsigned int lvl;
} ThreadArgs;

typedef struct Stack {
    ThreadArgs data[AMOUNT_THREADS * 10]; // Use struct ThreadArgs
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
    return s->top == AMOUNT_THREADS - 2;
}

// Push an integer onto the stack
// Push a ThreadArgs onto the stack
void push(Stack *s, ThreadArgs value) {
    if (isFull(s)) {
        printf("Stack overflow\n");
        return;
    }
    s->data[++(s->top)] = value;
    printf("Pushed to stack\n");
}

// Pop a ThreadArgs from the stack
ThreadArgs pop(Stack *s) {
    if (isEmpty(s)) {
        printf("Stack underflow\n");
        ThreadArgs empty = {0}; // Return an empty ThreadArgs if the stack is empty
        return empty;
    }
    return s->data[(s->top)--];
}

/////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////
void worker(bool stop){
    while (!stop){
        if (!isEmpty(sp)){
            printf("Taking on work");
            pthread_mutex_lock(&lock);
            ThreadArgs *new_arg = (ThreadArgs *)malloc(sizeof(ThreadArgs));
            *new_arg = pop(sp);
            pthread_mutex_unlock(&lock);
            quick_sort(new_arg);
            free(new_arg);
        }
    }
}


/////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////



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

void
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
        if (size > 100000){
            printf("Assigning work from lvl: %d of size: %d ",arg->lvl, argsleft->high - argsleft->low);
            push(sp, *argsleft);
            // pthread_create(&threads[thread_nr], NULL, (void *)quick_sort, (void *)argsleft);
        }
        else{
            pthread_mutex_unlock(&lock);
            quick_sort(argsleft);
        }
    }
    if (pivot_index < high)
        quick_sort(argsright);

    // if (initiated_thread != -1){
    //     printf("Joining thread nr: %d\n", initiated_thread);
    //     pthread_join(threads[initiated_thread], NULL);
    //     pthread_mutex_lock(&lock);
    //     push(arg->s, initiated_thread);
    //     threads_left++;
    //     pthread_mutex_unlock(&lock);
    // }

    free(argsleft);
    free(argsright);
}

int
main(int argc, char **argv)
{

    init_array();
    Stack s;
    sp = &s;
    initStack(sp);
    //print_array();
    ThreadArgs arg;
    arg.v = v;
    arg.low = 0;
    arg.high = MAX_ITEMS - 1;
    arg.lvl = 0;

    bool stop = false;
    //Creating a thread
    for (int i = 0; i < AMOUNT_THREADS; i++){
        pthread_create(&threads[i], NULL, worker, (void *)&stop);
    }

    quick_sort(&arg);
    if (DEBUG)
        print_array();
    //print_array();
}
