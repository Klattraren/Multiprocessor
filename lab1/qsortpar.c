/***************************************************************************
 *
 * Parallel version of Quick sort
 *
 ***************************************************************************/
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdbool.h>
#include<unistd.h>

#define DEBUG false

#define KILO (1024)
#define MEGA (1024*1024)
#define MAX_ITEMS (64*MEGA)
#define AMOUNT_THREADS 4
#define swap(v, a, b) {unsigned tmp; tmp=v[a]; v[a]=v[b]; v[b]=tmp;}
#define MAX_LEVELS (int)ceil(log2(AMOUNT_THREADS + 1))-1
#define MIN_ITEMS 50000

static int *v;
bool stop_threads = false;

pthread_t threads[AMOUNT_THREADS-1];
pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;


/////////////////////////////////////////////////////////////////////////////////////
//////////////////////QUEUE IMPLEMENTATION//////////////////////////////////////////
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

typedef struct Queue {
    ThreadArgs data[(MAX_ITEMS / MIN_ITEMS)]; // Use struct ThreadArgs
    int front;
    int rear;
    int size;
} Queue;

// Queue pointer
Queue *qp;

// Initialize the queue
void initQueue(Queue *q) {
    q->front = 0;
    q->rear = -1;
    q->size = 0;
}

// Check if the queue is empty
bool isQueueEmpty(Queue *q) {
    return q->size == 0;
}

// Check if the queue is full
bool isQueueFull(Queue *q) {
    return q->size == (MAX_ITEMS / MIN_ITEMS);
}

// Enqueue a ThreadArgs into the queue
void enqueue(Queue *q, ThreadArgs value) {
    if (isQueueFull(q)) {
        return;
    }
    q->rear = (q->rear + 1) % (MAX_ITEMS / MIN_ITEMS); // Wrap around using modulo
    q->data[q->rear] = value;
    q->size++;
}

// Dequeue a ThreadArgs from the queue
ThreadArgs dequeue(Queue *q) {
    if (isQueueEmpty(q)) {
        ThreadArgs empty = {0}; // Return an empty ThreadArgs if the queue is empty
        return empty;
    }
    else{
        ThreadArgs value = q->data[q->front];
        q->front = (q->front + 1) % (MAX_ITEMS / MIN_ITEMS); // Wrap around using modulo
        q->size--;
        return value;
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

    //Flag if we have started a task, used to controll if we should free the memory

    /* sort the two sub arrays */
    if (low < pivot_index){
        if ((argsleft->high-argsleft->low) > MIN_ITEMS && AMOUNT_THREADS > 1){
            pthread_mutex_lock(&lock);
            enqueue(qp, *argsleft);
            pthread_mutex_unlock(&lock);
            free(argsleft);
        }
        else{
            quick_sort(argsleft);
            free(argsleft);
        }
    }
    if (pivot_index < high){
        quick_sort(argsright);
    }

    free(argsright);
}

/////////////////////////////////////////////////////////////////////////////////////
//////////////////////WORKER FUNCTION FOR THREADS////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////

void worker(){
    while (!stop_threads){
        pthread_mutex_lock(&lock);
        if (!isQueueEmpty(qp)){
            ThreadArgs *task = (ThreadArgs *)malloc(sizeof(ThreadArgs));
            *task = dequeue(qp);
            pthread_mutex_unlock(&lock);
            quick_sort(task);
            free(task);
        }
        else{
            pthread_mutex_unlock(&lock);
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////

int
main(int argc, char **argv)
{

    init_array();
    Queue q;
    qp = &q;
    initQueue(qp);

    ThreadArgs arg;
    arg.v = v;
    arg.low = 0;
    arg.high = MAX_ITEMS - 1;
    arg.lvl = 0;

    //Creating a thread
    for (int i = 0; i < AMOUNT_THREADS-1; i++){
        pthread_create(&threads[i], NULL, worker, NULL);
    }

    //Initialize the quicksort
    quick_sort(&arg);


    // Letting main thread take work from the que
    while(!isQueueEmpty(qp)){
        ThreadArgs *task = (ThreadArgs *)malloc(sizeof(ThreadArgs));
        *task = dequeue(qp);
        pthread_mutex_unlock(&lock);
        quick_sort(task);
        free(task);
    }

    //When the stack is empty, we can stop the threads
    stop_threads = true;

    //Joining the threads
    for (int i = 0; i < AMOUNT_THREADS-1; i++){
        pthread_join(threads[i], NULL);
    }

    //Pringint the array
    if (DEBUG)
        print_array();

    //Freeing the array of numbers
    free(v);
}
