/***************************************************************************
 *
 * Paralell version of Gauss-Jordan row reduction
 *
 ***************************************************************************/

#include <stdio.h>

#define MAX_SIZE 4096
#define THREADS_PER_BLOCK 1024

typedef double matrix[MAX_SIZE * MAX_SIZE];

int	N;		/* matrix size		*/
int	maxnum;		/* max number of element*/
char* Init;		/* matrix init type	*/
int	PRINT;		/* print switch		*/
matrix	A;		/* matrix A		*/
double	b[MAX_SIZE];	/* vector b             */
double	y[MAX_SIZE];	/* vector y             */

/* forward declarations */
void work(void);
void Init_Matrix(void);
void Print_Matrix(void);
void Init_Default(void);
int Read_Options(int, char**);

int
main(int argc, char** argv)
{
    printf("Gauss Jordan MultiProcessed\n");
    int i, timestart, timeend, iter;

    Init_Default();		/* Init default values	*/
    Read_Options(argc, argv);	/* Read arguments	*/
    Init_Matrix();		/* Init the matrix	*/
    work();
    if (PRINT == 1)
        Print_Matrix();
}

__global__ void
division_step(double* d_A, double* d_b, double* d_y, int N, int k) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;

    // Ensure j is within bounds for the current row k
    if (j > k && j < N) {
        d_A[k * N + j] /= d_A[k * N + k];  // Division step for row k
    }

    __syncthreads(); // Ensure all threads in the block finish before proceeding

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_y[k] = d_b[k] / d_A[k * N + k];
        d_A[k * N + k] = 1.0; // Set the diagonal element to 1.0
    }
}


__global__ void
elimination_step(double* d_A, double* d_b, double* d_y, int N, int k){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int i = index / N;
    int j = index % N;

    if ((i > k && i < N)&&(j > k && j < N)) {
        d_A[i * N + j] -= (d_A[i * N + k] * d_A[k * N + j]);  // Elimination
    }
    __syncthreads(); // Ensure all threads in the block finish before proceeding
    if (i > k && i < N && j == k) {
        d_b[i] -= d_A[i * N + k] * d_y[k];
        d_A[i * N + k] = 0.0;
    }
}


__global__ void
additional_step(double* d_A, double* d_b, double* d_y, int N, int k) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int i = index / N;
    int j = index % N;
    if (i < k && j > k && j < N) {
        d_A[i * N + j] -= (d_A[i * N + k] * d_A[k * N + j]);  // Additional Elimination
    }
    __syncthreads(); // Ensure all threads in the block finish before proceeding
    if (i < k && j == k) {
        d_y[i] -= d_A[i * N + k] * d_y[k];
        d_A[i * N + k] = 0.0;
    }
}




void
work(void)
{
    double* d_A;
    double* d_b;
    double* d_y;
    int k;
    int nr_blocks = (N*N + THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK;

    // Allocate memory on device
    cudaMalloc((void**)&d_A, N * N * sizeof(double));
    cudaMalloc(&d_b, N * sizeof(double));
    cudaMalloc(&d_y, N * sizeof(double));

    // Copy data to device
    cudaMemcpy(d_A, A, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N * sizeof(double), cudaMemcpyHostToDevice);

    // Call the kernel
    for (k = 0; k < N; k++) {
        division_step<<<nr_blocks, THREADS_PER_BLOCK>>>(d_A, d_b, d_y, N, k);
        cudaDeviceSynchronize();

        elimination_step<<<nr_blocks, THREADS_PER_BLOCK>>>(d_A, d_b, d_y, N, k);
        cudaDeviceSynchronize();

        additional_step<<<nr_blocks, THREADS_PER_BLOCK>>>(d_A, d_b, d_y, N, k);
        cudaDeviceSynchronize();
    }
    // Copy data back to host
    cudaMemcpy(A, d_A, N * N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(b, d_b, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(y, d_y, N * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_y);
}

void
Init_Matrix()
{
    int i, j;

    printf("\nsize      = %dx%d ", N, N);
    printf("\nmaxnum    = %d \n", maxnum);
    printf("Init	  = %s \n", Init);
    printf("Initializing matrix...");

    if (strcmp(Init, "rand") == 0) {
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                if (i == j) /* diagonal dominance */
                    A[i * N + j] = (double)(rand() % maxnum) + 5.0;
                else
                    A[i * N + j] = (double)(rand() % maxnum) + 1.0;
            }
        }
    }
    if (strcmp(Init, "fast") == 0) {
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                if (i == j) /* diagonal dominance */
                    A[i * N + j] = 5.0;
                else
                    A[i * N + j] = 2.0;
            }
        }
    }

    /* Initialize vectors b and y */
    for (i = 0; i < N; i++) {
        b[i] = 2.0;
        y[i] = 1.0;
    }

    printf("done \n\n");
    if (PRINT == 1)
        Print_Matrix();
}

void
Print_Matrix()
{
    int i, j;

    printf("Matrix A:\n");
    for (i = 0; i < N; i++) {
        printf("[");
        for (j = 0; j < N; j++)
            printf(" %5.2f,", A[i * N + j]);
        printf("]\n");
    }
    printf("Vector y:\n[");
    for (j = 0; j < N; j++)
        printf(" %5.2f,", y[j]);
    printf("]\n");
    printf("\n\n");
}

void
Init_Default()
{
    N = 2048;
    Init = "fast";
    maxnum = 15.0;
    PRINT = 0;
}

int
Read_Options(int argc, char** argv)
{
    char* prog;

    prog = *argv;
    while (++argv, --argc > 0)
        if (**argv == '-')
            switch (*++ * argv) {
            case 'n':
                --argc;
                N = atoi(*++argv);
                break;
            case 'h':
                printf("\nHELP: try sor -u \n\n");
                exit(0);
                break;
            case 'u':
                printf("\nUsage: gaussian [-n problemsize]\n");
                printf("           [-D] show default values \n");
                printf("           [-h] help \n");
                printf("           [-I init_type] fast/rand \n");
                printf("           [-m maxnum] max random no \n");
                printf("           [-P print_switch] 0/1 \n");
                exit(0);
                break;
            case 'D':
                printf("\nDefault:  n         = %d ", N);
                printf("\n          Init      = rand");
                printf("\n          maxnum    = 5 ");
                printf("\n          P         = 0 \n\n");
                exit(0);
                break;
            case 'I':
                --argc;
                Init = *++argv;
                break;
            case 'm':
                --argc;
                maxnum = atoi(*++argv);
                break;
            case 'P':
                --argc;
                PRINT = atoi(*++argv);
                break;
            default:
                printf("%s: ignored option: -%s\n", prog, *argv);
                printf("HELP: try %s -u \n\n", prog);
                break;
            }
}
