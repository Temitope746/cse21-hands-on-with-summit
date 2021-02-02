#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

int main()
{
    const int N = 1e8;

    int num_threads;
    double start, stop;

	// Bytes in arrays
	size_t bytes_in_array = N*sizeof(double);

	// Allocate memory for arrays on host
	double *A = (double*)malloc(bytes_in_array);
	double *B = (double*)malloc(bytes_in_array);
	double *C = (double*)malloc(bytes_in_array);

	// Initialize vector values
	for(int i=0; i<N; i++){
		A[i] = 1.0;
		B[i] = 2.0;
	}

    start = omp_get_wtime();

	// Perform element-wise addition of vectors on GPU
    #pragma omp target map(to:A[:N],B[:N]) map(tofrom:C[:N])
    {
    #pragma omp teams distribute parallel for
	for(int i=0; i<N; i++){
	    C[i] = A[i] + B[i];
    }
    }

    stop = omp_get_wtime();

	// Check for correctness
	for(int i=0; i<N; i++){
		if(C[i] != 3.0){
			printf("Error: Element C[%d] = %f instead of 3.0\n", i, C[i]);
			exit(1);
		}
	}

    printf("__SUCCESS__\n");
    printf("Elapsed Time (s): %.06f\n", stop - start);	

    return 0;
}

