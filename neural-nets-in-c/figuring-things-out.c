/*
    In this file, I test things out to remember how to program in C.
*/

#include <stdio.h>
#include <stdlib.h>


// declare struct
typedef struct {
    float *data;
    int size;
    int *dims;
    int n_dims;
    int bits; // switch for an interface with actual data types
} Tensor;

int main(void) {

    // print stuff
    printf("Hello world!\n");

    // allocate a matrix in memory
    int rows = 16;
    int cols = 8;
    float *m = malloc(rows * cols * sizeof(float));
    
    return 0;
}