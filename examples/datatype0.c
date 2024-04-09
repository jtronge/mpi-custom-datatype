/*
 * Simple example using custom datatypes.
 */
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

#define COUNT 100

/* Struct to be packed */
struct pack_type {
    int a;
    double b[2];
};

int main(void)
{
    int size, rank;
    struct pack_type *buf;
    MPI_Status status;
    MPI_Datatype cd; /* Custom datatype */

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* Create the type */

    buf = malloc(sizeof(*buf) * COUNT);

    if (rank == 0) {
        /* Initialize the buffer */
        for (int i = 0; i < COUNT; i++) {
            buf[i].a = i;
            buf[i].b[0] = 0.1 * i;
            buf[i].b[1] = 0.2 * i;
        }

        MPI_Send(buf, COUNT, cd, 1, 0, MPI_COMM_WORLD);
    } else {
        MPI_Recv(buf, COUNT, cd, 0, 0, MPI_COMM_WORLD, &status);
        printf("first element = { .a = %d, .b[] = {%g, %g} }\n", buf[0].a, buf[0].b[0], buf[0].b[1]);
    }

    free(buf);
    MPI_Finalize();
    return 0;
}
