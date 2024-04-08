#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

int main()
{
    int size, rank;
    uint8_t buf0[] = {1, 2, 3, 4};
    uint8_t buf1[] = {0, 0, 0, 0};
    MPI_Status status;
    MPI_Datatype dt = 0;

    MPI_Init(NULL, NULL);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    printf("hello from rank %d of %d\n", rank, size);

    if (rank == 0) {
        MPI_Send(buf0, 4, 0, 1, 0, MPI_COMM_WORLD);
    } else {
        MPI_Recv(buf1, 4, 0, 1, 0, MPI_COMM_WORLD, &status);
        printf("received: [%x, %x, %x, %x]\n", buf1[0], buf1[1], buf1[2], buf1[3]);
    }

    MPI_Finalize();
}
