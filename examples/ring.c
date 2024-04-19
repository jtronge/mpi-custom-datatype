#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <mpi.h>

int main(void)
{
    int size, rank, prev, next, i;
    uint8_t buf[10];
    MPI_Request reqs[10];
    MPI_Status statuses[10];

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank % 2 == 0) {
        for (i = 0; i < 10; ++i) {
            buf[i] = 0;
        }

        next = (rank + 1) % size;

        for (i = 0; i < 10; ++i) {
            MPI_Isend(&buf[i], 1, MPI_BYTE, next, 0, MPI_COMM_WORLD, &reqs[i]);
        }
    } else {
        prev = (rank + size - 1) % size;

        for (i = 0; i < 10; ++i) {
            MPI_Irecv(&buf[i], 1, MPI_BYTE, prev, 0, MPI_COMM_WORLD, &reqs[i]);
        }
    }

    MPI_Waitall(10, reqs, statuses);
    for (i = 0; i < 10; ++i) {
        assert(buf[i] == 0);
    }

    MPI_Finalize();
    return 0;
}
