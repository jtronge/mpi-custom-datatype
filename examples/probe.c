/* Simple probe test */
#include <stdio.h>
#include <assert.h>
#include <mpi.h>

#define COUNT 16

int main(void)
{
    int size, rank;
    char buf[COUNT];

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    assert(size == 2);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        for (int i = 0; i < COUNT; ++i) {
            buf[i] = i;
        }
        MPI_Send(buf, COUNT, MPI_BYTE, 1, 0, MPI_COMM_WORLD);
    } else {
        int count;
        MPI_Status status;

        /* Test probe with and without the source */
        MPI_Probe(0, 0, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, MPI_BYTE, &count);
        assert(count == COUNT);

        MPI_Probe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, MPI_BYTE, &count);
        assert(count == COUNT);

        MPI_Recv(buf, COUNT, MPI_BYTE, 0, 0, MPI_COMM_WORLD, &status);
        for (int i = 0; i < COUNT; ++i) {
            assert(buf[i] == i);
        }
    }

    MPI_Finalize();
    return 0;
}
