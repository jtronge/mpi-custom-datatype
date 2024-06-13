/*
 * Simple example using custom datatypes.
 */
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <mpi.h>

#define COUNT 1000000
/* Size of one packed element */
#define PACKED_ELEMENT_SIZE sizeof(int)

int query(void *context, const void *buf, MPI_Count count, MPI_Count *size);
int pack(void *state, const void *buf, MPI_Count count, MPI_Count offset, void *dst, MPI_Count dst_size, MPI_Count *used);
int unpack(void *state, void *buf, MPI_Count count, MPI_Count offset, const void *src, MPI_Count src_size);

int main(void)
{
    int size, rank;
    int *buf;
    MPI_Datatype cd; /* Custom datatype */
    MPI_Status status;

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* Create the type */
    MPI_Type_create_custom(NULL, NULL, &query, &pack,
                           &unpack, NULL, NULL, NULL, 0, &cd);

    buf = malloc(sizeof(*buf) * COUNT);

    if (rank == 0) {
        /* Initialize the buffer */
        for (size_t i = 0; i < COUNT; i++) {
            buf[i] = i;
        }
        MPI_Send(buf, COUNT, cd, 1, 0, MPI_COMM_WORLD);
    } else {
        MPI_Recv(buf, COUNT, cd, 0, 0, MPI_COMM_WORLD, &status);
        for (size_t i = 0; i < COUNT; i++) {
            assert(buf[i] == i);
        }
    }

    free(buf);
    MPI_Finalize();
    return 0;
}

int query(void *context, const void *buf, MPI_Count count, MPI_Count *packed_size)
{
    *packed_size = count * sizeof(int);
    return 0;
}

int pack(void *state, const void *buf, MPI_Count count, MPI_Count offset, void *dst, MPI_Count dst_size, MPI_Count *used)
{
    size_t rem = dst_size % sizeof(int);
    size_t size = dst_size - rem;

    memcpy(dst, buf + offset, size);
    *used = size;
    printf("packed %zu bytes\n", size);

    return 0;
}

int unpack(void *state, void *buf, MPI_Count count, MPI_Count offset, const void *src, MPI_Count src_size)
{
    assert(src_size % sizeof(int) == 0);

    memcpy(buf + offset, src, src_size);
    printf("unpacked %zu bytes\n", src_size);

    return 0;
}
