/*
 * Example datatype using memory regions (or iovecs).
 */
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <mpi.h>

#define COUNT 8192

struct iovec_type {
    int a;
    int b[BUFSIZ];
};

int regions_count(void *buf, MPI_Count count, MPI_Count *region_count);
int regions(void *buf, MPI_Count count, MPI_Count region_count,
            MPI_Count reg_lens[], void *reg_bases[], MPI_Datatype types[]);

int main(void)
{
    int size, rank;
    struct iovec_type *buf;
    MPI_Datatype cd;
    MPI_Status status;

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Type_create_custom(NULL, NULL, NULL, NULL, NULL, NULL, NULL, regions_count, regions, NULL, &cd);

    buf = malloc(sizeof(*buf) * COUNT);

    if (rank == 0) {
        /* Initialize the buffer */
        for (size_t i = 0; i < COUNT; ++i) {
            buf[i].a = i;
            for (size_t j = 0; j < BUFSIZ; ++j) {
                buf[i].b[j] = i + j;
            }
        }
        MPI_Send(buf, COUNT, cd, 1, 0, MPI_COMM_WORLD);
    } else {
        MPI_Recv(buf, COUNT, cd, 0, 0, MPI_COMM_WORLD, &status);
        for (size_t i = 0; i < COUNT; ++i) {
            assert(buf[i].a == i);
            for (size_t j = 0; j < BUFSIZ; ++j) {
                assert(buf[i].b[j] == (i + j));
            }
        }
    }

    free(buf);
    MPI_Finalize();
    return 0;
}

int regions_count(void *buf, MPI_Count count, MPI_Count *region_count)
{
    *region_count = 2 * count;
    return 0;
}

int regions(void *buf, MPI_Count count, MPI_Count region_count,
            MPI_Count reg_lens[], void *reg_bases[], MPI_Datatype types[])
{
    struct iovec_type *real_buf = buf;
    assert((2 * count) == region_count);

    for (size_t i = 0; i < count; ++i) {
        size_t reg_pos = 2 * i;
        reg_lens[reg_pos] = sizeof(real_buf[i].a);
        reg_bases[reg_pos] = &real_buf[i].a;
        types[reg_pos] = MPI_BYTE;
        reg_lens[reg_pos + 1] = sizeof(real_buf[i].b);
        reg_bases[reg_pos + 1] = &real_buf[i].b;
        types[reg_pos + 1] = MPI_BYTE;
    }
    return 0;
}
