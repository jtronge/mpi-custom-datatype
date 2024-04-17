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

static int packfn(MPI_Count src_size, const void *src,
                  MPI_Count dst_size, void *dst, void **resume);
static int unpackfn(MPI_Count src_count, const void *src,
                    MPI_Count dst_size, void *dst, void **resume);
static int queryfn(const void *buf, MPI_Count size, MPI_Count *packed_size);

int main(void)
{
    int size, rank;
    int *buf;
    MPI_Status status;
    MPI_Datatype cd; /* Custom datatype */

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* Create the type */
    MPI_Type_create_custom(&packfn, &unpackfn, &queryfn, PACKED_ELEMENT_SIZE, NULL, 0, &cd);

    buf = malloc(sizeof(*buf) * COUNT);

    if (rank == 0) {
        /* Initialize the buffer */
        for (size_t i = 0; i < COUNT; i++) {
            buf[i] = i;
        }

        MPI_Send(buf, COUNT * sizeof(*buf), cd, 1, 0, MPI_COMM_WORLD);
    } else {
        MPI_Recv(buf, COUNT * sizeof(*buf), cd, 0, 0, MPI_COMM_WORLD, &status);
        for (size_t i = 0; i < COUNT; i++) {
            assert(buf[i] == i);
        }
    }

    free(buf);
    MPI_Finalize();
    return 0;
}

static int packfn(MPI_Count src_size, const void *src,
                  MPI_Count dst_size, void *dst, void **resume)
{
    size_t count = dst_size < src_size ? dst_size : src_size;
    // Dest size should be a multiple of PACKED_ELEMENT_SIZE
    memcpy(dst, src, count);
    // *used = count - 1;
    return 0;
}

static int unpackfn(MPI_Count src_size, const void *src,
                    MPI_Count dst_size, void *dst, void **resume)
{
    /* TODO: What happens if we get a sub buffer of the input, how do we know when to free resume? */
    size_t count = dst_size < src_size ? dst_size : src_size;
    memcpy(dst, src, count);
    return 0;
}

static int queryfn(const void *buf, MPI_Count size, MPI_Count *packed_size)
{
    *packed_size = size;
    return 0;
}
