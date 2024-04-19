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

int pack_state(void *context, const void *src, MPI_Count src_count, void **state);
int unpack_state(void *context, void *dst, MPI_Count dst_count, void **state);
int query(void *context, const void *buf, MPI_Count count, MPI_Count *size);
int pack(void *state, MPI_Count offset, void *dst, MPI_Count dst_size, MPI_Count *used);
int unpack(void *state, MPI_Count offset, const void *src, MPI_Count src_size);
int pack_state_free(void *state);
int unpack_state_free(void *state);

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
    MPI_Type_create_custom(&pack_state, &unpack_state, &query, &pack,
                           &unpack, &pack_state_free, &unpack_state_free, NULL,
                           &cd);

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

struct pack_state {
    const char *src;
    MPI_Count src_size;
};

int pack_state(void *context, const void *src, MPI_Count src_count, void **state)
{
    struct pack_state *state_tmp = malloc(sizeof(struct pack_state));
    state_tmp->src = src;
    state_tmp->src_size = src_count * sizeof(int);
    *state = state_tmp;
    return 0;
}

struct unpack_state {
    char *dst;
    MPI_Count dst_size;
};

int unpack_state(void *context, void *dst, MPI_Count dst_count, void **state)
{
    struct unpack_state *state_tmp = malloc(sizeof(struct unpack_state));
    state_tmp->dst = dst;
    state_tmp->dst_size = dst_count * sizeof(int);
    *state = state_tmp;
    return 0;
}

int query(void *context, const void *buf, MPI_Count count, MPI_Count *packed_size)
{
    *packed_size = count * sizeof(int);
    return 0;
}

int pack(void *state, MPI_Count offset, void *dst, MPI_Count dst_size, MPI_Count *used)
{
    struct pack_state *pstate = state;
    size_t rem = dst_size % sizeof(int);
    size_t size = dst_size - rem;

    memcpy(dst, pstate->src + offset, size);
    *used = size;
    printf("packed %zu bytes\n", size);

    return 0;
}

int unpack(void *state, MPI_Count offset, const void *src, MPI_Count src_size)
{
    struct unpack_state *ustate = state;
    assert(src_size % sizeof(int) == 0);

    memcpy(ustate->dst + offset, src, src_size);
    printf("unpacked %zu bytes\n", src_size);

    return 0;
}

int pack_state_free(void *state)
{
    free(state);
    return 0;
}

int unpack_state_free(void *state)
{
    free(state);
    return 0;
}
