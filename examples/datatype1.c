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
#define PACKED_ELEMENT_SIZE (sizeof(int) + 2 * sizeof(double))

struct datatype1 {
    int a;
    double b[2];
};

int state_new(void *context, const void *src, MPI_Count src_count, void **state);
int state_free(void *state);
int query(void *context, const void *buf, MPI_Count count, MPI_Count *size);
int pack(void *state, const void *buf, MPI_Count count, MPI_Count offset, void *dst, MPI_Count dst_size, MPI_Count *used);
int unpack(void *state, void *buf, MPI_Count count, MPI_Count offset, const void *src, MPI_Count src_size);

int main(void)
{
    int size, rank;
    struct datatype1 *buf;
    MPI_Status status;
    MPI_Datatype cd; /* Custom datatype */

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* Create the type */
    MPI_Type_create_custom(&state_new, &state_free, &query, &pack,
                           &unpack, NULL, NULL, NULL, 0, &cd);

    buf = malloc(sizeof(*buf) * COUNT);

    if (rank == 0) {
        /* Initialize the buffer */
        for (size_t i = 0; i < COUNT; i++) {
            buf[i].a = i;
            buf[i].b[0] = 0.2 * i;
            buf[i].b[1] = 0.4 * i;
        }
        MPI_Send(buf, COUNT, cd, 1, 0, MPI_COMM_WORLD);
    } else {
        MPI_Recv(buf, COUNT, cd, 0, 0, MPI_COMM_WORLD, &status);
        for (size_t i = 0; i < COUNT; i++) {
            assert(buf[i].a == i);
            assert(buf[i].b[0] == 0.2 * i);
            assert(buf[i].b[1] == 0.4 * i);
        }
    }

    free(buf);
    MPI_Finalize();
    return 0;
}

struct pack_state {
    MPI_Count last_offset;
};

int state_new(void *context, const void *src, MPI_Count src_count, void **state)
{
    struct pack_state *state_tmp = malloc(sizeof(struct pack_state));
    state_tmp->last_offset = 0;
    *state = state_tmp;
    return 0;
}

int state_free(void *state)
{
    free(state);
    return 0;
}

int query(void *context, const void *buf, MPI_Count count, MPI_Count *packed_size)
{
    *packed_size = count * PACKED_ELEMENT_SIZE;
    return 0;
}

int pack(void *state, const void *buf, MPI_Count count, MPI_Count offset,
         void *dst, MPI_Count dst_size, MPI_Count *used)
{
    struct pack_state *pstate = state;
    const struct datatype1 *tbuf = buf;
    size_t size = dst_size - (dst_size % PACKED_ELEMENT_SIZE);
    size_t elem_offset = offset / PACKED_ELEMENT_SIZE;
    size_t total = size / PACKED_ELEMENT_SIZE;

    assert(offset < (count * PACKED_ELEMENT_SIZE));

    for (size_t i = 0; i < total; ++i) {
        size_t tmp_off = i * PACKED_ELEMENT_SIZE;
        memcpy(dst + tmp_off, &tbuf[elem_offset + i].a, sizeof(int));
        memcpy(dst + tmp_off + sizeof(int), tbuf[elem_offset + i].b, 2 * sizeof(double));
    }
    *used = size;

    pstate->last_offset = offset;
    return 0;
}

int unpack(void *state, void *buf, MPI_Count count, MPI_Count offset,
           const void *src, MPI_Count src_size)
{
    struct pack_state *pstate = state;
    struct datatype1 *tbuf = buf;
    assert(src_size % PACKED_ELEMENT_SIZE == 0);
    size_t total = src_size / PACKED_ELEMENT_SIZE;
    size_t elem_offset = offset / PACKED_ELEMENT_SIZE;

    for (size_t i = 0; i < total; ++i) {
        size_t tmp_off = i * PACKED_ELEMENT_SIZE;
        memcpy(&tbuf[elem_offset + i].a, src + tmp_off, sizeof(int));
        memcpy(tbuf[elem_offset + i].b, src + tmp_off + sizeof(int), 2 * sizeof(double));
    }

    pstate->last_offset = offset;
    return 0;
}
