/*
 * Simple example using custom datatypes.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <mpi.h>

#define COUNT 100

/* Struct to be packed */
struct pack_type {
    int a;
    double b[2];
};

static int packfn(MPI_Count src_size, const void *src,
                  MPI_Count dst_size, void *dst,
                  MPI_Count *used, void **resume);
static int unpackfn(MPI_Count src_count, const void *src,
                    MPI_Count dst_size, void *dst, void **resume);
static int queryfn(const void *buf, MPI_Count size, MPI_Count *packed_size);

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
    MPI_Type_create_custom(&packfn, &unpackfn, &queryfn, NULL, 0, &cd);

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

/* Size of one packed element */
#define PACKED_ELEMENT_SIZE (sizeof(int) + 2 * sizeof(double))

static int packfn(MPI_Count src_size, const void *src,
                  MPI_Count dst_count, void *dst,
                  MPI_Count *used, void **resume)
{
    const struct pack_type *src_buf = src;
    char *dst_buf = dst;
    size_t complete = 0;
    size_t total_elems = src_size / PACKED_ELEMENT_SIZE;

    if (NULL == *resume) {
        *resume = malloc(sizeof(size_t));
    }

    complete = *((size_t *) *resume);

    for (size_t i = 0;
         (i + PACKED_ELEMENT_SIZE) < dst_count && complete < total_elems;
         i += PACKED_ELEMENT_SIZE)
    {
        const struct pack_type *elem = src_buf + complete;
        memcpy(dst_buf + i, &elem->a, sizeof(elem->a));
        memcpy(dst_buf + i + sizeof(elem->a), &elem->b, sizeof(elem->b));
        complete += 1;
    }

    if (complete < total_elems) {
        *((size_t *) *resume) = complete;
    } else {
        free(*resume);
    }
    return 0;
}

static int unpackfn(MPI_Count src_size, const void *src,
                    MPI_Count dst_size, void *dst, void **resume)
{
    /* TODO: What happens if we get a sub buffer of the input, how do we know when to free resume? */
/*
    size_t complete = 0;
    size_t total_elems = dst_size / PACKED_ELEMENT_SIZE;
    assert((dst_size % PACKED_ELEMENT_SIZE) == 0);

    if (NULL == *resume) {
        *resume = malloc(sizeof(size_t));
    }

    complete = *((size_t *) *resume);

    for (size_t i = 0; i < total_elem)

    if (complete < total_elems) {
        *((size_t *) *resume) = complete;
    } else {
        free(*resume);
    }
*/
    return 0;
}

static int queryfn(const void *buf, MPI_Count size, MPI_Count *packed_size)
{
    /* size is in bytes of the *unpacked* buffer */
    *packed_size = size / sizeof(struct pack_type);
    return 0;
}
