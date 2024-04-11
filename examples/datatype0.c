/*
 * Simple example using custom datatypes.
 */
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <mpi.h>

#define COUNT 1

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
        for (size_t i = 0; i < COUNT; i++) {
            buf[i].a = i;
            buf[i].b[0] = 0.1 * i;
            buf[i].b[1] = 0.2 * i;
        }

        MPI_Send(buf, COUNT * sizeof(*buf), cd, 1, 0, MPI_COMM_WORLD);
    } else {
        MPI_Recv(buf, COUNT * sizeof(*buf), cd, 0, 0, MPI_COMM_WORLD, &status);
        printf("first element = { .a = %d, .b[] = {%g, %g} }\n", buf[0].a, buf[0].b[0], buf[0].b[1]);
    }

    free(buf);
    MPI_Finalize();
    return 0;
}

/* Size of one packed element */
#define PACKED_ELEMENT_SIZE (sizeof(int) + 2 * sizeof(double))

static int packfn(MPI_Count src_size, const void *src,
                  MPI_Count dst_size, void *dst,
                  MPI_Count *used, void **resume)
{
    const struct pack_type *src_buf = src;
    char *dst_buf = dst;
    size_t complete = 0;
    size_t total_elems = src_size / sizeof(*src_buf);

    printf("packfn(src_size=%zu, src=., dst_size=%zu, dst=., used=., resume=.)\n", src_size, dst_size);

    if (NULL == *resume) {
        *resume = malloc(sizeof(size_t));
        *((size_t *) *resume) = 0;
    }

    complete = *((size_t *) *resume);

    for (size_t i = 0;
         (i + PACKED_ELEMENT_SIZE) < dst_size && complete < total_elems;
         i += PACKED_ELEMENT_SIZE)
    {
        const struct pack_type *elem = src_buf + complete;
        printf("packing....\n");
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
    const char *src_buf = src;
    struct pack_type *dst_buf = dst;
    size_t complete = 0;
    size_t total_elems = dst_size / sizeof(*dst_buf);

    assert((src_size % PACKED_ELEMENT_SIZE) == 0);
    printf("unpackfn(src_size=%zu, src=., dst_size=%zu, dst=., resume=.)\n", src_size, dst_size);

    if (NULL == *resume) {
        *resume = malloc(sizeof(size_t));
        *((size_t *) *resume) = 0;
    }

    complete = *((size_t *) *resume);

    for (size_t i = 0;
         (i + PACKED_ELEMENT_SIZE) < src_size && complete < total_elems;
         i += PACKED_ELEMENT_SIZE)
    {
        struct pack_type *elem = dst_buf + complete;
        printf("unpacking....\n");
        memcpy(&elem->a, src_buf + i, sizeof(elem->a));
        memcpy(&elem->b, src_buf + i + sizeof(elem->a), sizeof(elem->b));
        complete += 1;
    }

    if (complete < total_elems) {
        *((size_t *) *resume) = complete;
    } else {
        free(*resume);
    }
    return 0;
}

static int queryfn(const void *buf, MPI_Count size, MPI_Count *packed_size)
{
    *packed_size = (size / sizeof(struct pack_type)) * PACKED_ELEMENT_SIZE;
    return 0;
}
