#ifndef _MPI_H_
#define _MPI_H_

typedef size_t MPI_Count;

/* For simplicity MPI_Comm and other handles are defined to be integers */
typedef int MPI_Comm;
typedef int MPI_Datatype;
/* Handle constants */
#define MPI_COMM_WORLD 1

#define MPI_BYTE 1

typedef struct MPI_Status {
    int MPI_SOURCE;
    int MPI_TAG;
    int MPI_ERROR;
} MPI_Status;

int MPI_Init(int *argc, char **argv[]);
int MPI_Finalize(void);
int MPI_Comm_size(MPI_Comm comm, int *size);
int MPI_Comm_rank(MPI_Comm comm, int *rank);

/* P2P functions */
int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest,
             int tag, MPI_Comm comm);
int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag,
             MPI_Comm comm, MPI_Status *status);

/* Datatype functions
 *
 * Some changes to note:
 *   * renaming of count to size (all units in bytes)
 *   * addition of int return for error handling
 *   * addition of query function to custom api (how else would we know full
 *     length of packed data without packing first?)
 *   * removal of elem_size and elem_extent
 *   * Q: When to deallocate the resume pointer, if allocated?
 */
typedef int (MPI_Type_custom_pack_function)(MPI_Count src_size, const void *src,
                                            MPI_Count dst_size, void *dst, void **resume);
typedef int (MPI_Type_custom_unpack_function)(MPI_Count src_size, const void *src,
                                              MPI_Count dst_size, void *dst,
                                              void **resume);
typedef int (MPI_Type_custom_query_function)(const void *buf, MPI_Count size,
                                             MPI_Count *packed_size);
typedef int (MPI_Type_custom_reg_function)(const void *src, MPI_Count max_regions,
                                           MPI_Count region_lengths[],
                                           void *region_bases[], void **resume);

int MPI_Type_create_custom(MPI_Type_custom_pack_function *packfn,
                           MPI_Type_custom_unpack_function *unpackfn,
                           MPI_Type_custom_query_function *queryfn,
                           MPI_Count packed_elem_size,
                           MPI_Type_custom_reg_function *regfn,
                           MPI_Count reg_count, MPI_Datatype *type);

/* Idea: use a builder-like interface */

/* Constants */
#define MPI_SUCCESS 0
#define MPI_ERR_INTERNAL 1

#endif /* _MPI_H_ */
