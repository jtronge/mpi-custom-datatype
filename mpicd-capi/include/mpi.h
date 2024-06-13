#ifndef _MPI_H_
#define _MPI_H_

#if __cplusplus
extern "C" {
#endif

typedef size_t MPI_Count;

/* For simplicity MPI_Comm and other handles are defined to be integers */
typedef int MPI_Comm;
typedef int MPI_Datatype;

/* MPI_Request corresponds to Rust's isize */
typedef intptr_t MPI_Request;

/* Handle constants */
#define MPI_COMM_WORLD 1

#define MPI_BYTE 1

typedef struct MPI_Status {
    int count;
    int cancelled;
    int MPI_SOURCE;
    int MPI_TAG;
    int MPI_ERROR;
} MPI_Status;

int MPI_Init(int *argc, char **argv[]);
int MPI_Finalize(void);
int MPI_Comm_size(MPI_Comm comm, int *size);
int MPI_Comm_rank(MPI_Comm comm, int *rank);
double MPI_Wtime(void);
int MPI_Barrier(MPI_Comm comm);

/* P2P functions */
int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest,
             int tag, MPI_Comm comm);
int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag,
             MPI_Comm comm, MPI_Status *status);
int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest,
              int tag, MPI_Comm comm, MPI_Request *request);
int MPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source, int tag,
              MPI_Comm comm, MPI_Request *request);

int MPI_Wait(MPI_Request *request, MPI_Status *status);
int MPI_Waitall(int count, MPI_Request array_of_requests[], MPI_Status *array_of_statuses);

/*
 * All functions return 0 on success and non-zero on failure.
 */
typedef int (MPI_Type_custom_state_function)(
    // Input context, as passed in create function
    void *context,
    // Object buffer
    const void *buf,
    // Number of elements in buffer; could represent bytes, element counts, etc.
    MPI_Count count,
    // State to be created for packing
    void **state
);
/* Query the packed size of the buffer */
typedef int (MPI_Type_custom_query_function)(
    // State object
    void *state,
    // User-provided buffer (not packed)
    const void *buf,
    // Number of elements in buffer; could represent bytes, element counts, etc.
    MPI_Count count,
    // Output number of bytes to be packed or expected on receive
    MPI_Count *packed_size
);
typedef int (MPI_Type_custom_pack_function)(
    // State information for packing
    void *state,
    const void *buf,
    MPI_Count count,
    // Virtual offset in bytes into the packed buffer
    MPI_Count offset,
    // Destination buffer
    void *dst,
    // Number of bytes to be written to destination buffer
    MPI_Count dst_size,
    MPI_Count *used
);
typedef int (MPI_Type_custom_unpack_function)(
    // State information for unpacking
    void *state,
    void *buf,
    MPI_Count count,
    // Virtual offset in bytes into the buffer being unpacked
    MPI_Count offset,
    // Incoming buffer to be unpacked
    const void *src,
    // Number of bytes in current buffer to be unpacked
    MPI_Count src_size
);
typedef int (MPI_Type_custom_region_count_function)(
    void *state,
    // Buffer pointer
    void *buf,
    // Number of elements in send buffer.
    MPI_Count count,
    // Number of memory regions
    MPI_Count *region_count
);
typedef int (MPI_Type_custom_region_function)(
    void *state,
    // Buffer pointer
    void *buf,
    // Number of elements in send buffer
    MPI_Count count,
    // Number of regions
    MPI_Count region_count,
    // Lengths of each region (out)
    MPI_Count reg_lens[],
    // Pointers to each region (out)
    void *reg_bases[],
    // Types for each region
    MPI_Datatype types[]
);
typedef int (MPI_Type_custom_state_free_function)(void *state);

int MPI_Type_create_custom(MPI_Type_custom_state_function *statefn,
                           MPI_Type_custom_state_free_function *state_freefn,
                           MPI_Type_custom_query_function *queryfn,
                           MPI_Type_custom_pack_function *packfn,
                           MPI_Type_custom_unpack_function *unpackfn,
                           MPI_Type_custom_region_count_function *region_countfn,
                           MPI_Type_custom_region_function *regionfn,
                           void *context, // Context pointer to be stored for initializing state
                           int inorder, // Flag indicating in-order pack requirement
                           MPI_Datatype *type);

/* Idea: use a builder-like interface */

/* Constants */
#define MPI_SUCCESS 0
#define MPI_ERR_INTERNAL 1

#if __cplusplus
};
#endif

#endif /* _MPI_H_ */
