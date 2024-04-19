#ifndef _MPI_H_
#define _MPI_H_

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
int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest,
              int tag, MPI_Comm comm, MPI_Request *request);
int MPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source, int tag,
              MPI_Comm comm, MPI_Request *request);

int MPI_Waitall(int count, MPI_Request array_of_requests[], MPI_Status *array_of_statuses);

/*
 * All functions return 0 on success and non-zero on failure.
 */
typedef int (MPI_Type_custom_pack_state_function)(
    void *context, // Input context, as passed in create function
    const void *src, // Source buffer to unpack from
    MPI_Count src_count, // Number of elements in buffer; could represent bytes, element counts, etc.
    void **state // State to be created for packing
);
typedef int (MPI_Type_custom_unpack_state_function)(
    void *context, // Input context, as passed in create function
    void *dst, // Destination buffer
    MPI_Count dst_count, // Number of element in destination buffer; could represent bytes, element counts, etc.
    void **state // State to be created for unpacking
);
typedef int (MPI_Type_custom_query_function)(
    void *context, // Input context, as passed to create function
    const void *buf, // User-provided buffer (not packed)
    MPI_Count count, // Number of elements in buffer; could represent bytes, element counts, etc.
    MPI_Count *packed_size // Output number of bytes to be packed or expected on receive
);
typedef int (MPI_Type_custom_pack_function)(
    void *state, // State information for packing
    MPI_Count offset, // Virtual offset in bytes into the packed buffer
    void *dst, // Destination buffer
    MPI_Count dst_size, // Number of bytes to be written to destination buffer
    MPI_Count *used
);
typedef int (MPI_Type_custom_unpack_function)(
    void *state, // State information for unpacking
    MPI_Count offset, // Virtual offset in bytes into the buffer being unpacked
    const void *src, // Incoming buffer to be unpacked
    MPI_Count src_size // Number of bytes in current buffer to be unpacked
);
typedef int (MPI_Type_custom_pack_state_free_function)(void *state);
typedef int (MPI_Type_custom_unpack_state_free_function)(void *state);

int MPI_Type_create_custom(MPI_Type_custom_pack_state_function *pack_statefn,
                           MPI_Type_custom_unpack_state_function *unpack_statefn,
                           MPI_Type_custom_query_function *queryfn,
                           MPI_Type_custom_pack_function *packfn,
                           MPI_Type_custom_unpack_function *unpackfn,
                           MPI_Type_custom_pack_state_free_function *pack_freefn,
                           MPI_Type_custom_unpack_state_free_function *unpack_freefn,
                           void *context, // Context pointer to be stored for initializing state
                           MPI_Datatype *type);

/* Idea: use a builder-like interface */

/* Constants */
#define MPI_SUCCESS 0
#define MPI_ERR_INTERNAL 1

#endif /* _MPI_H_ */
