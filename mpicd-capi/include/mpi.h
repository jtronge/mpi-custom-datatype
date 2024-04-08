#ifndef _MPI_H_
#define _MPI_H_

/* For simplicity MPI_Comm and other handles are defined to be integers */
typedef int MPI_Comm;
typedef int MPI_Datatype;

typedef struct MPI_Status {
    int MPI_SOURCE;
    int MPI_TAG;
    int MPI_ERROR;
} MPI_Status;

#define MPI_COMM_WORLD 1

int MPI_Init(int *argc, char **argv[]);
int MPI_Finalize(void);
int MPI_Comm_size(MPI_Comm comm, int *size);
int MPI_Comm_rank(MPI_Comm comm, int *rank);

/* P2P functions */
int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm);
int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm, MPI_Status *status);

#endif /* _MPI_H_ */
