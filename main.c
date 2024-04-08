#include <stdlib.h>
#include <mpi.h>

int main()
{
    MPI_Init(NULL, NULL);

    MPI_Finalize();
}
