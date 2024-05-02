/*
 * Copyright (C) 2002-2023 the Network-Based Computing Laboratory
 * (NBCL), The Ohio State University.
 *
 * Contact: Dr. D. K. Panda (panda@cse.ohio-state.edu)
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level OMB directory.
 */
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <mpi.h>

#define FIELD_WIDTH 10
#define FLOAT_PRECISION 2

void parse_arg(int argc, char *argv[], const char *long_opt,
               const char *short_opt, int *value, int default_value);
double calculate_total(double, double, double, int);

int main(int argc, char *argv[])
{
    int myid, numprocs, i, j, k;
    int size;
    char **s_buf, **r_buf;
    double t_start = 0.0, t_end = 0.0, t_lo = 0.0, t_total = 0.0;
    double tmp_total = 0.0;
    size_t num_elements = 0;
    MPI_Comm omb_comm = MPI_COMM_WORLD;
    MPI_Datatype datatype = MPI_BYTE;
    MPI_Request *request = NULL;
    MPI_Status *reqstat = NULL;
    int warmup_validation;
    int window_size;
    int skip;
    int iterations;
    int min_message_size;
    int max_message_size;
    int mpi_type_size = 1;

    parse_arg(argc, argv, "--warmup", NULL, &warmup_validation, 16);
    parse_arg(argc, argv, "--skip", "-s", &skip, 10);
    parse_arg(argc, argv, "--iterations", "-i", &iterations, 128);
    parse_arg(argc, argv, "--window-size", "-w", &window_size, 64);
    parse_arg(argc, argv, "--min-size", NULL, &min_message_size, 2);
    parse_arg(argc, argv, "--max-size", NULL, &max_message_size, 128);

    s_buf = malloc(sizeof(char *) * window_size);
    r_buf = malloc(sizeof(char *) * window_size);
    request = malloc(sizeof(*request) * window_size);
    reqstat = malloc(sizeof(*reqstat) * window_size);

    MPI_Init(&argc, &argv);
    
    MPI_Comm_rank(omb_comm, &myid);
    MPI_Comm_size(omb_comm, &numprocs);

    if (numprocs != 2) {
        if (myid == 0) {
            fprintf(stderr, "This test requires exactly two processes\n");
        }

        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    /* Bandwidth test */
    fflush(stdout);
    for (size = min_message_size; size <= max_message_size; size *= 2) {
        num_elements = size / mpi_type_size;
        if (0 == num_elements) {
            continue;
        }

        for (i = 0; i < window_size; i++) {
            s_buf[i] = malloc(sizeof(char) * size);
            r_buf[i] = malloc(sizeof(char) * size);
        }

        MPI_Barrier(omb_comm);
        t_total = 0.0;

        for (i = 0; i < iterations + skip; i++) {
            if (myid == 0) {
                for (k = 0; k <= warmup_validation; k++) {
                    if (i >= skip &&
                        k == warmup_validation) {
                        t_start = MPI_Wtime();
                    }

                    for (j = 0; j < window_size; j++) {
                        MPI_Isend(s_buf[j], num_elements,
                                  datatype, 1, 100,
                                  omb_comm, request + j);
                    }
                    MPI_Waitall(window_size, request, reqstat);

                    MPI_Recv(r_buf[0], 1, MPI_BYTE, 1, 101, omb_comm, &reqstat[0]);

                    if (i >= skip &&
                        k == warmup_validation) {
                        t_end = MPI_Wtime();
                        t_total += calculate_total(t_start, t_end, t_lo,
                                                   window_size);
                            /* tmp_total =
                                omb_ddt_transmit_size / 1e6 * window_size; */
                        tmp_total = size / 1e6 * window_size;
                    }
                }
            } else if (myid == 1) {
                for (k = 0; k <= warmup_validation; k++) {
                    for (j = 0; j < window_size; j++) {
                        MPI_Irecv(r_buf[j], num_elements, datatype, 0, 100, omb_comm, request + j);
                    }
                    MPI_Waitall(window_size, request, reqstat);

                    MPI_Send(s_buf[0], 1, MPI_BYTE, 0, 101, omb_comm);
                }
            }
        }

        if (myid == 0) {
            /* tmp_total = omb_ddt_transmit_size / 1e6 * options.iterations * window_size; */
            tmp_total = size / 1e6 * iterations * window_size;
            fprintf(stdout, "%-*d", 10, size);
            fprintf(stdout, "%*.*f", FIELD_WIDTH, FLOAT_PRECISION,
                    tmp_total / t_total);
            fprintf(stdout, "\n");
            fflush(stdout);
        }
        for (i = 0; i < window_size; i++) {
            free(s_buf[i]);
            free(r_buf[i]);
        }
    }

    free(s_buf);
    free(r_buf);
    free(request);
    free(reqstat);
    MPI_Finalize();

    return EXIT_SUCCESS;
}

double calculate_total(double t_start, double t_end, double t_lo,
                       int window_size)
{
/*
    double t_total;

    if (options.dst == 'M' && options.MMdst == 'D') {
        t_total = ((t_end - t_start) - (t_lo * window_size));
    } else {
        t_total = (t_end - t_start);
    }

    return t_total;
*/
    return t_end - t_start;
}

void parse_arg(int argc, char *argv[], const char *long_opt, const char *short_opt, int *value, int default_value)
{
    *value = default_value;
    for (int i = 1; i < argc; ++i) {
        if ((strcmp(argv[i], long_opt) == 0 || (short_opt != NULL && strcmp(argv[i], short_opt) == 0)) && (i + 1) < argc) {
            *value = strtol(argv[i + 1], NULL, 10);
        }
    }
}
