// Copyright (c) 2012 The Trustees of University of Illinois. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include <stdio.h>
#include <stdlib.h>

#include "mpi.h"
#include "../src_c/ddtbench.h"

#define itag 0

void timing_basic_ping_pong_nelements( int DIM1, int loop, char* testname, MPI_Comm local_communicator) {

  float* array;
  int myrank;
  int base, typesize, bytes, i;
  char method[50];

  MPI_Win win;

  MPI_Comm_rank( local_communicator, &myrank );

  array = malloc( DIM1 * sizeof(float) );
  MPI_Win_create( array, DIM1 * sizeof(float), sizeof(float), MPI_INFO_NULL, local_communicator, &win );
  MPI_Win_fence( 0 /* assert */, win ); /* initial fence to open epoch */
  
  base = myrank * DIM1 + 1;
  utilities_fill_unique_array_1D_float( &array[0], DIM1, base );

  if ( myrank == 0 ) {
    snprintf(&method[0], 50, "reference");

    MPI_Type_size( MPI_FLOAT, &typesize );
    bytes = typesize * DIM1;

    timing_init( testname, &method[0], bytes );
  }

  for( i=0 ; i<loop ; i++ ){
    if ( myrank == 0 ) {
      MPI_Put( &array[0], DIM1, MPI_FLOAT, 1, 0, DIM1, MPI_FLOAT, win );
      MPI_Win_fence( 0 /* assert */, win );
      MPI_Win_fence( 0 /* assert */, win );
      timing_record(3);
    } else {
      MPI_Win_fence( 0 /* assert */, win );
      MPI_Put( &array[0], DIM1, MPI_FLOAT, 0, 0, DIM1, MPI_FLOAT, win );
      MPI_Win_fence( 0 /* assert */, win );
    }
  }

  if ( myrank == 0 ) {
    timing_print( 1 );
  }

  MPI_Win_free( &win );

  free( array );
}

void timing_basic_alltoall_nelements( int DIM1, int procs, int loop, char* testname, MPI_Comm local_communicator) {
      
  float* send_array;
  float* recv_array;
  
  int myrank;
  int commsize;
  int base, typesize, bytes, i, j;
  char method[50];

  MPI_Win win1, win2;

  MPI_Comm_rank( local_communicator, &myrank );
  MPI_Comm_size( local_communicator, &commsize );

  send_array = malloc( DIM1 * procs * sizeof(float) );
  recv_array = malloc( DIM1 * procs * sizeof(float) );

  MPI_Win_create( send_array, DIM1 * procs * sizeof(float), sizeof(float), MPI_INFO_NULL, local_communicator, &win1 );
  MPI_Win_create( recv_array, DIM1 * procs * sizeof(float), sizeof(float), MPI_INFO_NULL, local_communicator, &win2 );

  MPI_Win_fence( 0 /* assert */, win1 ); /* initial fence to open epoch */
  MPI_Win_fence( 0 /* assert */, win2 ); /* initial fence to open epoch */

  base = myrank * DIM1 + 1;
  utilities_fill_unique_array_1D_float( &send_array[0], DIM1, base );

  if ( myrank == 0 ) {
    snprintf(method, 50, "reference");
        
    MPI_Type_size( MPI_FLOAT, &typesize );
    bytes = typesize * DIM1 * procs;

    timing_init( testname, &method[0], bytes );
  }

  for( i=0 ; i<loop ; i++ ) {
    for( j=0 ; j<commsize ; j++ ) {
      MPI_Put( &send_array[j*DIM1], DIM1, MPI_FLOAT, j /* target rank */, myrank*DIM1 /* offset */, DIM1, MPI_FLOAT, win2 );
    }
    MPI_Win_fence( 0 /* assert */, win2 );
    for( j=0 ; j<commsize ; j++ ) {
      MPI_Put( &recv_array[j*DIM1], DIM1, MPI_FLOAT, j /* target rank */, myrank*DIM1 /* offset */, DIM1, MPI_FLOAT, win1 );
    }
    MPI_Win_fence( 0 /* assert */, win1 );
    if ( myrank == 0 ) {
      timing_record(3);
    }
  }

  if ( myrank == 0 ) {
    timing_print( 1 );
  }

  MPI_Win_free( &win1 );
  MPI_Win_free( &win2 );

  free( send_array );
  free( recv_array );
}
