// Copyright (c) 2012 The Trustees of University of Illinois. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include <stdio.h>
#include <stdlib.h>
#include <cassert>
#include <iostream>

#include "mpi.h"

#include "ddtbench.h"

#define itag 0

static inline int idx5D(int x, int y, int z, int t, int u, int DIM1, int DIM2, int DIM3, int DIM4) {
  return x+DIM1*(y+DIM2*(z+DIM3*(t+DIM4*u)));
}


struct pack_info_t {
  int DIM2, DIM3, DIM4, DIM5;
};

int region_count_cb(
  void *state,
  // Buffer pointer
  void *buf,
  // Number of elements in send buffer.
  MPI_Count count,
  // Number of memory regions
  MPI_Count *region_count)
{
  pack_info_t *info = (pack_info_t*)state;
  *region_count = info->DIM5*2;

  return MPI_SUCCESS;
}

int region_query_cb(
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
  MPI_Datatype reg_types[])
{
  pack_info_t *info = (pack_info_t*)state;
  assert(region_count == info->DIM5*2); // make sure we can store them all at once
  assert(count == 1); // we know we only transfer one at a time

  float *array = (float*)buf;
  int pos = 0;
  for( int k=0 ; k<info->DIM5 ; k++ ) {
    for( int l=0 ; l<info->DIM4 ; l+=info->DIM4/2 ) {
      reg_bases[pos] = &array[idx5D(0,0,0,l,k,6,info->DIM2,info->DIM3,info->DIM4)];
      reg_lens[pos]  = info->DIM3/2 * info->DIM2 * 6 * sizeof(float);
      reg_types[pos] = MPI_BYTE;
      pos++;
    }
  }

  return MPI_SUCCESS;
}

static int state_cb(void *context, const void *buf, MPI_Count count, void **state) {
  pack_info_t *info = (pack_info_t*)context;
  *state = info;
  return MPI_SUCCESS;
}

void timing_milc_su3_zdown_custom( int DIM2, int DIM3, int DIM4, int DIM5, int outer_loop, int inner_loop, int* correct_flag, int* ptypesize, char* testname, MPI_Comm local_communicator ) {

  float* array;
  float* buffer;

  int myrank;
  int i, j;
  int base, bytes, pos, dsize;

  int typesize = sizeof(float);

  char method[50];

//! just some statements to prevent compiler warnings of unused variables
//! those parameter are included for future features
  *correct_flag = 0;
  *ptypesize = 0;
//  typesize = filehandle_debug;

  array = (float*)malloc( 2 * 3 * DIM2 * DIM3 * DIM4 * DIM5 * sizeof(float) );

  MPI_Comm_rank( local_communicator, &myrank );

  base = myrank * 3 * DIM2 * DIM3 * DIM4 * DIM5 * 2 + 1;
  utilities_fill_unique_array_5D_float( &array[0], 6, DIM2, DIM3, DIM4, DIM5, base );

  if ( myrank == 0 ) {
    snprintf( &method[0], 50, "custom" );

    bytes = 2 * DIM5 * DIM2*DIM3/2 * 3 * 2 * typesize;

    timing_init( testname, &method[0], bytes );
  }

  pack_info_t info = {
    .DIM2 = DIM2,
    .DIM3 = DIM3,
    .DIM4 = DIM4,
    .DIM5 = DIM5
  };

  MPI_Datatype type;
  MPI_Type_create_custom(state_cb, NULL, NULL, NULL, NULL,
                         &region_count_cb, &region_query_cb, &info, 0, &type);

  MPI_Status status;

  for( i=0 ; i<outer_loop ; i++ ) {

//! modelling the zdown direction
    if ( myrank == 0 ) {
      timing_record(1);
    }

    for( j=0 ; j<inner_loop ; j++) {

      if ( myrank == 0 ) {
        MPI_Send( &array[0], 1, type, 1, itag, local_communicator );
        MPI_Recv( &array[0], 1, type, 1, itag, local_communicator, &status );
        timing_record(3);
      } else {
        MPI_Recv( &array[0], 1, type, 0, itag, local_communicator, &status );
        MPI_Send( &array[0], 1, type, 0, itag, local_communicator );
      }

    } //! inner loop

    if ( myrank == 0 ) {
      timing_record(5);
    }

  } //! outer loop

  if ( myrank == 0 ) {
    timing_print( 1 );
  }

  free(array);
}
