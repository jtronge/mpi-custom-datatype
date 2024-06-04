// Copyright (c) 2012 The Trustees of University of Illinois. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include <stdio.h>
#include <stdlib.h>

#include "mpi.h"

#include "ddtbench.h"

#define itag 0

static inline int idx2D(int x, int y, int DIM1) {
  return x+y*DIM1;
}

typedef struct field_info_t {
  double *atag;
  double *atype;
  double *amask;
  double *amolecule;
  double *aq;
  double *ax;
  int *temp_displacement;
  int i, icount, DIM1;
} field_info_t;

int state_cb(void *context, const void *buf, MPI_Count count, void **state) {
  *state = context;
  return MPI_SUCCESS;
}
//int free_cb(void *state) { return MPI_SUCCESS; }
int query_cb(void *state, const void *buf, MPI_Count count, MPI_Count *packed_size) {
  field_info_t *info = (field_info_t*)state;
  *packed_size = 8*sizeof(double) * state->icount;
  return MPI_SUCCESS;
}
int pack_cb(
  void *state, const void *buf,
  MPI_Count count, MPI_Count offset,
  void *dst, MPI_Count dst_size,
  MPI_Count *used)
{
  MPI_Count pos = 0;
  field_info_t *info = (field_info_t*)state;
  // how many iteration do we do? Only do full iterations
  MPI_Count mycount = dst_size / (8*sizeof(double));
  if (count < mycount) {
    mycount = count;
  }

  double *__restrict__ atag = state->atag;
  double *__restrict__ atype = state->atype;
  double *__restrict__ amask = state->amask;
  double *__restrict__ amolecule = state->amolecule;
  double *__restrict__ aq = state->aq;
  double *__restrict__ ax = state->ax;
  // compute the k to start with
  MPI_Count k = offset / (8*sizeof(double));
  for(; k<mycount ; k++ ) {
    int l=temp_displacement[idx2D(k, state->i, state->icount)];
    dst[pos++] = ax[idx2D(0,l,3)];
    dst[pos++] = ax[idx2D(1,l,3)];
    dst[pos++] = ax[idx2D(2,l,3)];
    dst[pos++] = atag[l];
    dst[pos++] = atype[l];
    dst[pos++] = amask[l];
    dst[pos++] = amolecule[l];
    dst[pos++] = aq[l];
  }

  *used = mycount*8*sizeof(double);

  return MPI_SUCCESS;
}

int unpack_cb(
  void *state, void *buf, MPI_Count count,
  MPI_Count offset, const void *src,
  MPI_Count src_size)
{
  MPI_Count pos = 0;
  field_info_t *info = (field_info_t*)state;

  MPI_Count mycount = src_size / (8*sizeof(double));
  if (count < mycount) {
    mycount = count;
  }

  double *__restrict__ atag = state->atag;
  double *__restrict__ atype = state->atype;
  double *__restrict__ amask = state->amask;
  double *__restrict__ amolecule = state->amolecule;
  double *__restrict__ aq = state->aq;
  double *__restrict__ ax = state->ax;

  MPI_Count k = offset / (8*sizeof(double));

  for(; k<mycount ; k++ ) {
    int l=state->DIM1+k;
    ax[idx2D(0,l,3)] = src[pos++];
    ax[idx2D(1,l,3)] = src[pos++];
    ax[idx2D(2,l,3)] = src[pos++];
    atag[l] = src[pos++];
    atype[l] = src[pos++];
    amask[l] = src[pos++];
    amolecule[l] = src[pos++];
    aq[l] = src[pos++];
  }

  return MPI_SUCCESS;
}


void timing_lammps_custom_dt( int DIM1, int icount, int* list, int outer_loop, int inner_loop, int* correct_flag, int* ptypesize, char* testname) {

  double* atag;
  double* atype;
  double* amask;
  double* amolecule;
  double* aq;
  double* ax;

  double* buffer;

  int myrank;
  int i, j, k, l, typesize, bytes, base, pos, isize;

  int* temp_displacement;

  char method[50];

//! just some statements to prevent compiler warnings of unused variables
//! those parameter are included for future features
  *correct_flag = 0;
  *ptypesize = 0;
//  typesize = filehandle_debug;

  atag = malloc( (DIM1+icount) * sizeof(double) );
  atype = malloc( (DIM1+icount) * sizeof(double) );
  amask = malloc( (DIM1+icount) * sizeof(double) );
  amolecule = malloc( (DIM1+icount) * sizeof(double) );
  aq = malloc( (DIM1+icount) * sizeof(double) );
  ax  = malloc( 3 * (DIM1+icount) * sizeof(double) );

//conversion from fortran to c
  temp_displacement = malloc( icount * outer_loop * sizeof(int) );
  for( i = 0 ; i<outer_loop ; i++ ) {
    for( j = 0 ; j<icount ; j++ ) {
      temp_displacement[idx2D(j,i,icount)] = list[idx2D(j,i,icount)] - 1;
    }
  }

  field_info_t info = {
    .atag = atag,
    .atype = atype,
    .amask = amask,
    .amolecule = amolecule,
    .aq = aq,
    .ax = ax,
    .temp_displacement = temp_displacement,
    .icount = icount,
    .DIM1 = DIM1
  };

  MPI_Comm_rank( local_communicator, &myrank );

  base = myrank * (8*(DIM1+icount)) + 1;
  utilities_fill_unique_array_2D_double( &ax[0], 3, DIM1+icount, base );
  base = base + 3*(DIM1+icount);
  utilities_fill_unique_array_1D_double( &atag[0], DIM1+icount, base );
  base = base + DIM1 + icount;
  utilities_fill_unique_array_1D_double( &atype[0], DIM1+icount, base );
  base = base + DIM1 + icount;
  utilities_fill_unique_array_1D_double( &amask[0], DIM1+icount, base );
  base = base + DIM1 + icount;
  utilities_fill_unique_array_1D_double( &aq[0], DIM1+icount, base );
  base = base + DIM1 + icount;
  utilities_fill_unique_array_1D_double( &amolecule[0], DIM1+icount, base );

  if ( myrank == 0 ) {
    snprintf( method, 50, "manual" );

    bytes = icount * 8 * sizeof(double);

    timing_init( testname, &method[0], bytes );
  }

  MPI_Datatype type;
  MPI_Type_create_custom(state_cb, NULL, query_cb, pack_cb, unpack_cb,
                         NULL, NULL, &info, &type);

  for( i=0 ; i<outer_loop ; i++ ) {

    info.i = i;

    isize = 8*icount;
    bytes = isize * sizeof(double);
    buffer = malloc( isize * sizeof(double) );

    if ( myrank == 0 ) {
      timing_record(1);
    }

    for( j=0 ; j<inner_loop ; j++ ) {

      if ( myrank == 0 ) {
        timing_record(2);
        MPI_Send( &buffer[0], icount, type, 1, itag, local_communicator );
        MPI_Recv( &buffer[0], icount, type, 1, itag, local_communicator, MPI_STATUS_IGNORE );
        timing_record(3);
        timing_record(4);
      } else {
        MPI_Recv( &buffer[0], icount, type, 0, itag, local_communicator, MPI_STATUS_IGNORE );
        MPI_Send( &buffer[0], icount, type, 0, itag, local_communicator );
      }

    } //! inner loop

    free( buffer );

    if ( myrank == 0 ) {
      timing_record(5);
    }

  } //! outer loop

  if ( myrank == 0 ) {
    timing_print( 1 );
  }

  free(temp_displacement);

  free(atag);
  free(atype);
  free(amask);
  free(amolecule);
  free(aq);
  free(ax);
}
