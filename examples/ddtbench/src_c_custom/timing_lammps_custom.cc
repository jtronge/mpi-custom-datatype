// Copyright (c) 2012 The Trustees of University of Illinois. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include <stdio.h>
#include <stdlib.h>

#include "ddtbench.h"

#include <mpi.h>

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

static int state_cb(void *context, const void *buf, MPI_Count count, void **state) {
  *state = context;
  return MPI_SUCCESS;
}
//int free_cb(void *state) { return MPI_SUCCESS; }
static int query_cb(void *state, const void *buf, MPI_Count count, MPI_Count *packed_size) {
  field_info_t *info = (field_info_t*)state;
  *packed_size = 8*sizeof(double) * info->icount;
  return MPI_SUCCESS;
}
static int pack_cb(
  void *state, const void *buf,
  MPI_Count count, MPI_Count offset,
  void *dst_v, MPI_Count dst_size,
  MPI_Count *used)
{
  MPI_Count pos = 0;
  field_info_t *info = (field_info_t*)state;
  // how many iteration do we do? Only do full iterations
  MPI_Count mycount = dst_size / (8*sizeof(double));
  if (count < mycount) {
    mycount = count;
  }

  double *dst = (double*) dst_v;

  double *__restrict__ atag = info->atag;
  double *__restrict__ atype = info->atype;
  double *__restrict__ amask = info->amask;
  double *__restrict__ amolecule = info->amolecule;
  double *__restrict__ aq = info->aq;
  double *__restrict__ ax = info->ax;
  // compute the k to start with
  MPI_Count k = offset / (8*sizeof(double));
  for(; k<mycount ; k++ ) {
    int l=info->temp_displacement[idx2D(k, info->i, info->icount)];
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

static int unpack_cb(
  void *state, void *buf, MPI_Count count,
  MPI_Count offset, const void *src_v,
  MPI_Count src_size)
{
  MPI_Count pos = 0;
  field_info_t *info = (field_info_t*)state;
  char *src = (char*) src_v;

  MPI_Count mycount = src_size / (8*sizeof(double));
  if (count < mycount) {
    mycount = count;
  }

  double *__restrict__ atag = info->atag;
  double *__restrict__ atype = info->atype;
  double *__restrict__ amask = info->amask;
  double *__restrict__ amolecule = info->amolecule;
  double *__restrict__ aq = info->aq;
  double *__restrict__ ax = info->ax;

  MPI_Count k = offset / (8*sizeof(double));

  for(; k<mycount ; k++ ) {
    int l=info->DIM1+k;
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


void timing_lammps_full_custom(int DIM1, int icount, int* list, int outer_loop, int inner_loop, int* correct_flag, int* ptypesize, char* testname, MPI_Comm local_communicator) {

  double* atag;
  double* atype;
  double* amask;
  double* amolecule;
  double* aq;
  double* ax;

  double* buffer;

  int myrank;
  int typesize, bytes, base, pos, isize;

  int* temp_displacement;

  char method[50];

//! just some statements to prevent compiler warnings of unused variables
//! those parameter are included for future features
  *correct_flag = 0;
  *ptypesize = 0;
//  typesize = filehandle_debug;

  atag = (double*)malloc( (DIM1+icount) * sizeof(double) );
  atype = (double*)malloc( (DIM1+icount) * sizeof(double) );
  amask = (double*)malloc( (DIM1+icount) * sizeof(double) );
  amolecule = (double*)malloc( (DIM1+icount) * sizeof(double) );
  aq = (double*)malloc( (DIM1+icount) * sizeof(double) );
  ax  = (double*)malloc( 3 * (DIM1+icount) * sizeof(double) );

//conversion from fortran to c
  temp_displacement = (int*)malloc( icount * outer_loop * sizeof(int) );
  for( int i = 0 ; i<outer_loop ; i++ ) {
    for( int j = 0 ; j<icount ; j++ ) {
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
    snprintf( method, 50, "custom" );

    bytes = icount * 8 * sizeof(double);

    timing_init( testname, &method[0], bytes );
  }

  MPI_Datatype type;
  MPI_Type_create_custom(state_cb, NULL, query_cb, pack_cb, unpack_cb,
                         NULL, NULL, &info, 0, &type);

  for( int i=0 ; i<outer_loop ; i++ ) {
    MPI_Status status;
    info.i = i;

    isize = 8*icount;
    bytes = isize * sizeof(double);
    buffer = (double*)malloc( isize * sizeof(double) );

    if ( myrank == 0 ) {
      timing_record(1);
    }

    for( int j=0 ; j<inner_loop ; j++ ) {

      if ( myrank == 0 ) {
        MPI_Send( &buffer[0], icount, type, 1, itag, local_communicator );
        MPI_Recv( &buffer[0], icount, type, 1, itag, local_communicator, &status );
        timing_record(3);
      } else {
        MPI_Recv( &buffer[0], icount, type, 0, itag, local_communicator, &status );
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
