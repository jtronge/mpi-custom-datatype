// Copyright (c) 2012 The Trustees of University of Illinois. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include <stdio.h>
#include <stdlib.h>

#include "generator.h"

#include "ddtbench.h"

#include <mpi.h>

#define itag 0

static inline int idx2D(int x, int y, int DIM1) {
  return x+DIM1*y;
}
static inline int idx3D(int x, int y, int z, int DIM1, int DIM2) {
  return x+DIM1*(y+z*DIM2);
}
static inline int idx4D(int x, int y, int z, int t, int DIM1, int DIM2, int DIM3) {
  return x+DIM1*(y+DIM2*(z+DIM3*t));
}

struct pack_info_t {
  coro::generator<MPI_Count> coro;
  size_t buf_size;
  int ie, is, je, js, ke, ks, number_2D, number_3D, number_4D, dim1, dim2, dim3;
  int pack_unpack; // 0: pack, 1: unpack
  double* buffer;
  float **array2Ds;
  float **array3Ds;
  float **array4Ds;
  int *limit_4D_arrays;
  int param_first_scalar;
};

/* coroutine used for packing, yields packed bytes count when full*/
static coro::generator<MPI_Count> pack_unpack_coro(pack_info_t *info)
{
  ssize_t size = info->buf_size;
  /* the unit packing size */
  int ie = info->ie, is = info->is;
  int je = info->je, js = info->js;
  int ke = info->ke, ks = info->ks;
  int dim1 = info->dim1;
  int dim2 = info->dim2;
  int dim3 = info->dim3;
  double *buffer = info->buffer;
  int ilen = ie-is+1;
  size_t unit_pack_size = ilen*sizeof(double);
  float **array2Ds = info->array2Ds;
  float **array3Ds = info->array3Ds;
  float **array4Ds = info->array4Ds;
  int counter = 0;

  if (info->pack_unpack == 0) {
    /* pack 2D */
    for(int m = 0; m<info->number_2D ; m++) {
      for(int k=js ; k<=je ; k++) {
        for(int l=is ; l<=ie ; l++) {
          buffer[counter++] = *(array2Ds[m]+idx2D(l,k,dim1));
        }
        size -= unit_pack_size;
        assert(size >= 0);
        if (size < unit_pack_size) {
          co_yield info->buf_size - size; // insufficient space for another round
          /* we're back, reset state */
          size = info->buf_size;
          buffer = info->buffer;
          counter = 0;
        }
      }
    }
    for(int m=0 ; m<info->number_3D ; m++ ) {
      for(int k=js ; k<=je ; k++ ) {
        for(int l=ks ; l<=ke ; l++ ) {
          for(int n=is ; n<=ie ; n++ ) {
            buffer[counter++] = *(array3Ds[m]+idx3D(n,l,k,dim1,dim2));
          }
          size -= unit_pack_size;
          assert(size >= 0);
          if (size < unit_pack_size) {
            co_yield info->buf_size - size; // insufficient space for another round
            /* we're back, reset state */
            size = info->buf_size;
            buffer = info->buffer;
            counter = 0;
          }
        }
      }
    }
    for(int m=0; m<info->number_4D; m++) {
      for(int k=info->param_first_scalar; k<info->limit_4D_arrays[m]; k++) {
        for(int l=js; l<=je; l++) {
          for(int n=ks ; n<=ke; n++) {
            for(int o=is; o<=ie; o++) {
              buffer[counter++] = *(array4Ds[m]+idx4D(o,n,l,k,dim1,dim2,dim3));
            }
            size -= unit_pack_size;
            assert(size >= 0);
            if (size < unit_pack_size) {
              co_yield info->buf_size - size; // insufficient space for another round
              /* we're back, reset state */
              size = info->buf_size;
              buffer = info->buffer;
              counter = 0;
            }
          }
        }
      }
    }
  } else {
    /* unpack */
    for(int m=0 ; m<info->number_2D ; m++) {
      for(int k=js ; k<=je ; k++) {
        for(int l=is ; l<=ie ; l++) {
          *(array2Ds[m]+idx2D(l,k,dim1)) = buffer[counter++];
        }
        size -= unit_pack_size;
        assert(size >= 0);
        if (size < unit_pack_size) {
          co_yield info->buf_size - size; // insufficient space for another round
          /* we're back, reset state */
          size = info->buf_size;
          buffer = info->buffer;
          counter = 0;
        }
      }
    }
    for(int m=0 ; m<info->number_3D ; m++ ) {
      for(int k=js ; k<=je ; k++ ) {
        for(int l=ks ; l<=ke ; l++ ) {
          for(int n=is ; n<=ie ; n++ ) {
            *(array3Ds[m]+idx3D(n,l,k,dim1,dim2)) = buffer[counter++];
          }
          size -= unit_pack_size;
          assert(size >= 0);
          if (size < unit_pack_size) {
            co_yield info->buf_size - size; // insufficient space for another round
            /* we're back, reset state */
            size = info->buf_size;
            buffer = info->buffer;
            counter = 0;
          }
        }
      }
    }
    for(int m=0 ; m<info->number_4D ; m++ ) {
      for(int k=info->param_first_scalar ; k<info->limit_4D_arrays[m] ; k++) {
        for(int l=js ; l<=je ; l++ ) {
          for(int n=ks ; n<=ke ; n++ ) {
            for(int o=is ; o<=ie ; o++ ) {
              *(array4Ds[m]+idx4D(o,n,l,k,dim1,dim2,dim3)) = buffer[counter++];
            }
            size -= unit_pack_size;
            assert(size >= 0);
            if (size < unit_pack_size) {
              co_yield info->buf_size - size; // insufficient space for another round
              /* we're back, reset state */
              size = info->buf_size;
              buffer = info->buffer;
              counter = 0;
            }
          }
        }
      }
    }
  }

  co_yield info->buf_size - size;
}

static int state_cb(void *context, const void *buf, MPI_Count count, void **state) {
  pack_info_t *info = (pack_info_t*)context;
  info->coro = pack_unpack_coro(info);
  *state = info;
  return MPI_SUCCESS;
}

static int query_cb(void *state, const void *buf, MPI_Count count, MPI_Count *packed_size) {
  pack_info_t *info = (pack_info_t*)state;
  int ie = info->ie, is = info->is;
  int je = info->je, js = info->js;
  int ke = info->ke, ks = info->ks;
  int ilen = ie-is+1;
  int jlen = je-js+1;
  int klen = ke-ks+1;

  // 2D and 3D
  size_t elem_count = (ilen*jlen*info->number_2D) + (ilen*jlen*klen*info->number_3D);
  // 4D
  for(int m=0; m<info->number_4D; m++) {
    for(int k=info->param_first_scalar; k<info->limit_4D_arrays[m]; k++) {
      elem_count += (ilen*jlen*klen);
    }
  }
  *packed_size = sizeof(double) * elem_count * count;
  return MPI_SUCCESS;
}

static int pack_cb(
  void *state, const void *buf,
  MPI_Count count, MPI_Count offset,
  void *dst, MPI_Count dst_size,
  MPI_Count *used)
{
  MPI_Count pos = 0;
  pack_info_t *info = (pack_info_t*)state;
  info->pack_unpack = 0;
  info->buf_size = dst_size;
  info->buffer = (double*)dst;
  if (info->coro.next()) {
    *used = info->coro.getValue().value();
    return MPI_SUCCESS;
  } else {
    throw std::runtime_error("Pack called without data left!");
    //return MPI_ERROR;
  }
}

static int unpack_cb(
  void *state, void *buf, MPI_Count count,
  MPI_Count offset, const void *src,
  MPI_Count src_size)
{
  pack_info_t *info = (pack_info_t*)state;
  info->pack_unpack = 1; // unpack
  info->buf_size = src_size;
  info->buffer = (double*)src;
  if (info->coro.next()) {
    info->coro.getValue().value();
    return MPI_SUCCESS;
  } else {
    throw std::runtime_error("Unpack called without data left!");
  }
}


void timing_wrf_custom ( int number_2D, int number_3D, int number_4D, int ims, int ime, int jms, int jme, int kms, int kme, int* limit_4D_arrays, int is, int ie, int js, int je,
  int ks, int ke, int param_first_scalar, int outer_loop, int inner_loop, int* correct_flag, int* ptypesize, char* testname, MPI_Comm local_communicator ) {

  float** array2Ds;
  float** array3Ds;
  float** array4Ds;

  float* buffer;

  int counter, bytes, typesize = sizeof(float), base;
  int element_number;
  int myrank;
  int dim1, dim2, dim3;
  int sub_dim1, sub_dim2, sub_dim3;

  char method[50];

//! just some statements to prevent compiler warnings of unused variables
//! those parameter are included for future features
  *correct_flag = 0;
  *ptypesize = 0;
//  typesize = filehandle_debug

  MPI_Comm_rank( local_communicator, &myrank );

//! some conversion from fortran to c
  dim1 = ime-ims+1;
  dim2 = kme-kms+1;
  dim3 = jme-jms+1;

  sub_dim1 = ie-is+1;
  sub_dim2 = ke-ks+1;
  sub_dim3 = je-js+1;

  is = is - ims;
  ks = ks - kms;
  js = js - jms;

  ie = ie - ims;
  ke = ke - kms;
  je = je - jms;

  param_first_scalar--;

//! ================= initialize the arrays =================

//! allocate all needed arrays first

  array2Ds = (float**)malloc( number_2D * sizeof(float*) );
  array3Ds = (float**)malloc( number_3D * sizeof(float*) );
  array4Ds = (float**)malloc( number_4D * sizeof(float*) );

//! allocate and initialize the arrays
//! compute the number of elements in the arrays
  counter = ( number_2D + number_3D * dim2 ) * dim1 * dim3 ;
  for( int m=0 ; m<number_4D ; m++ ) {
    counter = counter + limit_4D_arrays[m] * dim1 * dim2 * dim3;
  }
  base = myrank * counter + 1;

  for( int m=0 ; m<number_2D ; m++ ) {
    array2Ds[m] = (float*)malloc( dim1 * dim3 * sizeof(float) );
    utilities_fill_unique_array_2D_float( array2Ds[m], dim1, dim3, base );
    base = base + dim1 * dim3;
  }

  for( int m=0 ; m<number_3D ; m++ ) {
    array3Ds[m] = (float*)malloc( dim1 * dim2 * dim3 * sizeof(float) );
    utilities_fill_unique_array_3D_float( array3Ds[m], dim1, dim2, dim3, base );
    base = base + dim1 * dim2 * dim3;
  }

  for( int m=0 ; m<number_4D ; m++ ) {
    array4Ds[m] = (float*)malloc( dim1 * dim2 * dim3 * limit_4D_arrays[m] * sizeof(float) );
    utilities_fill_unique_array_4D_float( array4Ds[m], dim1, dim2, dim3, limit_4D_arrays[m], base );
    base = base + limit_4D_arrays[m] * dim1 * dim2  * dim3;
  }

  if ( myrank == 0 ) {
    snprintf( &method[0], 50, "custom" );

//! compute the number of bytes to be communicated
//! first compute the number of elements in the subarrays
    counter = number_2D * sub_dim1 * sub_dim3 + number_3D * sub_dim1 * sub_dim2 * sub_dim3;
    for( int m=0 ; m<number_4D ; m++ ) {
      if (limit_4D_arrays[m] > param_first_scalar) {
        counter = counter + (limit_4D_arrays[m]-param_first_scalar) * sub_dim1 * sub_dim2 * sub_dim3;
      }
    }
    bytes = counter * typesize;

    timing_init( testname, method, bytes );
  }

  for( int i=0 ; i<outer_loop ; i++ ) {

//! compute the number of elements in the subarray
    element_number = number_2D * sub_dim1 * sub_dim3 + number_3D * sub_dim1 * sub_dim2 * sub_dim3;
    for( int m=0 ; m<number_4D ; m++ ) {
      if (limit_4D_arrays[m] > param_first_scalar) {
        element_number = element_number + (limit_4D_arrays[m]-param_first_scalar) * sub_dim1 * sub_dim2 * sub_dim3;
      }
    }

    buffer = (float*)malloc( element_number * sizeof(float) );

    pack_info_t info = {
      .ie = ie, .is = is, .je = je, .js = js, .ke = ke, .ks = ks,
      .number_2D = number_2D, .number_3D = number_3D, .number_4D = number_4D,
      .dim1 = dim1, .dim2 = dim2, .dim3 = dim3,
      .array2Ds = array2Ds,
      .array3Ds = array3Ds,
      .array4Ds = array4Ds,
      .limit_4D_arrays = limit_4D_arrays,
      .param_first_scalar = param_first_scalar
    };

    if ( myrank == 0 ) {
      timing_record(1);
    }

    MPI_Datatype type;
    MPI_Type_create_custom(state_cb, NULL, query_cb, pack_cb, unpack_cb,
                            NULL, NULL, &info, 1, &type);
    MPI_Status status;

    for( int j=0 ; j<inner_loop ; j++ ) {

//! =============== ping pong communication =================

//! send the data from rank 0 to rank 1
      if ( myrank == 0 ) {
        timing_record(2);
        MPI_Send( &buffer[0], 1, type, 1, itag, local_communicator );
//! receive the data back from rank 1
        MPI_Recv( &buffer[0], 1, type, 1, itag, local_communicator, &status );
        timing_record(3);
        timing_record(4);
//! now for rank 1
      } else {
//! receive from rank 0
        MPI_Recv( &buffer[0], 1, type, 0, itag, local_communicator, &status );
//!> send to rank 0
        MPI_Send( buffer, 1, type, 0, itag, local_communicator );
      } //! of myrank .EQ. 0?

    } //! inner_loop

//! ======================= clean up ========================

    free( buffer );

    if ( myrank == 0 ) {
      timing_record(5);
    }

  } //! outer_loop

  if ( myrank == 0 ) {
    timing_print( 1 );
  }

//! ======================= clean up ========================

  for( int m=0 ; m<number_2D ; m++ ) {
    free( array2Ds[m] );
  }

  for( int m=0 ; m<number_3D ; m++ ) {
    free( array3Ds[m] );
  }

  for( int m=0 ; m<number_4D ; m++ ) {
    free( array4Ds[m] );
  }

  free( array2Ds );
  free( array3Ds );
  free( array4Ds );
}
