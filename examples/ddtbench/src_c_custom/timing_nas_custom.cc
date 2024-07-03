// Copyright (c) 2012 The Trustees of University of Illinois. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

//! contains the timing benchmarks for NAS tests (MG/LU)

#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

#include "ddtbench.h"
#include "generator.h"

#define itag 0

static inline int idx3D(int x, int y, int z, int DIM1, int DIM2) {
    return x+DIM1*(y+z*DIM2);
}

template<int OuterLB, int InnerLB, typename PackIndexFn, typename UnpackIndexFn>
struct pack_info_t {
  coro::generator<MPI_Count> coro;
  MPI_Count buf_size;
  static constexpr const int outer_lb = OuterLB;
  static constexpr const int inner_lb = InnerLB;
  double* buffer;
  double* array;
  PackIndexFn packidx;
  UnpackIndexFn unpackidx;
  int outer_ub;
  int inner_ub;
  int pack_unpack; // 0: pack, 1: unpack

  template<typename PackIndexFn_, typename UnpackIndexFn_>
  pack_info_t(PackIndexFn_&& pidx, UnpackIndexFn_&& uidx, int outer_ub, int inner_ub, double *array)
  : array(array)
  , packidx(std::forward<PackIndexFn_>(pidx))
  , unpackidx(std::forward<UnpackIndexFn_>(uidx))
  , outer_ub(outer_ub)
  , inner_ub(inner_ub)
  { }
};

//int free_cb(void *state) { return MPI_SUCCESS; }

template<typename PackInfoT>
static int query_cb(void *state, const void *buf, MPI_Count count, MPI_Count *packed_size) {
  PackInfoT *info = (PackInfoT*)state;
  *packed_size = (info->outer_ub - info->outer_lb)*(info->inner_ub - info->inner_lb)*sizeof(double);
  return MPI_SUCCESS;
}

#define UNIT_PACK_SIZE 16

/* coroutine used for packing, yields packed bytes count when full*/
template<typename PackInfoT>
static coro::generator<MPI_Count> pack_unpack_coro(PackInfoT *info)
{
  MPI_Count pos = 0;
  int outer_ub = info->outer_ub;
  int inner_ub = info->inner_ub;
  MPI_Count size = info->buf_size;
  double *__restrict__ array = info->array;
  double *__restrict__ buffer = info->buffer;
  if (info->pack_unpack == 0) {
    /* pack */
    for( int k=info->outer_lb ; k<outer_ub ; k++ ) {
      for( int l=info->inner_lb; l<inner_ub; l += UNIT_PACK_SIZE ) {
        if ((l+UNIT_PACK_SIZE) <= inner_ub) {
          for (int i = 0; i < UNIT_PACK_SIZE; i++) {
            buffer[pos++] = array[info->packidx(k, l+i)];
          }
          size -= UNIT_PACK_SIZE*sizeof(double);
        } else {
          for (int i = 0; i < inner_ub - l; i++) {
            buffer[pos++] = array[info->packidx(k, l+i)];
          }
          size -= (inner_ub - l)*sizeof(double);
        }
        assert(size >= 0);
        if (size < UNIT_PACK_SIZE*sizeof(double)) {
          co_yield info->buf_size - size; // insufficient space for another round
          /* we're back, reset state */
          size = info->buf_size;
          buffer = info->buffer;
          pos = 0;
        }
      }
    }
  } else {
    /* unpack */
    for( int k=info->outer_lb ; k<outer_ub ; k++ ) {
      for( int l=0 ; l<((inner_ub-info->inner_lb) + UNIT_PACK_SIZE-1)/UNIT_PACK_SIZE; l++ ) {
        int ll = (l+info->inner_lb)*UNIT_PACK_SIZE;
        if ((ll+UNIT_PACK_SIZE) <= inner_ub) {
          for (int i = 0; i < UNIT_PACK_SIZE; i++) {
            array[info->unpackidx(k, ll+i)] = buffer[pos++];
          }
          size -= UNIT_PACK_SIZE*sizeof(double);
        } else {
          for (int i = 0; i < inner_ub - ll; i++) {
            array[info->unpackidx(k, ll+i)] = buffer[pos++];
          }
          size -= (inner_ub - ll)*sizeof(double);
        }
        assert(size >= 0);
        if (size < UNIT_PACK_SIZE*sizeof(double)) {
          co_yield info->buf_size - size; // insufficient space for another round
          /* we're back, reset state */
          size = info->buf_size;
          buffer = info->buffer;
          pos = 0;
        }
      }
    }
  }

  co_yield info->buf_size - size;
}

template<typename PackInfoT>
static int state_cb(void *context, const void *buf, MPI_Count count, void **state) {
  PackInfoT *info = (PackInfoT*)context;
  info->coro = pack_unpack_coro(info);
  *state = info;
  return MPI_SUCCESS;
}

template<typename PackInfoT>
static int pack_cb(
  void *state, const void *buf,
  MPI_Count count, MPI_Count offset,
  void *dst, MPI_Count dst_size,
  MPI_Count *used)
{
  PackInfoT *info = (PackInfoT*)state;
  info->pack_unpack = 0; // pack
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

template<typename PackInfoT>
static int unpack_cb(
  void *state, void *buf, MPI_Count count,
  MPI_Count offset, const void *src,
  MPI_Count src_size)
{
  PackInfoT *info = (PackInfoT*)state;
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

template<int OuterLB, int InnerLB, typename PackIndexFn, typename UnpackIndexFn>
auto make_pack_info(PackIndexFn&& pix, UnpackIndexFn&& uix, int outerub, int innerub, double *array)
{
  return new pack_info_t<OuterLB, InnerLB,
                     PackIndexFn,
                     UnpackIndexFn>(std::forward<PackIndexFn>(pix),
                                    std::forward<UnpackIndexFn>(uix),
                                    outerub, innerub, array);
}

template<typename PackInfoT>
MPI_Datatype create_mpi_datatype(PackInfoT* info)
{
  using pack_info = std::decay_t<PackInfoT>;
  MPI_Datatype res;
  MPI_Type_create_custom(state_cb<pack_info>, NULL,
                         query_cb<pack_info>, pack_cb<pack_info>,
                         unpack_cb<pack_info>,
                         NULL, NULL, info, 1, &res);
  return res;
}


void timing_nas_lu_y_custom( int DIM2, int DIM3, int outer_loop, int inner_loop, int* correct_flag, int* ptypesize, char* testname, MPI_Comm local_communicator ) {

  int DIM1 = 5;

  double* array;
  double* buffer;

  int myrank;
  int i, j, base, bytes, typesize = sizeof(double);

  char method[50];

//! just some statements to prevent compiler warnings of unused variables
//! those parameter are included for future features
  *correct_flag = 0;
  *ptypesize = 0;
//  typesize = filehandle_debug

  array = (double*)malloc( DIM1*(DIM2+2)*(DIM3+2) * sizeof(double) );

  MPI_Comm_rank( local_communicator, &myrank );

  base = myrank * DIM1 * (DIM2+2) * (DIM3+2) + 1;
  utilities_fill_unique_array_3D_double( &array[0], DIM1, DIM2+2, DIM3+2, base );

  if ( myrank == 0 ) {
    snprintf(method, 50, "custom");
    bytes = 5 * DIM3 * typesize;
    timing_init( testname, &method[0], bytes );
  }

  auto *info = make_pack_info<1, 0>(
    [&](int k, int l){
      return idx3D(l,DIM2,k,DIM1,DIM2+2);
    },
    [&](int k, int l){
      return idx3D(l,0,k,DIM1,DIM2+2);
    }, DIM3, DIM1, array);
  MPI_Datatype type = create_mpi_datatype(info);
  MPI_Status status;

  for( i=0 ; i<outer_loop ; i++ ) {

    buffer = (double*)malloc( DIM1 * DIM3 * sizeof(double));

    if ( myrank == 0 ) {
       timing_record(1);
    }

    for( j=0 ; j<inner_loop ; j++ ) {
      if ( myrank == 0 ) {
        MPI_Send( &buffer[0], 1, type, 1, itag, local_communicator );
        MPI_Recv( &buffer[0], 1, type, 1, itag, local_communicator, &status );
        timing_record(3);
      } else {
        MPI_Recv( &buffer[0], 1, type, 0, itag, local_communicator, &status );
        MPI_Send( &buffer[0], 1, type, 0, itag, local_communicator );
      }
    } //! inner loop

    free( buffer );

    if ( myrank == 0 ) {
      timing_record(5);
    }
  }

  if ( myrank == 0 ) {
    timing_print( 1 );
  }

  delete info;

  free(array);
}

void timing_nas_lu_x_custom( int DIM2, int DIM3, int outer_loop, int inner_loop, int* correct_flag, int* ptypesize, char* testname, MPI_Comm local_communicator ) {

  int DIM1 = 5;

  double* array;
  double* buffer;

  int myrank;
  int i, j, base, bytes, typesize = sizeof(double);

  char method[50];

//! just some statements to prevent compiler warnings of unused variables
//! those parameter are included for future features
  *correct_flag = 0;
  *ptypesize = 0;
//  typesize = filehandle_debug;

  array = (double*)malloc( DIM1 * (DIM2+2) * (DIM3+2) * sizeof(double));

  MPI_Comm_rank( local_communicator, &myrank );

  base = myrank * DIM1 * (DIM2+2) * (DIM3+2) + 1;

  utilities_fill_unique_array_3D_double( &array[0], DIM1, DIM2+2, DIM3+2, base );

  if ( myrank == 0 ) {
    snprintf(method, 50, "custom");
    bytes = DIM1 * DIM2 * typesize;

    timing_init( testname, &method[0], bytes );
  }

  auto *info = make_pack_info<1, 0>(
    [&](int k, int l){
      return idx3D(l,k,DIM3,DIM1,DIM2+2);
    },
    [&](int k, int l){
      return idx3D(l,k,0,DIM1,DIM2+2);
    }, DIM2+1, DIM1, array);
  MPI_Datatype type = create_mpi_datatype(info);
  MPI_Status status;

  for( i=0 ; i<outer_loop ; i++ ) {

    buffer = (double*)malloc(DIM1 * DIM2 * sizeof(double));

    if ( myrank == 0 ) {
      timing_record(1);
    }

    for( j=0 ; j<inner_loop ; j++ ) {
      if ( myrank == 0 ) {
        MPI_Send( &buffer[0], 1, type, 1, itag, local_communicator );
        MPI_Recv( &buffer[0], 1, type, 1, itag, local_communicator, &status );
        timing_record(3);
      } else {
        MPI_Recv( &buffer[0], 1, type, 0, itag, local_communicator, &status );
        MPI_Send( &buffer[0], 1, type, 0, itag, local_communicator );
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

  delete info;

  free(array);
}

void timing_nas_mg_x_custom( int DIM1, int DIM2, int DIM3, int outer_loop, int inner_loop, int* correct_flag, int* ptypesize, char* testname, MPI_Comm local_communicator ) {

  double* array;
  double* buffer;

  int myrank;
  int base, i, j, typesize = sizeof(double), bytes, psize;

  char method[50];

//! just some statements to prevent compiler warnings of unused variables
//! those parameter are included for future features
  *correct_flag = 0;
  *ptypesize = 0;
//  typesize = filehandle_debug

  array = (double*)malloc( DIM1 * DIM2 * DIM3 * sizeof(double));

  MPI_Comm_rank( local_communicator, &myrank );

  base = myrank * DIM1 * DIM2 * DIM3 + 1;
  utilities_fill_unique_array_3D_double( &array[0], DIM1, DIM2, DIM3, base);

  if ( myrank == 0 ) {
    snprintf(method, 50, "custom");
    bytes = (DIM2-2)*(DIM3-2) * typesize;

    timing_init( testname, &method[0], bytes );
  }

  auto *info = make_pack_info<1, 1>(
    [&](int k, int l){
      return idx3D(DIM1-2,l,k,DIM1,DIM2);
    },
    [&](int k, int l){
      return idx3D(DIM1-1,l,k,DIM1,DIM2);
    }, DIM3-1, DIM2-1, array);
  MPI_Datatype type = create_mpi_datatype(info);
  MPI_Status status;

  for( i=0 ; i<outer_loop ; i++ ) {

    psize = (DIM2-2)*(DIM3-2);
    buffer = (double*)malloc( psize * sizeof(double) );

    if ( myrank == 0 ) {
      timing_record(1);
    }

    for( j=0 ; j<inner_loop ; j++ ) {

      if ( myrank == 0 ) {
        MPI_Send( &buffer[0], 1, type, 1, itag, local_communicator );
        MPI_Recv( &buffer[0], 1, type, 1, itag, local_communicator, &status );
        timing_record(3);
      } else {
        MPI_Recv( &buffer[0], 1, type, 0, itag, local_communicator, &status );
        MPI_Send( &buffer[0], 1, type, 0, itag, local_communicator );
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

  delete info;

  free(array);
}

void timing_nas_mg_y_custom( int DIM1, int DIM2, int DIM3, int outer_loop, int inner_loop, int* correct_flag, int* ptypesize, char* testname, MPI_Comm local_communicator ) {

 double* array;
 double* buffer;

 int myrank;
 int base, i, j, typesize = sizeof(double), bytes, psize;

 char method[50];

//! just some statements to prevent compiler warnings of unused variables
//! those parameter are included for future features
 *correct_flag = 0;
 *ptypesize = 0;
// typesize = filehandle_debug

  array = (double*)malloc( DIM1 * DIM2 * DIM3 * sizeof(double) );

  MPI_Comm_rank( local_communicator, &myrank );

  base = myrank * DIM1 * DIM2 * DIM3 + 1;
  utilities_fill_unique_array_3D_double( &array[0], DIM1, DIM2, DIM3, base);

  if ( myrank == 0 ) {
    snprintf(method, 50, "custom");
    bytes = (DIM1-2)*(DIM3-2) * typesize;

    timing_init( testname, method, bytes );
  }

  auto *info = make_pack_info<1, 1>(
    [&](int k, int l){
      return idx3D(l,DIM2-2,k,DIM1,DIM2);
    },
    [&](int k, int l){
      return idx3D(l,DIM2-1,k,DIM1,DIM2);
    }, DIM3-1, DIM1-1, array);
  MPI_Datatype type = create_mpi_datatype(info);
  MPI_Status status;

  for( i=0 ; i<outer_loop ; i++ ) {

    psize = (DIM1-2)*(DIM3-2);
    buffer = (double*)malloc( psize * sizeof(double) );

    if ( myrank == 0 ) {
      timing_record(1);
    }

    for( j=0 ; j<inner_loop ; j++ ) {

      if ( myrank == 0 ) {
        MPI_Send( &buffer[0], 1, type, 1, itag, local_communicator );
        MPI_Recv( &buffer[0], 1, type, 1, itag, local_communicator, &status );
        timing_record(3);
      } else {
        MPI_Recv( &buffer[0], 1, type, 0, itag, local_communicator, &status );
        MPI_Send( &buffer[0], 1, type, 0, itag, local_communicator );
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

  delete info;

  free(array);
}

void timing_nas_mg_z_custom( int DIM1, int DIM2, int DIM3, int outer_loop, int inner_loop, int* correct_flag, int* ptypesize, char* testname, MPI_Comm local_communicator ) {

  double* array;
  double* buffer;

  int myrank;
  int base, i, j, typesize = sizeof(double), bytes, psize;

  char method[50];

//! just some statements to prevent compiler warnings of unused variables
//! those parameter are included for future features
  *correct_flag = 0;
  *ptypesize = 0;
//typesize = filehandle_debug

  array = (double*)malloc( DIM1 * DIM2 * DIM3 * sizeof(double) );

  MPI_Comm_rank( local_communicator, &myrank );

  base = myrank * DIM1 * DIM2 * DIM3 + 1;
  utilities_fill_unique_array_3D_double( &array[0], DIM1, DIM2, DIM3, base );

  if ( myrank == 0 ) {
    snprintf( method, 50, "custom" );
    bytes = (DIM1-2) * (DIM2-2) * typesize;

    timing_init( testname, &method[0], bytes );
  }

  auto *info = make_pack_info<1, 1>(
    [&](int k, int l){
      return idx3D(l,k,1,DIM1,DIM2);
    },
    [&](int k, int l){
      return idx3D(l,k,0,DIM1,DIM2);
    }, DIM2-1, DIM1-1, array);
  MPI_Datatype type = create_mpi_datatype(info);
  MPI_Status status;

  for( i=0 ; i<outer_loop ; i++ ) {

    psize = (DIM1-2) * (DIM2-2);
    buffer = (double*)malloc( psize * sizeof(double) );

    if ( myrank == 0 ) {
      timing_record(1);
    }

    for( j=0 ; j<inner_loop ; j++ ) {

      if ( myrank == 0 ) {
        MPI_Send( &buffer[0], 1, type, 1, itag, local_communicator );
        MPI_Recv( &buffer[0], 1, type, 1, itag, local_communicator, &status );
        timing_record(3);
      } else {
        MPI_Recv( &buffer[0], 1, type, 0, itag, local_communicator, &status );
        MPI_Send( &buffer[0], 1, type, 0, itag, local_communicator );
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

  delete info;

  free(array);
}
