// Copyright (c) 2012 The Trustees of University of Illinois. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

//! contains the timing benchmarks for NAS tests (MG/LU)

#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <cstring>

#include "ddtbench.h"
#include "generator.h"

#define itag 0

static inline int idx3D(int x, int y, int z, int DIM1, int DIM2) {
    return x+DIM1*(y+z*DIM2);
}

template<typename RegionCountFn, typename RegionGetFn>
struct mem_info_t {
  RegionCountFn countfn;
  RegionGetFn getfn;

  template<typename C, typename G>
  mem_info_t(C&& count, G&& get)
  : countfn(std::forward<C>(count))
  , getfn(std::forward<G>(g))
  { }
};

//int free_cb(void *state) { return MPI_SUCCESS; }

/**
 * Memory region functions
 */

template<typename PackInfoT>
static int state_mem_cb(void *context, const void *buf, MPI_Count count, void **state) {
  PackInfoT *info = (PackInfoT*)context;
  *state = info;
  return MPI_SUCCESS;
}

template<typename MemInfoT>
int region_count_cb(
  void *state,
  // Buffer pointer
  void *buf,
  // Number of elements in send buffer.
  MPI_Count count,
  // Number of memory regions
  MPI_Count *region_count)
{
  MemInfoT *info = (MemInfoT*)state;
  *region_count = info->countfn();
  return MPI_SUCCESS;
}


template<typename MemInfoT>
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
  MemInfoT *info = (MemInfoT*)state;
  assert(region_count == info->countfn()); // make sure we can store them all at once
  assert(count == 1); // we know we only transfer one at a time
  info->getfn(reg_lens, reg_bases, reg_types);

  return MPI_SUCCESS;
}

template<typename CountFn, typename GetFn>
static MPI_Datatype make_region_info(CountFn&& c, GetFn&& g)
{
  auto info = new mem_info_t(std::forward<CountFn>(c), std::forward<GetFn>(g));
  return info;
}

template<typename MemInfoT>
MPI_Datatype create_region_datatype(MemInfoT* info)
{
  using mem_info = std::decay_t<PackInfoT>;
  MPI_Datatype res;
  MPI_Type_create_custom(state_mem_cb<mem_info>,
                         NULL, NULL, NULL, NULL,
                         region_count_cb<mem_info>, region_query_cb<mem_info>,
                         info, 1, &res);
  return res;
}


/**
 * Pack functions
 */

#define VAR_INNER_UB (std::numeric_limits<int>::max())

template<int OuterLB, int InnerLB, int InnerUB, typename PackIndexFn, typename UnpackIndexFn>
struct pack_info_t {
  coro::generator<MPI_Count> coro;
  MPI_Count buf_size;
  double* buffer;
  double* array;
  PackIndexFn packidx;
  UnpackIndexFn unpackidx;
  int m_outer_ub;
  int m_inner_ub;
  int pack_unpack; // 0: pack, 1: unpack

  template<typename PackIndexFn_, typename UnpackIndexFn_>
  pack_info_t(PackIndexFn_&& pidx, UnpackIndexFn_&& uidx, int oub, int iub, double *array)
  : array(array)
  , packidx(std::forward<PackIndexFn_>(pidx))
  , unpackidx(std::forward<UnpackIndexFn_>(uidx))
  , m_outer_ub(oub)
  , m_inner_ub(iub)
  { }

  constexpr int inner_lb() const {
    return InnerLB;
  }
  constexpr int inner_ub() const {
    if constexpr(VAR_INNER_UB == InnerUB) {
      return inner_ub; // only used if variable
    } else {
      return InnerUB;
    }
  }

  constexpr int outer_lb() const {
    return OuterLB;
  }

  int outer_ub() const {
    return outer_ub;
  }
};


template<typename PackInfoT>
static int query_pack_cb(void *state, const void *buf, MPI_Count count, MPI_Count *packed_size) {
  PackInfoT *info = (PackInfoT*)state;
  *packed_size = (info->outer_ub() - info->outer_lb())*(info->inner_ub() - info->inner_lb())*sizeof(double);
  return MPI_SUCCESS;
}

#define UNIT_PACK_SIZE 16

template<typename PackInfoT>
static MPI_Count pack(PackInfoT *info)
{
  MPI_Count pos = 0;
  MPI_Count size = info->buf_size;
  double *__restrict__ array = info->array;
  double *__restrict__ buffer = info->buffer;
    /* pack */
    for( int k=info->outer_lb() ; k<info->outer_ub() ; k++ ) {
      for( int l=info->inner_lb(); l<info->inner_ub(); l++ ) {
            buffer[pos++] = array[info->packidx(k, l)];
      }
    }
  return pos*sizeof(double);
}



template<typename PackInfoT>
static MPI_Count unpack(PackInfoT *info)
{
  MPI_Count pos = 0;
  MPI_Count size = info->buf_size;
  double *__restrict__ array = info->array;
  double *__restrict__ buffer = info->buffer;
    /* unpack */
    for( int k=info->outer_lb() ; k<info->outer_ub(); k++ ) {
      for( int l=info->inner_lb() ; l<info->inner_ub(); l++ ) {
            array[info->unpackidx(k, l)] = buffer[pos++];
      }
    }

  return pos*sizeof(double);
}



/* coroutine used for packing, yields packed bytes count when full*/
template<typename PackInfoT>
static coro::generator<MPI_Count> pack_unpack_coro(PackInfoT *info)
{
  MPI_Count pos = 0;
  MPI_Count size = info->buf_size;
  double *__restrict__ array = info->array;
  double *__restrict__ buffer = info->buffer;
  if (info->pack_unpack == 0) {
    /* pack */
    for( int k=info->outer_lb() ; k<info->outer_ub() ; k++ ) {
      for( int l=info->inner_lb(); l<info->inner_ub(); l += UNIT_PACK_SIZE ) {
        if ((l+UNIT_PACK_SIZE) <= inner_ub) {
#ifndef USE_MEMCPY
          for (int i = 0; i < UNIT_PACK_SIZE; i++) {
            buffer[pos++] = array[info->packidx(k, l+i)];
          }
#else
          memcpy(&buffer[pos], &array[info->packidx(k, l)], UNIT_PACK_SIZE);
          pos += UNIT_PACK_SIZE;
#endif // USE_MEMCPY
          size -= UNIT_PACK_SIZE*sizeof(double);
        } else {
#ifndef USE_MEMCPY
          for (int i = 0; i < info->inner_ub() - l; i++) {
            buffer[pos++] = array[info->packidx(k, l+i)];
          }
#else
          memcpy(&buffer[pos], &array[info->packidx(k, l)], info->inner_ub() - l);
          pos += info->inner_ub() - l;
#endif // USE_MEMCPY
          size -= (info->inner_ub() - l)*sizeof(double);
        }
        assert(size >= 0);
        if (size < std::min(UNIT_PACK_SIZE, info->inner_ub() - info->inner_lb())*sizeof(double)) {
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
    for( int k=info->outer_lb() ; k<info->outer_ub() ; k++ ) {
      for( int l=0 ; l<info->inner_ub(); l += UNIT_PACK_SIZE ) {
        if (l+UNIT_PACK_SIZE <= info->inner_ub()) {
#ifndef USE_MEMCPY
          for (int i = 0; i < UNIT_PACK_SIZE; i++) {
            array[info->unpackidx(k, l+i)] = buffer[pos++];
          }
#else
          memcpy(&array[info->unpackidx(k, l)], &buffer[pos], UNIT_PACK_SIZE);
          pos += UNIT_PACK_SIZE;
#endif // USE_MEMCPY
          size -= UNIT_PACK_SIZE*sizeof(double);
        } else {
#ifndef USE_MEMCPY
          for (int i = 0; i < info->inner_ub() - l; i++) {
            array[info->unpackidx(k, l+i)] = buffer[pos++];
          }
#else
          memcpy(&array[info->unpackidx(k, l)], &buffer[pos], info->inner_ub() - l);
          pos += info->inner_ub() - l;
#endif // USE_MEMCPY
          size -= (info->inner_ub() - l)*sizeof(double);
        }
        assert(size >= 0);
        if (size < std::min(UNIT_PACK_SIZE, info->inner_ub() - info->inner_lb())*sizeof(double)) {
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
static int state_pack_cb(void *context, const void *buf, MPI_Count count, void **state) {
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
  MPI_Count packed_size = (info->outer_ub() - info->outer_lb())*(info->inner_ub() - info->inner_lb())*sizeof(double);
  if (packed_size <= dst_size && offset == 0) {
    *used = pack(info);
    return MPI_SUCCESS;
  } else if (info->coro.next()) {
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
  MPI_Count packed_size = (info->outer_ub() - info->outer_lb())*(info->inner_ub() - info->inner_lb())*sizeof(double);
  if (packed_size == src_size && offset == 0) {
    unpack(info);
    return MPI_SUCCESS;
  } else if (info->coro.next()) {
    info->coro.getValue().value();
    return MPI_SUCCESS;
  } else {
    throw std::runtime_error("Unpack called without data left!");
  }
}

template<int OuterLB, int InnerLB, int InnerUB, typename PackIndexFn, typename UnpackIndexFn>
auto make_pack_info(PackIndexFn&& pix, UnpackIndexFn&& uix, int outerub, int innerub, double *array)
{
  return new pack_info_t<OuterLB, InnerLB, InnerUB,
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

  auto *info = make_pack_info<1, 0, 5>(
    [=](int k, int l){
      return idx3D(l,DIM2,k,DIM1,DIM2+2);
    },
    [=](int k, int l){
      return idx3D(l,0,k,DIM1,DIM2+2);
    }, DIM3, DIM1, array);
  MPI_Datatype type = create_mpi_datatype(info);
  MPI_Status status;

  for( i=0 ; i<outer_loop ; i++ ) {

    if ( myrank == 0 ) {
       timing_record(1);
    }

    for( j=0 ; j<inner_loop ; j++ ) {
      if ( myrank == 0 ) {
        MPI_Send( &array[idx3D(0,DIM2,1,DIM1,DIM2+2)], 1, type, 1, itag, local_communicator );
        MPI_Recv( &array[idx3D(0,0,1,DIM1,DIM2+2)], 1, type, 1, itag, local_communicator, &status );
        timing_record(3);
      } else {
        MPI_Recv( &array[idx3D(0,0,1,DIM1,DIM2+2)], 1, type, 0, itag, local_communicator, &status );
        MPI_Send( &array[idx3D(0,DIM2,1,DIM1,DIM2+2)], 1, type, 0, itag, local_communicator );
      }
    } //! inner loop

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

  auto *info = make_pack_info<1, 0, 5>(
    [=](int k, int l){
      return idx3D(l,k,DIM3,DIM1,DIM2+2);
    },
    [=](int k, int l){
      return idx3D(l,k,0,DIM1,DIM2+2);
    }, DIM2+1, DIM1, array);
  MPI_Datatype type = create_mpi_datatype(info);
  MPI_Status status;

  for( i=0 ; i<outer_loop ; i++ ) {

    if ( myrank == 0 ) {
      timing_record(1);
    }

    for( j=0 ; j<inner_loop ; j++ ) {
      if ( myrank == 0 ) {
        MPI_Send( &array[idx3D(0,1,DIM3,DIM1,DIM2+2)], 1, type, 1, itag, local_communicator );
        MPI_Recv( &array[idx3D(0,1,0,DIM1,DIM2+2)], 1, type, 1, itag, local_communicator, &status );
        timing_record(3);
      } else {
        MPI_Recv( &array[idx3D(0,1,0,DIM1,DIM2+2)], 1, type, 0, itag, local_communicator, &status );
        MPI_Send( &array[idx3D(0,1,DIM3,DIM1,DIM2+2)], 1, type, 0, itag, local_communicator );
      }
    } //! inner loop

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

  int myrank;
  int base, i, j, typesize = sizeof(double), bytes;

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

  auto *info = make_pack_info<1, 1, VAR_INNER_UB>(
    [=](int k, int l){
      return idx3D(DIM1-2,l,k,DIM1,DIM2);
    },
    [=](int k, int l){
      return idx3D(DIM1-1,l,k,DIM1,DIM2);
    }, DIM3-1, DIM2-1, array);
  MPI_Datatype type = create_mpi_datatype(info);
  MPI_Status status;

  for( i=0 ; i<outer_loop ; i++ ) {

    if ( myrank == 0 ) {
      timing_record(1);
    }

    for( j=0 ; j<inner_loop ; j++ ) {

      if ( myrank == 0 ) {
        MPI_Send( &array[idx3D(DIM1-2,1,1,DIM1,DIM2)], 1, type, 1, itag, local_communicator );
        MPI_Recv( &array[idx3D(DIM1-1,1,1,DIM1,DIM2)], 1, type, 1, itag, local_communicator, &status );
        timing_record(3);
      } else {
        MPI_Recv( &array[idx3D(DIM1-1,1,1,DIM1,DIM2)], 1, type, 0, itag, local_communicator, &status );
        MPI_Send( &array[idx3D(DIM1-2,1,1,DIM1,DIM2)], 1, type, 0, itag, local_communicator );
      }

    } //! inner loop

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

 int myrank;
 int base, i, j, typesize = sizeof(double), bytes;

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

  auto *info = make_pack_info<1, 1, VAR_INNER_UB>(
    [=](int k, int l){
      return idx3D(l,DIM2-2,k,DIM1,DIM2);
    },
    [=](int k, int l){
      return idx3D(l,DIM2-1,k,DIM1,DIM2);
    }, DIM3-1, DIM1-1, array);
  MPI_Datatype type = create_mpi_datatype(info);
  MPI_Status status;

  for( i=0 ; i<outer_loop ; i++ ) {

    if ( myrank == 0 ) {
      timing_record(1);
    }

    for( j=0 ; j<inner_loop ; j++ ) {

      if ( myrank == 0 ) {
        MPI_Send( &array[idx3D(1,DIM2-2,1,DIM1,DIM2)], 1, type, 1, itag, local_communicator );
        MPI_Recv( &array[idx3D(1,DIM2-1,1,DIM1,DIM2)], 1, type, 1, itag, local_communicator, &status );
        timing_record(3);
      } else {
        MPI_Recv( &array[idx3D(1,DIM2-1,1,DIM1,DIM2)], 1, type, 0, itag, local_communicator, &status );
        MPI_Send( &array[idx3D(1,DIM2-2,1,DIM1,DIM2)], 1, type, 0, itag, local_communicator );
      }

    } //! inner loop

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

  int myrank;
  int base, i, j, typesize = sizeof(double), bytes;

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

  auto *info = make_pack_info<1, 1, VAR_INNER_UB>(
    [=](int k, int l){
      return idx3D(l,k,1,DIM1,DIM2);
    },
    [=](int k, int l){
      return idx3D(l,k,0,DIM1,DIM2);
    }, DIM2-1, DIM1-1, array);
  MPI_Datatype type = create_mpi_datatype(info);
  MPI_Status status;

  for( i=0 ; i<outer_loop ; i++ ) {

    if ( myrank == 0 ) {
      timing_record(1);
    }

    for( j=0 ; j<inner_loop ; j++ ) {

      if ( myrank == 0 ) {
        MPI_Send( &array[idx3D(1,1,1,DIM1,DIM2)], 1, type, 1, itag, local_communicator );
        MPI_Recv( &array[idx3D(1,1,0,DIM1,DIM2)], 1, type, 1, itag, local_communicator, &status );
        timing_record(3);
      } else {
        MPI_Recv( &array[idx3D(1,1,0,DIM1,DIM2)], 1, type, 0, itag, local_communicator, &status );
        MPI_Send( &array[idx3D(1,1,1,DIM1,DIM2)], 1, type, 0, itag, local_communicator );
      }

    } //! inner loop

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



/***
 * Memory region functions
 */


void timing_nas_lu_y_region( int DIM2, int DIM3, int outer_loop, int inner_loop, int* correct_flag, int* ptypesize, char* testname, MPI_Comm local_communicator ) {

  int DIM1 = 5;

  double* array;

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

  auto *info = make_mem_info<1, 0>(
    [=](){ // region count
      return DIM3;
    },
    [=](void *buffer, void **bases, MPI_Count counts[], MPI_Datatype types[]){
      /* fill in the region information */
      double *db = (double*)buffer;
      for (int i = 0; i < DIM3; ++i) {
        bases[i]  = &db[i*((DIM2+2)*5)];
        counts[i] = 5*sizeof(double);
        types[i]  = MPI_BYTE;
      }
    });
  MPI_Datatype type = create_region_datatype(info);
  MPI_Status status;

  for( i=0 ; i<outer_loop ; i++ ) {

    if ( myrank == 0 ) {
       timing_record(1);
    }

    for( j=0 ; j<inner_loop ; j++ ) {
      if ( myrank == 0 ) {
        MPI_Send( &array[idx3D(0,DIM2,1,DIM1,DIM2+2)], 1, type, 1, itag, local_communicator );
        MPI_Recv( &array[idx3D(0,0,1,DIM1,DIM2+2)], 1, type, 1, itag, local_communicator, &status );
        timing_record(3);
      } else {
        MPI_Recv( &array[idx3D(0,0,1,DIM1,DIM2+2)], 1, type, 0, itag, local_communicator, &status );
        MPI_Send( &array[idx3D(0,DIM2,1,DIM1,DIM2+2)], 1, type, 0, itag, local_communicator );
      }
    } //! inner loop

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

void timing_nas_lu_x_region( int DIM2, int DIM3, int outer_loop, int inner_loop, int* correct_flag, int* ptypesize, char* testname, MPI_Comm local_communicator ) {

  int DIM1 = 5;

  double* array;

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

  auto *info = make_mem_info<1, 0>(
    [=](){ // region count
      return 1;
    },
    [=](void *buffer, void **bases, MPI_Count counts[], MPI_Datatype types[]){
      /* fill in the region information */
      bases[0]  = buffer;
      counts[0] = DIM1*DIM2*sizeof(double);
      types[0]  = MPI_BYTE;
    });
  MPI_Datatype type = create_region_datatype(info);
  MPI_Status status;

  for( i=0 ; i<outer_loop ; i++ ) {

    if ( myrank == 0 ) {
      timing_record(1);
    }

    for( j=0 ; j<inner_loop ; j++ ) {
      if ( myrank == 0 ) {
        MPI_Send( &array[idx3D(0,1,DIM3,DIM1,DIM2+2)], 1, type, 1, itag, local_communicator );
        MPI_Recv( &array[idx3D(0,1,0,DIM1,DIM2+2)], 1, type, 1, itag, local_communicator, &status );
        timing_record(3);
      } else {
        MPI_Recv( &array[idx3D(0,1,0,DIM1,DIM2+2)], 1, type, 0, itag, local_communicator, &status );
        MPI_Send( &array[idx3D(0,1,DIM3,DIM1,DIM2+2)], 1, type, 0, itag, local_communicator );
      }
    } //! inner loop

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

void timing_nas_mg_x_region( int DIM1, int DIM2, int DIM3, int outer_loop, int inner_loop, int* correct_flag, int* ptypesize, char* testname, MPI_Comm local_communicator ) {

  double* array;

  int myrank;
  int base, i, j, typesize = sizeof(double), bytes;

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

  auto *info = make_mem_info<1, 0>(
    [=](){ // region count
      return (DIM3-2)*(DIM2-2);
    },
    [=](void *buffer, void **bases, MPI_Count counts[], MPI_Datatype types[]){
      /* fill in the region information */
      double *db = (double*)buffer;
      int outer_stride = DIM1 * DIM2;
      for (int i = 0; i < DIM3-2; ++i) {
        for (int j = 0; j < DIM2-2; ++j) {
          int idx = i*(DIM3-2) + j;
          bases[idx]  = &db[i*outer_stride+(DIM1*j)];
          counts[idx] = sizeof(double);
          types[idx]  = MPI_BYTE;
        }
      }
    });
  MPI_Datatype type = create_region_datatype(info);
  MPI_Status status;

  for( i=0 ; i<outer_loop ; i++ ) {

    if ( myrank == 0 ) {
      timing_record(1);
    }

    for( j=0 ; j<inner_loop ; j++ ) {

      if ( myrank == 0 ) {
        MPI_Send( &array[idx3D(DIM1-2,1,1,DIM1,DIM2)], 1, type, 1, itag, local_communicator );
        MPI_Recv( &array[idx3D(DIM1-1,1,1,DIM1,DIM2)], 1, type, 1, itag, local_communicator, &status );
        timing_record(3);
      } else {
        MPI_Recv( &array[idx3D(DIM1-1,1,1,DIM1,DIM2)], 1, type, 0, itag, local_communicator, &status );
        MPI_Send( &array[idx3D(DIM1-2,1,1,DIM1,DIM2)], 1, type, 0, itag, local_communicator );
      }

    } //! inner loop

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

void timing_nas_mg_y_region( int DIM1, int DIM2, int DIM3, int outer_loop, int inner_loop, int* correct_flag, int* ptypesize, char* testname, MPI_Comm local_communicator ) {

 double* array;

 int myrank;
 int base, i, j, typesize = sizeof(double), bytes;

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

  auto *info = make_mem_info<1, 0>(
    [=](){ // region count
      return (DIM3-2);
    },
    [=](void *buffer, void **bases, MPI_Count counts[], MPI_Datatype types[]){
      /* fill in the region information */
      double *db = (double*)buffer;
      for (int i = 0; i < DIM3-2; ++i) {
        bases[i]  = &db[i*DIM1*DIM2)];
        counts[i] = (DIM1-2)*sizeof(double);
        types[i]  = MPI_BYTE;
      }
    });
  MPI_Datatype type = create_region_datatype(info);
  MPI_Status status;

  for( i=0 ; i<outer_loop ; i++ ) {

    if ( myrank == 0 ) {
      timing_record(1);
    }

    for( j=0 ; j<inner_loop ; j++ ) {

      if ( myrank == 0 ) {
        MPI_Send( &array[idx3D(1,DIM2-2,1,DIM1,DIM2)], 1, type, 1, itag, local_communicator );
        MPI_Recv( &array[idx3D(1,DIM2-1,1,DIM1,DIM2)], 1, type, 1, itag, local_communicator, &status );
        timing_record(3);
      } else {
        MPI_Recv( &array[idx3D(1,DIM2-1,1,DIM1,DIM2)], 1, type, 0, itag, local_communicator, &status );
        MPI_Send( &array[idx3D(1,DIM2-2,1,DIM1,DIM2)], 1, type, 0, itag, local_communicator );
      }

    } //! inner loop

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

void timing_nas_mg_z_region( int DIM1, int DIM2, int DIM3, int outer_loop, int inner_loop, int* correct_flag, int* ptypesize, char* testname, MPI_Comm local_communicator ) {

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

  auto *info = make_mem_info<1, 0>(
    [=](){ // region count
      return (DIM2-2);
    },
    [=](void *buffer, void **bases, MPI_Count counts[], MPI_Datatype types[]){
      /* fill in the region information */
      double *db = (double*)buffer;
      for (int i = 0; i < DIM2-2; ++i) {
        bases[i]  = &db[i*DIM1)];
        counts[i] = (DIM1-2)*sizeof(double);
        types[i]  = MPI_BYTE;
      }
    });
  MPI_Datatype type = create_region_datatype(info);
  MPI_Status status;

  for( i=0 ; i<outer_loop ; i++ ) {

    if ( myrank == 0 ) {
      timing_record(1);
    }

    for( j=0 ; j<inner_loop ; j++ ) {

      if ( myrank == 0 ) {
        MPI_Send( &array[idx3D(1,1,1,DIM1,DIM2)], 1, type, 1, itag, local_communicator );
        MPI_Recv( &array[idx3D(1,1,0,DIM1,DIM2)], 1, type, 1, itag, local_communicator, &status );
        timing_record(3);
      } else {
        MPI_Recv( &array[idx3D(1,1,0,DIM1,DIM2)], 1, type, 0, itag, local_communicator, &status );
        MPI_Send( &array[idx3D(1,1,1,DIM1,DIM2)], 1, type, 0, itag, local_communicator );
      }

    } //! inner loop

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





/**
 * Manual packing functions
 */
void timing_nas_lu_y_manual( int DIM2, int DIM3, int outer_loop, int inner_loop, int* correct_flag, int* ptypesize, char* testname, MPI_Comm local_communicator ) {

  int DIM1 = 5;

  double* array;
  double* buffer;

  int myrank;
  int i, j, k, l,  base, bytes, typesize = sizeof(double);
  MPI_Status status;

  char method[50];

//! just some statements to prevent compiler warnings of unused variables
//! those parameter are included for future features
  *correct_flag = 0;
  *ptypesize = 0;
//  typesize = filehandle_debug

  array =(double*)malloc( DIM1*(DIM2+2)*(DIM3+2) * sizeof(double) );

  MPI_Comm_rank( local_communicator, &myrank );

  base = myrank * DIM1 * (DIM2+2) * (DIM3+2) + 1;
  utilities_fill_unique_array_3D_double( &array[0], DIM1, DIM2+2, DIM3+2, base );

  if ( myrank == 0 ) {
    snprintf(method, 50, "mpicd_manual");

    bytes = 5 * DIM3 * typesize;

    timing_init( testname, &method[0], bytes );
  }

  for( i=0 ; i<outer_loop ; i++ ) {

    buffer =(double*)malloc( DIM1 * DIM3 * sizeof(double));

    if ( myrank == 0 ) {
       timing_record(1);
    }

    for( j=0 ; j<inner_loop ; j++ ) {
      if ( myrank == 0 ) {
//! pack the data
        base = 0;
        for( k=1 ; k<DIM3 ; k++ ) {
          for( l=0 ; l<DIM1 ; l++ ) {
            buffer[base++] = array[idx3D(l,DIM2,k,DIM1,DIM2+2)];
          }
        }
        timing_record(2);
        MPI_Send( &buffer[0], DIM1*DIM3*sizeof(double), MPI_BYTE, 1, itag, local_communicator );
        MPI_Recv( &buffer[0], DIM1*DIM3*sizeof(double), MPI_BYTE, 1, itag, local_communicator, &status );
        timing_record(3);
//! unpack the data
        base = 0;
        for( k=1 ; k<DIM3 ; k++ ) {
          for( l=0 ; l<DIM1 ; l++ ) {
            array[idx3D(l,0,k,DIM1,DIM2+2)] = buffer[base++];
          }
        }
        timing_record(4);
      } else {
        MPI_Recv( &buffer[0], DIM1*DIM3*sizeof(double), MPI_BYTE, 0, itag, local_communicator, &status );
//! unpack the data
        base = 0;
        for( k=1 ; k<DIM3 ; k++ ) {
          for( l=0 ; l<DIM1 ; l++ ) {
            array[idx3D(l,0,k,DIM1,DIM2+2)] = buffer[base++];
          }
        }
//! pack the data
        base = 0;
        for( k=1 ; k<DIM3 ; k++ ) {
          for( l=0 ; l<DIM1 ; l++ ) {
            buffer[base++] = array[idx3D(l,DIM2,k,DIM1,DIM2+2)];
          }
        }
        MPI_Send( &buffer[0], DIM1*DIM3*sizeof(double), MPI_BYTE, 0, itag, local_communicator );
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

  free(array);
}


void timing_nas_lu_x_manual( int DIM2, int DIM3, int outer_loop, int inner_loop, int* correct_flag, int* ptypesize, char* testname, MPI_Comm local_communicator ) {

  int DIM1 = 5;

  double* array;
  double* buffer;

  int myrank;
  int i, j, k, l,  base, bytes, typesize = sizeof(double);
  MPI_Status status;

  char method[50];

//! just some statements to prevent compiler warnings of unused variables
//! those parameter are included for future features
  *correct_flag = 0;
  *ptypesize = 0;
//  typesize = filehandle_debug;

  array =(double*)malloc( DIM1 * (DIM2+2) * (DIM3+2) * sizeof(double));

  MPI_Comm_rank( local_communicator, &myrank );

  base = myrank * DIM1 * (DIM2+2) * (DIM3+2) + 1;

  utilities_fill_unique_array_3D_double( &array[0], DIM1, DIM2+2, DIM3+2, base );

  if ( myrank == 0 ) {
    snprintf(method, 50, "mpicd_manual");

    bytes = DIM1 * DIM2 * typesize;

    timing_init( testname, &method[0], bytes );
  }

  for( i=0 ; i<outer_loop ; i++ ) {

    buffer =(double*)malloc(DIM1 * DIM2 * sizeof(double));

    if ( myrank == 0 ) {
      timing_record(1);
    }

    for( j=0 ; j<inner_loop ; j++ ) {
      if ( myrank == 0 ) {
//! pack the data
        base = 0;
        for( k=1 ; k<DIM2+1 ; k++ ) {
          for( l=0 ; l<DIM1 ; l++ ) {
            buffer[base++] = array[idx3D(l,k,DIM3,DIM1,DIM2+2)];
          }
        }
        timing_record(2);
        MPI_Send( &buffer[0], DIM1*DIM2*sizeof(double), MPI_BYTE, 1, itag, local_communicator );
        MPI_Recv( &buffer[0], DIM1*DIM2*sizeof(double), MPI_BYTE, 1, itag, local_communicator, &status );
        timing_record(3);
//! unpack the data
        base = 0;
        for( k=1 ; k<DIM2+1 ; k++ ) {
          for( l=0 ; l<DIM1 ; l++ ) {
            array[idx3D(l,k,0,DIM1,DIM2+2)] = buffer[base++];
          }
        }
        timing_record(4);
      } else {
        MPI_Recv( &buffer[0], DIM1*DIM2*sizeof(double), MPI_BYTE, 0, itag, local_communicator, &status );
//! unpack the data
        base = 0;
        for( k=1 ; k<DIM2+1 ; k++ ) {
          for( l=0 ; l<DIM1 ; l++ ) {
            array[idx3D(l,k,0,DIM1,DIM2+2)] = buffer[base++];
          }
        }
//! pack the data
        base = 0;
        for( k=1 ; k<DIM2+1 ; k++ ) {
          for( l=0 ; l<DIM1 ; l++ ) {
            buffer[base++] = array[idx3D(l,k,DIM3,DIM1,DIM2+2)];
          }
        }
        MPI_Send( &buffer[0], DIM1*DIM2*sizeof(double), MPI_BYTE, 0, itag, local_communicator );
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

  free(array);
}


void timing_nas_mg_x_manual( int DIM1, int DIM2, int DIM3, int outer_loop, int inner_loop, int* correct_flag, int* ptypesize, char* testname, MPI_Comm local_communicator ) {

  double* array;
  double* buffer;

  int myrank;
  int base, i, j, k, l, typesize = sizeof(double), bytes, psize;
  MPI_Status status;

  char method[50];

//! just some statements to prevent compiler warnings of unused variables
//! those parameter are included for future features
  *correct_flag = 0;
  *ptypesize = 0;
//  typesize = filehandle_debug

  array =(double*)malloc( DIM1 * DIM2 * DIM3 * sizeof(double));

  MPI_Comm_rank( local_communicator, &myrank );

  base = myrank * DIM1 * DIM2 * DIM3 + 1;
  utilities_fill_unique_array_3D_double( &array[0], DIM1, DIM2, DIM3, base);

  if ( myrank == 0 ) {
    snprintf(method, 50, "mpicd_manual");

    bytes = (DIM2-2)*(DIM3-2) * typesize;

    timing_init( testname, &method[0], bytes );
  }

  for( i=0 ; i<outer_loop ; i++ ) {

    psize = (DIM2-2)*(DIM3-2);
    buffer =(double*)malloc( psize * sizeof(double) );

    if ( myrank == 0 ) {
      timing_record(1);
    }

    for( j=0 ; j<inner_loop ; j++ ) {

      if ( myrank == 0 ) {
        base = 0;
        for( k=1 ; k<DIM3-1 ; k++ ) {
          for( l=1 ; l<DIM2-1 ; l++ ) {
            buffer[base++] = array[idx3D(DIM1-2,l,k,DIM1,DIM2)];
          }
        }
        timing_record(2);
        MPI_Send( &buffer[0], psize*sizeof(double), MPI_BYTE, 1, itag, local_communicator );
        MPI_Recv( &buffer[0], psize*sizeof(double), MPI_BYTE, 1, itag, local_communicator, &status );
        timing_record(3);
        base = 0;
        for( k=1 ; k<DIM3-1 ; k++ ) {
          for( l=1 ; l<DIM2-1 ; l++ ) {
            array[idx3D(DIM1-1,l,k,DIM1,DIM2)] = buffer[base++];
          }
        }
        timing_record(4);
      } else {
        MPI_Recv( &buffer[0], psize*sizeof(double), MPI_BYTE, 0, itag, local_communicator, &status );
        base = 0;
        for( k=1 ; k<DIM3-1 ; k++ ) {
          for( l=1 ; l<DIM2-1 ; l++ ) {
            array[idx3D(DIM1-1,l,k,DIM1,DIM2)] = buffer[base++];
          }
        }
        base = 0;
        for( k=1 ; k<DIM3-1 ; k++ ) {
          for( l=1 ; l<DIM2-1 ; l++ ) {
            buffer[base++] = array[idx3D(DIM1-2,l,k,DIM1,DIM2)];
          }
        }
        MPI_Send( &buffer[0], psize*sizeof(double), MPI_BYTE, 0, itag, local_communicator );
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

  free(array);
}


void timing_nas_mg_y_manual( int DIM1, int DIM2, int DIM3, int outer_loop, int inner_loop, int* correct_flag, int* ptypesize, char* testname, MPI_Comm local_communicator ) {

 double* array;
 double* buffer;

 int myrank;
 int base, i, j, k, l, typesize = sizeof(double), bytes, psize;
  MPI_Status status;

 char method[50];

//! just some statements to prevent compiler warnings of unused variables
//! those parameter are included for future features
 *correct_flag = 0;
 *ptypesize = 0;
// typesize = filehandle_debug

  array =(double*)malloc( DIM1 * DIM2 * DIM3 * sizeof(double) );

  MPI_Comm_rank( local_communicator, &myrank );

  base = myrank * DIM1 * DIM2 * DIM3 + 1;
  utilities_fill_unique_array_3D_double( &array[0], DIM1, DIM2, DIM3, base);

  if ( myrank == 0 ) {
    snprintf(method, 50, "mpicd_manual");

    bytes = (DIM1-2)*(DIM3-2) * typesize;

    timing_init( testname, method, bytes );
  }

  for( i=0 ; i<outer_loop ; i++ ) {

    psize = (DIM1-2)*(DIM3-2);
    buffer =(double*)malloc( psize * sizeof(double) );

    if ( myrank == 0 ) {
      timing_record(1);
    }

    for( j=0 ; j<inner_loop ; j++ ) {

      if ( myrank == 0 ) {
        base = 0;
        for( k=1 ; k<DIM3-1 ; k++ ) {
          for( l=1 ; l<DIM1-1 ; l++ ) {
            buffer[base++] = array[idx3D(l,DIM2-2,k,DIM1,DIM2)];
          }
        }
        timing_record(2);
        MPI_Send( &buffer[0], psize*sizeof(double), MPI_BYTE, 1, itag, local_communicator );
        MPI_Recv( &buffer[0], psize*sizeof(double), MPI_BYTE, 1, itag, local_communicator, &status );
        timing_record(3);
        base = 0;
        for( k=1 ; k<DIM3-1 ; k++ ) {
          for( l=1 ; l<DIM1-1 ; l++ ) {
            array[idx3D(l,DIM2-1,k,DIM1,DIM2)] = buffer[base++];
          }
        }
        timing_record(4);
      } else {
        MPI_Recv( &buffer[0], psize*sizeof(double), MPI_BYTE, 0, itag, local_communicator, &status );
        base = 0;
        for( k=1 ; k<DIM3-1 ; k++ ) {
          for( l=1 ; l<DIM1-1 ; l++ ) {
            array[idx3D(l,DIM2-1,k,DIM1,DIM2)] = buffer[base++];
          }
        }
        base = 0;
        for( k=1 ; k<DIM3-1 ; k++ ) {
          for( l=1 ; l<DIM1-1 ; l++ ) {
            buffer[base++] = array[idx3D(l,DIM2-2,k,DIM1,DIM2)];
          }
        }
        MPI_Send( &buffer[0], psize*sizeof(double), MPI_BYTE, 0, itag, local_communicator );
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

  free(array);
}


void timing_nas_mg_z_manual( int DIM1, int DIM2, int DIM3, int outer_loop, int inner_loop, int* correct_flag, int* ptypesize, char* testname, MPI_Comm local_communicator ) {

  double* array;
  double* buffer;

  int myrank;
  int base, i, j, k, l, typesize = sizeof(double), bytes, psize;
  MPI_Status status;

  char method[50];

//! just some statements to prevent compiler warnings of unused variables
//! those parameter are included for future features
  *correct_flag = 0;
  *ptypesize = 0;
//typesize = filehandle_debug

  array =(double*)malloc( DIM1 * DIM2 * DIM3 * sizeof(double) );

  MPI_Comm_rank( local_communicator, &myrank );

  base = myrank * DIM1 * DIM2 * DIM3 + 1;
  utilities_fill_unique_array_3D_double( &array[0], DIM1, DIM2, DIM3, base );

  if ( myrank == 0 ) {
    snprintf( method, 50, "mpicd_manual" );

    bytes = (DIM1-2) * (DIM2-2) * typesize;

    timing_init( testname, &method[0], bytes );
  }

  for( i=0 ; i<outer_loop ; i++ ) {

    psize = (DIM1-2) * (DIM2-2);
    buffer =(double*)malloc( psize * sizeof(double) );

    if ( myrank == 0 ) {
      timing_record(1);
    }

    for( j=0 ; j<inner_loop ; j++ ) {

      if ( myrank == 0 ) {
        base = 0;
        for( k=1 ; k<DIM2-1 ; k++ ) {
          for( l=1 ; l<DIM1-1 ; l++ ) {
            buffer[base++] = array[idx3D(l,k,1,DIM1,DIM2)];
          }
        }
        timing_record(2);
        MPI_Send( &buffer[0], psize*sizeof(double), MPI_BYTE, 1, itag, local_communicator );
        MPI_Recv( &buffer[0], psize*sizeof(double), MPI_BYTE, 1, itag, local_communicator, &status );
        timing_record(3);
        base = 0;
        for( k=1 ; k<DIM2-1 ; k++ ) {
          for( l=1 ; l<DIM1-1 ; l++ ) {
            array[idx3D(l,k,0,DIM1,DIM2)] = buffer[base++];
          }
        }
        timing_record(4);
      } else {
        MPI_Recv( &buffer[0], psize*sizeof(double), MPI_BYTE, 0, itag, local_communicator, &status );
        base = 0;
        for( k=1 ; k<DIM2-1 ; k++ ) {
          for( l=1 ; l<DIM1-1 ; l++ ) {
            array[idx3D(l,k,0,DIM1,DIM2)] = buffer[base++];
          }
        }
        base = 0;
        for( k=1 ; k<DIM2-1 ; k++ ) {
          for( l=1 ; l<DIM1-1 ; l++ ) {
            buffer[base++] = array[idx3D(l,k,1,DIM1,DIM2)];
          }
        }
        MPI_Send( &buffer[0], psize*sizeof(double), MPI_BYTE, 0, itag, local_communicator );
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

  free(array);
}
