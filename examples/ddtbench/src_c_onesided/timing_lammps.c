// Copyright (c) 2012 The Trustees of University of Illinois. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include <stdio.h>
#include <stdlib.h>

#include "mpi.h"
#include "../src_c/ddtbench.h"

#define itag 0

static inline int idx2D(int x, int y, int DIM1) {
  return x+y*DIM1;
}

void timing_lammps_full_ddt( int DIM1, int icount, int* list, int outer_loop, int inner_loop, int* correct_flag, int* ptypesize, char* testname, MPI_File filehandle_debug, MPI_Comm local_communicator) {

  double* array;

  int myrank;
  int i, j, typesize, bytes, base;
  int counter;

  MPI_Datatype dtype_indexed1_t, dtype_indexed3_t, dtype_send_t, dtype_cont1_t, dtype_cont3_t, dtype_recv_t, oldtype[6];
  MPI_Aint address_displacement[6];
  int blocklength[6];
  int* index_displacement;

  int* temp_displacement;

  char method[50];

  MPI_Win win;

//! just some statements to prevent compiler warnings of unused variables
//! those parameter are included for future features
  *correct_flag = 0;
  *ptypesize = 0;
//  typesize = filehandle_debug

//conversion from fortran to c
  temp_displacement = malloc( icount * outer_loop * sizeof(int) );
  for( i = 0 ; i<outer_loop ; i++ ) {
    for( j = 0 ; j<icount ; j++ ) {
      temp_displacement[idx2D(j,i,icount)] = list[idx2D(j,i,icount)] - 1;
    }
  }
 
  MPI_Comm_rank( local_communicator, &myrank );

  array = malloc( 8 * (DIM1+icount) * sizeof(double) );

  MPI_Win_create( array, 8 * (DIM1+icount) * sizeof(double), sizeof(double), MPI_INFO_NULL, local_communicator, &win );
  MPI_Win_fence( 0 /* assert */, win ); /* initial fence to open epoch */

  counter = 0;
  base = myrank * (8*(DIM1+icount)) + 1;
  utilities_fill_unique_array_2D_double( &array[counter], 3, DIM1+icount, base+counter );
  counter += 3*(DIM1+icount);
  utilities_fill_unique_array_1D_double( &array[counter], DIM1+icount, base+counter );
  counter += DIM1 + icount;
  utilities_fill_unique_array_1D_double( &array[counter], DIM1+icount, base+counter );
  counter += DIM1 + icount;
  utilities_fill_unique_array_1D_double( &array[counter], DIM1+icount, base+counter );
  counter += DIM1 + icount;
  utilities_fill_unique_array_1D_double( &array[counter], DIM1+icount, base+counter );
  counter += DIM1 + icount;
  utilities_fill_unique_array_1D_double( &array[counter], DIM1+icount, base+counter );

  if ( myrank == 0 ) {
    snprintf( method, 50, "mpi_ddt" );

    MPI_Type_size( MPI_DOUBLE, &typesize );
    bytes = icount * 8 * typesize;
 
    timing_init( testname, &method[0], bytes );
  }

  for( i=0 ; i<outer_loop ; i++ ) {

    index_displacement = malloc( icount * sizeof(int) );

    MPI_Type_create_indexed_block( icount, 1, &temp_displacement[idx2D(0,i,icount)], MPI_DOUBLE, &dtype_indexed1_t );
    
    for( j=0 ; j<icount ; j++ ) {
      index_displacement[j] = 3 * temp_displacement[idx2D(j,i,icount)];
    }
    MPI_Type_create_indexed_block( icount, 3, &index_displacement[0], MPI_DOUBLE, &dtype_indexed3_t );

    address_displacement[0] = 0;
    address_displacement[1] = 3 * (icount+DIM1) * sizeof(double);
    address_displacement[2] = 4 * (icount+DIM1) * sizeof(double);
    address_displacement[3] = 5 * (icount+DIM1) * sizeof(double);
    address_displacement[4] = 6 * (icount+DIM1) * sizeof(double);
    address_displacement[5] = 7 * (icount+DIM1) * sizeof(double);

    oldtype[0] = dtype_indexed3_t;
    blocklength[0] = 1;
    for( j=1 ; j<6 ; j++ ) {
      oldtype[j] = dtype_indexed1_t;
      blocklength[j] = 1;
    }
 
    MPI_Type_create_struct( 6, &blocklength[0], &address_displacement[0], &oldtype[0], &dtype_send_t );
    MPI_Type_commit( &dtype_send_t );

    MPI_Type_free( &dtype_indexed1_t );
    MPI_Type_free( &dtype_indexed3_t );

    MPI_Type_contiguous( icount, MPI_DOUBLE, &dtype_cont1_t );
	  MPI_Type_contiguous( 3*icount, MPI_DOUBLE, &dtype_cont3_t );

    address_displacement[0] = (3 * DIM1) * sizeof(double);
    address_displacement[1] = (3 * (DIM1+icount) + DIM1) * sizeof(double);
    address_displacement[2] = (4 * (DIM1+icount) + DIM1) * sizeof(double);
    address_displacement[3] = (5 * (DIM1+icount) + DIM1) * sizeof(double);
    address_displacement[4] = (6 * (DIM1+icount) + DIM1) * sizeof(double);
    address_displacement[5] = (7 * (DIM1+icount) + DIM1) * sizeof(double);

    oldtype[0] = dtype_cont3_t;
    for( j=1 ; j<6 ; j++ ) {
      oldtype[j] = dtype_cont1_t;
    }
 
    MPI_Type_create_struct( 6, &blocklength[0], &address_displacement[0], &oldtype[0], &dtype_recv_t );
    MPI_Type_commit( &dtype_recv_t );

    MPI_Type_free( &dtype_cont1_t );
    MPI_Type_free( &dtype_cont3_t );

    free(index_displacement );

    if ( myrank == 0 ) {
      timing_record(1);
    }

    for( j=0 ; j<inner_loop ; j++ ) {

      if ( myrank == 0 ) {
        MPI_Put( &array[0], 1, dtype_send_t, 1 /* target */, 0 /* offset */, 1 /* count */, dtype_recv_t, win );
        MPI_Win_fence( 0 /* assert */, win );
        MPI_Win_fence( 0 /* assert */, win );
        timing_record(3);
      } else {
        MPI_Win_fence( 0 /* assert */, win );
        MPI_Put( &array[0], 1, dtype_send_t, 0 /* target */, 0 /* offset */, 1 /* count */, dtype_recv_t, win );
        MPI_Win_fence( 0 /* assert */, win );
      }

    } //! inner loop

    MPI_Type_free( &dtype_send_t );
	  MPI_Type_free( &dtype_recv_t );

    if ( myrank == 0 ) {
      timing_record(5);
    }

  } //! outer loop

  if ( myrank == 0 ) {
    timing_print( 1 );
  }

  free(temp_displacement);

  MPI_Win_free( &win );

  free( array );
}

void timing_lammps_full_manual( int DIM1, int icount, int* list, int outer_loop, int inner_loop, int* correct_flag, int* ptypesize, char* testname, MPI_File filehandle_debug, MPI_Comm local_communicator) {

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

  MPI_Win win;

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

  MPI_Comm_rank( local_communicator, &myrank );

  isize = 8*icount;
  MPI_Type_size( MPI_DOUBLE, &typesize );
  bytes = isize * typesize;

  buffer = malloc( isize * sizeof(double) );
  MPI_Win_create( buffer, isize * sizeof(double), sizeof(double), MPI_INFO_NULL, local_communicator, &win );
  MPI_Win_fence( 0 /* assert */, win ); /* initial fence to open epoch */
  
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

    timing_init( testname, &method[0], bytes );
  }

  for( i=0 ; i<outer_loop ; i++ ) {

    for( j=0 ; j<inner_loop ; j++ ) {

      if ( myrank == 0 ) {
        pos = 0;
        for( k=0 ; k<icount ; k++ ) {
          l=temp_displacement[idx2D(k,i,icount)];
          buffer[pos++] = ax[idx2D(0,l,3)];
          buffer[pos++] = ax[idx2D(1,l,3)];
          buffer[pos++] = ax[idx2D(2,l,3)];
          buffer[pos++] = atag[l];
          buffer[pos++] = atype[l];
          buffer[pos++] = amask[l];
          buffer[pos++] = amolecule[l];
          buffer[pos++] = aq[l];
        }
        timing_record(2);
        MPI_Put( &buffer[0], isize, MPI_DOUBLE, 1 /* target */, 0 /* offset */, isize /* count */, MPI_DOUBLE, win );
        MPI_Win_fence( 0 /* assert */, win );
        MPI_Win_fence( 0 /* assert */, win );
        timing_record(3);
        pos = 0;
        for( k=0 ; k<icount ; k++ ) {
          l=DIM1+k;
          ax[idx2D(0,l,3)] = buffer[pos++];
          ax[idx2D(1,l,3)] = buffer[pos++];
          ax[idx2D(2,l,3)] = buffer[pos++];
          atag[l] = buffer[pos++];
          atype[l] = buffer[pos++];
          amask[l] = buffer[pos++];
          amolecule[l] = buffer[pos++];
          aq[l] = buffer[pos++];
        }
        timing_record(4);
      } else {
        MPI_Win_fence( 0 /* assert */, win );
        pos = 0;
        for( k=0 ; k<icount ; k++ ) {
          l=DIM1+k;
          ax[idx2D(0,l,3)] = buffer[pos++];
          ax[idx2D(1,l,3)] = buffer[pos++];
          ax[idx2D(2,l,3)] = buffer[pos++];
          atag[l] = buffer[pos++];
          atype[l] = buffer[pos++];
          amask[l] = buffer[pos++];
          amolecule[l] = buffer[pos++];
          aq[l] = buffer[pos++];
        }
        pos = 0;
        for( k=0 ; k<icount ; k++ ) {
          l=temp_displacement[idx2D(k,i,icount)];
          buffer[pos++] = ax[idx2D(0,l,3)];
          buffer[pos++] = ax[idx2D(1,l,3)];
          buffer[pos++] = ax[idx2D(2,l,3)];
          buffer[pos++] = atag[l];
          buffer[pos++] = atype[l];
          buffer[pos++] = amask[l];
          buffer[pos++] = amolecule[l];
          buffer[pos++] = aq[l];
        }
        MPI_Put( &buffer[0], isize, MPI_DOUBLE, 0 /* target */, 0 /* offset */, isize /* count */, MPI_DOUBLE, win );
        MPI_Win_fence( 0 /* assert */, win );
      }

    } //! inner loop

  } //! outer loop

  if ( myrank == 0 ) {
    timing_print( 1 );
  }

  MPI_Win_free( &win );

  free( buffer );

  free(temp_displacement);

  free(atag);
  free(atype);
  free(amask);
  free(amolecule);
  free(aq);
  free(ax);
} 

void timing_lammps_full_mpi_pack_ddt( int DIM1, int icount, int* list, int outer_loop, int inner_loop, int* correct_flag, int* ptypesize, char* testname, MPI_File filehandle_debug, MPI_Comm local_communicator) {

  double* atag;
  double* atype;
  double* amask;
  double* amolecule;
  double* aq;
  double* ax;

  double* buffer;

  int myrank;
  int i, j, typesize, bytes, base, pos;

  MPI_Datatype dtype_indexed1_t, dtype_indexed3_t, dtype_send_t, dtype_cont1_t, dtype_cont3_t, dtype_recv_t, oldtype[6];
  MPI_Aint address_displacement[6];
  int blocklength[6];
  int* index_displacement;

  int* temp_displacement;

  char method[50];

  MPI_Win win;

//! just some statements to prevent compiler warnings of unused variables
//! those parameter are included for future features
  *correct_flag = 0;
  *ptypesize = 0;
//  typesize = filehandle_debug

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

  MPI_Comm_rank( local_communicator, &myrank  );

  MPI_Type_size( MPI_DOUBLE, &typesize );
  bytes = 8 * icount * typesize ;

  buffer = malloc( 8 * icount * sizeof(double) );
  MPI_Win_create( buffer, 8 * icount * sizeof(double), sizeof(double), MPI_INFO_NULL, local_communicator, &win );
  MPI_Win_fence( 0 /* assert */, win ); /* initial fence to open epoch */

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
    snprintf(method, 50, "mpi_pack_ddt" );

    timing_init( testname, &method[0], bytes );
  }

  for( i=0 ; i<outer_loop ; i++ ) {

    index_displacement = malloc( icount * sizeof(int) );
    MPI_Type_create_indexed_block( icount, 1, &temp_displacement[idx2D(0,i,icount)], MPI_DOUBLE, &dtype_indexed1_t );
    
    for( j = 0 ; j < icount ; j++ ) {  
      index_displacement[j] = 3 * temp_displacement[idx2D(j,i,icount)];
    }
    MPI_Type_create_indexed_block( icount, 3, &index_displacement[0], MPI_DOUBLE, &dtype_indexed3_t );

    MPI_Get_address( &ax[0], &address_displacement[0] );
    MPI_Get_address( &atag[0], &address_displacement[1] );
    MPI_Get_address( &atype[0], &address_displacement[2] );
    MPI_Get_address( &amask[0], &address_displacement[3] );
    MPI_Get_address( &amolecule[0], &address_displacement[4] );
    MPI_Get_address( &aq[0], &address_displacement[5] );

    oldtype[0] = dtype_indexed3_t;
    blocklength[0] = 1;
    for( j = 1 ; j < 6 ; j++ ) {
      oldtype[j] = dtype_indexed1_t;
      blocklength[j] = 1;
    }
 
    MPI_Type_create_struct( 6, &blocklength[0], &address_displacement[0], &oldtype[0], &dtype_send_t );
    MPI_Type_commit( &dtype_send_t );

    MPI_Type_free( &dtype_indexed1_t );
    MPI_Type_free( &dtype_indexed3_t );

    MPI_Type_contiguous( icount, MPI_DOUBLE, &dtype_cont1_t );
  	MPI_Type_contiguous( 3*icount, MPI_DOUBLE, &dtype_cont3_t );

    MPI_Get_address( &ax[3*DIM1], &address_displacement[0] );
    MPI_Get_address( &atag[DIM1], &address_displacement[1] );
    MPI_Get_address( &atype[DIM1], &address_displacement[2] );
    MPI_Get_address( &amask[DIM1], &address_displacement[3] );
    MPI_Get_address( &aq[DIM1], &address_displacement[4] );
    MPI_Get_address( &amolecule[DIM1], &address_displacement[5] );

    oldtype[0] = dtype_cont3_t;
    for( j=1 ; j<6 ; j++ ) {
      oldtype[j] = dtype_cont1_t;
    }
 
    MPI_Type_create_struct( 6, &blocklength[0], &address_displacement[0], &oldtype[0], &dtype_recv_t );
    MPI_Type_commit( &dtype_recv_t );

    MPI_Type_free( &dtype_cont1_t );
    MPI_Type_free( &dtype_cont3_t );

    free( index_displacement );

    if ( myrank == 0 ) {
      timing_record(1);
    }

    for( j=0 ; j<inner_loop ; j++ ) {

      if ( myrank == 0 ) {
        pos = 0;
        MPI_Pack( MPI_BOTTOM, 1, dtype_send_t, &buffer[0], bytes, &pos, local_communicator );
        timing_record(2);
        MPI_Put( &buffer[0], pos, MPI_PACKED, 1 /* target */, 0 /* offset */, pos /* count */, MPI_PACKED, win );
        MPI_Win_fence( 0 /* assert */, win );
        MPI_Win_fence( 0 /* assert */, win );
        timing_record(3);
        pos = 0;
        MPI_Unpack( &buffer[0], bytes, &pos, MPI_BOTTOM, 1, dtype_recv_t, local_communicator );
        timing_record(4);
      } else {
        MPI_Win_fence( 0 /* assert */, win );
        pos = 0;
        MPI_Unpack( &buffer[0], bytes, &pos, MPI_BOTTOM, 1, dtype_recv_t, local_communicator );
        pos = 0;
        MPI_Pack( MPI_BOTTOM, 1, dtype_send_t, &buffer[0], bytes, &pos, local_communicator );
        MPI_Put( &buffer[0], pos, MPI_PACKED, 0 /* target */, 0 /* offset */, pos /* count */, MPI_PACKED, win );
        MPI_Win_fence( 0 /* assert */, win );
      }

    } //! inner loop

    MPI_Type_free( &dtype_send_t );
  	MPI_Type_free( &dtype_recv_t );

    if ( myrank == 0 ) {
      timing_record(5);
    }

  } //! outer loop

  if ( myrank == 0 ) {
    timing_print( 1 );
  }

  MPI_Win_free( &win );

  free( buffer );

  free(temp_displacement);

  free(atag);
  free(atype);
  free(amask);
  free(amolecule);
  free(aq);
  free(ax);
}

void timing_lammps_atomic_ddt( int DIM1, int icount, int* list, int outer_loop, int inner_loop, int* correct_flag, int* ptypesize, char* testname, MPI_File filehandle_debug, MPI_Comm local_communicator) {

  double* array;

  int myrank;
  int i, j, typesize, bytes, base;
  int counter;

  MPI_Datatype dtype_indexed1_t, dtype_indexed3_t, dtype_send_t, dtype_cont1_t, dtype_cont3_t, dtype_recv_t, oldtype[4];
  MPI_Aint address_displacement[4];
  int blocklength[4];
  int* index_displacement;

  int* temp_displacement;

  char method[50];

  MPI_Win win;

//! just some statements to prevent compiler warnings of unused variables
//! those parameter are included for future features
  *correct_flag = 0;
  *ptypesize = 0;
//  typesize = filehandle_debug

//conversion from fortran to c
  temp_displacement = malloc( icount * outer_loop * sizeof(int) );
  for( i = 0 ; i<outer_loop ; i++ ) {
    for( j = 0 ; j<icount ; j++ ) {
      temp_displacement[idx2D(j,i,icount)] = list[idx2D(j,i,icount)] - 1;
    }
  }

  MPI_Comm_rank( local_communicator, &myrank );

  array = malloc( 6 * (DIM1+icount) * sizeof(double) );
  MPI_Win_create( array, 6 * (DIM1+icount) * sizeof(double), sizeof(double), MPI_INFO_NULL, local_communicator, &win );
  MPI_Win_fence( 0 /* assert */, win ); /* initial fence to open epoch */

  counter = 0;
  base = myrank * (6*(DIM1+icount)) + 1;
  utilities_fill_unique_array_2D_double( &array[counter], 3, DIM1+icount, base+counter );
  counter += 3*(DIM1+icount);
  utilities_fill_unique_array_1D_double( &array[counter], DIM1+icount, base+counter );
  counter += DIM1 + icount;
  utilities_fill_unique_array_1D_double( &array[counter], DIM1+icount, base+counter );
  counter += DIM1 + icount;
  utilities_fill_unique_array_1D_double( &array[counter], DIM1+icount, base+counter );

  if ( myrank == 0 ) {
    snprintf( method, 50, "mpi_ddt" );

    MPI_Type_size( MPI_DOUBLE, &typesize );
    bytes = icount * 6 * typesize;

    timing_init( testname, &method[0], bytes );
  }

  for( i=0 ; i<outer_loop ; i++ ) {

    index_displacement = malloc( icount * sizeof(int) );

    MPI_Type_create_indexed_block( icount, 1, &temp_displacement[idx2D(0,i,icount)], MPI_DOUBLE, &dtype_indexed1_t );
      
    for( j=0 ; j<icount ; j++ ) {
      index_displacement[j] = 3 * temp_displacement[idx2D(j,i,icount)];
    }
    MPI_Type_create_indexed_block( icount, 3, &index_displacement[0], MPI_DOUBLE, &dtype_indexed3_t );

    address_displacement[0] = 0;
    address_displacement[1] = 3 * (icount+DIM1) * sizeof(double);
    address_displacement[2] = 4 * (icount+DIM1) * sizeof(double);
    address_displacement[3] = 5 * (icount+DIM1) * sizeof(double);

    oldtype[0] = dtype_indexed3_t;
    blocklength[0] = 1;
    for( j=1 ; j < 4 ; j++ ) {
      oldtype[j] = dtype_indexed1_t;
      blocklength[j] = 1;
    }
 
    MPI_Type_create_struct( 4, &blocklength[0], &address_displacement[0], &oldtype[0], &dtype_send_t );
    MPI_Type_commit( &dtype_send_t );

    MPI_Type_free( &dtype_indexed1_t );
    MPI_Type_free( &dtype_indexed3_t );

	  MPI_Type_contiguous( icount, MPI_DOUBLE, &dtype_cont1_t );
	  MPI_Type_contiguous( 3*icount, MPI_DOUBLE, &dtype_cont3_t );

    address_displacement[0] = (3 * DIM1) * sizeof(double);
    address_displacement[1] = (3 * (DIM1+icount) + DIM1) * sizeof(double);
    address_displacement[2] = (4 * (DIM1+icount) + DIM1) * sizeof(double);
    address_displacement[3] = (5 * (DIM1+icount) + DIM1) * sizeof(double);

    oldtype[0] = dtype_cont3_t;
    for( j=1 ; j < 4 ; j++ ) {
      oldtype[j] = dtype_cont1_t;
    }
 
    MPI_Type_create_struct( 4, &blocklength[0], &address_displacement[0], &oldtype[0], &dtype_recv_t );
    MPI_Type_commit( &dtype_recv_t );

    MPI_Type_free( &dtype_cont1_t );
    MPI_Type_free( &dtype_cont3_t );

    free(index_displacement);

    if ( myrank == 0 ) {
      timing_record(1);
    }

    for( j=0 ; j<inner_loop ; j++ ) { 
      if ( myrank == 0 ) {
        MPI_Put( &array[0], 1, dtype_send_t, 1 /* target */, 0 /* offset */, 1 /* count */, dtype_recv_t, win );
        MPI_Win_fence( 0 /* assert */, win );
        MPI_Win_fence( 0 /* assert */, win );
        timing_record(3);
      } else {
        MPI_Win_fence( 0 /* assert */, win );
        MPI_Put( &array[0], 1, dtype_send_t, 0 /* target */, 0 /* offset */, 1 /* count */, dtype_recv_t, win );
        MPI_Win_fence( 0 /* assert */, win );
      }

    } //! inner loop

    MPI_Type_free( &dtype_send_t );
	  MPI_Type_free( &dtype_recv_t );

    if ( myrank == 0 ) {
      timing_record(5);
    }

  } //! outer loop

  if ( myrank == 0 ) {
    timing_print( 1 );
  }

  free(temp_displacement);

  MPI_Win_free( &win );

  free( array );
}

void timing_lammps_atomic_manual( int DIM1, int icount, int* list, int outer_loop, int inner_loop, int* correct_flag, int* ptypesize, char* testname, MPI_File filehandle_debug, MPI_Comm local_communicator) {

  double* atag;
  double* atype;
  double* amask;
  double* ax;

  double* buffer;

  int myrank;
  int i, j, k, l, typesize, bytes, base, pos, isize;

  int* temp_displacement;

  char method[50];

  MPI_Win win;

//! just some statements to prevent compiler warnings of unused variables
//! those parameter are included for future features
  *correct_flag = 0;
  *ptypesize = 0;
// typesize = filehandle_debug

  atag = malloc( (DIM1+icount) * sizeof(double) );
  atype = malloc( (DIM1+icount) * sizeof(double) );
  amask = malloc( (DIM1+icount) * sizeof(double) );
  ax = malloc( 3 * (DIM1+icount) * sizeof(double) );

//conversion from fortran to c
  temp_displacement = malloc( icount * outer_loop * sizeof(int) );
  for( i = 0 ; i<outer_loop ; i++ ) {
    for( j = 0 ; j<icount ; j++ ) {
      temp_displacement[idx2D(j,i,icount)] = list[idx2D(j,i,icount)] - 1;
    }
  }

  MPI_Comm_rank( local_communicator, &myrank );

  isize = 6*icount;
  MPI_Type_size( MPI_DOUBLE, &typesize );
  bytes = isize * typesize;

  buffer = malloc( isize * sizeof(double) );
  MPI_Win_create( buffer, isize * sizeof(double), sizeof(double), MPI_INFO_NULL, local_communicator, &win );
  MPI_Win_fence( 0 /* assert */, win ); /* initial fence to open epoch */

  base = myrank * (6*(DIM1+icount)) + 1;
  utilities_fill_unique_array_2D_double( &ax[0], 3, DIM1+icount, base );
  base = base + 3*(DIM1+icount);
  utilities_fill_unique_array_1D_double( &atag[0], DIM1+icount, base );
  base = base + DIM1 + icount;
  utilities_fill_unique_array_1D_double( &atype[0], DIM1+icount, base );
  base = base + DIM1 + icount;
  utilities_fill_unique_array_1D_double( &amask[0], DIM1+icount, base );

  if ( myrank == 0 ) {
    snprintf(method, 50, "manual");

    timing_init( testname, &method[0], bytes );
  }

  for( i=0 ; i<outer_loop ; i++ ) {

    for( j=0 ; j<inner_loop ; j++ ) {

      if ( myrank == 0 ) {
        pos = 0;
        for( k=0 ; k<icount ; k++ ) {
          l = temp_displacement[idx2D(k,i,icount)];
          buffer[pos++] = ax[idx2D(0,l,3)];
          buffer[pos++] = ax[idx2D(1,l,3)];
          buffer[pos++] = ax[idx2D(2,l,3)];
          buffer[pos++] = atag[l];
          buffer[pos++] = atype[l];
          buffer[pos++] = amask[l];
        }
        timing_record(2);
        MPI_Put( &buffer[0], isize, MPI_DOUBLE, 1 /* target */, 0 /* offset */, isize /* count */, MPI_DOUBLE, win );
        MPI_Win_fence( 0 /* assert */, win );
        MPI_Win_fence( 0 /* assert */, win );
        timing_record(3);
        pos = 0;
        for( k=0 ; k<icount ; k++ ) {
          l = DIM1+k;
          ax[idx2D(0,l,3)] = buffer[pos++];
          ax[idx2D(1,l,3)] = buffer[pos++];
          ax[idx2D(2,l,3)] = buffer[pos++];
          atag[l] = buffer[pos++];
          atype[l] = buffer[pos++];
          amask[l] = buffer[pos++];
        }
        timing_record(4);
      } else {
        MPI_Win_fence( 0 /* assert */, win );
        pos = 0;
        for( k=0 ; k<icount ; k++ ) {
          l = DIM1+k;
          buffer[pos++] = ax[idx2D(0,l,3)];
          buffer[pos++] = ax[idx2D(1,l,3)];
          buffer[pos++] = ax[idx2D(2,l,3)];
          buffer[pos++] = atag[l];
          buffer[pos++] = atype[l];
          buffer[pos++] = amask[l];
        }
        pos = 0;
        for( k=0 ; k<icount ; k++ ) {
          l = temp_displacement[idx2D(k,i,icount)];
          ax[idx2D(0,l,3)] = buffer[pos++];
          ax[idx2D(1,l,3)] = buffer[pos++];
          ax[idx2D(2,l,3)] = buffer[pos++];
          atag[l] = buffer[pos++];
          atype[l] = buffer[pos++];
          amask[l] = buffer[pos++];
        }
        MPI_Put( &buffer[0], isize, MPI_DOUBLE, 0 /* target */, 0 /* offset */, isize /* count */, MPI_DOUBLE, win );
        MPI_Win_fence( 0 /* assert */, win );
      }

    } //! inner loop

  } //! outer loop

  if ( myrank == 0 ) {
    timing_print( 1 );
  }

  MPI_Win_free( &win );

  free( buffer );

  free(temp_displacement);

  free(ax);
  free(atag);
  free(atype);
  free(amask);
}

void timing_lammps_atomic_mpi_pack_ddt( int DIM1, int icount, int* list, int outer_loop, int inner_loop, int* correct_flag, int* ptypesize, char* testname, MPI_File filehandle_debug, MPI_Comm local_communicator ){

  double* atag;
  double* atype;
  double* amask;
  double* ax;

  double* buffer;

  int myrank;
  int i, j, typesize, bytes, base, pos;

  MPI_Datatype dtype_indexed1_t, dtype_indexed3_t, dtype_send_t, dtype_cont1_t, dtype_cont3_t, dtype_recv_t, oldtype[4];
  MPI_Aint address_displacement[4];
  int blocklength[4];
  int* index_displacement;

  int* temp_displacement;

  char method[50];

  MPI_Win win;

//! just some statements to prevent compiler warnings of unused variables
//! those parameter are included for future features
  *correct_flag = 0;
  *ptypesize = 0;
//  typesize = filehandle_debug

  atag = malloc( (DIM1+icount) * sizeof(double) );
  atype = malloc( (DIM1+icount) * sizeof(double) );
  amask = malloc( (DIM1+icount) * sizeof(double) );
  ax = malloc( 3 * (DIM1+icount) * sizeof(double) );

//conversion from fortran to c
  temp_displacement = malloc( icount * outer_loop * sizeof(int) );
  for( i = 0 ; i<outer_loop ; i++ ) {
    for( j = 0 ; j<icount ; j++ ) {
      temp_displacement[idx2D(j,i,icount)] = list[idx2D(j,i,icount)] - 1;
    }
  }

  MPI_Comm_rank( local_communicator, &myrank );

  buffer = malloc( 6 * icount * sizeof(double) );
  MPI_Win_create( buffer, 6 * icount * sizeof(double), sizeof(double), MPI_INFO_NULL, local_communicator, &win );
  MPI_Win_fence( 0 /* assert */, win ); /* initial fence to open epoch */
  
  base = myrank * (6*(DIM1+icount)) + 1;
  utilities_fill_unique_array_2D_double( &ax[0], 3, DIM1+icount, base );
  base = base + 3*(DIM1+icount);
  utilities_fill_unique_array_1D_double( &atag[0], DIM1+icount, base );
  base = base + DIM1 + icount;
  utilities_fill_unique_array_1D_double( &atype[0], DIM1+icount, base );
  base = base + DIM1 + icount;
  utilities_fill_unique_array_1D_double( &amask[0], DIM1+icount, base );

  if ( myrank == 0 ) {
    snprintf( method, 50, "mpi_pack_ddt" );

    MPI_Type_size( MPI_DOUBLE, &typesize );
    bytes = icount * 6 * typesize;

    timing_init( testname, &method[0], bytes );
  }

  for( i=0 ; i<outer_loop ; i++ ) {

    MPI_Type_size( MPI_DOUBLE, &typesize );
    bytes = 6 * icount * typesize;

    index_displacement = malloc( icount * sizeof(int) );
    
    MPI_Type_create_indexed_block( icount, 1, &temp_displacement[idx2D(0,i,icount)], MPI_DOUBLE, &dtype_indexed1_t );

    for( j = 0 ; j<icount ; j++ ) {    
      index_displacement[j] = 3 * temp_displacement[idx2D(j,i,icount)];
    }
    MPI_Type_create_indexed_block( icount, 3, &index_displacement[0], MPI_DOUBLE, &dtype_indexed3_t );

    MPI_Get_address( &ax[0], &address_displacement[0] );
    MPI_Get_address( &atag[0], &address_displacement[1] );
    MPI_Get_address( &atype[0], &address_displacement[2] );
    MPI_Get_address( &amask[0], &address_displacement[3] );

    oldtype[0] = dtype_indexed3_t;
    blocklength[0] = 1;
    for( j=1 ; j<4 ; j++ ) {
      oldtype[j] = dtype_indexed1_t;
      blocklength[j] = 1;
    }
 
    MPI_Type_create_struct( 4, &blocklength[0], &address_displacement[0], &oldtype[0], &dtype_send_t );
    MPI_Type_commit( &dtype_send_t );

    MPI_Type_free( &dtype_indexed1_t );
    MPI_Type_free( &dtype_indexed3_t );

	  MPI_Type_contiguous( icount, MPI_DOUBLE, &dtype_cont1_t );
	  MPI_Type_contiguous( 3*icount, MPI_DOUBLE, &dtype_cont3_t );

    MPI_Get_address( &ax[3*DIM1], &address_displacement[0] );
    MPI_Get_address( &atag[DIM1], &address_displacement[1] );
    MPI_Get_address( &atype[DIM1], &address_displacement[2] );
    MPI_Get_address( &amask[DIM1], &address_displacement[3] );

    oldtype[0] = dtype_cont3_t;
    for( j=1 ; j < 4 ; j++ ) {
      oldtype[j] = dtype_cont1_t;
    }
 
    MPI_Type_create_struct( 4, &blocklength[0], &address_displacement[0], &oldtype[0], &dtype_recv_t );
    MPI_Type_commit( &dtype_recv_t );

    MPI_Type_free( &dtype_cont1_t );
    MPI_Type_free( &dtype_cont3_t );

    free(index_displacement );

    if ( myrank == 0 ) {
      timing_record(1);
    }

    for( j=0 ; j<inner_loop ; j++ ) {

      if ( myrank == 0 ) {
        pos = 0;
        MPI_Pack( MPI_BOTTOM, 1, dtype_send_t, &buffer[0], bytes, &pos, local_communicator );
        timing_record(2);
        MPI_Put( &buffer[0], pos, MPI_PACKED, 1 /* target */, 0 /* offset */, pos /* count */, MPI_PACKED, win );
        MPI_Win_fence( 0 /* assert */, win );
        MPI_Win_fence( 0 /* assert */, win );
        timing_record(3);
        pos = 0;
        MPI_Unpack( &buffer[0], bytes, &pos, MPI_BOTTOM, 1, dtype_recv_t, local_communicator );
        timing_record(4);
      } else {
        MPI_Win_fence( 0 /* assert */, win );
        pos = 0;
        MPI_Unpack( &buffer[0], bytes, &pos, MPI_BOTTOM, 1, dtype_recv_t, local_communicator );
        pos = 0;
        MPI_Pack( MPI_BOTTOM, 1, dtype_send_t, &buffer[0], bytes, &pos, local_communicator );
        MPI_Put( &buffer[0], pos, MPI_PACKED, 0 /* target */, 0 /* offset */, pos /* count */, MPI_PACKED, win );
        MPI_Win_fence( 0 /* assert */, win );
      }

    } //! inner loop

    MPI_Type_free( &dtype_send_t );
	  MPI_Type_free( &dtype_recv_t );

    if ( myrank == 0 ) {
      timing_record(5);
    }

  } //! outer loop

  if ( myrank == 0 ) {
    timing_print( 1 );
  }

  MPI_Win_free( &win );

  free( buffer );

  free(temp_displacement);

  free(ax);
  free(atag);
  free(atype);
  free(amask);
}
