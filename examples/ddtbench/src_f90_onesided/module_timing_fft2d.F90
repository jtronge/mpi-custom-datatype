!     Copyright (c) 2012 The Trustees of University of Illinois. All 
!     rights reserved. Use of this source code is governed by a 
!     BSD-style license that can be found in the LICENSE file.
      
      module timing_fft2d

      private

      public :: timing_fft2d_ddt
      public :: timing_fft2d_manual
      public :: timing_fft2d_mpi_pack_ddt

      contains

      subroutine timing_fft2d_ddt( DIM1, procs, outer_loop, &
        inner_loop, correct_flag, ptypesize, testname, &
        filehandle_debug, local_communicator )

!      use timing, only: timing_init, timing_record, timing_print
      use utilities, only: fill_unique_array_2D_double_complex

      implicit none

      include 'mpif.h'

      integer, intent(in) :: DIM1
      integer, intent(in) :: procs

      integer, intent(in) :: outer_loop, inner_loop

      logical, intent(out) :: correct_flag
      integer, intent(out) :: ptypesize

      character(50), intent(in) :: testname

      integer, intent(in) :: filehandle_debug

      integer, intent(in) :: local_communicator

!> local variables
      double complex, dimension(DIM1, DIM1/procs) :: matrix, recv_array

      integer :: myrank, commsize, ier
      integer :: i, j, k, base, typesize, bytes

!> variables for the datatype construction
      integer :: dtype_vector_t, dtype_resize_t
      integer :: dtype_scatter_t, dtype_gather_t
      integer :: t_complex_size
      integer(kind=MPI_ADDRESS_KIND) :: lb, extent

      character(50) :: method

      integer :: win1, win2
      integer(kind=MPI_ADDRESS_KIND) :: win_size
      integer(kind=MPI_ADDRESS_KIND) :: target_disp

!> just some statements to prevent compiler warnings of unused variables
!> those parameter are included for future features
      correct_flag = .false.
      ptypesize = 0
      typesize = filehandle_debug

      call MPI_Comm_rank( local_communicator, myrank, ier )
      call MPI_Comm_size( local_communicator, commsize, ier )

!> ================= initialize the arrays =================

! assume that one double complex element has a size of 16 byte
      win_size = DIM1 * DIM1/procs * 16
      call MPI_Win_create( matrix(1,1), win_size, 16, MPI_INFO_NULL, &
        local_communicator, win1, ier )
      call MPI_Win_create( recv_array(1,1), win_size, 16, MPI_INFO_NULL, &
        local_communicator, win2, ier )

! initial fence to open epoch
      call MPI_Win_fence( 0, win1, ier )
      call MPI_Win_fence( 0, win2, ier )

      base = myrank * DIM1 * DIM1/procs * 2 + 1
      call fill_unique_array_2D_double_complex( matrix, DIM1, &
        DIM1/procs, base )
 
      if ( myrank .EQ. 0 ) then
        write (method,'(A)') "mpi_ddt"

        call MPI_Type_size( MPI_DOUBLE_COMPLEX, typesize, ier )
        bytes = DIM1/procs * DIM1 * typesize
 
        call timing_init( testname, method, bytes )
      endif
     
      do i=1,outer_loop

        call MPI_Type_vector( DIM1/procs, 1, DIM1, MPI_DOUBLE_COMPLEX, &
          dtype_vector_t, ier )
        call MPI_Type_size( MPI_DOUBLE_COMPLEX, t_complex_size, ier )
        lb = 0
        extent = t_complex_size
        call MPI_Type_create_resized( dtype_vector_t, lb, extent, &
          dtype_resize_t, ier )
        call MPI_Type_contiguous( DIM1/procs, dtype_resize_t, &
          dtype_scatter_t, ier )
        call MPI_Type_commit( dtype_scatter_t, ier )

        call MPI_Type_free( dtype_vector_t, ier )
        call MPI_Type_free( dtype_resize_t, ier )

        call MPI_Type_vector( DIM1/procs, DIM1/procs, DIM1, &
          MPI_DOUBLE_COMPLEX, dtype_vector_t, ier )
        lb = 0
        extent = DIM1/procs * t_complex_size
        call MPI_Type_create_resized( dtype_vector_t, lb, &
          extent, dtype_gather_t, ier )
        call MPI_Type_commit( dtype_gather_t, ier )

        call MPI_Type_free( dtype_vector_t, ier )

        if ( myrank .EQ. 0 ) then
          call timing_record(1)
        endif

        do j=1,inner_loop
          do k=0,commsize-1
            target_disp = k * DIM1/procs
            call MPI_Put( matrix(target_disp+1,1), 1, dtype_gather_t, k, &
              target_disp, 1, dtype_scatter_t, win2, ier )
          enddo
          call MPI_Win_fence( 0, win2, ier )
          
          do k=0,commsize-1
            target_disp = k * DIM1/procs
            call MPI_Put( recv_array(target_disp+1,1), 1, &
              dtype_gather_t, k, target_disp, 1, dtype_scatter_t, win1, &
              ier )
          enddo
          call MPI_Win_fence( 0, win1, ier )

          if ( myrank .EQ. 0 ) then
            call timing_record(3)
          endif

        enddo

        call MPI_Type_free( dtype_gather_t, ier )
        call MPI_Type_free( dtype_scatter_t, ier )

        if ( myrank .EQ. 0 ) then
          call timing_record(5)
        endif

      enddo !> outer loop

      if ( myrank .EQ. 0 ) then
        call timing_print( .true. )
      endif

      call MPI_Win_free( win1, ier )
      call MPI_Win_free( win2, ier )

      end subroutine timing_fft2d_ddt

      subroutine timing_fft2d_manual( DIM1, procs, outer_loop, &
        inner_loop, correct_flag, ptypesize, testname, &
        filehandle_debug, local_communicator )

!      use timing, only: timing_init, timing_record, timing_print
      use utilities, only: fill_unique_array_2D_double_complex

      implicit none

      include 'mpif.h'

      integer, intent(in) :: DIM1
      integer, intent(in) :: procs
      
      integer, intent(in) :: outer_loop, inner_loop

      logical, intent(out) :: correct_flag
      integer, intent(out) :: ptypesize

      character(50), intent(in) :: testname

      integer, intent(in) :: filehandle_debug

      integer, intent(in) :: local_communicator

!> local variables
      double complex, dimension(DIM1,DIM1/procs) :: matrix
      double complex, dimension(:), allocatable :: buffer1, buffer2

      integer :: myrank, commsize, ier
      integer :: i, j, k, l, base, base2, typesize, bytes

      character(50) :: method

      integer :: win1, win2
      integer(kind=MPI_ADDRESS_KIND) :: win_size
      integer(kind=MPI_ADDRESS_KIND) :: target_disp

!> just some statements to prevent compiler warnings of unused variables
!> those parameter are included for future features
      correct_flag = .false.
      ptypesize = 0
      typesize = filehandle_debug

      call MPI_Comm_rank( local_communicator, myrank, ier )
      call MPI_Comm_size( local_communicator, commsize, ier )

!> ================= initialize the arrays =================

      allocate( buffer1(DIM1*DIM1/procs), stat=ier )
      allocate( buffer2(DIM1*DIM1/procs), stat=ier )

! assume that one double complex element has a size of 16 byte
      win_size = DIM1 * DIM1/procs * 16
      call MPI_Win_create( buffer1(1), win_size, 16, MPI_INFO_NULL, &
        local_communicator, win1, ier )
      call MPI_Win_create( buffer2(1), win_size, 16, MPI_INFO_NULL, &
        local_communicator, win2, ier )

! initial fence to open epoch
      call MPI_Win_fence( 0, win1, ier )
      call MPI_Win_fence( 0, win2, ier )

      base = myrank * DIM1 * DIM1/procs * 2 + 1
      call fill_unique_array_2D_double_complex( matrix, DIM1, &
        DIM1/procs, base )

      if ( myrank .EQ. 0 ) then
        write (method,'(A)') "manual"

        call MPI_Type_size( MPI_DOUBLE_COMPLEX, typesize, ier )
        bytes = DIM1/procs * DIM1 * typesize

        call timing_init( testname, method, bytes )
      endif
     
      do i=1,outer_loop

        do j=1,inner_loop
!> pack the data
          do k=0,procs-1
            do l=1,DIM1/procs
              base = k * DIM1/procs * DIM1/procs + (l-1) * DIM1/procs + 1
              base2 = k * DIM1/procs + 1
              buffer1(base:base+DIM1/procs-1) = &
                matrix(base2:base2+DIM1/procs-1,l)
            enddo
          enddo

          if ( myrank .EQ. 0 ) then
            call timing_record(2)            
          endif

          do k=0,commsize-1
            target_disp = k * DIM1/procs*DIM1/procs
            call MPI_Put( buffer1(target_disp+1), DIM1/procs*DIM1/procs, &
              MPI_DOUBLE_COMPLEX, k, target_disp, DIM1/procs*DIM1/procs, &
              MPI_DOUBLE_COMPLEX, win2, ier )
          enddo
          call MPI_Win_fence( 0, win2, ier )

!> unpack the data
          do k=1,DIM1/procs
            do l=1,DIM1
              base = (k-1) + (l-1) * DIM1/procs + 1
              matrix(l,k) = buffer2(base)
            enddo
          enddo

!> pack the data            
          do k=0,procs-1
            do l=1,DIM1/procs
              base = k * DIM1/procs * DIM1/procs + (l-1) * DIM1/procs + 1
              base2 = k * DIM1/procs + 1
              buffer2(base:base+DIM1/procs-1) = &
                matrix(base2:base2+DIM1/procs-1,l)
            enddo
          enddo

          do k=0,commsize-1
            target_disp = k * DIM1/procs*DIM1/procs
            call MPI_Put( buffer2(target_disp+1), DIM1/procs*DIM1/procs, &
              MPI_DOUBLE_COMPLEX, k, target_disp, DIM1/procs*DIM1/procs, &
              MPI_DOUBLE_COMPLEX, win1, ier )
          enddo
          call MPI_Win_fence( 0, win1, ier )

          if ( myrank .EQ. 0 ) then
            call timing_record(3)
          endif

!> unpack the data
          do k=1,DIM1/procs
            do l=1,DIM1
              base = (k-1) + (l-1) * DIM1/procs + 1
              matrix(l,k) = buffer2(base)
            enddo
          enddo          

          if ( myrank .EQ. 0 ) then
            call timing_record(4)
          endif

        enddo !> inner loop

      enddo !> outer loop

      if ( myrank .EQ. 0 ) then
        call timing_print( .true. )
      endif

      call MPI_Win_free( win1, ier )
      call MPI_Win_free( win2, ier )

      deallocate( buffer1, stat=ier )
      deallocate( buffer2, stat=ier )

      end subroutine timing_fft2d_manual

      subroutine timing_fft2d_mpi_pack_ddt( DIM1, procs, outer_loop, &
        inner_loop, correct_flag, ptypesize, testname, &
        filehandle_debug, local_communicator )

!      use timing, only: timing_init, timing_record, timing_print
      use utilities, only: fill_unique_array_2D_double_complex

      implicit none

      include 'mpif.h'

      integer, intent(in) :: DIM1
      integer, intent(in) :: procs

      integer, intent(in) :: outer_loop, inner_loop

      logical, intent(out) :: correct_flag
      integer, intent(out) :: ptypesize

      character(50), intent(in) :: testname

      integer, intent(in) :: filehandle_debug

      integer, intent(in) :: local_communicator

!> local variables
      double complex, dimension(DIM1,DIM1/procs) :: matrix
      double complex, dimension(:), allocatable :: buffer1, buffer2

      integer :: myrank, commsize, ier
      integer :: i, j, k, base, typesize, bytes
      integer :: pos

!> variables for the datatype construction
      integer :: dtype_vector_t, dtype_resize_t
      integer :: dtype_scatter_t, dtype_gather_t
      integer(kind=MPI_ADDRESS_KIND) :: lb, extent

      character(50) ::  method

      integer :: win1, win2
      integer(kind=MPI_ADDRESS_KIND) :: win_size
      integer(kind=MPI_ADDRESS_KIND) :: target_disp

!> just some statements to prevent compiler warnings of unused variables
!> those parameter are included for future features
      correct_flag = .false.
      ptypesize = 0
      typesize = filehandle_debug

      call MPI_Comm_rank( local_communicator, myrank, ier )
      call MPI_Comm_size( local_communicator, commsize, ier )

!> ================= initialize the arrays =================

      allocate( buffer1(DIM1*DIM1/procs), stat=ier )
      allocate( buffer2(DIM1*DIM1/procs), stat=ier )

! assume that one double complex element has a size of 16 byte
      win_size = DIM1*DIM1/procs*16
      call MPI_Win_create( buffer1(1), win_size, 16, MPI_INFO_NULL, &
        local_communicator, win1, ier )
      call MPI_Win_create( buffer2(1), win_size, 16, MPI_INFO_NULL, &
        local_communicator, win2, ier )

! initial fence to open epoch
      call MPI_Win_fence( 0, win1, ier )
      call MPI_Win_fence( 0, win2, ier )

      base = myrank * DIM1 * DIM1/procs * 2 + 1
      call fill_unique_array_2D_double_complex( matrix, DIM1, &
        DIM1/procs, base )
 
      if ( myrank .EQ. 0 ) then
        write (method,'(A)') "mpi_pack_ddt"

        call MPI_Type_size( MPI_DOUBLE_COMPLEX, typesize, ier )
        bytes = DIM1/procs * DIM1 * typesize

        call timing_init( testname, method, bytes )
      endif
     
      do i=1,outer_loop

        call MPI_Type_size( MPI_DOUBLE_COMPLEX, typesize, ier )
        bytes = DIM1/procs*DIM1/procs * typesize

        call MPI_Type_vector( DIM1/procs, 1, DIM1, MPI_DOUBLE_COMPLEX, &
          dtype_vector_t, ier )
        lb = 0
        extent = typesize
        call MPI_Type_create_resized( dtype_vector_t, lb, extent, &
          dtype_resize_t, ier )
        call MPI_Type_contiguous( DIM1/procs, dtype_resize_t, &
          dtype_scatter_t, ier )
        call MPI_Type_commit( dtype_scatter_t, ier )

        call MPI_Type_free( dtype_vector_t, ier )
        call MPI_Type_free( dtype_resize_t, ier )

        call MPI_Type_vector( DIM1/procs, DIM1/procs, DIM1, &
          MPI_DOUBLE_COMPLEX, dtype_vector_t, ier )
        lb = 0
        extent = DIM1/procs * typesize
        call MPI_Type_create_resized( dtype_vector_t, lb, &
          extent, dtype_gather_t, ier )
        call MPI_Type_commit( dtype_gather_t, ier )

        call MPI_Type_free( dtype_vector_t, ier )

        if ( myrank .EQ. 0 ) then
          call timing_record(1)
        endif

        do j=1,inner_loop
!> pack the data
          pos = 0
          call MPI_Pack( matrix(1,1), 1, dtype_gather_t, &
            buffer1(1), bytes, pos, local_communicator, ier )
          if ( myrank .EQ. 0 ) then
            call timing_record(2)            
          endif

          do k=0,commsize-1
            target_disp = k * DIM1/procs*DIM1/procs
            call MPI_Put( buffer1(target_disp+1), bytes, MPI_PACKED, &
              k, target_disp, bytes, MPI_PACKED, win2, ier )
          enddo
          call MPI_Win_fence( 0, win2, ier )

!> unpack the data
          pos = 0
          call MPI_Unpack( buffer2(1), bytes, pos, matrix(1,1), 1, &
            dtype_scatter_t, local_communicator, ier )

!> pack the data
          pos = 0
          call MPI_Pack( matrix(1,1), 1, dtype_gather_t, &
            buffer2(1), bytes, pos, local_communicator, ier )

          do k=0,commsize-1
            target_disp = k * DIM1/procs*DIM1/procs
            call MPI_Put( buffer2(target_disp+1), bytes, MPI_PACKED, &
              k, target_disp, bytes, MPI_PACKED, win1, ier )
          enddo
          call MPI_Win_fence( 0, win1, ier )

          if ( myrank .EQ. 0 ) then
            call timing_record(3)
          endif

!> unpack the data
          pos = 0
          call MPI_Unpack( buffer1(1), bytes, pos, matrix(1,1), 1, &
            dtype_scatter_t, local_communicator, ier )

          if ( myrank .EQ. 0 ) then
            call timing_record(4)
          endif
        
        enddo !> inner loop

        call MPI_Type_free( dtype_gather_t, ier )
        call MPI_Type_free( dtype_scatter_t, ier )

        if ( myrank .EQ. 0 ) then
          call timing_record(5)
        endif

      enddo !> outer loop

      if ( myrank .EQ. 0 ) then
        call timing_print( .true. )
      endif

      call MPI_Win_free( win1, ier )
      call MPI_Win_free( win2, ier )

      deallocate( buffer1, stat=ier )
      deallocate( buffer2, stat=ier )

      end subroutine timing_fft2d_mpi_pack_ddt

      end module timing_fft2d
