!     Copyright (c) 2012 The Trustees of University of Illinois.
!     All rights reserved. Use of this source code is governed by a
!     BSD-style license that can be found in the LICENSE file.
      
      module timing_nas
!> module contains the timing benchmarks for NAS tests (MG/LU)
      
      private

      public :: timing_nas_lu_x_ddt
      public :: timing_nas_lu_x_manual
      public :: timing_nas_lu_x_mpi_pack_ddt
      public :: timing_nas_lu_y_ddt
      public :: timing_nas_lu_y_manual
      public :: timing_nas_lu_y_mpi_pack_ddt
      public :: timing_nas_mg_x_ddt
      public :: timing_nas_mg_x_manual
      public :: timing_nas_mg_x_mpi_pack_ddt
      public :: timing_nas_mg_y_ddt
      public :: timing_nas_mg_y_manual
      public :: timing_nas_mg_y_mpi_pack_ddt
      public :: timing_nas_mg_z_ddt
      public :: timing_nas_mg_z_manual
      public :: timing_nas_mg_z_mpi_pack_ddt

      contains

      subroutine timing_nas_lu_y_ddt( DIM2, DIM3, outer_loop, &
        inner_loop, correct_flag, ptypesize, testname, &
        filehandle_debug, local_communicator )

!      use timing, only: timing_init, timing_record, timing_print
      use utilities, only: fill_unique_array_3D_double

      implicit none

      include 'mpif.h'

      integer, intent(in) :: DIM2, DIM3
      integer, intent(in) :: inner_loop, outer_loop

      logical, intent(out) :: correct_flag
      integer, intent(out) :: ptypesize

      character(50), intent(in) :: testname

      integer, intent(in) :: filehandle_debug

      integer, intent(in) :: local_communicator

      double precision, dimension(5,DIM2+2,DIM3+2) :: array

      integer :: myrank, ier
      integer :: i, j, base, bytes, typesize

      integer :: dtype_y_t, dtype_temp_t

      character(50) :: method

      integer :: win
      integer(kind=MPI_ADDRESS_KIND) :: win_size
      integer(kind=MPI_ADDRESS_KIND) :: target_disp

      target_disp = 5 * (DIM2+2)

!> just some statements to prevent compiler warnings of unused variables      
!> those parameter are included for future features
      correct_flag = .false.
      ptypesize = 0
      typesize = filehandle_debug
      
      call MPI_Comm_rank( local_communicator, myrank, ier )

! assume that one double precision element has a size of 8 byte
      win_size = 5 * (DIM2+2) * (DIM3+2) * 8
      call MPI_Win_create( array(1,1,1), win_size, 8, MPI_INFO_NULL, &
        local_communicator, win, ier )

! initial fence to open epoch
      call MPI_Win_fence( 0, win, ier )

      base = myrank * 5 * (DIM2+2) * (DIM3+2) + 1
      call fill_unique_array_3D_double( array, 5, (DIM2+2), (DIM3+2), &
        base )

      if ( myrank .EQ. 0 ) then
        write (method,'(A)') "mpi_ddt"

        call MPI_Type_size( MPI_DOUBLE_PRECISION, typesize, ier )
        bytes = 5 * DIM3 * typesize

        call timing_init( testname, method, bytes )
      endif

      do i=1,outer_loop

        call MPI_Type_contiguous(5, MPI_DOUBLE_PRECISION, &
          dtype_temp_t, ier )

        call MPI_Type_vector( DIM3, 1, DIM2+2, dtype_temp_t, &
          dtype_y_t, ier )
        call MPI_Type_commit( dtype_y_t, ier )

        call MPI_Type_free( dtype_temp_t, ier )

        if ( myrank .EQ. 0 ) then
          call timing_record(1)
        endif

        do j=1,inner_loop

          if ( myrank .EQ. 0 ) then
            call MPI_Put( array(1,DIM2+1,2), 1, dtype_y_t, 1, &
              target_disp, 1, dtype_y_t, win, ier )
            call MPI_Win_fence( 0, win, ier )
            call MPI_Win_fence( 0, win, ier )
            call timing_record(3)
          else
            call MPI_Win_fence( 0, win, ier )
            call MPI_Put( array(1,DIM2+1,2), 1, dtype_y_t, 0, &
              target_disp, 1, dtype_y_t, win, ier )
            call MPI_Win_fence( 0, win, ier )
          endif

        enddo

        call MPI_Type_free( dtype_y_t, ier )

        if ( myrank .EQ. 0 ) then
          call timing_record(5)
        endif

      enddo 

      if ( myrank .EQ. 0 ) then
        call timing_print( .true. )
      endif

      call MPI_Win_free( win, ier )

      end subroutine timing_nas_lu_y_ddt

      subroutine timing_nas_lu_y_manual( DIM2, DIM3, outer_loop, &
        inner_loop, correct_flag, ptypesize, testname, &
        filehandle_debug, local_communicator )

!      use timing, only: timing_init, timing_record, timing_print
      use utilities, only: fill_unique_array_3D_double

      implicit none

      include 'mpif.h'

      integer, intent(in) :: DIM2, DIM3
      integer, intent(in) :: inner_loop, outer_loop

      logical, intent(out) :: correct_flag
      integer, intent(out) :: ptypesize

      character(50), intent(in) :: testname

      integer, intent(in) :: filehandle_debug

      integer, intent(in) :: local_communicator

      double precision, dimension(5,DIM2+2,DIM3+2) :: array
      double precision, dimension(:), allocatable :: buffer

      integer :: myrank, ier
      integer :: i, j, k, l,  base, bytes, typesize

      character(50) :: method

      integer :: win
      integer(kind=MPI_ADDRESS_KIND) :: win_size
      integer(kind=MPI_ADDRESS_KIND) :: target_disp = 0

!> just some statements to prevent compiler warnings of unused variables      
!> those parameter are included for future features
      correct_flag = .false.
      ptypesize = 0
      typesize = filehandle_debug

      call MPI_Comm_rank( local_communicator, myrank, ier )

      allocate( buffer(5*DIM3), stat=ier )
! assume that one double precision element has a size of 8 byte
      win_size = 5 * DIM3 * 8
      call MPI_Win_create( buffer(1), win_size, 8, MPI_INFO_NULL, &
        local_communicator, win, ier )

! initial fence to open epoch
      call MPI_Win_fence( 0, win, ier )

      base = myrank * 5 * (DIM2+2) * (DIM3+2) + 1
      call fill_unique_array_3D_double( array, 5, (DIM2+2), (DIM3+2), &
        base )
      
      if ( myrank .EQ. 0 ) then
        write (method,'(A)') "manual"

        call MPI_Type_size( MPI_DOUBLE_PRECISION, typesize, ier )
        bytes = 5 * DIM3 * typesize

        call timing_init( testname, method, bytes )
      endif

      do i=1,outer_loop

        do j=1,inner_loop
          if ( myrank .EQ. 0 ) then
!> pack the data
            base = 1
            do k=2,DIM3+1
              do l=1,5
                buffer(base) = array(l,DIM2+1,k)
                base = base + 1
              enddo
            enddo
            call timing_record(2)
            call MPI_Put( buffer(1), 5*DIM3, MPI_DOUBLE_PRECISION, 1, &
              target_disp, 5*DIM3, MPI_DOUBLE_PRECISION, win, ier )
            call MPI_Win_fence( 0, win, ier )
            call MPI_Win_fence( 0, win, ier )
            call timing_record(3)
!> unpack the data
            base = 1
            do k=2,DIM3+1
              do l=1,5
                array(l,1,k) = buffer(base)
                base = base + 1
              enddo
            enddo
            call timing_record(4)
          else
            call MPI_Win_fence( 0, win, ier )
!> unpack the data
            base = 1
            do k=2,DIM3+1
              do l=1,5
                array(l,1,k) = buffer(base)
                base = base + 1
              enddo
            enddo
!> pack the data
            base = 1
            do k=2,DIM3+1
              do l=1,5
                buffer(base) = array(l,DIM2+1,k)
                base = base + 1
              enddo
            enddo
            call MPI_Put( buffer(1), 5*DIM3, MPI_DOUBLE_PRECISION, 0, &
              target_disp, 5*DIM3, MPI_DOUBLE_PRECISION, win, ier )
            call MPI_Win_fence( 0, win, ier )
          endif

        enddo

      enddo 

      if ( myrank .EQ. 0 ) then
        call timing_print( .true. )
      endif

      call MPI_Win_free( win, ier ) 

      deallocate( buffer, stat=ier )

      end subroutine timing_nas_lu_y_manual

      subroutine timing_nas_lu_y_mpi_pack_ddt( DIM2, DIM3, outer_loop, &
        inner_loop, correct_flag, ptypesize, testname, &
        filehandle_debug, local_communicator )

!      use timing, only: timing_init, timing_record, timing_print
      use utilities, only: fill_unique_array_3D_double

      implicit none

      include 'mpif.h'

      integer, intent(in) :: DIM2, DIM3
      integer, intent(in) :: inner_loop, outer_loop

      logical, intent(out) :: correct_flag
      integer, intent(out) :: ptypesize

      character(50), intent(in) :: testname

      integer, intent(in) :: filehandle_debug

      integer, intent(in) :: local_communicator

      double precision, dimension(5,DIM2+2,DIM3+2) :: array
      double precision, dimension(:), allocatable :: buffer

      integer :: myrank, ier
      integer :: i, j, base, bytes, typesize, pos

      integer :: dtype_y_t, dtype_temp_t

      character(50) :: method

      integer :: win
      integer(kind=MPI_ADDRESS_KIND) :: win_size
      integer(kind=MPI_ADDRESS_KIND) :: target_disp = 0 

!> just some statements to prevent compiler warnings of unused variables      
!> those parameter are included for future features
      correct_flag = .false.
      ptypesize = 0
      typesize = filehandle_debug
     
      call MPI_Comm_rank( local_communicator, myrank, ier )

      allocate( buffer(5*DIM3), stat=ier )
      call MPI_Type_size( MPI_DOUBLE_PRECISION, typesize, ier )
      bytes = 5 * DIM3 * typesize

      win_size = bytes
      call MPI_Win_create( buffer(1), win_size, typesize, MPI_INFO_NULL, &
        local_communicator, win, ier )

! initial fence to open epoch
      call MPI_Win_fence( 0, win, ier )

      base = myrank * 5 * (DIM2+2) * (DIM3+2) + 1
      call fill_unique_array_3D_double( array, 5, (DIM2+2), (DIM3+2), base)

      if ( myrank .EQ. 0 ) then
        write (method,'(A)') "mpi_pack_ddt"

        call timing_init( testname, method, bytes )
      endif

      do i=1,outer_loop

        call MPI_Type_contiguous(5, MPI_DOUBLE_PRECISION, &
          dtype_temp_t, ier )

        call MPI_Type_vector( DIM3, 1, DIM2+2, dtype_temp_t, &
          dtype_y_t, ier )
        call MPI_Type_commit( dtype_y_t, ier )

        call MPI_Type_free( dtype_temp_t, ier )
       
        if ( myrank .EQ. 0 ) then
          call timing_record(1)
        endif

        do j=1,inner_loop
          if ( myrank .EQ. 0 ) then
!> pack the data
            pos = 0
            call MPI_Pack( array(1,DIM2+1,2), 1, dtype_y_t, &
              buffer(1), bytes, pos, local_communicator, ier )
            call timing_record(2)
            call MPI_Put( buffer(1), pos, MPI_PACKED, 1, target_disp, &
              pos, MPI_PACKED, win, ier )
            call MPI_Win_fence( 0, win, ier )
            call MPI_Win_fence( 0, win, ier )
            call timing_record(3)
!> unpack the data
            pos = 0
            call MPI_Unpack( buffer, bytes, pos, array(1,1,2), 1, &
              dtype_y_t, local_communicator, ier )
            call timing_record(4)
          else
            call MPI_Win_fence( 0, win, ier )
!> unpack the data
            pos = 0
            call MPI_Unpack( buffer, bytes, pos, array(1,1,2), 1, &
              dtype_y_t, local_communicator, ier )
!> pack the data
            pos = 0
            call MPI_Pack( array(1,DIM2+1,2), 1, dtype_y_t, &
              buffer(1), bytes, pos, local_communicator, ier )
            call MPI_Put( buffer(1), pos, MPI_PACKED, 0, target_disp, &
              pos, MPI_PACKED, win, ier )
            call MPI_Win_fence( 0, win, ier )
          endif

        enddo

        call MPI_Type_free( dtype_y_t, ier )

        if ( myrank .EQ. 0 ) then
          call timing_record(5)
        endif

      enddo 

      if ( myrank .EQ. 0 ) then
        call timing_print( .true. )
      endif

      call MPI_Win_free( win, ier )      
      
      deallocate( buffer, stat=ier )

      end subroutine timing_nas_lu_y_mpi_pack_ddt

      subroutine timing_nas_lu_x_ddt( DIM2, DIM3, outer_loop, &
        inner_loop, correct_flag, ptypesize, testname, &
        filehandle_debug, local_communicator )

!      use timing, only: timing_init, timing_record, timing_print
      use utilities, only: fill_unique_array_3D_double

      implicit none

      include 'mpif.h'

      integer, intent(in) :: DIM2, DIM3
      integer, intent(in) :: inner_loop, outer_loop

      logical, intent(out) :: correct_flag
      integer, intent(out) :: ptypesize

      character(50), intent(in) :: testname

      integer, intent(in) :: filehandle_debug

      integer, intent(in) :: local_communicator

      double precision, dimension(5,DIM2+2,DIM3+2) :: array

      integer :: myrank, ier
      integer :: i, j, base, bytes, typesize

      integer :: dtype_x_t

      character(50) :: method

      integer :: win
      integer(kind=MPI_ADDRESS_KIND) :: win_size
      integer(kind=MPI_ADDRESS_KIND) :: target_disp = 5

!> just some statements to prevent compiler warnings of unused variables      
!> those parameter are included for future features
      correct_flag = .false.
      ptypesize = 0
      typesize = filehandle_debug

      call MPI_Comm_rank( local_communicator, myrank, ier )

! assume that one double precision element has a size of 8 byte
      win_size = 5 * (DIM2+2) * (DIM3+2) * 8
      call MPI_Win_create( array(1,1,1), win_size, 8, MPI_INFO_NULL, &
        local_communicator, win, ier )

! initial fence to open epoch
      call MPI_Win_fence( 0, win, ier )

      base = myrank * 5 * (DIM2+2) * (DIM3+2) + 1
      call fill_unique_array_3D_double( array, 5, (DIM2+2), (DIM3+2), base )

      if ( myrank .EQ. 0 ) then
        write (method,'(A)') "mpi_ddt"

        call MPI_Type_size( MPI_DOUBLE_PRECISION, typesize, ier )
        bytes = 5 * DIM2 * typesize

        call timing_init( testname, method, bytes )
      endif

      do i=1,outer_loop

        call MPI_Type_contiguous(5*DIM2, MPI_DOUBLE_PRECISION, &
          dtype_x_t, ier )

        call MPI_Type_commit( dtype_x_t, ier )

        if ( myrank .EQ. 0 ) then
          call timing_record(1)
        endif

        do j=1,inner_loop

          if ( myrank .EQ. 0 ) then
            call MPI_Put( array(1,2,DIM3+1), 1, dtype_x_t, 1, &
              target_disp, 1, dtype_x_t, win, ier )
            call MPI_Win_fence( 0, win, ier )
            call MPI_Win_fence( 0, win, ier )
            call timing_record(3)
          else
            call MPI_Win_fence( 0, win, ier )
            call MPI_Put( array(1,2,DIM3+1), 1, dtype_x_t, 0, &
              target_disp, 1, dtype_x_t, win, ier )
            call MPI_Win_fence( 0, win, ier )
          endif

        enddo

        call MPI_Type_free( dtype_x_t, ier )

        if ( myrank .EQ. 0 ) then
          call timing_record(5)
        endif

      enddo 

      if ( myrank .EQ. 0 ) then
        call timing_print( .true. )
      endif

      call MPI_Win_free( win, ier )

      end subroutine timing_nas_lu_x_ddt

      subroutine timing_nas_lu_x_manual( DIM2, DIM3, outer_loop, &
        inner_loop, correct_flag, ptypesize, testname, &
        filehandle_debug, local_communicator )

!      use timing, only: timing_init, timing_record, timing_print
      use utilities, only: fill_unique_array_3D_double

      implicit none

      include 'mpif.h'

      integer, intent(in) :: DIM2, DIM3
      integer, intent(in) :: inner_loop, outer_loop

      logical, intent(out) :: correct_flag
      integer, intent(out) :: ptypesize

      character(50), intent(in) :: testname

      integer, intent(in) :: filehandle_debug

      integer, intent(in) :: local_communicator

      double precision, dimension(5,DIM2+2,DIM3+2) :: array
      double precision, dimension(:), allocatable :: buffer

      integer :: myrank, ier
      integer :: i, j, k, l,  base, bytes, typesize

      character(50) :: method

      integer :: win
      integer(kind=MPI_ADDRESS_KIND) :: win_size
      integer(kind=MPI_ADDRESS_KIND) :: target_disp = 0

!> just some statements to prevent compiler warnings of unused variables      
!> those parameter are included for future features
      correct_flag = .false.
      ptypesize = 0
      typesize = filehandle_debug

      call MPI_Comm_rank( local_communicator, myrank, ier )

      allocate( buffer(5*DIM2), stat=ier )
! assume that one double precision element has a size of 8 byte
      win_size = 5 * DIM2 * 8
      call MPI_Win_create( buffer(1), win_size, 8, MPI_INFO_NULL, &
        local_communicator, win, ier )

! initial fence to open epoch
      call MPI_Win_fence( 0, win, ier )
        
      base = myrank * 5 * (DIM2+2) * (DIM3+2) + 1
      call fill_unique_array_3D_double( array, 5, (DIM2+2), (DIM3+2), base )

      if ( myrank .EQ. 0 ) then
        write (method,'(A)') "manual"          

        call MPI_Type_size( MPI_DOUBLE_PRECISION, typesize, ier )
        bytes = 5 * DIM2 * typesize

        call timing_init( testname, method, bytes )
      endif

      do i=1,outer_loop

        do j=1,inner_loop
          if ( myrank .EQ. 0 ) then
!> pack the data
            base = 1
            do k=2,DIM2+1
              do l=1,5
                buffer(base) = array(l,k,DIM3+1)
                base = base + 1
              enddo
            enddo
            call timing_record(2)
            call MPI_Put( buffer(1), 5*DIM2, MPI_DOUBLE_PRECISION, 1, &
              target_disp, 5*DIM2, MPI_DOUBLE_PRECISION, win, ier )
            call MPI_Win_fence( 0, win, ier )
            call MPI_Win_fence( 0, win, ier )
            call timing_record(3)
!> unpack the data
            base = 1
            do k=2,DIM2+1
              do l=1,5
                array(l,k,1) = buffer(base)
                base = base + 1
              enddo
            enddo
            call timing_record(4)
          else
            call MPI_Win_fence( 0, win, ier )
!> unpack the data
            base = 1
            do k=2,DIM2+1
              do l=1,5
                array(l,k,1) = buffer(base)
                base = base + 1
              enddo
            enddo
!> pack the data
            base = 1
            do k=2,DIM2+1
              do l=1,5
                buffer(base) = array(l,k,DIM3+1)
                base = base + 1
              enddo
            enddo
            call MPI_Put( buffer(1), 5*DIM2, MPI_DOUBLE_PRECISION, 0, &
              target_disp, 5*DIM2, MPI_DOUBLE_PRECISION, win, ier )
            call MPI_Win_fence( 0, win, ier )
          endif

        enddo

      enddo 

      if ( myrank .EQ. 0 ) then
        call timing_print( .true. )
      endif

      call MPI_Win_free( win, ier )

      deallocate( buffer, stat=ier )

      end subroutine timing_nas_lu_x_manual

      subroutine timing_nas_lu_x_mpi_pack_ddt( DIM2, DIM3, outer_loop, &
        inner_loop, correct_flag, ptypesize, testname, &
        filehandle_debug, local_communicator )

!      use timing, only: timing_init, timing_record, timing_print
      use utilities, only: fill_unique_array_3D_double

      implicit none

      include 'mpif.h'

      integer, intent(in) :: DIM2, DIM3
      integer, intent(in) :: inner_loop, outer_loop

      logical, intent(out) :: correct_flag
      integer, intent(out) :: ptypesize

      character(50), intent(in) :: testname

      integer, intent(in) :: filehandle_debug

      integer, intent(in) :: local_communicator

      double precision, dimension(5,DIM2+2,DIM3+2) :: array
      double precision, dimension(:), allocatable :: buffer

      integer :: myrank, ier
      integer :: i, j, base, bytes, typesize, pos

      integer :: dtype_x_t

      character(50) :: method

      integer :: win
      integer(kind=MPI_ADDRESS_KIND) :: win_size
      integer(kind=MPI_ADDRESS_KIND) :: target_disp = 0 

!> just some statements to prevent compiler warnings of unused variables      
!> those parameter are included for future features
      correct_flag = .false.
      ptypesize = 0
      typesize = filehandle_debug
       
      call MPI_Comm_rank( local_communicator, myrank, ier )

      allocate( buffer(5*DIM2), stat=ier )
      call MPI_Type_size( MPI_DOUBLE_PRECISION, typesize, ier )
      bytes = 5 * DIM2 * typesize

      win_size = bytes
      call MPI_Win_create( buffer(1), win_size, typesize, MPI_INFO_NULL, &
        local_communicator, win, ier )

! initial fence to open epoch
      call MPI_Win_fence( 0, win, ier )

      base = myrank * 5 * (DIM2+2) * (DIM3+2) + 1
      call fill_unique_array_3D_double( array, 5, (DIM2+2), (DIM3+2), base )

      if ( myrank .EQ. 0 ) then
        write (method,'(A)') "mpi_pack_ddt"
        
        call timing_init( testname, method, bytes )
      endif

      do i=1,outer_loop

        call MPI_Type_contiguous(5*DIM2, MPI_DOUBLE_PRECISION, &
          dtype_x_t, ier )
        call MPI_Type_commit( dtype_x_t, ier )

        if ( myrank .EQ. 0 ) then
          call timing_record(1)
        endif

        do j=1,inner_loop
          if ( myrank .EQ. 0 ) then
!> pack the data
            pos = 0
            call MPI_Pack( array(1,2,DIM3+1), 1, dtype_x_t, &
              buffer(1), bytes, pos, local_communicator, ier )
            call timing_record(2)
            call MPI_Put( buffer(1), pos, MPI_PACKED, 1, target_disp, &
              pos, MPI_PACKED, win, ier )
            call MPI_Win_fence( 0, win, ier )
            call MPI_Win_fence( 0, win, ier )
            call timing_record(3)
!> unpack the data
            pos = 0
            call MPI_Unpack( buffer, bytes, pos, array(1,2,1), 1, &
              dtype_x_t, local_communicator, ier )
            call timing_record(4)
          else
            call MPI_Win_fence( 0, win, ier )
!> unpack the data
            pos = 0
            call MPI_Unpack( buffer, bytes, pos, array(1,2,1), 1, &
              dtype_x_t, local_communicator, ier )
!> pack the data
            pos = 0
            call MPI_Pack( array(1,2,DIM3+1), 1, dtype_x_t, &
              buffer(1), bytes, pos, local_communicator, ier )
            call MPI_Put( buffer(1), pos, MPI_PACKED, 0, target_disp, &
              pos, MPI_PACKED, win, ier )
            call MPI_Win_fence( 0, win, ier )
          endif

        enddo

        call MPI_Type_free( dtype_x_t, ier )

        if ( myrank .EQ. 0 ) then
          call timing_record(5)
        endif

      enddo 

      if ( myrank .EQ. 0 ) then
        call timing_print( .true. )
      endif

      call MPI_Win_free( win, ier )

      deallocate( buffer, stat=ier )

      end subroutine timing_nas_lu_x_mpi_pack_ddt

      subroutine timing_nas_mg_x_ddt( DIM1, DIM2, DIM3, outer_loop, &
        inner_loop, correct_flag, ptypesize, testname, filehandle_debug, &
        local_communicator )

!      use timing, only: timing_init, timing_record, timing_print
      use utilities, only: fill_unique_array_3D_double

      implicit none

      include 'mpif.h'

      integer, intent(in) :: DIM1, DIM2, DIM3
      integer, intent(in) :: outer_loop, inner_loop

      logical, intent(out) :: correct_flag
      integer, intent(out) :: ptypesize

      character(50), intent(in) :: testname

      integer, intent(in) :: filehandle_debug

      integer, intent(in) :: local_communicator

!> local variables
      double precision, dimension(DIM1,DIM2,DIM3) :: array

      integer :: myrank, ier
      integer :: base, i, j, typesize, bytes

      integer(kind=MPI_ADDRESS_KIND) :: stride
      integer :: dtype_face_x_t, dtype_temp_t

      character(50) :: method

      integer :: win
      integer(kind=MPI_ADDRESS_KIND) :: win_size
      integer(kind=MPI_ADDRESS_KIND) :: target_disp

      target_disp = DIM1*(DIM2+2) - 1

!> just some statements to prevent compiler warnings of unused variables      
!> those parameter are included for future features
      correct_flag = .false.
      ptypesize = 0
      typesize = filehandle_debug

      call MPI_Comm_rank( local_communicator, myrank, ier )

! assume that one double precision element has a size of 8 byte
      win_size = DIM1 * DIM2 * DIM3 * 8
      call MPI_Win_create( array(1,1,1), win_size, 8, MPI_INFO_NULL, &
        local_communicator, win, ier )

! initial fence to open epoch
      call MPI_Win_fence( 0, win, ier )

      base = myrank * DIM1 * DIM2 * DIM3 + 1
      call fill_unique_array_3D_double( array, DIM1, DIM2, DIM3, base )

      if ( myrank .EQ. 0 ) then
         write (method,'(A)') "mpi_ddt"
  
         call MPI_Type_size( MPI_DOUBLE_PRECISION, typesize, ier )
         bytes = (DIM2-2)*(DIM3-2) * typesize

         call timing_init( testname, method, bytes )
      endif

      do i=1,outer_loop
  
        call MPI_Type_vector( DIM2-2, 1, DIM1, MPI_DOUBLE_PRECISION, &
          dtype_temp_t, ier )

        call MPI_Type_size( MPI_DOUBLE_PRECISION, typesize, ier )
        stride = DIM1*DIM2*typesize

        call MPI_Type_create_hvector( DIM3-2, 1, stride, &
          dtype_temp_t, dtype_face_x_t, ier )
        call MPI_Type_commit( dtype_face_x_t, ier )

        call MPI_Type_free( dtype_temp_t, ier )

        if ( myrank .EQ. 0 ) then
          call timing_record(1)
        endif

        do j=1,inner_loop

          if ( myrank .EQ. 0 ) then
            call MPI_Put( array(DIM1-1,2,2), 1, dtype_face_x_t, 1, &
              target_disp, 1, dtype_face_x_t, win, ier )
            call MPI_Win_fence( 0, win, ier )
            call MPI_Win_fence( 0, win, ier )
            call timing_record(3)
          else
            call MPI_Win_fence( 0, win, ier )
            call MPI_Put( array(DIM1-1,2,2), 1, dtype_face_x_t, 0, &
              target_disp, 1, dtype_face_x_t, win, ier )
            call MPI_Win_fence( 0, win, ier )
          endif

        enddo

        call MPI_Type_free( dtype_face_x_t, ier )

        if ( myrank .EQ. 0 ) then
          call timing_record(5)
        endif

      enddo

      if ( myrank .EQ. 0 ) then
        call timing_print( .true. )
      endif

      call MPI_Win_free( win, ier )

      end subroutine timing_nas_mg_x_ddt

      subroutine timing_nas_mg_x_manual( DIM1, DIM2, DIM3, outer_loop, &
        inner_loop, correct_flag, ptypesize, testname, filehandle_debug, &
        local_communicator )

!      use timing, only: timing_init, timing_record, timing_print
      use utilities, only: fill_unique_array_3D_double

      implicit none

      include 'mpif.h'

      integer, intent(in) :: DIM1, DIM2, DIM3
      integer, intent(in) :: outer_loop, inner_loop

      logical, intent(out) :: correct_flag
      integer, intent(out) :: ptypesize

      character(50), intent(in) :: testname

      integer, intent(in) :: filehandle_debug

      integer, intent(in) :: local_communicator

!> local variables
      double precision, dimension(DIM1,DIM2,DIM3) :: array
      double precision, dimension(:), allocatable :: buffer

      integer :: myrank, ier
      integer :: base, i, j, k, l, typesize, bytes, psize

      character(50) :: method

      integer :: win
      integer(kind=MPI_ADDRESS_KIND) :: win_size
      integer(kind=MPI_ADDRESS_KIND) :: target_disp = 0

!> just some statements to prevent compiler warnings of unused variables      
!> those parameter are included for future features
      correct_flag = .false.
      ptypesize = 0
      typesize = filehandle_debug

      call MPI_Comm_rank( local_communicator, myrank, ier )

      psize = (DIM2-2)*(DIM3-2)
      allocate( buffer(psize), stat=ier )

! assume that one double precision element has a size of 8 byte
      win_size = psize * 8
      call MPI_Win_create( buffer(1), win_size, 8, MPI_INFO_NULL, &
        local_communicator, win, ier )

! initial fence to open epoch
      call MPI_Win_fence( 0, win, ier )

      base = myrank * DIM1 * DIM2 * DIM3 + 1
      call fill_unique_array_3D_double( array, DIM1, DIM2, DIM3, base)

      if ( myrank .EQ. 0 ) then
         write (method,'(A)') "manual"

         call MPI_Type_size( MPI_DOUBLE_PRECISION, typesize, ier )
         bytes = (DIM2-2)*(DIM3-2) * typesize

         call timing_init( testname, method, bytes )
      endif

      do i=1,outer_loop

        do j=1,inner_loop

          if ( myrank .EQ. 0 ) then
            base =1
            do k=2,DIM3-1
              do l=2,DIM2-1
                buffer(base) = array(DIM1-1,l,k)
                base = base + 1
              enddo
            enddo
            call timing_record(2)
            call MPI_Put( buffer(1), psize, MPI_DOUBLE_PRECISION, 1, &
               target_disp, psize, MPI_DOUBLE_PRECISION, win, ier )
            call MPI_Win_fence( 0, win, ier )
            call MPI_Win_fence( 0, win, ier )
            call timing_record(3)
            base = 1
            do k=2,DIM3-1
              do l=2,DIM2-1
                array(DIM1,l,k) = buffer(base)
                base = base + 1
              enddo
            enddo
            call timing_record(4)
          else
            call MPI_Win_fence( 0, win, ier )
            base = 1
            do k=2,DIM3-1
              do l=2,DIM2-1
                array(DIM1,l,k) = buffer(base)
                base = base + 1
              enddo
            enddo
            base =1
            do k=2,DIM3-1
              do l=2,DIM2-1
                buffer(base) = array(DIM1-1,l,k)
                base = base + 1
              enddo
            enddo
            call MPI_Put( buffer(1), psize, MPI_DOUBLE_PRECISION, 0, &
              target_disp, psize, MPI_DOUBLE_PRECISION, win, ier )
            call MPI_Win_fence( 0, win, ier )
          endif

        enddo

      enddo

      if ( myrank .EQ. 0 ) then
        call timing_print( .true. )
      endif

      call MPI_Win_free( win, ier )

      deallocate( buffer, stat=ier )

      end subroutine timing_nas_mg_x_manual
      
      subroutine timing_nas_mg_x_mpi_pack_ddt( DIM1, DIM2, DIM3, &
        outer_loop, inner_loop, correct_flag, ptypesize, testname, &
        filehandle_debug, local_communicator )

!      use timing, only: timing_init, timing_record, timing_print
      use utilities, only: fill_unique_array_3D_double

      implicit none

      include 'mpif.h'

      integer, intent(in) :: DIM1, DIM2, DIM3
      integer, intent(in) :: outer_loop, inner_loop

      logical, intent(out) :: correct_flag
      integer, intent(out) :: ptypesize

      character(50), intent(in) :: testname

      integer, intent(in) :: filehandle_debug

      integer, intent(in) :: local_communicator

!> local variables
      double precision, dimension(DIM1,DIM2,DIM3) :: array
      double precision, dimension(:), allocatable :: buffer

      integer :: myrank, ier
      integer :: base, i, j, typesize, bytes, pos

      integer(kind=MPI_ADDRESS_KIND) :: stride
      integer :: dtype_face_x_t, dtype_temp_t

      character(50) :: method

      integer :: win
      integer(kind=MPI_ADDRESS_KIND) :: win_size
      integer(kind=MPI_ADDRESS_KIND) :: target_disp = 0

!> just some statements to prevent compiler warnings of unused variables      
!> those parameter are included for future features
      correct_flag = .false.
      ptypesize = 0
      typesize = filehandle_debug

      call MPI_Comm_rank( local_communicator, myrank, ier )

      allocate( buffer((DIM2-2)*(DIM3-2)), stat=ier )
      call MPI_Type_size( MPI_DOUBLE_PRECISION, typesize, ier )
      bytes = (DIM2-2)*(DIM3-2) * typesize

      win_size = bytes
      call MPI_Win_create( buffer(1), win_size, typesize, MPI_INFO_NULL, &
        local_communicator, win, ier )

! initial fence to open epoch
      call MPI_Win_fence( 0, win, ier )
  
      base = myrank * DIM1 * DIM2 * DIM3 + 1
      call fill_unique_array_3D_double( array, DIM1, DIM2, DIM3, base)

      if ( myrank .EQ. 0 ) then
         write (method,'(A)') "mpi_pack_ddt"

         call timing_init( testname, method, bytes )
      endif

      do i=1,outer_loop

        call MPI_Type_vector( DIM2-2, 1, DIM1, MPI_DOUBLE_PRECISION, &
          dtype_temp_t, ier )

        call MPI_Type_size( MPI_DOUBLE_PRECISION, typesize, ier )
        stride = DIM1*DIM2*typesize

        call MPI_Type_create_hvector( DIM3-2, 1, stride, &
          dtype_temp_t, dtype_face_x_t, ier )
        call MPI_Type_commit( dtype_face_x_t, ier )

        call MPI_Type_free( dtype_temp_t, ier )

        if ( myrank .EQ. 0 ) then
          call timing_record(1)
        endif

        do j=1,inner_loop

          if ( myrank .EQ. 0 ) then
            pos = 0
            call MPI_Pack( array(DIM1-1,2,2), 1, dtype_face_x_t, &
              buffer, bytes, pos, local_communicator, ier )
            call timing_record(2)
            call MPI_Put( buffer(1), pos, MPI_PACKED, 1, target_disp, &
              pos, MPI_PACKED, win, ier )
            call MPI_Win_fence( 0, win, ier )
            call MPI_Win_fence( 0, win, ier ) 
            call timing_record(3)
            pos = 0
            call MPI_Unpack( buffer, bytes, pos, array(DIM1,2,2), 1, &
              dtype_face_x_t, local_communicator, ier )
            call timing_record(4)
          else
            call MPI_Win_fence( 0, win, ier )
            pos = 0
            call MPI_Unpack( buffer, bytes, pos, array(DIM1,2,2), 1, &
              dtype_face_x_t, local_communicator, ier )
            pos = 0
            call MPI_Pack( array(DIM1-1,2,2), 1, dtype_face_x_t, &
              buffer, bytes, pos, local_communicator, ier )
            call MPI_Put( buffer(1), pos, MPI_PACKED, 0, target_disp, &
              pos, MPI_PACKED, win, ier )
            call MPI_Win_fence( 0, win, ier )
          endif

        enddo

        call MPI_Type_free( dtype_face_x_t, ier )

        if ( myrank .EQ. 0 ) then
          call timing_record(5)
        endif

      enddo

      if ( myrank .EQ. 0 ) then
        call timing_print( .true. )
      endif

      call MPI_Win_free( win, ier )
      
      deallocate( buffer, stat=ier )

      end subroutine timing_nas_mg_x_mpi_pack_ddt

      subroutine timing_nas_mg_y_ddt( DIM1, DIM2, DIM3, outer_loop, &
        inner_loop, correct_flag, ptypesize, testname, &
        filehandle_debug, local_communicator )

!      use timing, only: timing_init, timing_record, timing_print
      use utilities, only: fill_unique_array_3D_double

      implicit none

      include 'mpif.h'

      integer, intent(in) :: DIM1, DIM2, DIM3
      integer, intent(in) :: outer_loop, inner_loop

      logical, intent(out) :: correct_flag
      integer, intent(out) :: ptypesize

      character(50), intent(in) :: testname

      integer, intent(in) :: filehandle_debug

      integer, intent(in) :: local_communicator

!> local variables
      double precision, dimension(DIM1,DIM2,DIM3) :: array

      integer :: myrank, ier
      integer :: base, i, j, typesize, bytes

      integer :: dtype_face_y_t

      character(50) :: method

      integer :: win
      integer(kind=MPI_ADDRESS_KIND) :: win_size
      integer(kind=MPI_ADDRESS_KIND) :: target_disp

      target_disp = DIM1*(2*DIM2-1)+1

!> just some statements to prevent compiler warnings of unused variables      
!> those parameter are included for future features
      correct_flag = .false.
      ptypesize = 0
      typesize = filehandle_debug

      call MPI_Comm_rank( local_communicator, myrank, ier )

! assume that one double precision element has a size of 8 byte
      win_size = DIM1 * DIM2 * DIM3 * 8
      call MPI_Win_create( array(1,1,1), win_size, 8, MPI_INFO_NULL, &
        local_communicator, win, ier )

! initial fence to open epoch
      call MPI_Win_fence( 0, win, ier )

      base = myrank * DIM1 * DIM2 * DIM3 + 1
      call fill_unique_array_3D_double( array, DIM1, DIM2, DIM3, base)

      if ( myrank .EQ. 0 ) then
        write (method,'(A)') "mpi_ddt"

        call MPI_Type_size( MPI_DOUBLE_PRECISION, typesize, ier )
        bytes = (DIM1-2)*(DIM3-2) * typesize

        call timing_init( testname, method, bytes )
      endif

      do i=1,outer_loop
  
        call MPI_Type_vector( DIM3-2, DIM1-2, DIM1*DIM2, &
          MPI_DOUBLE_PRECISION, dtype_face_y_t, ier )

        call MPI_Type_commit( dtype_face_y_t, ier )

        if ( myrank .EQ. 0 ) then
          call timing_record(1)
        endif

        do j=1,inner_loop

          if ( myrank .EQ. 0 ) then
            call MPI_Put( array(2,DIM2-1,2), 1, dtype_face_y_t, 1, &
              target_disp, 1, dtype_face_y_t, win, ier )
            call MPI_Win_fence( 0, win, ier )
            call MPI_Win_fence( 0, win, ier )
            call timing_record(3)
          else
            call MPI_Win_fence( 0, win, ier )
            call MPI_Put( array(2,DIM2-1,2), 1, dtype_face_y_t, 0, &
              target_disp, 1, dtype_face_y_t, win, ier )
            call MPI_Win_fence( 0, win, ier )
          endif

        enddo

        call MPI_Type_free( dtype_face_y_t, ier )

        if ( myrank .EQ. 0 ) then
          call timing_record(5)
        endif

      enddo

      if ( myrank .EQ. 0 ) then
        call timing_print( .true. )
      endif

      call MPI_Win_free( win, ier )

      end subroutine timing_nas_mg_y_ddt

      subroutine timing_nas_mg_y_manual( DIM1, DIM2, DIM3, &
        outer_loop, inner_loop, correct_flag, ptypesize, testname, &
        filehandle_debug, local_communicator )

!      use timing, only: timing_init, timing_record, timing_print
      use utilities, only: fill_unique_array_3D_double

      implicit none

      include 'mpif.h'

      integer, intent(in) :: DIM1, DIM2, DIM3
      integer, intent(in) :: outer_loop, inner_loop

      logical, intent(out) :: correct_flag
      integer, intent(out) :: ptypesize

      character(50), intent(in) :: testname

      integer, intent(in) :: filehandle_debug

      integer, intent(in) :: local_communicator

!> local variables
      double precision, dimension(DIM1,DIM2,DIM3) :: array
      double precision, dimension(:), allocatable :: buffer

      integer :: myrank, ier
      integer :: base, i, j, k, l, typesize, bytes, psize

      character(50) :: method

      integer :: win
      integer(kind=MPI_ADDRESS_KIND) :: win_size
      integer(kind=MPI_ADDRESS_KIND) :: target_disp = 0

!> just some statements to prevent compiler warnings of unused variables      
!> those parameter are included for future features
      correct_flag = .false.
      ptypesize = 0
      typesize = filehandle_debug

      call MPI_Comm_rank( local_communicator, myrank, ier )

      psize = (DIM1-2)*(DIM3-2)
      allocate( buffer(psize), stat=ier )
      
! assume that one double precision element has a size of 8 byte
      win_size = psize * 8
      call MPI_Win_create( buffer(1), win_size, 8, MPI_INFO_NULL, &
        local_communicator, win, ier )

! initial fence to open epoch
      call MPI_Win_fence( 0, win, ier )

      base = myrank * DIM1 * DIM2 * DIM3 + 1
      call fill_unique_array_3D_double( array, DIM1, DIM2, DIM3, base)

      if ( myrank .EQ. 0 ) then
        write (method,'(A)') "manual"

        call MPI_Type_size( MPI_DOUBLE_PRECISION, typesize, ier )
        bytes = (DIM1-2)*(DIM3-2) * typesize

        call timing_init( testname, method, bytes )
      endif

      do i=1,outer_loop

        do j=1,inner_loop

          if ( myrank .EQ. 0 ) then
            base =1
            do k=2,DIM3-1
              do l=2,DIM1-1
                buffer(base) = array(l,DIM2-1,k)
                base = base + 1
              enddo
            enddo
            call timing_record(2)
            call MPI_Put( buffer(1), psize, MPI_DOUBLE_PRECISION, 1, &
              target_disp, psize, MPI_DOUBLE_PRECISION, win, ier )
            call MPI_Win_fence( 0, win, ier )
            call MPI_Win_fence( 0, win, ier )
            call timing_record(3)
            base = 1
            do k=2,DIM3-1
              do l=2,DIM1-1
                array(l,DIM2,k) = buffer(base)
                base = base + 1
              enddo
            enddo
            call timing_record(4)
          else
            call MPI_Win_fence( 0, win, ier )
            base = 1
            do k=2,DIM3-1
              do l=2,DIM1-1
                array(l,DIM2,k) = buffer(base)
                base = base + 1
              enddo
            enddo
            base =1
            do k=2,DIM3-1
              do l=2,DIM1-1
                buffer(base) = array(l,DIM2-1,k)
                base = base + 1
              enddo
            enddo
            call MPI_Put( buffer(1), psize, MPI_DOUBLE_PRECISION, 0, &
              target_disp, psize, MPI_DOUBLE_PRECISION, win, ier )
            call MPI_Win_fence( 0, win, ier )
          endif

        enddo

      enddo

      if ( myrank .EQ. 0 ) then
        call timing_print( .true. )
      endif

      call MPI_Win_free( win, ier )
      
      deallocate( buffer, stat=ier )

      end subroutine timing_nas_mg_y_manual
      
      subroutine timing_nas_mg_y_mpi_pack_ddt( DIM1, DIM2, DIM3, &
        outer_loop, inner_loop, correct_flag, ptypesize, testname, &
        filehandle_debug, local_communicator )

!      use timing, only: timing_init, timing_record, timing_print
      use utilities, only: fill_unique_array_3D_double

      implicit none

      include 'mpif.h'

      integer, intent(in) :: DIM1, DIM2, DIM3
      integer, intent(in) :: outer_loop, inner_loop

      logical, intent(out) :: correct_flag
      integer, intent(out) :: ptypesize

      character(50), intent(in) :: testname

      integer, intent(in) :: filehandle_debug

      integer, intent(in) :: local_communicator

!> local variables
      double precision, dimension(DIM1,DIM2,DIM3) :: array
      double precision, dimension(:), allocatable :: buffer

      integer :: myrank, ier
      integer :: base, i, j, typesize, bytes, pos

      integer :: dtype_face_y_t

      character(50) :: method

      integer :: win
      integer(kind=MPI_ADDRESS_KIND) :: win_size
      integer(kind=MPI_ADDRESS_KIND) :: target_disp = 0

!> just some statements to prevent compiler warnings of unused variables      
!> those parameter are included for future features
      correct_flag = .false.
      ptypesize = 0
      typesize = filehandle_debug

      call MPI_Comm_rank( local_communicator, myrank, ier )

      allocate( buffer((DIM1-2)*(DIM3-2)), stat=ier )
      call MPI_Type_size( MPI_DOUBLE_PRECISION, typesize, ier )
      bytes = (DIM1-2)*(DIM3-2) * typesize
  
      win_size = bytes
      call MPI_Win_create( buffer(1), win_size, typesize, MPI_INFO_NULL, &
        local_communicator, win, ier )

! initial fence to open epoch
      call MPI_Win_fence( 0, win, ier )

      base = myrank * DIM1 * DIM2 * DIM3 + 1
      call fill_unique_array_3D_double( array, DIM1, DIM2, DIM3, base)

      if ( myrank .EQ. 0 ) then
        write (method,'(A)') "mpi_pack_ddt"

        call timing_init( testname, method, bytes )
      endif

      do i=1,outer_loop

        call MPI_Type_vector( DIM3-2, DIM1-2, DIM1*DIM2, &
          MPI_DOUBLE_PRECISION, dtype_face_y_t, ier )
        call MPI_Type_commit( dtype_face_y_t, ier )

        if ( myrank .EQ. 0 ) then
          call timing_record(1)
        endif

        do j=1,inner_loop

          if ( myrank .EQ. 0 ) then
            pos = 0
            call MPI_Pack( array(2,DIM2-1,2), 1, dtype_face_y_t, &
              buffer, bytes, pos, local_communicator, ier )
            call timing_record(2)
            call MPI_Put( buffer(1), pos, MPI_PACKED, 1, target_disp, &
              pos, MPI_PACKED, win, ier )
            call MPI_Win_fence( 0, win, ier )
            call MPI_Win_fence( 0, win, ier )
            call timing_record(3)
            pos = 0
            call MPI_Unpack( buffer, bytes, pos, array(2,DIM2,2), 1, &
              dtype_face_y_t, local_communicator, ier )
            call timing_record(4)
          else
            call MPI_Win_fence( 0, win, ier )
            pos = 0
            call MPI_Unpack( buffer, bytes, pos, array(2,DIM2,2), 1, &
              dtype_face_y_t, local_communicator, ier )
            pos = 0
            call MPI_Pack( array(2,DIM2-1,2), 1, dtype_face_y_t, &
              buffer, bytes, pos, local_communicator, ier )
            call MPI_Put( buffer(1), pos, MPI_PACKED, 0, target_disp, &
              pos, MPI_PACKED, win, ier )
            call MPI_Win_fence( 0, win, ier )
          endif

        enddo

        call MPI_Type_free( dtype_face_y_t, ier )

        if ( myrank .EQ. 0 ) then
          call timing_record(5)
        endif

      enddo

      if ( myrank .EQ. 0 ) then
        call timing_print( .true. )
      endif

      call MPI_Win_free( win, ier )

      deallocate( buffer, stat=ier )

      end subroutine timing_nas_mg_y_mpi_pack_ddt

      subroutine timing_nas_mg_z_ddt( DIM1, DIM2, DIM3, outer_loop, &
        inner_loop, correct_flag, ptypesize, testname, filehandle_debug, &
        local_communicator )

!      use timing, only: timing_init, timing_record, timing_print
      use utilities, only: fill_unique_array_3D_double

      implicit none

      include 'mpif.h'

      integer, intent(in) :: DIM1, DIM2, DIM3
      integer, intent(in) :: outer_loop, inner_loop

      logical, intent(out) :: correct_flag
      integer, intent(out) :: ptypesize

      character(50), intent(in) :: testname

      integer, intent(in) :: filehandle_debug

      integer, intent(in) :: local_communicator

!> local variables
      double precision, dimension(DIM1,DIM2,DIM3) :: array

      integer :: myrank, ier
      integer :: base, i, j, typesize, bytes

      integer :: dtype_face_z_t

      character(50) :: method

      integer :: win
      integer(kind=MPI_ADDRESS_KIND) :: win_size
      integer(kind=MPI_ADDRESS_KIND) :: target_disp

      target_disp = DIM1+1

!> just some statements to prevent compiler warnings of unused variables      
!> those parameter are included for future features
      correct_flag = .false.
      ptypesize = 0
      typesize = filehandle_debug

      call MPI_Comm_rank( local_communicator, myrank, ier )

! assume that one double precision element has a size of 8 byte
      win_size = DIM1 * DIM2 * DIM3 * 8
      call MPI_Win_create( array(1,1,1), win_size, 8, MPI_INFO_NULL, &
        local_communicator, win, ier )

! initial fence to open epoch
      call MPI_Win_fence( 0, win, ier )

      base = myrank * DIM1 * DIM2 * DIM3 + 1
      call fill_unique_array_3D_double( array, DIM1, DIM2, DIM3, base)

      if ( myrank .EQ. 0 ) then
        write (method,'(A)') "mpi_ddt"

        call MPI_Type_size( MPI_DOUBLE_PRECISION, typesize, ier )
        bytes = (DIM1-2)*(DIM2-2) * typesize

        call timing_init( testname, method, bytes )
      endif

      do i=1,outer_loop
  
        call MPI_Type_vector( DIM2-2, DIM1-2, DIM1, &
          MPI_DOUBLE_PRECISION, dtype_face_z_t, ier )
        call MPI_Type_commit( dtype_face_z_t, ier )

        if ( myrank .EQ. 0 ) then
          call timing_record(1)
        endif

        do j=1,inner_loop

          if ( myrank .EQ. 0 ) then
            call MPI_Put( array(2,2,2), 1, dtype_face_z_t, 1, &
              target_disp, 1, dtype_face_z_t, win, ier )
            call MPI_Win_fence( 0, win, ier )
            call MPI_Win_fence( 0, win, ier )
            call timing_record(3)
          else
            call MPI_Win_fence( 0, win, ier )
            call MPI_Put( array(2,2,2), 1, dtype_face_z_t, 0, &
              target_disp, 1, dtype_face_z_t, win, ier )
            call MPI_Win_fence( 0, win, ier )
          endif

        enddo

        call MPI_Type_free( dtype_face_z_t, ier )

        if ( myrank .EQ. 0 ) then
          call timing_record(5)
        endif

      enddo

      if ( myrank .EQ. 0 ) then
        call timing_print( .true. )
      endif

      call MPI_Win_free( win, ier )

      end subroutine timing_nas_mg_z_ddt

      subroutine timing_nas_mg_z_manual( DIM1, DIM2, DIM3, &
        outer_loop, inner_loop, correct_flag, ptypesize, testname, &
        filehandle_debug, local_communicator )

!      use timing, only: timing_init, timing_record, timing_print
      use utilities, only: fill_unique_array_3D_double

      implicit none

      include 'mpif.h'

      integer, intent(in) :: DIM1, DIM2, DIM3
      integer, intent(in) :: outer_loop, inner_loop

      logical, intent(out) :: correct_flag
      integer, intent(out) :: ptypesize

      character(50), intent(in) :: testname

      integer, intent(in) :: filehandle_debug

      integer, intent(in) :: local_communicator

!> local variables
      double precision, dimension(DIM1,DIM2,DIM3) :: array
      double precision, dimension(:), allocatable :: buffer

      integer :: myrank, ier
      integer :: base, i, j, k, l, typesize, bytes, psize

      character(50) :: method

      integer :: win
      integer(kind=MPI_ADDRESS_KIND) :: win_size
      integer(kind=MPI_ADDRESS_KIND) :: target_disp = 0

!> just some statements to prevent compiler warnings of unused variables      
!> those parameter are included for future features
      correct_flag = .false.
      ptypesize = 0
      typesize = filehandle_debug

      call MPI_Comm_rank( local_communicator, myrank, ier )

      psize = (DIM1-2)*(DIM2-2)
      allocate( buffer(psize), stat=ier )

! assume that one double precision element has a size of 8 byte
      win_size = psize * 8
      call MPI_Win_create( buffer(1), win_size, 8, MPI_INFO_NULL, &
        local_communicator, win, ier )

! initial fence to open epoch
      call MPI_Win_fence( 0, win, ier )

      base = myrank * DIM1 * DIM2 * DIM3 + 1
      call fill_unique_array_3D_double( array, DIM1, DIM2, DIM3, base )

      if ( myrank .EQ. 0 ) then
        write (method,'(A)') "manual"

        call MPI_Type_size( MPI_DOUBLE_PRECISION, typesize, ier )
        bytes = (DIM1-2)*(DIM2-2) * typesize
         
        call timing_init( testname, method, bytes )
      endif

      do i=1,outer_loop

        do j=1,inner_loop

          if ( myrank .EQ. 0 ) then
            base =1
            do k=2,DIM2-1
              do l=2,DIM1-1
                buffer(base) = array(l,k,2)
                base = base + 1
              enddo
            enddo
            call timing_record(2)
            call MPI_Put( buffer(1), psize, MPI_DOUBLE_PRECISION, 1, &
              target_disp, psize, MPI_DOUBLE_PRECISION, win, ier )
            call MPI_Win_fence( 0, win, ier )
            call MPI_Win_fence( 0, win, ier )
            call timing_record(3)
            base = 1
            do k=2,DIM2-1
              do l=2,DIM1-1
                array(l,k,1) = buffer(base)
                base = base + 1
              enddo
            enddo
            call timing_record(4)
          else
            call MPI_Win_fence( 0, win, ier )
            base = 1
            do k=2,DIM2-1
              do l=2,DIM1-1
                array(l,k,1) = buffer(base)
                base = base + 1
              enddo
            enddo
            base =1
            do k=2,DIM2-1
              do l=2,DIM1-1
                buffer(base) = array(l,k,2)
                base = base + 1
              enddo
            enddo
            call MPI_Put( buffer(1), psize, MPI_DOUBLE_PRECISION, 0, &
              target_disp, psize, MPI_DOUBLE_PRECISION, win, ier )
            call MPI_Win_fence( 0, win, ier )
          endif

        enddo

      enddo

      if ( myrank .EQ. 0 ) then
        call timing_print( .true. )
      endif

      call MPI_Win_free( win, ier )

      deallocate( buffer, stat=ier )

      end subroutine timing_nas_mg_z_manual
      
      subroutine timing_nas_mg_z_mpi_pack_ddt( DIM1, DIM2, DIM3, &
        outer_loop, inner_loop, correct_flag, ptypesize, testname, &
        filehandle_debug, local_communicator )

!      use timing, only: timing_init, timing_record, timing_print
      use utilities, only: fill_unique_array_3D_double

      implicit none

      include 'mpif.h'

      integer, intent(in) :: DIM1, DIM2, DIM3
      integer, intent(in) :: outer_loop, inner_loop

      logical, intent(out) :: correct_flag
      integer, intent(out) :: ptypesize

      character(50), intent(in) :: testname

      integer, intent(in) :: filehandle_debug

      integer, intent(in) :: local_communicator

!> local variables
      double precision, dimension(DIM1,DIM2,DIM3) :: array
      double precision, dimension(:), allocatable :: buffer

      integer :: myrank, ier
      integer :: base, i, j, typesize, bytes, pos

      integer :: dtype_face_z_t

      character(50) :: method

      integer :: win
      integer(kind=MPI_ADDRESS_KIND) :: win_size
      integer(kind=MPI_ADDRESS_KIND) :: target_disp = 0

!> just some statements to prevent compiler warnings of unused variables      
!> those parameter are included for future features
      correct_flag = .false.
      ptypesize = 0
      typesize = filehandle_debug

      call MPI_Comm_rank( local_communicator, myrank, ier )

      allocate( buffer((DIM1-2)*(DIM2-2)), stat=ier )
      call MPI_Type_size( MPI_DOUBLE_PRECISION, typesize, ier )
      bytes = (DIM1-2)*(DIM2-2) * typesize

      win_size = bytes
      call MPI_Win_create( buffer(1), win_size, typesize, MPI_INFO_NULL, &
        local_communicator, win, ier )

! initial fence to open epoch
      call MPI_Win_fence( 0, win, ier )

      base = myrank * DIM1 * DIM2 * DIM3 + 1
      call fill_unique_array_3D_double( array, DIM1, DIM2, DIM3, base )

      if ( myrank .EQ. 0 ) then
        write (method,'(A)') "mpi_pack_ddt"

        call timing_init( testname, method, bytes )
      endif

      do i=1,outer_loop

        call MPI_Type_vector( DIM2-2, DIM1-2, DIM1, &
          MPI_DOUBLE_PRECISION, dtype_face_z_t, ier )
        call MPI_Type_commit( dtype_face_z_t, ier )

        if ( myrank .EQ. 0 ) then
          call timing_record(1)
        endif

        do j=1,inner_loop

          if ( myrank .EQ. 0 ) then
            pos = 0
            call MPI_Pack( array(2,2,2), 1, dtype_face_z_t, &
              buffer, bytes, pos, local_communicator, ier )
            call timing_record(2)
            call MPI_Put( buffer(1), pos, MPI_PACKED, 1, target_disp, &
              pos, MPI_PACKED, win, ier )
            call MPI_Win_fence( 0, win, ier )
            call MPI_Win_fence( 0, win, ier ) 
            call timing_record(3)
            pos = 0
            call MPI_Unpack( buffer, bytes, pos, array(2,2,1), 1, &
              dtype_face_z_t, local_communicator, ier )
            call timing_record(4)
          else
            call MPI_Win_fence( 0, win, ier )
            pos = 0
            call MPI_Unpack( buffer, bytes, pos, array(2,2,1), 1, &
              dtype_face_z_t, local_communicator, ier )
            pos = 0
            call MPI_Pack( array(2,2,2), 1, dtype_face_z_t, &
              buffer, bytes, pos, local_communicator, ier )
            call MPI_Put( buffer(1), pos, MPI_PACKED, 0, target_disp, &
              pos, MPI_PACKED, win, ier )
            call MPI_Win_fence( 0, win, ier )
          endif

        enddo

        call MPI_Type_free( dtype_face_z_t, ier )

        if ( myrank .EQ. 0 ) then
          call timing_record(5)
        endif

      enddo

      if ( myrank .EQ. 0 ) then
        call timing_print( .true. )
      endif

      deallocate( buffer, stat=ier )

      call MPI_Win_free( win, ier )

      end subroutine timing_nas_mg_z_mpi_pack_ddt

      end module timing_nas
