!     Copyright (c) 2012 The Trustees of University of Illinois. All 
!     rights reserved. Use of this source code is governed by a 
!     BSD-style license that can be found in the LICENSE file.
      
      module timing_basic

      private 

      public :: time_ping_pong_nelements
      public :: time_alltoall_nelements

      contains

      subroutine time_ping_pong_nelements(DIM1, loop, testname, &
        local_communicator)

!      use timing, only: timing_init, timing_record, timing_print
      use utilities, only: fill_unique_array_1D_real

      implicit none

      include 'mpif.h'

      integer, intent(in) :: DIM1
      integer, intent(in) :: loop

      character(50), intent(in) :: testname

      integer, intent(in) :: local_communicator

! local variables
      real, dimension(DIM1) :: array
      integer :: myrank, ier
      integer :: base, typesize, bytes, i
      character(50) :: method

      integer :: win
      integer(kind=MPI_ADDRESS_KIND) :: win_size
      integer(kind=MPI_ADDRESS_KIND) :: target_disp = 0

      call MPI_Comm_rank( local_communicator, myrank, ier )

! assume that one real element has a size of 4 byte
      win_size = DIM1 * 4
      call MPI_Win_create( array(1), win_size, 4, MPI_INFO_NULL, &
        local_communicator, win, ier )

! initial fence to open epoch
      call MPI_Win_fence( 0, win, ier )

      base = myrank * DIM1 + 1
      call fill_unique_array_1D_real( array, DIM1, base )

      if ( myrank .EQ. 0 ) then
        write (method,'(A)') "reference"

        call MPI_Type_size(MPI_REAL, typesize, ier )
        bytes = typesize * DIM1

        call timing_init( testname, method, bytes )
      endif

      do i=1,loop
        if ( myrank .EQ. 0 ) then
          call MPI_Put( array(1), DIM1, MPI_REAL, 1, target_disp, DIM1, &
            MPI_REAL, win, ier )
          call MPI_Win_fence( 0, win, ier )
          call MPI_Win_fence( 0, win, ier )
          call timing_record(3)
        else
          call MPI_Win_fence( 0, win, ier )
          call MPI_Put( array(1), DIM1, MPI_REAL, 0, target_disp, DIM1, &
            MPI_REAL, win, ier )
          call MPI_Win_fence( 0, win, ier )
        endif
      enddo

      if ( myrank .EQ. 0 ) then
        call timing_print(.true.)
      endif

      call MPI_Win_free( win, ier )
     
      end subroutine time_ping_pong_nelements

      subroutine time_alltoall_nelements(DIM1, procs, loop, testname, &
        local_communicator)

!      use timing, only: timing_init, timing_record, timing_print
      use utilities, only: fill_unique_array_1D_real

      implicit none

      include 'mpif.h'

      integer, intent(in) :: DIM1
      integer, intent(in) :: procs
      integer, intent(in) :: loop

      character(50), intent(in) :: testname

      integer, intent(in) :: local_communicator

! local variables
      real, dimension(DIM1*procs) :: send_array, recv_array
      integer :: myrank, commsize, ier
      integer :: base, typesize, bytes, i, j
      character(50) :: method

      integer :: win1, win2
      integer(kind=MPI_ADDRESS_KIND) :: win_size
      integer(kind=MPI_ADDRESS_KIND) :: target_disp = 0

      call MPI_Comm_rank( local_communicator, myrank, ier )
      call MPI_Comm_size( local_communicator, commsize, ier )

! assume that one real element has a size of 4 byte
      win_size = DIM1 * procs * 4 
      call MPI_Win_create( send_array(1), win_size, 4, MPI_INFO_NULL, &
        local_communicator, win1, ier )
      call MPI_Win_create( recv_array(1), win_size, 4, MPI_INFO_NULL, &
        local_communicator, win2, ier )

! initial fence to open epoch
      call MPI_Win_fence( 0, win1, ier )
      call MPI_Win_fence( 0, win2, ier )

      base = myrank * DIM1 + 1
      call fill_unique_array_1D_real( send_array, DIM1, base )

      if ( myrank .EQ. 0 ) then
        write (method,'(A)') "reference"
        
        call MPI_Type_size(MPI_REAL, typesize, ier )
        bytes = typesize * DIM1 * procs

        call timing_init( testname, method, bytes )
      endif

      do i=1,loop
        target_disp = myrank * DIM1
        do j=0,commsize-1
          call MPI_Put( send_array(j*DIM1+1), DIM1, MPI_REAL, j, target_disp, &
            DIM1, MPI_REAL, win2, ier )
        enddo
        call MPI_Win_fence( 0, win2, ier )
        
        do j=0,commsize-1
          call MPI_Put( recv_array(j*DIM1+1), DIM1, MPI_REAL, j, target_disp, &
            DIM1, MPI_REAL, win1, ier )
        enddo
        call MPI_Win_fence( 0, win1, ier )

        if ( myrank .EQ. 0 ) then
          call timing_record(3)
        endif
      enddo

      if ( myrank .EQ. 0 ) then
        call timing_print(.true.)
      endif
     
      call MPI_Win_free( win1, ier )
      call MPI_Win_free( win2, ier )

      end subroutine time_alltoall_nelements

      end module timing_basic
