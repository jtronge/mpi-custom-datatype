!     Copyright (c) 2012 The Trustees of University of Illinois. All 
!     rights reserved. Use of this source code is governed by a 
!     BSD-style license that can be found in the LICENSE file.
      
      module timing_lammps

        private 

        public :: timing_lammps_full_ddt
        public :: timing_lammps_full_manual
        public :: timing_lammps_full_mpi_pack_ddt
        public :: timing_lammps_atomic_ddt
        public :: timing_lammps_atomic_manual
        public :: timing_lammps_atomic_mpi_pack_ddt

      contains

      subroutine timing_lammps_full_ddt( DIM1, icount, list, &
        outer_loop, inner_loop, correct_flag, ptypesize, testname, &
        filehandle_debug, local_communicator)

!      use timing, only: timing_init, timing_record, timing_print
      use utilities, only: fill_unique_array_1D_double, &
        fill_unique_array_2D_double

      implicit none

      include 'mpif.h'

      integer, intent(in) :: DIM1
      integer, intent(in) :: icount

      integer, intent(in) :: outer_loop, inner_loop

      integer, intent(in), dimension(icount,outer_loop) :: list

      logical, intent(out) :: correct_flag
      integer, intent(out) :: ptypesize

      character(50), intent(in) :: testname

      integer, intent(in) :: filehandle_debug

      integer, intent(in) :: local_communicator

      double precision, dimension(8*(DIM1+icount)) :: array

      integer :: myrank, ier
      integer :: i, j
      integer :: typesize, bytes, base

      integer :: dtype_indexed1_t, dtype_indexed3_t, &
        dtype_send_t, dtype_cont1_t, dtype_cont3_t, dtype_recv_t
      integer(kind=MPI_ADDRESS_KIND), dimension(6) :: &
        address_displacement
      integer, dimension(6) :: blocklength, oldtype
      integer, dimension(:), allocatable :: index_displacement

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

! assume that one double precision element has a size of 8 byte
      win_size = 8 * (DIM1+icount) * 8
      call MPI_Win_create( array(1), win_size, 8, MPI_INFO_NULL, &
        local_communicator, win, ier )

! initial fence to open epoch
      call MPI_Win_fence( 0, win, ier )

      base = myrank * (8*(DIM1+icount)) + 1
      call fill_unique_array_1D_double( array(1), 3*(DIM1+icount), base )
      base = base + 3*(DIM1+icount)
      call fill_unique_array_1D_double( array(3*(DIM1+icount)+1), &
        DIM1+icount, base )
      base = base + DIM1 + icount
      call fill_unique_array_1D_double( array(4*(DIM1+icount)+1), &
        DIM1+icount, base )
      base = base + DIM1 + icount
      call fill_unique_array_1D_double( array(5*(DIM1+icount)+1), &
        DIM1+icount, base )
      base = base + DIM1 + icount
      call fill_unique_array_1D_double( array(6*(DIM1+icount)+1), &
        DIM1+icount, base )
      base = base + DIM1 + icount
      call fill_unique_array_1D_double( array(7*(DIM1+icount)+1), &
        DIM1+icount, base )

      if ( myrank .EQ. 0 ) then
        write (method,'(A)') "mpi_ddt"

        call MPI_Type_size( MPI_DOUBLE_PRECISION, typesize, ier )
        bytes = icount * 8 * typesize
 
        call timing_init( testname, method, bytes )
      endif

      do i=1,outer_loop

        allocate(index_displacement(icount), stat=ier)

        index_displacement(1:icount) = list(1:icount,i) - 1
        call MPI_Type_create_indexed_block( icount, 1, index_displacement, &
          MPI_DOUBLE_PRECISION, dtype_indexed1_t, ier )
      
        index_displacement(1:icount) = 3 * index_displacement(1:icount)
        call MPI_Type_create_indexed_block( icount, 3, index_displacement, &
          MPI_DOUBLE_PRECISION, dtype_indexed3_t, ier )

! assume that one double precision element has a size of 8 byte
        address_displacement(1) = 0
        address_displacement(2) = 3 * (icount+DIM1) * 8
        address_displacement(3) = 4 * (icount+DIM1) * 8
        address_displacement(4) = 5 * (icount+DIM1) * 8
        address_displacement(5) = 6 * (icount+DIM1) * 8
        address_displacement(6) = 7 * (icount+DIM1) * 8

        oldtype(1) = dtype_indexed3_t
        oldtype(2:6) = dtype_indexed1_t
 
        blocklength = 1
 
        call MPI_Type_create_struct( 6, blocklength, address_displacement, &
          oldtype, dtype_send_t, ier )
        call MPI_Type_commit( dtype_send_t, ier )

        call MPI_Type_contiguous( icount, MPI_DOUBLE_PRECISION, &
          dtype_cont1_t, ier );
        call MPI_Type_contiguous( 3*icount, MPI_DOUBLE_PRECISION, &
          dtype_cont3_t, ier );

! assume that one double precision element has a size of 8 byte
        address_displacement(1) = (3 * DIM1) * 8
        address_displacement(2) = (3 * (DIM1+icount) + DIM1) * 8
        address_displacement(3) = (4 * (DIM1+icount) + DIM1) * 8
        address_displacement(4) = (5 * (DIM1+icount) + DIM1) * 8
        address_displacement(5) = (6 * (DIM1+icount) + DIM1) * 8
        address_displacement(6) = (7 * (DIM1+icount) + DIM1) * 8

        oldtype(1) = dtype_cont3_t
        oldtype(2:6) = dtype_cont1_t

        call MPI_Type_create_struct( 6, blocklength, address_displacement, &
          oldtype, dtype_recv_t, ier )
        call MPI_Type_commit( dtype_recv_t, ier )

        call MPI_Type_free( dtype_indexed1_t, ier )
        call MPI_Type_free( dtype_indexed3_t, ier )
        call MPI_Type_free( dtype_cont1_t, ier )
        call MPI_Type_free( dtype_cont3_t, ier )

        deallocate(index_displacement, stat=ier )

        if ( myrank .EQ. 0 ) then
          call timing_record(1)
        endif

        do j=1,inner_loop

          if ( myrank .EQ. 0 ) then
            call MPI_Put( array(1), 1, dtype_send_t, 1, target_disp, &
              1, dtype_recv_t, win, ier )
            call MPI_Win_fence( 0, win, ier )
            call MPI_Win_fence( 0, win, ier )
            call timing_record(3)
          else
            call MPI_Win_fence( 0, win, ier )
            call MPI_Put( array(1), 1, dtype_send_t, 0, target_disp, &
              1, dtype_recv_t, win, ier )
            call MPI_Win_fence( 0, win, ier )
          endif

        enddo ! inner loop

        call MPI_Type_free( dtype_send_t, ier )
        call MPI_Type_free( dtype_recv_t, ier )

        if ( myrank .EQ. 0 ) then
          call timing_record(5)
        endif

      enddo ! outer loop

      if ( myrank .EQ. 0 ) then
        call timing_print( .true. )
      endif

      call MPI_Win_free( win, ier )

      end subroutine timing_lammps_full_ddt

      subroutine timing_lammps_full_manual( DIM1, icount, list, &
        outer_loop, inner_loop, correct_flag, ptypesize, testname, &
        filehandle_debug, local_communicator)

!      use timing, only: timing_init, timing_record, timing_print
      use utilities, only: fill_unique_array_1D_double, &
        fill_unique_array_2D_double

      implicit none

      include 'mpif.h'

      integer, intent(in) :: DIM1
      integer, intent(in) :: icount

      integer, intent(in) :: outer_loop, inner_loop

      integer, dimension(icount,outer_loop) :: list

      logical, intent(out) :: correct_flag
      integer, intent(out) :: ptypesize

      character(50), intent(in) :: testname

      integer, intent(in) :: filehandle_debug

      integer, intent(in) :: local_communicator

      double precision, dimension(DIM1+icount) :: atag, atype, amask, &
        amolecule, aq
      double precision, dimension(3,DIM1+icount) :: ax
      double precision, dimension(:), allocatable :: buffer

      integer :: myrank, ier
      integer :: i, j, k, l
      integer :: typesize, bytes, base, pos, isize

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

      isize = 8*icount
      call MPI_Type_size( MPI_DOUBLE_PRECISION, typesize, ier )
      bytes = isize * typesize

      allocate( buffer(isize), stat=ier ) 
! assume that one double precision element has a size of 8 byte
      win_size = isize * 8
      call MPI_Win_create( buffer(1), win_size, 8, MPI_INFO_NULL, &
        local_communicator, win, ier )

! initial fence to open epoch
      call MPI_Win_fence( 0, win, ier )

      base = myrank * (8*(DIM1+icount)) + 1
      call fill_unique_array_2D_double( ax, 3, DIM1+icount, base )
      base = base + 3*(DIM1+icount)
      call fill_unique_array_1D_double( atag, DIM1+icount, base )
      base = base + DIM1 + icount
      call fill_unique_array_1D_double( atype, DIM1+icount, base )
      base = base + DIM1 + icount
      call fill_unique_array_1D_double( amask, DIM1+icount, base )
      base = base + DIM1 + icount
      call fill_unique_array_1D_double( aq, DIM1+icount, base )
      base = base + DIM1 + icount
      call fill_unique_array_1D_double( amolecule, DIM1+icount, base )
     
      if ( myrank .EQ. 0 ) then
        write (method,'(A)') "manual"

        call timing_init( testname, method, bytes )
      endif

      do k=1,outer_loop

        do l=1,inner_loop

          if ( myrank .EQ. 0 ) then
            pos = 1
            do i=1,icount
              j=list(i,k)
              buffer(pos) = ax(1,j)
              buffer(pos+1) = ax(2,j)
              buffer(pos+2) = ax(3,j)
              buffer(pos+3) = atag(j)
              buffer(pos+4) = atype(j)
              buffer(pos+5) = amask(j)
              buffer(pos+6) = amolecule(j)
              buffer(pos+7) = aq(j)
              pos = pos + 8
            enddo
            call timing_record(2)
            call MPI_Put( buffer(1), isize, MPI_DOUBLE_PRECISION, 1, &
              target_disp, isize, MPI_DOUBLE_PRECISION, win, ier )
            call MPI_Win_fence( 0, win, ier )
            call MPI_Win_fence( 0, win, ier )
            call timing_record(3)
            pos = 1
            do i=1,icount
              j=DIM1+i
              ax(1,j) = buffer(pos)
              ax(2,j) = buffer(pos+1)
              ax(3,j) = buffer(pos+2)
              atag(j) = buffer(pos+3)
              atype(j) = buffer(pos+4)
              amask(j) = buffer(pos+5)
              amolecule(j) = buffer(pos+6)
              aq(j) = buffer(pos+7)
              pos = pos + 8
            enddo
            call timing_record(4)
          else
            call MPI_Win_fence( 0, win, ier )
            pos = 1
            do i=1,icount
              j=DIM1+i
              ax(1,j) = buffer(pos)
              ax(2,j) = buffer(pos+1)
              ax(3,j) = buffer(pos+2)
              atag(j) = buffer(pos+3)
              atype(j) = buffer(pos+4)
              amask(j) = buffer(pos+5)
              amolecule(j) = buffer(pos+6)
              aq(j) = buffer(pos+7)
              pos = pos + 8
            enddo
            pos = 1
            do i=1,icount
              j=list(i,k)
              buffer(pos) = ax(1,j)
              buffer(pos+1) = ax(2,j)
              buffer(pos+2) = ax(3,j)
              buffer(pos+3) = atag(j)
              buffer(pos+4) = atype(j)
              buffer(pos+5) = amask(j)
              buffer(pos+6) = amolecule(j)
              buffer(pos+7) = aq(j)
              pos = pos + 8
            enddo
            call MPI_Put( buffer(1), isize, MPI_DOUBLE_PRECISION, 0, &
              target_disp, isize, MPI_DOUBLE_PRECISION, win, ier )
            call MPI_Win_fence( 0, win, ier )
          endif

        enddo ! inner loop

      enddo ! outer loop

      if ( myrank .EQ. 0 ) then
        call timing_print( .true. )
      endif

      call MPI_Win_free( win, ier ) 

      deallocate( buffer, stat=ier )

      end subroutine timing_lammps_full_manual

      subroutine timing_lammps_full_mpi_pack_ddt( DIM1, icount, list, &
        outer_loop, inner_loop, correct_flag, ptypesize, testname, &
        filehandle_debug, local_communicator)

!      use timing, only: timing_init, timing_record, timing_print
      use utilities, only: fill_unique_array_1D_double, &
        fill_unique_array_2D_double

      implicit none

      include 'mpif.h'

      integer, intent(in) :: DIM1
      integer, intent(in) :: icount

      integer, intent(in) :: outer_loop, inner_loop

      integer, dimension(icount,outer_loop) :: list

      logical, intent(out) :: correct_flag
      integer, intent(out) :: ptypesize

      character(50), intent(in) :: testname

      integer, intent(in) :: filehandle_debug

      integer, intent(in) :: local_communicator

      double precision, dimension(DIM1+icount) :: atag, atype, amask, &
        amolecule, aq
      double precision, dimension(3,DIM1+icount) :: ax
      double precision, dimension(:), allocatable :: buffer

      integer :: myrank, ier
      integer :: i, j
      integer :: typesize, bytes, base, pos

      integer :: dtype_indexed1_t, dtype_indexed3_t, &
        dtype_send_t, dtype_cont1_t, dtype_cont3_t, dtype_recv_t
      integer(kind=MPI_ADDRESS_KIND), dimension(6) :: address_displacement
      integer, dimension(6) :: blocklength, oldtype
      integer, dimension(:), allocatable :: index_displacement

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

      call MPI_Type_size( MPI_DOUBLE_PRECISION, typesize, ier )
      bytes = 8 * icount * typesize 
      allocate( buffer(8*icount), stat=ier ) 

! assume that one double precision element has a size of 8 byte
      win_size = 8 * icount * 8
      call MPI_Win_create( buffer(1), win_size, 8, MPI_INFO_NULL, &
        local_communicator, win, ier )

! initial fence to open epoch
      call MPI_Win_fence( 0, win, ier )

      base = myrank * (8*(DIM1+icount)) + 1
      call fill_unique_array_2D_double( ax, 3, DIM1+icount, base )
      base = base + 3*(DIM1+icount)
      call fill_unique_array_1D_double( atag, DIM1+icount, base )
      base = base + DIM1 + icount
      call fill_unique_array_1D_double( atype, DIM1+icount, base )
      base = base + DIM1 + icount
      call fill_unique_array_1D_double( amask, DIM1+icount, base )
      base = base + DIM1 + icount
      call fill_unique_array_1D_double( aq, DIM1+icount, base )
      base = base + DIM1 + icount
      call fill_unique_array_1D_double( amolecule, DIM1+icount, base )

      if ( myrank .EQ. 0 ) then
        write (method,'(A)') "mpi_pack_ddt"

        call timing_init( testname, method, bytes )
      endif

      do i=1,outer_loop

        allocate(index_displacement(icount), stat=ier)
        index_displacement(1:icount) = list(1:icount,i) - 1
        call MPI_Type_create_indexed_block( icount, 1, index_displacement, &
          MPI_DOUBLE_PRECISION, dtype_indexed1_t, ier )
      
        index_displacement(1:icount) = 3 * index_displacement(1:icount)
        call MPI_Type_create_indexed_block( icount, 3, index_displacement, &
          MPI_DOUBLE_PRECISION, dtype_indexed3_t, ier )

        call MPI_Get_address( ax(1,1), address_displacement(1), ier )
        call MPI_Get_address( atag(1), address_displacement(2), ier )
        call MPI_Get_address( atype(1), address_displacement(3), ier )
        call MPI_Get_address( amask(1), address_displacement(4), ier )
        call MPI_Get_address( amolecule(1), address_displacement(5), ier )
        call MPI_Get_address( aq(1), address_displacement(6), ier )

        oldtype(1) = dtype_indexed3_t
        oldtype(2:6) = dtype_indexed1_t
 
        blocklength = 1
 
        call MPI_Type_create_struct( 6, blocklength, address_displacement, &
          oldtype, dtype_send_t, ier )
        call MPI_Type_commit( dtype_send_t, ier )

        call MPI_Type_contiguous( icount, MPI_DOUBLE_PRECISION, &
          dtype_cont1_t, ier )
        call MPI_Type_contiguous( 3*icount, MPI_DOUBLE_PRECISION, &
          dtype_cont3_t, ier )

        call MPI_Get_address( ax(1,DIM1+1), address_displacement(1), ier )
        call MPI_Get_address( atag(DIM1+1), address_displacement(2), ier )
        call MPI_Get_address( atype(DIM1+1), address_displacement(3), ier )
        call MPI_Get_address( amask(DIM1+1), address_displacement(4), ier )
        call MPI_Get_address( aq(DIM1+1), address_displacement(5), ier )
        call MPI_Get_address( amolecule(DIM1+1), address_displacement(6), ier )

        oldtype(1) = dtype_cont3_t
        oldtype(2:6) = dtype_cont1_t

        call MPI_Type_create_struct( 6, blocklength, address_displacement, &
          oldtype, dtype_recv_t, ier )
        call MPI_Type_commit( dtype_recv_t, ier )

        call MPI_Type_free( dtype_indexed1_t, ier )
        call MPI_Type_free( dtype_indexed3_t, ier )
        call MPI_Type_free( dtype_cont1_t, ier )
        call MPI_Type_free( dtype_cont3_t, ier )

        deallocate(index_displacement, stat=ier )

        if ( myrank .EQ. 0 ) then
          call timing_record(1)
        endif

        do j=1,inner_loop

          if ( myrank .EQ. 0 ) then
            pos = 0
            call MPI_Pack( MPI_BOTTOM, 1, dtype_send_t, buffer, &
              bytes, pos, local_communicator, ier )
            call timing_record(2)
            call MPI_Put( buffer(1), pos, MPI_PACKED, 1, target_disp, &
              pos, MPI_PACKED, win, ier )
            call MPI_Win_fence( 0, win, ier )
            call MPI_Win_fence( 0, win, ier )
            call timing_record(3)
            pos = 0
            call MPI_Unpack( buffer, bytes, pos, MPI_BOTTOM, 1, &
              dtype_recv_t, local_communicator, ier )
            call timing_record(4)
          else
            call MPI_Win_fence( 0, win, ier )
            pos = 0
            call MPI_Unpack( buffer, bytes, pos, MPI_BOTTOM, 1, &
              dtype_recv_t, local_communicator, ier )
            pos = 0
            call MPI_Pack( MPI_BOTTOM, 1, dtype_send_t, buffer, &
              bytes, pos, local_communicator, ier )           
            call MPI_Put( buffer(1), pos, MPI_PACKED, 0, target_disp, &
              pos, MPI_PACKED, win, ier )
            call MPI_Win_fence( 0, win, ier )
          endif

        enddo ! inner loop

        call MPI_Type_free( dtype_send_t, ier )
        call MPI_Type_free( dtype_recv_t, ier )

        if ( myrank .EQ. 0 ) then
          call timing_record(5)
        endif

      enddo ! outer loop

      if ( myrank .EQ. 0 ) then
        call timing_print( .true. )
      endif

      call MPI_Win_free( win, ier )

      deallocate( buffer, stat=ier )

      end subroutine timing_lammps_full_mpi_pack_ddt

      subroutine timing_lammps_atomic_ddt( DIM1, icount, list, &
        outer_loop, inner_loop, correct_flag, ptypesize, testname, &
        filehandle_debug, local_communicator)

!      use timing, only: timing_init, timing_record, timing_print
      use utilities, only: fill_unique_array_1D_double, &
        fill_unique_array_2D_double

      implicit none

      include 'mpif.h'

      integer, intent(in) :: DIM1
      integer, intent(in) :: icount

      integer, intent(in) :: outer_loop, inner_loop

      integer, intent(in), dimension(icount,outer_loop) :: list

      logical, intent(out) :: correct_flag
      integer, intent(out) :: ptypesize

      character(50), intent(in) :: testname

      integer, intent(in) :: filehandle_debug

      integer, intent(in) :: local_communicator

      double precision, dimension(6*(DIM1+icount)) :: array

      integer :: myrank, ier
      integer :: i, j
      integer :: typesize, bytes, base

      integer :: dtype_indexed1_t, dtype_indexed3_t, &
        dtype_send_t, dtype_cont1_t, dtype_cont3_t, dtype_recv_t
      integer(kind=MPI_ADDRESS_KIND), dimension(4) :: address_displacement
      integer, dimension(4) :: blocklength, oldtype
      integer, dimension(:), allocatable :: index_displacement

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

! assume that one double precision element has a size of 8 byte
      win_size = 6 * (DIM1+icount) * 8
      call MPI_Win_create( array(1), win_size, 8, MPI_INFO_NULL, &
        local_communicator, win, ier )

! initial fence to open epoch
      call MPI_Win_fence( 0, win, ier )

      base = myrank * (6*(DIM1+icount)) + 1
      call fill_unique_array_1D_double( array(1), 3*(DIM1+icount), base )
      base = base + 3*(DIM1+icount)
      call fill_unique_array_1D_double( array(3*(DIM1+icount)+1), &
        DIM1+icount, base )
      base = base + DIM1 + icount
      call fill_unique_array_1D_double( array(4*(DIM1+icount)+1), &
        DIM1+icount, base )
      base = base + DIM1 + icount
      call fill_unique_array_1D_double( array(5*(DIM1+icount)+1), &
        DIM1+icount, base )

      if ( myrank .EQ. 0 ) then
        write (method,'(A)') "mpi_ddt"

        call MPI_Type_size( MPI_DOUBLE_PRECISION, typesize, ier )
        bytes = icount * 6 * typesize

        call timing_init( testname, method, bytes )
      endif

      do i=1,outer_loop

        allocate(index_displacement(icount), stat=ier)

        index_displacement(1:icount) = list(1:icount,i) - 1
        call MPI_Type_create_indexed_block( icount, 1, index_displacement, &
          MPI_DOUBLE_PRECISION, dtype_indexed1_t, ier )
      
        index_displacement(1:icount) = 3 * index_displacement(1:icount)
        call MPI_Type_create_indexed_block( icount, 3, index_displacement, &
          MPI_DOUBLE_PRECISION, dtype_indexed3_t, ier )

! assume that one double precision element has a size of 8 byte
        address_displacement(1) = 0
        address_displacement(2) = 3 * (icount+DIM1) * 8
        address_displacement(3) = 4 * (icount+DIM1) * 8
        address_displacement(4) = 5 * (icount+DIM1) * 8        

        oldtype(1) = dtype_indexed3_t
        oldtype(2:4) = dtype_indexed1_t
 
        blocklength = 1
 
        call MPI_Type_create_struct( 4, blocklength, address_displacement, &
          oldtype, dtype_send_t, ier )
        call MPI_Type_commit( dtype_send_t, ier )

        call MPI_Type_contiguous( icount, MPI_DOUBLE_PRECISION, &
          dtype_cont1_t, ier )
        call MPI_Type_contiguous( 3*icount, MPI_DOUBLE_PRECISION, &
          dtype_cont3_t, ier )

! assume that one double precision element has a size of 8 byte
        address_displacement(1) = (3 * DIM1) * 8
        address_displacement(2) = (3 * (DIM1+icount) + DIM1) * 8
        address_displacement(3) = (4 * (DIM1+icount) + DIM1) * 8
        address_displacement(4) = (5 * (DIM1+icount) + DIM1) * 8

        oldtype(1) = dtype_cont3_t
        oldtype(2:4) = dtype_cont1_t
 
        call MPI_Type_create_struct( 4, blocklength, address_displacement, &
          oldtype, dtype_recv_t, ier )
        call MPI_Type_commit( dtype_recv_t, ier )

        call MPI_Type_free( dtype_indexed1_t, ier )
        call MPI_Type_free( dtype_indexed3_t, ier )
        call MPI_Type_free( dtype_cont1_t, ier )
        call MPI_Type_free( dtype_cont3_t, ier )

        deallocate(index_displacement, stat=ier )

        if ( myrank .EQ. 0 ) then
          call timing_record(1)
        endif

        do j=1,inner_loop

          if ( myrank .EQ. 0 ) then
            call MPI_Put( array(1), 1, dtype_send_t, 1, target_disp, &
              1, dtype_recv_t, win, ier )
            call MPI_Win_fence( 0, win, ier )
            call MPI_Win_fence( 0, win, ier )
            call timing_record(3)
          else
            call MPI_Win_fence( 0, win, ier )
            call MPI_Put( array(1), 1, dtype_send_t, 0, target_disp, &
              1, dtype_recv_t, win, ier )
            call MPI_Win_fence( 0, win, ier )
          endif

        enddo ! inner loop

        call MPI_Type_free( dtype_send_t, ier )
        call MPI_Type_free( dtype_recv_t, ier )

        if ( myrank .EQ. 0 ) then
          call timing_record(5)
        endif

      enddo ! outer loop

      if ( myrank .EQ. 0 ) then
        call timing_print( .true. )
      endif

      call MPI_Win_free( win, ier )

      end subroutine timing_lammps_atomic_ddt

      subroutine timing_lammps_atomic_manual( DIM1, icount, list, &
        outer_loop, inner_loop, correct_flag, ptypesize, testname, &
        filehandle_debug, local_communicator)

!      use timing, only: timing_init, timing_record, timing_print
      use utilities, only: fill_unique_array_1D_double, &
        fill_unique_array_2D_double

      implicit none

      include 'mpif.h'

      integer, intent(in) :: DIM1
      integer, intent(in) :: icount

      integer, intent(in) :: outer_loop, inner_loop

      integer, dimension(icount,outer_loop) :: list

      logical, intent(out) :: correct_flag
      integer, intent(out) :: ptypesize

      character(50), intent(in) :: testname

      integer, intent(in) :: filehandle_debug

      integer, intent(in) :: local_communicator

      double precision, dimension(DIM1+icount) :: atag, atype, amask
      double precision, dimension(3,DIM1+icount) :: ax
      double precision, dimension(:), allocatable :: buffer

      integer :: myrank, ier
      integer :: i, j, k, l
      integer :: typesize, bytes, base, pos, isize

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

      isize = 6*icount
      call MPI_Type_size( MPI_DOUBLE_PRECISION, typesize, ier )
      bytes = isize * typesize

      allocate( buffer(isize), stat=ier ) 
! assume that one double precision element has a size of 8 byte
      win_size = isize * 8
      call MPI_Win_create( buffer(1), win_size, 8, MPI_INFO_NULL, &
        local_communicator, win, ier )

! initial fence to open epoch
      call MPI_Win_fence( 0, win, ier )

      base = myrank * (6*(DIM1+icount)) + 1
      call fill_unique_array_2D_double( ax, 3, DIM1+icount, base )
      base = base + 3*(DIM1+icount)
      call fill_unique_array_1D_double( atag, DIM1+icount, base )
      base = base + DIM1 + icount
      call fill_unique_array_1D_double( atype, DIM1+icount, base )
      base = base + DIM1 + icount
      call fill_unique_array_1D_double( amask, DIM1+icount, base )

      if ( myrank .EQ. 0 ) then
        write (method,'(A)') "manual"

        call timing_init( testname, method, bytes )
      endif

      do k=1,outer_loop

        do l=1,inner_loop

          if ( myrank .EQ. 0 ) then
            pos = 1
            do i=1,icount
              j = list(i,k)
              buffer(pos) = ax(1,j)
              buffer(pos+1) = ax(2,j)
              buffer(pos+2) = ax(3,j)
              buffer(pos+3) = atag(j)
              buffer(pos+4) = atype(j)
              buffer(pos+5) = amask(j)
              pos = pos + 6
            enddo
            call timing_record(2)
            call MPI_Put( buffer(1), isize, MPI_DOUBLE_PRECISION, 1, &
              target_disp, isize, MPI_DOUBLE_PRECISION, win, ier )
            call MPI_Win_fence( 0, win, ier )
            call MPI_Win_fence( 0, win, ier )
            call timing_record(3)
            pos = 1
            do i=1,icount
              j = DIM1+i
              ax(1,j) = buffer(pos)
              ax(2,j) = buffer(pos+1)
              ax(3,j) = buffer(pos+2)
              atag(j) = buffer(pos+3)
              atype(j) = buffer(pos+4)
              amask(j) = buffer(pos+5)
              pos = pos + 6
            enddo
            call timing_record(4)
          else
            call MPI_Win_fence( 0, win, ier )
            pos = 1
            do i=1,icount
              j = DIM1+i
              ax(1,j) = buffer(pos)
              ax(2,j) = buffer(pos+1)
              ax(3,j) = buffer(pos+2)
              atag(j) = buffer(pos+3)
              atype(j) = buffer(pos+4)
              amask(j) = buffer(pos+5)
              pos = pos + 6
            enddo
            pos = 1
            do i=1,icount
              j = list(i,k)
              buffer(pos) = ax(1,j)
              buffer(pos+1) = ax(2,j)
              buffer(pos+2) = ax(3,j)
              buffer(pos+3) = atag(j)
              buffer(pos+4) = atype(j)
              buffer(pos+5) = amask(j)
              pos = pos + 6
            enddo
            call MPI_Put( buffer(1), isize, MPI_DOUBLE_PRECISION, 0, &
              target_disp, isize, MPI_DOUBLE_PRECISION, win, ier )
            call MPI_Win_fence( 0, win, ier ) 
          endif

        enddo ! inner loop

      enddo ! outer loop

      if ( myrank .EQ. 0 ) then
        call timing_print( .true. )
      endif

      call MPI_Win_free( win, ier )

      deallocate( buffer, stat=ier )

      end subroutine timing_lammps_atomic_manual

      subroutine timing_lammps_atomic_mpi_pack_ddt( DIM1, icount, list,&
        outer_loop, inner_loop, correct_flag, ptypesize, testname, &
        filehandle_debug, local_communicator )

!      use timing, only: timing_init, timing_record, timing_print
      use utilities, only: fill_unique_array_1D_double, &
        fill_unique_array_2D_double

      implicit none

      include 'mpif.h'

      integer, intent(in) :: DIM1
      integer, intent(in) :: icount

      integer, intent(in) :: outer_loop, inner_loop

      integer, dimension(icount,outer_loop) :: list

      logical, intent(out) :: correct_flag
      integer, intent(out) :: ptypesize

      character(50), intent(in) :: testname

      integer, intent(in) :: filehandle_debug

      integer, intent(in) :: local_communicator

      double precision, dimension(DIM1+icount) :: atag, atype, amask
      double precision, dimension(3,DIM1+icount) :: ax
      double precision, dimension(:), allocatable :: buffer

      integer :: myrank, ier
      integer :: i, j
      integer :: typesize, bytes, base, pos

      integer :: dtype_indexed1_t, dtype_indexed3_t, &
        dtype_send_t, dtype_cont1_t, dtype_cont3_t, dtype_recv_t
      integer(kind=MPI_ADDRESS_KIND), dimension(4) :: address_displacement
      integer, dimension(4) :: blocklength, oldtype
      integer, dimension(:), allocatable :: index_displacement

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

      call MPI_Type_size( MPI_DOUBLE_PRECISION, typesize, ier )
      bytes = 6 * icount * typesize 
      allocate( buffer(6*icount), stat=ier ) 

! assume that one double precision element has a size of 8 byte
      win_size = 6 * icount * 8
      call MPI_Win_create( buffer(1), win_size, 8, MPI_INFO_NULL, &
        local_communicator, win, ier )

! initial fence to open epoch
      call MPI_Win_fence( 0, win, ier )

      base = myrank * (6*(DIM1+icount)) + 1
      call fill_unique_array_2D_double( ax, 3, DIM1+icount, base )
      base = base + 3*(DIM1+icount)
      call fill_unique_array_1D_double( atag, DIM1+icount, base )
      base = base + DIM1 + icount
      call fill_unique_array_1D_double( atype, DIM1+icount, base )
      base = base + DIM1 + icount
      call fill_unique_array_1D_double( amask, DIM1+icount, base )

      if ( myrank .EQ. 0 ) then
        write (method,'(A)') "mpi_pack_ddt"

        call timing_init( testname, method, bytes )
      endif

      do i=1,outer_loop

        allocate(index_displacement(icount), stat=ier)
        index_displacement(1:icount) = list(1:icount,i) - 1
        call MPI_Type_create_indexed_block( icount, 1, index_displacement, &
          MPI_DOUBLE_PRECISION, dtype_indexed1_t, ier )
      
        index_displacement(1:icount) = 3 * index_displacement(1:icount)
        call MPI_Type_create_indexed_block( icount, 3, index_displacement, &
          MPI_DOUBLE_PRECISION, dtype_indexed3_t, ier )

        call MPI_Get_address( ax(1,1), address_displacement(1), ier )
        call MPI_Get_address( atag(1), address_displacement(2), ier )
        call MPI_Get_address( atype(1), address_displacement(3), ier )
        call MPI_Get_address( amask(1), address_displacement(4), ier )

        oldtype(1) = dtype_indexed3_t
        oldtype(2:4) = dtype_indexed1_t
 
        blocklength = 1
 
        call MPI_Type_create_struct( 4, blocklength, address_displacement, &
          oldtype, dtype_send_t, ier )
        call MPI_Type_commit( dtype_send_t, ier )

        call MPI_Type_contiguous( icount, MPI_DOUBLE_PRECISION, &
          dtype_cont1_t, ier );
        call MPI_Type_contiguous( 3*icount, MPI_DOUBLE_PRECISION, &
          dtype_cont3_t, ier );

        call MPI_Get_address( ax(1,DIM1+1), address_displacement(1), ier )
        call MPI_Get_address( atag(DIM1+1), address_displacement(2), ier )
        call MPI_Get_address( atype(DIM1+1), address_displacement(3), ier )
        call MPI_Get_address( amask(DIM1+1), address_displacement(4), ier )

        oldtype(1) = dtype_cont3_t
        oldtype(2:4) = dtype_cont1_t

        call MPI_Type_create_struct( 4, blocklength, address_displacement, &
          oldtype, dtype_recv_t, ier )
        call MPI_Type_commit( dtype_recv_t, ier )

        call MPI_Type_free( dtype_indexed1_t, ier )
        call MPI_Type_free( dtype_indexed3_t, ier )
        call MPI_Type_free( dtype_cont1_t, ier )
        call MPI_Type_free( dtype_cont3_t, ier )

        deallocate(index_displacement, stat=ier )

        if ( myrank .EQ. 0 ) then
          call timing_record(1)
        endif

        do j=1,inner_loop

          if ( myrank .EQ. 0 ) then
            pos = 0
            call MPI_Pack( MPI_BOTTOM, 1, dtype_send_t, buffer, &
              bytes, pos, local_communicator, ier )
            call timing_record(2)
            call MPI_Put( buffer(1), pos, MPI_PACKED, 1, target_disp, &
              pos, MPI_PACKED, win, ier )
            call MPI_Win_fence( 0, win, ier )
            call MPI_Win_fence( 0, win, ier )
            call timing_record(3)
            pos = 0
            call MPI_Unpack( buffer, bytes, pos, MPI_BOTTOM, 1, &
              dtype_recv_t, local_communicator, ier )
            call timing_record(4)
          else
            call MPI_Win_fence( 0, win, ier )
            pos = 0
            call MPI_Unpack( buffer, bytes, pos, MPI_BOTTOM, 1, &
              dtype_recv_t, local_communicator, ier )
            pos = 0
            call MPI_Pack( MPI_BOTTOM, 1, dtype_send_t, buffer, &
              bytes, pos, local_communicator, ier )
            call MPI_Put( buffer(1), pos, MPI_PACKED, 0, target_disp, &
              pos, MPI_PACKED, win, ier )
            call MPI_Win_fence( 0, win, ier )
          endif

        enddo ! inner loop

        call MPI_Type_free( dtype_send_t, ier )
        call MPI_Type_free( dtype_recv_t, ier )

        if ( myrank .EQ. 0 ) then
          call timing_record(5)
        endif

      enddo ! outer loop

      if ( myrank .EQ. 0 ) then
        call timing_print( .true. )
      endif

      call MPI_Win_free( win, ier )

      deallocate( buffer, stat=ier )

      end subroutine timing_lammps_atomic_mpi_pack_ddt

      end module timing_lammps
