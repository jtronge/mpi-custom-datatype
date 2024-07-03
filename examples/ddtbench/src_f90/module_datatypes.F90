!     Copyright (c) 2012 The Trustees of University of Illinois. All 
!     rights reserved. Use of this source code is governed by a 
!     BSD-style license that can be found in the LICENSE file.

      module datatypes

      type arrayPointer2D
        real, pointer :: ptr(:,:)
      end type arrayPointer2D

      type arrayPointer3D
        real, pointer :: ptr(:,:,:)
      end type arrayPointer3D

      type arrayPointer4D
        real, pointer :: ptr(:,:,:,:)
      end type arrayPointer4D      

      end module datatypes
