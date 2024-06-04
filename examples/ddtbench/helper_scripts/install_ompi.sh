# Copyright (c) 2012 The Trustees of University of Illinois. All 
# rights reserved. Use of this source code is governed by a 
# BSD-style license that can be found in the LICENSE file.

#!/bin/sh

install=$1

mkdir -p local/$install
cd local/$install
TEMP=`pwd`
cd ../../src_mpi/
if [ ! -d "$install" ]; then
  tar xvfj $install.tar.bz2
fi
cd $install
./configure --prefix=$TEMP
make
make install
cd ../..
