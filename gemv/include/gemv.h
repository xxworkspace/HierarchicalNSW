/*
 * Copyright (c) BIGO, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <map>
#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <iostream>

namespace gemv{

#define CHECK_DIM(n,m)        \
  if(n < m || (n % m != 0)){  \
    printf("dimension of this data type is not supported"); \
    exit(0);                  \
  }

  typedef signed char int8;
  typedef signed short half;
  typedef unsigned long long usize_t;  

  enum DisT{
    IP = 0,
    L2 = 1,
  };

  template<class T1,class T2,class T3>
  struct Params{
    DisT dist;
    usize_t dim;
    std::vector<T1*> neighbour;

    T2* query;
    std::vector<T3> distance;
  };

  template<class T1,class T2,class T3>
  struct SubParams{
    usize_t dim;
    T2* query;

    T1** neighbour;
    T3* distance;
  };

  template<class T1,class T2,class T3>
  void GemvAVX(Params<T1,T2,T3>&);
}
