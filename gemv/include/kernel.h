/*
 * Copyright (c) BIGO, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
 
#pragma once

#include "gemv.h"

namespace gemv{
  template <class T1,class T2>
  struct KParams{
    usize_t dim;
    std::vector<T1*> src;
    std::vector<T2*> dst;	
  };

  template<class T1,class T2>
  struct SubKParams{
    usize_t dim;
    T1** src;
    T2** dst;
  };

  template<class T1,class T2>
  void L2NormAVX(KParams<T1,T2>& kp);
  void Float2HalfAVX(KParams<float,half>& kp);
}