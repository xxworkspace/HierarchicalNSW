/*
 * Copyright (c) BIGO, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "kernel.h"
#include "cpuinfo.h"
#include <type_traits>
#include "AVXKernel.h"

#include <memory.h>

namespace gemv{

static KernelInfo KInfo;

template<class T1,class T2>
inline void AVXKernel(KParams<T1,T2> &kp,const std::vector<KERNEL>& kernelPtr){
  SubKParams<T1,T2> sp;
  sp.dim = kp.dim;
  CHECK_DIM(sp.dim,32)
  
  unsigned size = kernelPtr.size() - 1;
  unsigned remain = kp.src.size();
  for(unsigned n = 0; remain > 0 ;){
    sp.src = &kp.src[n];
    sp.dst = &kp.dst[n];

    unsigned tmp = size;
    if(remain < size)
      tmp = remain;
    kernelPtr[tmp]((void*)&sp);
    remain -= tmp;
    n += tmp;
  }
}


static std::vector<KERNEL> NPTR;
template<class T1,class T2>
void L2NormAVX(KParams<T1,T2> &kp){
  if(NPTR.empty()){
    std::string ptrName = "L2Norm_";
    if(std::is_same<T1,half>::value && std::is_same<T2,half>::value)
      ptrName += "Half_";
    else if(std::is_same<T1,float>::value && std::is_same<T2,half>::value)
      ptrName += "Float_Half_";
    else
      ptrName += "Float_";
  
    if(hasAVX512())
      ptrName += "AVX512";
    else if(hasAVX2())
      ptrName += "AVX256";
    else{
      printf("This device is not supported!");
      exit(0);
    }
    NPTR = KInfo[ptrName];
  }
  AVXKernel<T1,T2>(kp,NPTR);
}

template void L2NormAVX<half,half>(KParams<half,half>&kp);
template void L2NormAVX<float,half>(KParams<float,half>&kp);
template void L2NormAVX<float,float>(KParams<float,float>&kp);

static std::vector<KERNEL> FPTR;
void Float2HalfAVX(KParams<float,half>&kp){
  if(FPTR.empty()){
    if(hasAVX512())
      FPTR = KInfo["Float2Half_Float_AVX512"];
    else if(hasAVX())
      FPTR = KInfo["Float2Half_Float_AVX256"];
    else{
      printf("This device is not supported!");
      exit(0);
    }
  }
  AVXKernel<float,half>(kp,FPTR);
}

}

