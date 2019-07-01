/*
 * Copyright (c) BIGO, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "cpuinfo.h"
//#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <cpuid.h>

namespace gemv{
  CpuInfo::CpuInfo(){
    unsigned int eax,ebx,ecx,edx;
    __cpuid(0,eax,ebx,ecx,edx);
    if(eax < 0){
      printf("This device is not supported!");
      exit(0);
    }

    __cpuid(1,eax,ebx,ecx,edx);
    has_fma  = (ecx & (1 << 12));
    has_avx  = (ecx & (1 << 28));
    has_f16c = (ecx & (1 << 29));
    //this is the minimum requirements
    if(!has_fma || !has_f16c || !has_avx){
      printf("This device is not supported!");
      exit(0);
    }
    
    __asm__ volatile("xgetbv"
                    : "=a"(eax), "=d"(edx)
                    : "c"(0));
    __cpuid(7,eax,ebx,ecx,edx);
    has_avx2   = (ebx & (1<<5));
    has_avx512 = (ebx & (1<<16));
  }
}
