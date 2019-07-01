/*
 * Copyright (c) BIGO, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

namespace gemv{

class CpuInfo{
public:
  static const CpuInfo& GetCpuInfo(){
    static CpuInfo cpuinfo;
    return cpuinfo;
  }
  bool HasFMA()const{return has_fma;}
  bool HasF16C()const{return has_f16c;}
  bool HasAVX()const{return has_avx;}
  bool HasAVX2()const{return has_avx2;}
  bool HasAVX512()const{return has_avx512;}
private:
  CpuInfo();
  bool has_fma{false};
  bool has_f16c{false};
  bool has_avx{false};
  bool has_avx2{false};
  bool has_avx512{false};
};
inline bool hasFMA(){
  CpuInfo::GetCpuInfo().HasFMA();
}
inline bool hasF16C(){
  CpuInfo::GetCpuInfo().HasF16C();
}
inline bool hasAVX(){
  CpuInfo::GetCpuInfo().HasAVX();
}
inline bool hasAVX2(){
  CpuInfo::GetCpuInfo().HasAVX2();
}
inline bool hasAVX512(){
  CpuInfo::GetCpuInfo().HasAVX512();
}

}
