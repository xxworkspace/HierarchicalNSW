/*
 * Copyright (c) BIGO, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "gemv.h"
#include "cpuinfo.h"
#include <type_traits>
#include "GemvKernel.h"

namespace gemv{
	
static GemvInfo GInfo;
std::vector<GEMV> PTR;

template<class T1,class T2,class T3>
std::vector<GEMV> GetGemvAVX(Params<T1,T2,T3>&p){
  std::string ptrName = "GEMV_";
  if(p.dist == 0)
    ptrName += "IP_";
  else
    ptrName += "L2_";

  if(std::is_same<T1,int8>::value)
    ptrName += "Int8_";
  else if(std::is_same<T1,uint8>::value)
    ptrName += "UInt8_";
  else if(std::is_same<T1,half>::value && std::is_same<T2,half>::value)
    ptrName += "Half_";
  else if(std::is_same<T1,half>::value && std::is_same<T2,float>::value)
    ptrName += "Half_Float_";
  else
    ptrName += "Float_";
  
  if(hasAVX512()){
    if(std::is_same<T1,int8>::value && (p.dim%64 != 0))
      ptrName += "AVX256";
    else
      ptrName += "AVX512";
  }
  else if(hasAVX2())
    ptrName += "AVX256";
  else if(hasAVX() && !std::is_same<T1, int8>::value)
    ptrName += "AVX256";
  else{
    printf("this device is not supported\n");
    exit(0);
  }
  if(!GInfo.hasKernel(ptrName)){
    printf("this type %s is not supported\n",ptrName.c_str());
    exit(0);
  }
  return GInfo[ptrName];
  //GemvAVXKernel<T1,T2,T3>(p,GInfo[ptrName]);
}

template<class T1,class T2,class T3>
//void GemvAVX(Params<T1,T2,T3>&p,const std::vector<GEMV>& gemvPtr){
inline void GemvAVX(Params<T1,T2,T3>&p){//const std::vector<GEMV>& gemvPtr){
  if(PTR.empty())
    PTR = GetGemvAVX<T1,T2,T3>(p);

  SubParams<T1,T2,T3> sp;
  sp.dim = p.dim;
  sp.query = p.query;
  CHECK_DIM(p.dim,32)

  uint32_t tmp; 
  uint32_t size = PTR.size() - 1;
  uint32_t remain = p.neighbour.size();
  p.distance.resize(p.neighbour.size());
  for(uint32_t n = 0 ; remain > 0;){
    sp.neighbour = &p.neighbour[n];
    sp.distance  = &p.distance[n];

    if(remain >= (size << 1))
      tmp = size;
    if(remain >= size)
      tmp = (size + 1)/2;
    else
      tmp = remain;

    PTR[tmp]((void*)&sp);
    remain -= tmp;
    n += tmp;
  }
}

template void GemvAVX<int8,int8,float>(Params<int8,int8,float>&);
template void GemvAVX<uint8,uint8,float>(Params<uint8,uint8,float>&);
template void GemvAVX<half,half,float>(Params<half,half,float>&);
template void GemvAVX<half,float,float>(Params<half,float,float>&);
template void GemvAVX<float,float,float>(Params<float,float,float>&);

}
