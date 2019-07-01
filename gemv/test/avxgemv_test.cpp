
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <random>
#include <vector>

#include "gemv.h"
#include <sstream>
#include "util.h"

using namespace gemv;
using namespace std;

int main(int argc,char **argv){
  default_random_engine e;
  uniform_real_distribution<double> u(-1, 1);

  if(argc != 2) exit(0);
  int n = to_t<int>(string(argv[1]));
/*
  gemv::Params<float,float,float> gp;
  gp.dist = gemv::DisT::L2;
  gp.dim = 1024;

  gp.query = (float*)aligned_alloc(64,sizeof(float)*gp.dim);
  for(int j = 0 ; j < gp.dim ; j++)
    gp.query[j] = u(e);
  for(int i = 0 ; i < n ; i++){
    float *tmp = (float*)aligned_alloc(64,sizeof(float)*gp.dim);
    gp.neighbour.push_back(tmp);
    for(int j = 0 ; j < gp.dim ; j++)
      tmp[j] = u(e);
  }
  float avx,cpp;
  {
    ST()
    gemv::GemvAVX<float,float,float>(gp);
    END()
    TIME("AVX2")
    avx = duration_cast<microseconds>(T2 - T1).count();
  }
  {
    ST()
    for(int i  = 0 ; i < n ; i++)
      float sim = _L2(gp.neighbour[i],gp.query,gp.dim);
    END()
    TIME("NORM")
    cpp = duration_cast<microseconds>(T2 - T1).count();
  }
  std::cout<<"speed up : "<<cpp/avx<<std::endl;
*/
  /*
  for(int i = 0 ; i < gp.neighbour.size() ; i++){
    float value = compute(gp.query,gp.neighbour[i],gp.dim);
    if(gp.distance[i] != value)
      std::cout<<i<<" - "<<gp.distance[i]<<" "<<value<<std::endl;
  }*/
/*
  gemv::Params<int8,int8,int> gp;
  gp.dist = gemv::DisT::IP;
  gp.dim = 1024;
  
  gp.query = (int8*)aligned_alloc(64,sizeof(int8)*gp.dim);
  for(int i = 0 ; i < gp.dim ; i ++)
    gp.query[i] = int8(u(e)*127);
  for(int j = 0 ; j < n ; j++){
    int8 *tmp = (int8*)aligned_alloc(64,sizeof(int8)*gp.dim);
    gp.neighbour.push_back(tmp);
	for(int i = 0 ; i < gp.dim ; i++)
      tmp[i] = int(u(e)*127);
  }

  float avx,cpp;
  {
    ST()
    gemv::GemvAVX<int8,int8,int>(gp);
    END()
    TIME("AVX2")
    avx = duration_cast<microseconds>(T2 - T1).count();
  }
  {
    ST()
    for(int i  = 0 ; i < n ; i++)
      float sim = _IP(gp.neighbour[i],gp.query,gp.dim);
    END()
    TIME("NORM")
    cpp = duration_cast<microseconds>(T2 - T1).count();
  }
  std::cout<<"speed up : "<<cpp/avx<<std::endl;
*/
  /*for(int i = 0 ; i < gp.neighbour.size() ; i++){
   int value = _IP(gp.query,gp.neighbour[i],gp.dim);
   if(gp.distance[i] != value)
     std::cout<<i<<" == "<<gp.distance[i]<<" "<<value<<std::endl;
  }*/
/*
  gemv::Params<half,float,float> gp;
  gp.dist = gemv::DisT::IP;
  gp.dim = 1024;

  gp.query = (float*)aligned_alloc(64,sizeof(float)*gp.dim);
  for(int i = 0 ; i < gp.dim ; i ++)
    gp.query[i] = u(e);

  for(int j = 0 ; j < n ; j++){
    half *tmp = (half*)aligned_alloc(64,sizeof(half)*gp.dim);
    gp.neighbour.push_back(tmp);
    for(int i = 0 ; i < gp.dim ; i++)
      tmp[i] = float2half(u(e));
  }

  float avx,cpp;
  {
    ST()
    gemv::GemvAVX<half,float,float>(gp);
    END()
    TIME("AVX2")
    avx = duration_cast<microseconds>(T2 - T1).count();
  }
  {
    ST()
    for(int i  = 0 ; i < n ; i++)
      float sim = _IP(gp.neighbour[i],gp.query,gp.dim);
    END()
    TIME("NORM")
    cpp = duration_cast<microseconds>(T2 - T1).count();
  }
  std::cout<<"speed up : "<<cpp/avx<<std::endl;
*/
  /*
  for(int i = 0 ; i < gp.neighbour.size() ; i++){
    float value = computeL2(gp.query,gp.neighbour[i],gp.dim);
    if(gp.distance[i] != value)
      std::cout<<i<<" - "<<gp.distance[i]<<" "<<value<<std::endl;
  }
  */
/*
  gemv::Params<half,half,float> gp;
  gp.dist = gemv::DisT::IP;
  gp.dim = 1024;
  
  gp.query = (half*)aligned_alloc(64,sizeof(half)*gp.dim);
  for(int i = 0 ; i < gp.dim ; i ++)
    gp.query[i] = float2half(u(e));

  for(int j = 0 ; j < n ; j++){
    half *tmp = (half*)aligned_alloc(64,sizeof(half)*gp.dim);
    gp.neighbour.push_back(tmp);
    for(int i = 0 ; i < gp.dim ; i++)
      tmp[i] = float2half(u(e));
  }
  
  float avx,cpp;
  {
    ST()
    gemv::GemvAVX<half,half,float>(gp);
    END()
    TIME("AVX2")
    avx = duration_cast<microseconds>(T2 - T1).count();
  }
  {
    ST()
    for(int i  = 0 ; i < n ; i++)
      float sim = _IP(gp.neighbour[i],gp.query,gp.dim);
    END()
    TIME("NORM")
    cpp = duration_cast<microseconds>(T2 - T1).count();
  }
  std::cout<<"speed up : "<<cpp/avx<<std::endl;
*/
  /*
  for(int i = 0 ; i < gp.neighbour.size() ; i++){
    float value = compute(gp.query,gp.neighbour[i],gp.dim);
    if(gp.distance[i] != value)
      std::cout<<i<<" - "<<gp.distance[i]<<" "<<value<<std::endl;
  }*/

  return 0;
}
