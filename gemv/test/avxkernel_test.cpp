
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <random>
#include <vector>
#include <chrono>

#include "kernel.h"
#include <sstream>
#include "util.h"

using namespace std::chrono;
using namespace gemv;
using namespace std;

using std::default_random_engine;
using std::uniform_real_distribution;

int main(int argc,char**argv){
  
  default_random_engine e;
  uniform_real_distribution<double> u(-1, 1);

  if(argc != 2) exit(0);
  string sn = argv[1];
  int n = to_t<int>(sn);
  /*
  //float2Half
  gemv::KParams<float,half> kp;
  kp.dim = 1024;
  for(int k = 0 ; k < n ; ++k){
    float* data = (float*)aligned_alloc(64,sizeof(float)*kp.dim);
    for(int i = 0 ; i < kp.dim ; i++){
      data[i] = u(e);
    }
    kp.src.push_back(data);
    
    half* output = (half*)aligned_alloc(64,sizeof(half)*kp.dim);
    kp.dst.push_back(output);
  }
  float avx,cpp;
  {
    ST
    gemv::Float2HalfAVX(kp);
    END
    TIME("AVX2")
	avx = duration_cast<microseconds>(T2 - T1).count();
  }
  {
	ST
	for(int i  = 0 ; i < n ; i++)
      float2half(kp.src[i],kp.dst[i],kp.dim);
    END
	TIME("NORM")
	cpp = duration_cast<microseconds>(T2 - T1).count();
  }
  std::cout <<"speed up : "<<cpp/avx<<std::endl;
  for(int i = 0 ; i < kp.src.size() ; i++){
    for(int j = 0 ; j < kp.dim ; j++){
      float tmp = (half2Float(kp.dst[i][j]) - kp.src[i][j])/kp.src[i][j];
      if(abs(tmp) > 0.0005)
        std::cout<<half2Float(kp.dst[i][j])<<" "<<kp.src[i][j]<<" "<<tmp<<std::endl;
    }
    //std::cout<<(half2Float(kp.dst[i][j]) - kp.src[i][j])/kp.src[i][j]<<"**";
  }*/
  /*
  gemv::KParams<float,float> kp;
  kp.dim = 1024;
  for(int k = 0 ; k < n ; ++k){
    float* data = (float*)aligned_alloc(64,sizeof(float)*kp.dim);
    for(int i = 0 ; i < kp.dim ; i++)
    data[i] = u(e);
    kp.src.push_back(data);
    
    float* output = (float*)aligned_alloc(64,sizeof(float)*kp.dim);
    kp.dst.push_back(output);
  }

  float avx,cpp;
  {
    ST
    gemv::L2NormAVX(kp);
    END
    TIME("AVX2")
	avx = duration_cast<microseconds>(T2 - T1).count();
  }
  {
	ST
	for(int i  = 0 ; i < n ; i++)
      normalization(kp.src[i],kp.dst[i],kp.dim);
    END
	TIME("NORM")
	cpp = duration_cast<microseconds>(T2 - T1).count();
  }
  std::cout <<"speed up : "<<cpp/avx<<std::endl;
  for(int k = 0 ; k < n ; ++k){
    float sum = 0;
    for(int i = 0 ; i < kp.dim ; i++)
      sum += (kp.src[k][i]*kp.src[k][i]);
    sum = 1.0/sqrtf(sum);

    for(int i = 0 ;i < kp.dim ; i++){
      float data = kp.src[k][i]*sum;
      float tmp = (kp.dst[k][i] - data)/kp.dst[k][i];
      if(abs(tmp) > 0.0001)
        std::cout<<k<<"-"<<i<<" "<<kp.dst[k][i]<<" "<<data<<std::endl;
    }
    //std::cout<<k<<"-"<<i<<" "<<kp.dst[k][i]<<" "<<kp.src[k][i]*sum<<"**";
    //std::cout<<std::endl;
  }*/
  /*
  gemv::KParams<float,half> kp;
  kp.dim = 1024;
  for(int k = 0 ; k < n ; ++k){
    float* data = (float*)aligned_alloc(64,sizeof(float)*kp.dim);
    for(int i = 0 ; i < kp.dim ; i++)
    data[i] = u(e);
    kp.src.push_back(data);
    
    half* output = (half*)aligned_alloc(64,sizeof(half)*kp.dim);
    kp.dst.push_back(output);
  }

  float avx,cpp;
  {
    ST
    gemv::L2NormAVX<float,half>(kp);
    END
    TIME("AVX2")
	avx = duration_cast<microseconds>(T2 - T1).count();
  }
  {
	ST
	float * data = (float*)aligned_alloc(64,sizeof(float)*kp.dim);
	for(int i  = 0 ; i < n ; i++){
      normalization(kp.src[i],data,kp.dim);
	  float2half(data,kp.dst[i],kp.dim);
	}
    END
	TIME("NORM")
	cpp = duration_cast<microseconds>(T2 - T1).count();
  }
  std::cout <<"speed up : "<<cpp/avx<<std::endl;
  */
  /*for(int k = 0 ; k < n ; ++k){
    float sum = 0;
    for(int i = 0 ; i < kp.dim ; i++)
      sum += (kp.src[k][i]*kp.src[k][i]);
    sum = 1.0/sqrt(sum);

    for(int i = 0 ;i < kp.dim ; i++){
      float data = kp.src[k][i]*sum;
      float tmp = (half2Float(kp.dst[k][i]) - data)/kp.dst[k][i];
      if(abs(tmp) > 0.0001)
        std::cout<<k<<"-"<<i<<" "<<half2Float(kp.dst[k][i])<<" "<<data<<std::endl;
    }
  }*/
  /*
  gemv::KParams<half,half> kp;
  kp.dim = 160;
  for(int k = 0 ; k < n ; ++k){
    half* data = (half*)aligned_alloc(64,sizeof(half)*kp.dim);
    for(int i = 0 ; i < kp.dim ; i++)
    data[i] = float2Half(u(e));
    kp.src.push_back(data);

    half* output = (half*)aligned_alloc(64,sizeof(half)*kp.dim);
    kp.dst.push_back(output);
  }
  
  float avx,cpp;
  {
    ST
    gemv::L2NormAVX<half,half>(kp);
    END
    TIME("AVX2")
	avx = duration_cast<microseconds>(T2 - T1).count();
  }
  {
	ST
	float * fdata = (float*)aligned_alloc(64,sizeof(float)*kp.dim);
	float * data = (float*)aligned_alloc(64,sizeof(float)*kp.dim);
	for(int i  = 0 ; i < n ; i++){
      half2float(kp.src[i],fdata,kp.dim);
	  normalization(fdata,data,kp.dim);
	  float2half(data,kp.dst[i],kp.dim);
	}
    END
	TIME("NORM")
	cpp = duration_cast<microseconds>(T2 - T1).count();
  }
  std::cout <<"speed up : "<<cpp/avx<<std::endl;
  for(int k = 0 ; k < n ; ++k){
    float sum = 0;
    for(int i = 0 ; i < kp.dim ; i++){
      float tmp = half2Float(kp.src[k][i]);
      sum += tmp*tmp;
    }
    sum = 1.0/sqrt(sum);
    for(int i = 0 ;i < kp.dim ; i++){
      float data = half2Float(kp.src[k][i])*sum;
      float tmp = (half2Float(kp.dst[k][i]) - data)/data;
      if(abs(tmp) > 0.0005)
        std::cout<<half2Float(kp.dst[k][i])<<" "<<data<<" "<<tmp<<std::endl;
    }
  }*/
  return 0;
}
