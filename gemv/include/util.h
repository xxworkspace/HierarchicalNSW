
#pragma once

#include <iostream>
#include <cstdlib>
#include <random>
#include <chrono>
#include <sstream>
#include <cstring>
#include "gemv.h"

using namespace std;
using namespace std::chrono;

using std::default_random_engine;
using std::uniform_real_distribution;

using namespace gemv;

void float2half(float *f,half *h,int number);
half float2half(float f);

void half2float(half *h,float *f,int numbers);
float half2float(half h);

void normalization(float*,float*,int number);

template<typename T>
T to_t(string str) {
  T val;
  std::stringstream sstr;
  sstr << str;
  sstr >> val;
  return val;
}

float _IP(float*,float*,size_t dim);
float _L2(float*,float*,size_t dim);

float _IP(int8*,int8*,size_t dim);
float _L2(int8*,int8*,size_t dim);

float _IP(uint8*,uint8*,size_t dim);
float _L2(uint8*,uint8*,size_t dim);

float _IP(half*,half*,size_t dim);
float _L2(half*,half*,size_t dim);

float _IP(half*,float*,size_t dim);
float _L2(half*,float*,size_t dim);

#define ST() high_resolution_clock::time_point T1 = high_resolution_clock::now();
#define END() high_resolution_clock::time_point T2 = high_resolution_clock::now();
#define TIME(log) std::cout<<log<<" : "<< duration_cast<microseconds>(T2 - T1).count()/1000.0 << "ms." << std::endl;
