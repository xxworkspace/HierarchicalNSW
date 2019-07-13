

#include "util.h"

inline half float2half(float f){
  unsigned short ret;
  unsigned int x,u, remainder, shift, lsb, lsb_s1, lsb_m1;
  unsigned int sign, exponent, mantissa;
  
  memcpy(&x,&f,sizeof(float));
  u = (x & 0x7fffffff);
  // Get rid of +NaN/-NaN case first.
  if (u > 0x7f800000) {
    ret = 0x7fffU;
  }
  sign = ((x >> 16) & 0x8000);
  // Get rid of +Inf/-Inf, +0/-0.
  if (u > 0x477fefff) {
    ret = sign | 0x7c00U;
  }
  if (u < 0x33000001) {
    ret = (sign | 0x0000);
  }
  exponent = ((u >> 23) & 0xff);
  mantissa = (u & 0x7fffff);

  if (exponent > 0x70) {
    shift = 13;
    exponent -= 0x70;
  } else {
    shift = 0x7e - exponent;
    exponent = 0;
    mantissa |= 0x800000;
  }
  lsb = (1 << shift);
  lsb_s1 = (lsb >> 1);
  lsb_m1 = (lsb - 1);
  // Round to nearest even
  remainder = (mantissa & lsb_m1);
  mantissa >>= shift;
  if (remainder > lsb_s1 || (remainder == lsb_s1 && (mantissa & 0x1))) {
    ++mantissa;
    if (!(mantissa & 0x3ff)) {
      ++exponent;
      mantissa = 0;
    }
  }
  ret = (sign | (exponent << 10) | mantissa);
  return ret;
}

void float2half(float *f,half * h,int number){
  for(int i = 0 ; i < number ; i++)
    h[i] = float2half(f[i]);
}

float half2float(half h){
  unsigned short x = h;
  unsigned sign = ((x >> 15) & 1);
  unsigned exponent = ((x >> 10) & 0x1f);
  unsigned mantissa = ((x & 0x3ff) << 13);

  if (exponent == 0x1f) {  /* NaN or Inf */
    mantissa = (mantissa ? (sign = 0, 0x7fffff) : 0);
    exponent = 0xff;
  } 
  else if (!exponent) {  /* Denorm or Zero */
    if (mantissa) {
      unsigned int msb;
      exponent = 0x71;
      do {
        msb = (mantissa & 0x400000);
        mantissa <<= 1;  /* normalize */
        --exponent;
      } while (!msb);
      mantissa &= 0x7fffff;  /* 1.mantissa is implicit */
    }
  }
  else{
    exponent += 0x70;
  }
  int temp = ((sign << 31) | (exponent << 23) | mantissa);
  return *((float*)((void*)&temp));
}

void half2float(half *h,float *f,int number){
  for(int i = 0 ; i < number ; i++)
    f[i] = half2float(h[i]);
}

void normalization(float* f,float* d,int number){
  float sum = 0;
  for(int i = 0 ; i < number ; i++)
    sum += f[i]*f[i];
  sum = 1.0/sqrtf(sum);
  
  for(int i = 0 ; i < number ; i++)
    d[i] = f[i]*sum;
}

void normalization(float* f,half* h,float*tmp,int number){
  float sum = 0;
  for(int i = 0 ; i < number ; i++)
    sum += f[i]*f[i];
  sum = 1/sqrtf(sum);
  
  for(int i = 0 ; i < number ; i++)
    tmp[i] = f[i]*sum;

  float2half(tmp,h,number);
}

void normalization(half* f,half* h,float*tmp,int number){
  half2float(f,tmp,number);
  float sum = 0;
  for(int i = 0 ; i < number ; i++)
    sum += tmp[i]*tmp[i];
  sum = 1.0/sqrtf(sum);

  for(int i = 0 ; i < number ; i++)
    tmp[i] *= sum;

  float2half(tmp,h,number);
}

float _IP(float*p,float*q,size_t dim){
  float sum = 0;
  for(int i = 0 ; i < dim ; i++)
    sum += p[i]*q[i];
  return -sum;
}

float _L2(float*p,float*q,size_t dim){
  float sum = 0;
  for(int i = 0 ; i < dim ; i++){
    float tmp = p[i] - q[i];
    sum += tmp*tmp;
  }
  return sqrtf(sum);
}

float _IP(int8*p,int8*q,size_t dim){
  int sum = 0;
  for(int i = 0 ; i < dim ; i++){
    int tmp = p[i]*q[i];
	sum += tmp;
  }
  return -sum;
}

float _L2(int8*p,int8*q,size_t dim){
  int sum = 0;
  for(int i = 0 ; i < dim ; i++){
    int tmp = p[i] - q[i];
    sum += tmp*tmp;
  }
  return sqrtf(float(sum));
}

float _IP(uint8*p,uint8*q,size_t dim){
  int sum = 0;
  for(int i = 0 ; i < dim ; i++){
    int tmp = p[i]*q[i];
	sum += tmp;
  }
  return -sum;
}

float _L2(uint8*p,uint8*q,size_t dim){
  int sum = 0;
  for(int i = 0 ; i < dim ; i++){
    int tmp = p[i] - q[i];
    sum += tmp*tmp;
  }
  return sqrtf(float(sum));
}

float _IP(half*p,half*q,size_t dim){
  float sum = 0;
  for(int i = 0 ;i < dim ; i++){
    float tmp = half2float(p[i])*half2float(q[i]);
    sum += tmp;
  }
  return -sum;
}

float _L2(half*p,half*q,size_t dim){
  float sum = 0;
  for(int i = 0 ;i < dim ; i++){
    float tmp = half2float(p[i]) - half2float(q[i]);
    sum += tmp*tmp;
  }
  return sqrtf(sum);
}

float _IP(half *p,float *q,size_t dim){
  float sum = 0;
  for(int i = 0 ;i < dim ; i++){
    float tmp = half2float(p[i]) * q[i];
    sum += tmp;
  }
  return -sum;
}

float _L2(half *p,float *q,size_t dim){
  float sum = 0;
  for(int i = 0 ;i < dim ; i++){
    float tmp = half2float(p[i]) - q[i];
    sum += tmp*tmp;
  }
  return sqrtf(sum);
}
