
#include "cpuinfo.h"
#include <iostream>

using std::cout;
using std::endl;
int main(){
  std::cout<<"FMA  : "<<gemv::hasFMA()<<std::endl;
  std::cout<<"F16C : "<<gemv::hasF16C()<<std::endl;
  std::cout<<"AVX  : "<<gemv::hasAVX()<<std::endl;
  std::cout<<"AVX2 : "<<gemv::hasAVX2()<<std::endl;
  std::cout<<"AVX512 : "<<gemv::hasAVX512()<<std::endl;
  return 0;
}
