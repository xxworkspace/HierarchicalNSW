/*
 * Copyright (c) BIGO, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

template<typename T>
string to_string(T value) {
  string str;
  stringstream sstr;
  sstr << value;
  sstr >> str;
  return str;
}

template<typename T>
T to_t(string str) {
  T val;
  stringstream sstr;
  sstr << str;
  sstr >> val;
  return val;
}

void addi(ofstream& of, string i, bool disable = false) {
  if (disable == false)
    of << "    \"" + i + "\\t\\n\"" + "\n";
}

int main(){
  std::vector<std::vector<string>> ats{
    {"L2Norm","Half","AVX512","16"},
    {"L2Norm","Float","AVX512","16"},
    {"L2Norm","Float_Half","AVX512","16"},

    {"L2Norm","Half","AVX256","16"},
    {"L2Norm","Float","AVX256","16"},
    {"L2Norm","Float_Half","AVX256","16"},

    {"Float2Half","Float","AVX512","16"},
    {"Float2Half","Float","AVX256","16"},
  };
  ofstream hfile("AVXKernel.h");
  ofstream srcfile("AVXKernel.cc");
  
  hfile
    << "/*\n"
    " * Copyright (c) BIGO, Inc. and its affiliates.\n"
    " * All rights reserved.\n"
    " * This source code is licensed under the BSD-style license found in the\n"
    " * LICENSE file in the root directory of this source tree.\n"
    " */\n";

  srcfile
    << "/*\n"
    " * Copyright (c) BIGO, Inc. and its affiliates.\n"
    " * All rights reserved.\n"
    " * This source code is licensed under the BSD-style license found in the\n"
    " * LICENSE file in the root directory of this source tree.\n"
    " */\n";

  hfile << "#pragma once" << std::endl;
  hfile << "#include \"kernel.h\"" << std::endl;
  srcfile << std::endl << "#include \"AVXKernel.h\"" << std::endl<<std::endl;

  hfile << "namespace gemv{" << std::endl << std::endl;
  srcfile << "namespace gemv{" << std::endl << std::endl;

  string hinfo = "typedef void (*KERNEL)(void*);\nstruct KernelInfo{\nprivate:\n  std::map<std::string,std::vector<KERNEL>> KernelPtr = {\n";

  vector<string> func_names;
  for(auto line : ats){
    int N = to_t<int>(line[3]);
    hinfo += "    {\"" + line[0] + "_" + line[1] + "_" + line[2] + "\",{\n      nullptr,\n";
    for(int n = 1 ; n <= N ; ++n){
      string func_name = "void " + line[0] + "_" + line[1] + "_" + line[2] + "_" + to_string(n) + "x1" + "(void*sp)";
      srcfile << func_name << "{\n  asm volatile(\n";
      func_names.push_back(func_name);
      hinfo += "      "+ line[0] + "_" + line[1] + "_" + line[2] + "_" + to_string(n) + "x1,\n";
    
      addi(srcfile, "mov r15,%[sp]");
      srcfile << "    // copy" << std::endl;
      srcfile << "    // dim" << std::endl;
      addi(srcfile, "mov r8,[r15]");
      srcfile << "    // src" << std::endl;
      addi(srcfile, "mov r9,[r15 + 8]");
      srcfile << "    // dst" << std::endl;
      addi(srcfile, "mov r10,[r15 + 16]");
	  
      srcfile << std::endl;

      if(line[0] == "L2Norm"){
        if(line[2] == "AVX512"){
          std::vector<std::string> r_file;
          std::vector<std::string> rr_file;

          for (int k = 8; k < 12; ++k)
            r_file.push_back("zmm" + to_string(k));
          for(int k = 0 ; k < 8 ; ++k)
            rr_file.push_back("zmm" + to_string(k));

          addi(srcfile,"add r8,r8");
          if(line[1] != "Half")
            addi(srcfile,"add r8,r8");

          for(int k = 0 ; k < n ; k += 4){
            addi(srcfile,"mov rax,0");
            for(int m =  k ; m < std::min(n,k + 4) ; ++m)
              addi(srcfile, "vxorps " + r_file[m - k] + "," + r_file[m - k] + "," + r_file[m - k]);
            srcfile<<std::endl;
            addi(srcfile, "loop_first_" + to_string(k) + "%=:");
            srcfile<<std::endl;

            for(int m = k ; m < std::min(n,k + 4) ; ++m){
              if(m == k)
                addi(srcfile,"mov r11,[r9 + " + to_string(m*8) + "]");

              if(n > 1 && m < (n - 1)){
                srcfile<<"    //prefetch"<<std::endl;
                addi(srcfile,"mov r13,[r9 + " + to_string(m*8 + 8) + "]");
                addi(srcfile,"prefetchnta [r13 + rax]");
                if(line[1] != "Half")
                  addi(srcfile,"prefetchnta [r13 + rax + 64]");
              }

              srcfile<<std::endl;
              if(line[1] == "Half"){
                addi(srcfile,"vcvtph2ps " + rr_file[(m-k)*2 + 0] + ",YMMWORD PTR [r11 + rax]");
                addi(srcfile,"vcvtph2ps " + rr_file[(m-k)*2 + 1] + ",YMMWORD PTR [r11 + rax + 32]");
              }else{
                addi(srcfile,"vmovaps " + rr_file[(m-k)*2 + 0] + ",ZMMWORD PTR [r11 + rax]");
                addi(srcfile,"vmovaps " + rr_file[(m-k)*2 + 1] + ",ZMMWORD PTR [r11 + rax + 64]");
              }
              if(m < std::min(n,k + 4) - 1)
                addi(srcfile,"mov r11,r13");
            }
            srcfile<<std::endl;

            for(int m = k ; m < std::min(n,k + 4) ; ++m)
              addi(srcfile,"vfmadd231ps " + r_file[m-k] + "," + rr_file[(m-k)*2 + 0] + "," + rr_file[(m-k)*2 + 0]);
            
            for(int m = k ; m < std::min(n,k + 4) ; ++m)
              addi(srcfile,"vfmadd231ps " + r_file[m-k] + "," + rr_file[(m-k)*2 + 1] + "," + rr_file[(m-k)*2 + 1]);
            
            srcfile<<std::endl;
            if(line[1] == "Half")
              addi(srcfile,"add rax,64");
            else
              addi(srcfile,"add rax,128");
            addi(srcfile,"cmp rax,r8");
            addi(srcfile,"jl loop_first_" + to_string(k) + "%=");
            srcfile<<std::endl;
            
            for(int m = k ; m < std::min(n,k + 4) ; ++m){
              addi(srcfile, "vextractf32x8 ymm0," + r_file[m - k] + ",0");
              addi(srcfile, "vextractf32x8 ymm1," + r_file[m - k] + ",1");
              addi(srcfile, "vaddps ymm2,ymm1,ymm0");
              addi(srcfile, "vhaddps ymm3,ymm2,ymm2");
              addi(srcfile, "vhaddps ymm4,ymm3,ymm3");
              addi(srcfile, "vextractf32x4 xmm5,ymm4,0");
              addi(srcfile, "vextractf32x4 xmm6,ymm4,1");
              addi(srcfile, "addss xmm6,xmm5");
              addi(srcfile, "sqrtss xmm6,xmm6");
              addi(srcfile, "vbroadcastss " + r_file[m - k] + ",xmm6");
              srcfile << std::endl;
            }
            
            addi(srcfile,"mov rax,0");
            addi(srcfile,"mov rbx,0");
            addi(srcfile, "loop_second_" + to_string(k) + "%=:");
            srcfile << std::endl;
            for(int m = k ; m < std::min(n,k + 4) ; ++m){
              if(m == k)
                addi(srcfile,"mov r11,[r9 + " + to_string(m*8) + "]");

              if(n > 1 && m < (n - 1)){
                srcfile<<"    //prefetch"<<std::endl;
                addi(srcfile,"mov r13,[r9 + " + to_string(m*8 + 8) + "]");
                addi(srcfile,"prefetchnta [r13 + rax]");
                if(line[1] != "Half")
                  addi(srcfile,"prefetchnta [r13 + rax + 64]");
              }
              srcfile<<std::endl;
              if(line[1] == "Half"){
                addi(srcfile,"vcvtph2ps " + rr_file[(m-k)*2 + 0] + ",YMMWORD PTR [r11 + rax]");
                addi(srcfile,"vcvtph2ps " + rr_file[(m-k)*2 + 1] + ",YMMWORD PTR [r11 + rax + 32]");
              }else{
                addi(srcfile,"vmovaps " + rr_file[(m-k)*2 + 0] + ",ZMMWORD PTR [r11 + rax]");
                addi(srcfile,"vmovaps " + rr_file[(m-k)*2 + 1] + ",ZMMWORD PTR [r11 + rax + 64]");
              }
              
              if(n > 1 && m < n - 1)
                addi(srcfile,"mov r11,r13");
            }

            srcfile<<std::endl;
            for(int m = k ; m < std::min(n,k + 4) ; ++m){
              addi(srcfile,"vdivps " + rr_file[(m-k)*2 + 0] + "," + rr_file[(m-k)*2 + 0] + "," + r_file[(m-k)]);
              addi(srcfile,"vdivps " + rr_file[(m-k)*2 + 1] + "," + rr_file[(m-k)*2 + 1] + "," + r_file[(m-k)]);
            }
            srcfile<<std::endl;
            for(int m = k ; m < std::min(n,k + 4) ; ++m){
              if(m == k)
                addi(srcfile,"mov r12,[r10 + " + to_string(m*8) + "]");

              if(n > 1 && m < (n - 1)){
                srcfile<<"      //prefetch"<<std::endl;
                addi(srcfile,"mov r13,[r10 + " + to_string(m*8 + 8) + "]");
                addi(srcfile,"prefetchnta [r13 + rax]");
                if(line[1] == "Float")
                  addi(srcfile,"prefetchnta [r13 + rax + 64]");
              }
              srcfile<<std::endl;
              if(line[1] != "Float"){
                addi(srcfile,"vcvtps2ph YMMWORD PTR [r12 + rbx],"    + rr_file[(m-k)*2 + 0] + ",0x0");
                addi(srcfile,"vcvtps2ph YMMWORD PTR [r12 + rbx + 32]," + rr_file[(m-k)*2 + 1] + ",0x0");
              }else {
                addi(srcfile,"vmovaps ZMMWORD PTR [r12 + rbx],"    + rr_file[(m-k)*2 + 0]);
                addi(srcfile,"vmovaps ZMMWORD PTR [r12 + rbx + 64]," + rr_file[(m-k)*2 + 1]);
              }
        
              if(n > 1 && m < n - 1)
                addi(srcfile,"mov r12,r13");
            }
            
            srcfile<<std::endl;
            if(line[1] == "Half")
              addi(srcfile,"add rax,64");
            else
              addi(srcfile,"add rax,128");
            
            if(line[1] != "Float")
              addi(srcfile,"add rbx,64");
            else
              addi(srcfile,"add rbx,128");
            
            addi(srcfile,"cmp rax,r8");
            addi(srcfile,"jl loop_second_" + to_string(k) + "%=");
            srcfile<<std::endl;
          }
        }else if(line[2] == "AVX256"){
          std::vector<std::string> r_file;
          std::vector<std::string> rr_file;
          
          for (int k = 8; k < 10; ++k)
            r_file.push_back("ymm" + to_string(k));
          for(int k = 0 ; k < 8 ; ++k)
            rr_file.push_back("ymm" + to_string(k));
          
		  addi(srcfile,"add r8,r8");
            if(line[1] != "Half")
              addi(srcfile,"add r8,r8");

          for(int k = 0 ; k < n ; k += 2){
            addi(srcfile,"mov rax,0");
            for(int m =  k ; m < std::min(n,k + 2) ; ++m)
              addi(srcfile, "vxorps " + r_file[m - k] + "," + r_file[m - k] + "," + r_file[m - k]);
            srcfile<<std::endl;
      
            addi(srcfile, "loop_first_" + to_string(k) + "%=:");
            srcfile<<std::endl;
            for(int m = k ; m < std::min(n,k + 2) ; ++m){
              if(m == k)
                addi(srcfile,"mov r11,[r9 +" + to_string(m*8) + "]");

              if(n > 1 && m < (n - 1)){
                srcfile<<"    //prefetch"<<std::endl;
                addi(srcfile,"mov r13,[r9 + " + to_string(m*8 + 8) + "]");
                addi(srcfile,"prefetchnta [r13 + rax]");
                if(line[1] != "Hlaf")
                  addi(srcfile,"prefetchnta [r13 + rax + 64]");
              }
              srcfile<<std::endl;

              if(line[1] == "Half"){
                addi(srcfile,"vcvtph2ps " + rr_file[(m-k)*4 + 0] + ",XMMWORD PTR [r11 + rax]");
                addi(srcfile,"vcvtph2ps " + rr_file[(m-k)*4 + 1] + ",xMMWORD PTR [r11 + rax + 16]");
                addi(srcfile,"vcvtph2ps " + rr_file[(m-k)*4 + 2] + ",XMMWORD PTR [r11 + rax + 32]");
                addi(srcfile,"vcvtph2ps " + rr_file[(m-k)*4 + 3] + ",xMMWORD PTR [r11 + rax + 48]");
              }else{
                addi(srcfile,"vmovaps " + rr_file[(m-k)*4 + 0] + ",YMMWORD PTR [r11 + rax + 0]");
                addi(srcfile,"vmovaps " + rr_file[(m-k)*4 + 1] + ",YMMWORD PTR [r11 + rax + 32]");
                addi(srcfile,"vmovaps " + rr_file[(m-k)*4 + 2] + ",YMMWORD PTR [r11 + rax + 64]");
                addi(srcfile,"vmovaps " + rr_file[(m-k)*4 + 3] + ",YMMWORD PTR [r11 + rax + 96]");
              }
              
              if(m < std::min(n,k + 4) - 1)
                addi(srcfile,"mov r11,r13");
            }
            
            srcfile << std::endl;
            for(int m = k ; m < std::min(n,k + 2) ; ++m)
              addi(srcfile,"vfmadd231ps " + r_file[m-k] + "," + rr_file[(m-k)*4 + 0] + "," + rr_file[(m-k)*4 + 0]);
            
            for(int m = k ; m < std::min(n,k + 2) ; ++m)
              addi(srcfile,"vfmadd231ps " + r_file[m-k] + "," + rr_file[(m-k)*4 + 1] + "," + rr_file[(m-k)*4 + 1]);
            
            for(int m = k ; m < std::min(n,k + 2) ; ++m)
              addi(srcfile,"vfmadd231ps " + r_file[m-k] + "," + rr_file[(m-k)*4 + 2] + "," + rr_file[(m-k)*4 + 2]);
            
            for(int m = k ; m < std::min(n,k + 2) ; ++m)
              addi(srcfile,"vfmadd231ps " + r_file[m-k] + "," + rr_file[(m-k)*4 + 3] + "," + rr_file[(m-k)*4 + 3]);
            
            srcfile << std::endl;
            if(line[1] == "Half")
              addi(srcfile,"add rax,64");
            else
              addi(srcfile,"add rax,128");

            addi(srcfile,"cmp rax,r8");
            addi(srcfile,"jl loop_first_" + to_string(k) + "%=");
            srcfile<<std::endl;
            
            for(int m = k ; m < std::min(n,k + 2) ; ++m){
              addi(srcfile, "vhaddps ymm0," + r_file[m-k] + "," + r_file[m-k]);
              addi(srcfile, "vhaddps ymm1,ymm0,ymm0");
              addi(srcfile, "vextractf128 xmm2,ymm1,0");
              addi(srcfile, "vextractf128 xmm3,ymm1,1");
              addi(srcfile, "addss xmm3,xmm2");
              addi(srcfile, "sqrtss xmm3,xmm3");
              addi(srcfile, "vbroadcastss " + r_file[m-k] + ",xmm3");
              srcfile<<std::endl;
            }
            
            addi(srcfile,"mov rax,0");
            addi(srcfile,"mov rbx,0");
            addi(srcfile, "loop_second_" + to_string(k) + "%=:");
            srcfile<<std::endl;
            for(int m = k ; m < std::min(n,k + 2) ; ++m){
              if(m == k)
                addi(srcfile,"mov r11,[r9+" + to_string(m*8) + "]");

              if(n > 1 && m < (n - 1)){
                srcfile<<"    //prefetch"<<std::endl;
                addi(srcfile,"mov r13,[r9 + " + to_string(m*8 + 8) + "]");
                addi(srcfile,"prefetchnta [r13 + rax]");
                if(line[1] != "Half")
                  addi(srcfile,"prefetchnta [r13 + rax + 64]");
              }
              srcfile<<std::endl;
              if(line[1] == "Half"){
                addi(srcfile,"vcvtph2ps " + rr_file[(m-k)*4 + 0] + ",XMMWORD PTR [r11 + rax + 0]");
                addi(srcfile,"vcvtph2ps " + rr_file[(m-k)*4 + 1] + ",XMMWORD PTR [r11 + rax + 16]");
                addi(srcfile,"vcvtph2ps " + rr_file[(m-k)*4 + 2] + ",XMMWORD PTR [r11 + rax + 32]");
                addi(srcfile,"vcvtph2ps " + rr_file[(m-k)*4 + 3] + ",XMMWORD PTR [r11 + rax + 48]");
              }else{
                addi(srcfile,"vmovaps " + rr_file[(m-k)*4 + 0] + ",YMMWORD PTR [r11 + rax + 0]");
                addi(srcfile,"vmovaps " + rr_file[(m-k)*4 + 1] + ",YMMWORD PTR [r11 + rax + 32]");
                addi(srcfile,"vmovaps " + rr_file[(m-k)*4 + 2] + ",YMMWORD PTR [r11 + rax + 64]");
                addi(srcfile,"vmovaps " + rr_file[(m-k)*4 + 3] + ",YMMWORD PTR [r11 + rax + 96]");
              }
              if(n > 1 && m < n - 1)
              addi(srcfile,"mov r11,r13");
            }
            
            srcfile<<std::endl;
            for(int m = k ; m < std::min(n,k + 2) ; ++m){
              addi(srcfile,"vdivps " + rr_file[(m-k)*4 + 0] + "," + rr_file[(m-k)*4 + 0] + "," + r_file[(m-k)]);
              addi(srcfile,"vdivps " + rr_file[(m-k)*4 + 1] + "," + rr_file[(m-k)*4 + 1] + "," + r_file[(m-k)]);
              addi(srcfile,"vdivps " + rr_file[(m-k)*4 + 2] + "," + rr_file[(m-k)*4 + 2] + "," + r_file[(m-k)]);
              addi(srcfile,"vdivps " + rr_file[(m-k)*4 + 3] + "," + rr_file[(m-k)*4 + 3] + "," + r_file[(m-k)]);
            }
            srcfile<<std::endl;
            for(int m = k ; m < std::min(n,k + 2) ; ++m){
              if(m == k)
                addi(srcfile,"mov r12,[r10 + " + to_string(m*8) + "]");

              if(n > 1 && m < (n - 1)){
                srcfile<<"    //prefetch"<<std::endl;
                addi(srcfile,"mov r13,[r10 + " + to_string(m*8 + 8) + "]");
                addi(srcfile,"prefetchnta [r13 + rax]");
                if(line[1] == "Float")
                  addi(srcfile,"prefetchnta [r13 + rax + 64]");
              }
              srcfile<<std::endl;
              if(line[1] != "Float"){
                addi(srcfile,"vcvtps2ph XMMWORD PTR [r12 + rbx + 0],"  + rr_file[(m-k)*4 + 0] + ",0x0");
                addi(srcfile,"vcvtps2ph XMMWORD PTR [r12 + rbx + 16]," + rr_file[(m-k)*4 + 1] + ",0x0");
                addi(srcfile,"vcvtps2ph XMMWORD PTR [r12 + rbx + 32]," + rr_file[(m-k)*4 + 2] + ",0x0");
                addi(srcfile,"vcvtps2ph XMMWORD PTR [r12 + rbx + 48]," + rr_file[(m-k)*4 + 3] + ",0x0");
              }else {
                addi(srcfile,"vmovaps YMMWORD PTR [r12 + rbx + 0],"  + rr_file[(m-k)*4 + 0]);
                addi(srcfile,"vmovaps YMMWORD PTR [r12 + rbx + 32]," + rr_file[(m-k)*4 + 1]);
                addi(srcfile,"vmovaps YMMWORD PTR [r12 + rbx + 64]," + rr_file[(m-k)*4 + 2]);
                addi(srcfile,"vmovaps YMMWORD PTR [r12 + rbx + 96]," + rr_file[(m-k)*4 + 3]);
              }
              if(n > 1 && m < n - 1)
                addi(srcfile,"mov r12,r13");
            }
            
            srcfile<<std::endl;
            if(line[1] == "Half")
              addi(srcfile,"add rax,64");
            else
              addi(srcfile,"add rax,128");
            
            if(line[1] != "Float")
              addi(srcfile,"add rbx,64");
            else
              addi(srcfile,"add rbx,128");
            
            addi(srcfile,"cmp rax,r8");
            addi(srcfile,"jl loop_second_" + to_string(k) +"%=");
            srcfile<<std::endl;
          }
        }
      }else if(line[0] == "Float2Half"){
        if(line[2] == "AVX512"){
          std::vector<string> r_file;
          for(int k = 0 ; k < 8 ; ++k)
            r_file.push_back("zmm" + to_string(k));
          
          addi(srcfile,"add r8,r8");
          addi(srcfile,"add r8,r8");
          addi(srcfile,"mov rax,0");
          addi(srcfile,"mov rbx,0");
          addi(srcfile, "loop%=:");
          srcfile << std::endl;
          for(int k = 0 ; k < n ; k += 4){
            for(int m = k ; m < std::min(n,k + 4) ; ++m){
              if(m == 0)
                addi(srcfile,"mov r11,[r9]");

              if(n > 1 && m < (n - 1)){
                srcfile<<"    //prefetch"<<std::endl;
                addi(srcfile,"mov r13,[r9 + " + to_string(m*8 + 8) + "]");
                addi(srcfile,"prefetchnta [r13 + rax]");
                addi(srcfile,"prefetchnta [r13 + rax + 64]");
              }
              srcfile<<std::endl;
              addi(srcfile,"vmovaps " + r_file[(m-k)*2 + 0] + ",ZMMWORD PTR [r11 + rax]");
              addi(srcfile,"vmovaps " + r_file[(m-k)*2 + 1] + ",ZMMWORD PTR [r11 + rax + 64]");

              if(n > 1 && m < n - 1)
                addi(srcfile,"mov r11,r13");
            }
            srcfile<<std::endl;
            for(int m = k ; m < std::min(n,k + 4) ; ++m){
              if(m == 0)
                addi(srcfile,"mov r12,[r10]");
              
              if(n > 1 && m < (n - 1)){
                srcfile<<"    //prefetch"<<std::endl;
                addi(srcfile,"mov r13,[r10 + " + to_string(m*8 + 8) + "]");
                addi(srcfile,"prefetchnta [r13 + rbx]");
              }
              srcfile<<std::endl;
              addi(srcfile,"vcvtps2ph YMMWORD PTR [r12 + rbx],"    + r_file[(m-k)*2 + 0] + ",0x0");
              addi(srcfile,"vcvtps2ph YMMWORD PTR [r12 + rbx + 32]," + r_file[(m-k)*2 + 1] + ",0x0");
              if(n > 1 && m < n - 1)
                addi(srcfile,"mov r12,r13");
            }
          }
          srcfile<<std::endl;
          addi(srcfile,"add rax,128");
          addi(srcfile,"add rbx,64");
          addi(srcfile,"cmp rax,r8");
          addi(srcfile,"jl loop%=");
          srcfile<<std::endl;
        }else if(line[2] == "AVX256"){
          std::vector<string> r_file;
          for(int k = 0 ; k < 16 ; ++k)
            r_file.push_back("ymm" + to_string(k));

          addi(srcfile,"add r8,r8");
          addi(srcfile,"add r8,r8");
          addi(srcfile,"mov rax,0");
          addi(srcfile,"mov rbx,0");
          addi(srcfile, "loop%=:");
          srcfile << std::endl;
          for(int k = 0 ; k < n ; k += 4){
            for(int m = k ; m < std::min(n,k + 4) ; ++m){
              if(m == 0)
                addi(srcfile,"mov r11,[r9]");

              if(n > 1 && m < (n - 1)){
                srcfile << "    //prefetch"<<std::endl;
                addi(srcfile,"mov r13,[r9 + " + to_string(m*8 + 8) + "]");
                addi(srcfile,"prefetchnta [r13 + rax]");
                addi(srcfile,"prefetchnta [r13 + rax + 64]");
              }
              srcfile<<std::endl;

              addi(srcfile,"vmovaps " + r_file[(m-k)*4 + 0] + ",YMMWORD PTR [r11 + rax]");
              addi(srcfile,"vmovaps " + r_file[(m-k)*4 + 1] + ",YMMWORD PTR [r11 + rax + 32]");
              addi(srcfile,"vmovaps " + r_file[(m-k)*4 + 2] + ",YMMWORD PTR [r11 + rax + 64]");
              addi(srcfile,"vmovaps " + r_file[(m-k)*4 + 3] + ",YMMWORD PTR [r11 + rax + 96]");
              
              if(n > 1 && m < n - 1)
                addi(srcfile,"mov r11,r13");
            }
            srcfile<<std::endl;
            for(int m = k ; m < std::min(n,k + 4) ; ++m){
              if(m == 0)
                addi(srcfile,"mov r12,[r10]");
              
              if(n > 1 && m < (n - 1)){
                srcfile<<"    //prefetch"<<std::endl;
                addi(srcfile,"mov r13,[r10 + " + to_string(m*8 + 8) + "]");
                addi(srcfile,"prefetchnta [r13 + rbx]");
              }
              srcfile<<std::endl;
              
              addi(srcfile,"vcvtps2ph XMMWORD PTR [r12 + rbx],"    + r_file[(m-k)*4 + 0] + ",0x0");
              addi(srcfile,"vcvtps2ph XMMWORD PTR [r12 + rbx + 16]," + r_file[(m-k)*4 + 1] + ",0x0");
              addi(srcfile,"vcvtps2ph XMMWORD PTR [r12 + rbx + 32]," + r_file[(m-k)*4 + 2] + ",0x0");
              addi(srcfile,"vcvtps2ph XMMWORD PTR [r12 + rbx + 48]," + r_file[(m-k)*4 + 3] + ",0x0");
              if(n > 1 && m < n - 1)
                addi(srcfile,"mov r12,r13");
            }
          }
          srcfile<<std::endl;
          addi(srcfile,"add rax,128");
          addi(srcfile,"add rbx,64");
          addi(srcfile,"cmp rax,r8");
          addi(srcfile,"jl loop%=");
          srcfile<<std::endl;
        }
      }else{}

      srcfile
        << "  :\n"
        << "  :[sp] \"rm\"(sp)\n"
        << "  : \"r8\",\n    \"r9\",\n    \"r10\",\n"
        "    \"r11\",\n    \"r12\",\n    \"r13\",\n"
        "    \"r15\",\n    \"rax\",\n    \"rbx\",\n"
        "    \"memory\");\n";
      srcfile << "}\n\n";
    }
    hinfo += "    }},\n";
  }
  
  hinfo += "  };\npublic:\n  const std::vector<KERNEL>& operator[](std::string name){\n    return KernelPtr[name];\n  }\n};\n";
  for(auto line : func_names)
    hfile<<line<<";\n";
  hfile << std::endl<<hinfo;
  hfile << std::endl << "}" << std::endl;
  srcfile << std::endl << "}" << std::endl;
  hfile.close();
  srcfile.close();
  return 0;
}
