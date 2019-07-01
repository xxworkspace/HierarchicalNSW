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
    of << "      \"" + i + "\\t\\n\"" + "\n";
}

int main() {
  std::vector<std::vector<string>> ats{
    {"GEMV","IP", "Int8", "AVX512","25"},
    {"GEMV","IP", "Half", "AVX512","25"},
    {"GEMV","IP", "Float","AVX512","25"},
    {"GEMV","IP", "Half_Float", "AVX512","25"},

    {"GEMV","L2", "Int8", "AVX512","25"},
    {"GEMV","L2", "Half", "AVX512","25"},
    {"GEMV","L2", "Float","AVX512","25"},
    {"GEMV","L2", "Half_Float", "AVX512","25"},

    {"GEMV","IP", "Int8", "AVX256","10"},
    {"GEMV","IP", "Half", "AVX256","10"},
    {"GEMV","IP", "Half_Float","AVX256","10"},
    {"GEMV","IP", "Float","AVX256","10"},

    {"GEMV","L2", "Int8", "AVX256","10"},
    {"GEMV","L2", "Half", "AVX256","10"},
    {"GEMV","L2", "Half_Float","AVX256","10"},
    {"GEMV","L2", "Float","AVX256","10"},
  };

  ofstream hfile("GemvKernel.h");
  ofstream srcfile("GemvKernel.cc");

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
  hfile << "#include \"gemv.h\"" << std::endl;
  srcfile << std::endl << "#include \"GemvKernel.h\"" << std::endl;

  hfile << "namespace gemv{" << std::endl << std::endl;
  srcfile << "namespace gemv{" << std::endl << std::endl;

  //hfile << "template <class T1,class T2,class T3,size_t Dist,size_t Inst,size_t N>" << std::endl
  //  << "void Gemv(SubParams<T1,T2,T3>*sp);" << std::endl << std::endl;
  string hinfo = "typedef void (*GEMV)(void*);\nstruct GemvInfo{\nprivate:\n  std::map<std::string,std::vector<GEMV>> GemvPtr = {\n";
  
  for (auto line : ats) {
    int N = to_t<int>(line[4]);
	hinfo += "    {\"" + line[0] + "_" + line[1] + "_" + line[2] + "_" + line[3] + "\",{\n      nullptr,\n";
    for (int n = 1; n <= N; n++) {
      string func_name = line[0] + "_"  + line[1] + "_" + line[2] + "_" + line[3] + "_" + to_string(n) + "x1";
      hinfo += "      "+ func_name + ",\n";//line[0] + "_" + line[1] + "_" + line[2] + "_" + to_string(n) + "x1,\n";

      srcfile << "void "<<func_name<<"(void*sp){\n";
	  hfile << "void "<<func_name<<"(void*);\n";

      srcfile <<"  asm volatile(" << std::endl;
      addi(srcfile, "mov r15,%[sp]");
      srcfile << "      // copy" << std::endl;
      srcfile << "      // dim" << std::endl;
      addi(srcfile, "mov r8,[r15]");
      srcfile << "      // q" << std::endl;
      addi(srcfile, "mov r9,[r15 + 8]");
      srcfile << "      // p" << std::endl;
      addi(srcfile, "mov r10,[r15 + 16]");
      srcfile << "      // r" << std::endl;
      addi(srcfile, "mov r11,[r15 + 24]");
      srcfile << std::endl;

      std::vector<string> r_file;
      if (line[3] == "AVX512") {
        for (int k = 32 - N; k < 32; ++k)
          r_file.push_back("zmm" + to_string(k));

        for (int k = 0; k < n; ++k)
          addi(srcfile, "vxorps " + r_file[k] + "," + r_file[k] + "," + r_file[k]);

        srcfile << std::endl;

        if (line[2] == "Int8") {
          addi(srcfile, "mov rax,0");
          addi(srcfile, "loop%=:");
          srcfile << std::endl;
          srcfile << "      //prefetch the first line " << std::endl;
          addi(srcfile, "mov r13,[r10]");
          addi(srcfile, "prefetchnta [r13 + rax]");
          srcfile << "      //load vector and convert" << std::endl;
          addi(srcfile, "vmovdqu8 ymm0,[r9 + rax]");
          addi(srcfile, "vmovdqu8 ymm1,[r9 + rax + 32]");
          addi(srcfile, "vpmovsxbw zmm0,ymm0");
          addi(srcfile, "vpmovsxbw zmm1,ymm1");
          srcfile << std::endl;

          for (int m = 0; m < n; ++m) {
            addi(srcfile, "mov r12,r13");

            //when n = 1 or last line
            if (n > 1 && m < n - 1) {
              srcfile << "      //prefetch the next line" << std::endl;
              addi(srcfile, "mov r13,[r10 + " + to_string(8 * m + 8) + "]");
              addi(srcfile, "prefetchnta [r13 + rax]");
            }
            srcfile << "      //compute" << std::endl;
            if (line[1] == "IP") {
              addi(srcfile, "vmovdqu8 ymm2,[r12 + rax]");
              addi(srcfile, "vmovdqu8 ymm3,[r12 + rax + 32]");
              addi(srcfile, "vpmovsxbw zmm2,ymm2");
              addi(srcfile, "vpmovsxbw zmm3,ymm3");
              addi(srcfile, "vpmaddwd zmm4,zmm2,zmm0");
              addi(srcfile, "vpmaddwd zmm5,zmm3,zmm1");
              addi(srcfile, "vpaddd " + r_file[m] + "," + r_file[m] + ",zmm4");
              addi(srcfile, "vpaddd " + r_file[m] + "," + r_file[m] + ",zmm5");
            }
            else {
              addi(srcfile, "vmovdqu8 ymm2,[r12 + rax]");
              addi(srcfile, "vmovdqu8 ymm3,[r12 + rax + 32]");
              addi(srcfile, "vpmovsxbw zmm2,ymm2");
              addi(srcfile, "vpmovsxbw zmm3,ymm3");
              addi(srcfile, "vpsubw zmm4,zmm2,zmm0");
              addi(srcfile, "vpsubw zmm5,zmm3,zmm1");
              addi(srcfile, "vpmaddwd zmm4,zmm4,zmm4");
              addi(srcfile, "vpmaddwd zmm5,zmm5,zmm5");
              addi(srcfile, "vpaddd " + r_file[m] + "," + r_file[m] + ",zmm4");
              addi(srcfile, "vpaddd " + r_file[m] + "," + r_file[m] + ",zmm5");
            }

            srcfile << std::endl;
          }

          srcfile << "      //judge the loop" << std::endl;
          addi(srcfile, "add rax,64");
          addi(srcfile, "cmp rax,r8");
          addi(srcfile, "jl loop%=");
          srcfile << std::endl;

          srcfile << "      //compute the sum" << std::endl;
          for (int m = 0; m < n; ++m) {
            addi(srcfile, "vextracti32x8 ymm0," + r_file[m] + ",0");
            addi(srcfile, "vextracti32x8 ymm1," + r_file[m] + ",1");
            addi(srcfile, "vpaddd ymm2,ymm1,ymm0");
            addi(srcfile, "vphaddd ymm3,ymm2,ymm2");
            addi(srcfile, "vphaddd ymm4,ymm3,ymm3");
            addi(srcfile, "vextracti32x4 xmm5,ymm4,0");
            addi(srcfile, "vextracti32x4 xmm6,ymm4,1");
            addi(srcfile, "paddd xmm6,xmm5");
            if (line[1] == "IP") {
              addi(srcfile, "xorps xmm0,xmm0");
              addi(srcfile, "psubd xmm0,xmm6");
              addi(srcfile, "movss DWORD PTR [r11 + " + to_string(m * 4) + "],xmm0");
			  addi(srcfile, "cvtsi2ss xmm0,DWORD PTR [r11 + " + to_string(m * 4) + "]");
              addi(srcfile, "movss DWORD PTR [r11 + " + to_string(m * 4) + "],xmm0");
            }
            else{
              addi(srcfile, "movss DWORD PTR [r11 + " + to_string(m * 4) + "],xmm6");
			  addi(srcfile, "cvtsi2ss xmm6,DWORD PTR [r11 + " + to_string(m * 4) + "]");
              addi(srcfile, "sqrtss xmm0,xmm6");
              addi(srcfile, "movss DWORD PTR [r11 + " + to_string(m * 4) + "],xmm0");
			}
            srcfile << std::endl;
          }
        }
        /*
        else if (line[2] == "Short") {
          addi(srcfile, "mul r8,2");
          addi(srcfile, "mov rax,0");
          addi(srcfile, "loop%=:");
          srcfile << std::endl;
          addi(srcfile, "mov r13,[r10]");
          addi(srcfile, "prefetchnta [r13 + rax]");

          addi(srcfile, "vmovdqu16 zmm0,[r9 + rax]");
          for (int m = 0; m < n; ++m) {
            addi(srcfile, "mov r12,r13");

            if (n > 1 && m < n - 1) {
              addi(srcfile, "mov r13,[r10 + " + to_string(8 * m) + "]");
              addi(srcfile, "prefetchnta [r13 + rax]");
            }

            addi(srcfile, "vmovdqu16 zmm1,[r12 + rax]");
            addi(srcfile, "VPMADDUBSW zmm2,zmm1,zmm0");
            addi(srcfile, "vpaddd + " + r_file[m] + "," + r_file[m] + ",zmm2");
            srcfile << std::endl;
          }

          addi(srcfile, "add rax,64");
          addi(srcfile, "cmp rax,r8");
          addi(srcfile, "jl loop%=");

          for (int m = 0; m < n; ++m) {
            addi(srcfile, "vextracti32x8 ymm0," + r_file[m] + ",0");
            addi(srcfile, "vextracti32x8 ymm1," + r_file[m] + ",1");
            addi(srcfile, "vpaddd ymm2,ymm1,ymm0");
            addi(srcfile, "vphaddd ymm3,ymm2,ymm2");
            addi(srcfile, "vphaddd ymm4,ymm3,ymm3");
            addi(srcfile, "vextracti32x4 xmm5,ymm4,0");
            addi(srcfile, "vextracti32x4 xmm6,ymm4,1");
            addi(srcfile, "paddd xmm7,xmm6,xmm5");
            addi(srcfile, "cvtss2si [r11 + " + to_string(m * 8) + "],xmm7");
          }
        }*/
        else if (line[2] == "Half") {
          addi(srcfile, "add r8,r8");
          addi(srcfile, "mov rax,0");
          addi(srcfile, "loop%=:");
          srcfile << "      //prefetch the first line" << std::endl;
          addi(srcfile, "mov r13,[r10]");
          addi(srcfile, "prefetchnta [r13 + rax]");
          srcfile << "      //load vector and convert" << std::endl;
          addi(srcfile, "vcvtph2ps zmm0,YMMWORD PTR [r9 + rax]");
          addi(srcfile, "vcvtph2ps zmm1,YMMWORD PTR [r9 + rax + 32]");
          srcfile << std::endl;
          for (int m = 0; m < n; ++m) {
            addi(srcfile, "mov r12,r13");

            if (n > 1 && m < n - 1) {
              srcfile << "      //prefetch the next line" << std::endl;
              addi(srcfile, "mov r13,[r10 + " + to_string(8 * m + 8) + "]");
              addi(srcfile, "prefetchnta [r13 + rax]");
            }

            srcfile << "      //compute" << std::endl;
            if (line[1] == "IP") {
              addi(srcfile, "vcvtph2ps zmm2,YMMWORD PTR [r12 + rax]");
              addi(srcfile, "vcvtph2ps zmm3,YMMWORD PTR [r12 + rax + 32]");
              addi(srcfile, "vfmadd231ps " + r_file[m] + ",zmm0,zmm2");
              addi(srcfile, "vfmadd231ps " + r_file[m] + ",zmm1,zmm3");
            }
            else {
              addi(srcfile, "vcvtph2ps zmm2,YMMWORD PTR [r12 + rax]");
              addi(srcfile, "vcvtph2ps zmm3,YMMWORD PTR [r12 + rax + 32]");
              addi(srcfile, "vsubps zmm4,zmm2,zmm0");
              addi(srcfile, "vsubps zmm5,zmm3,zmm1");
              addi(srcfile, "vfmadd231ps " + r_file[m] + ",zmm4,zmm4");
              addi(srcfile, "vfmadd231ps " + r_file[m] + ",zmm5,zmm5");
            }
            srcfile << std::endl;
          }
          srcfile << "      //judge the loop" << std::endl;
          addi(srcfile, "add rax,64");
          addi(srcfile, "cmp rax,r8");
          addi(srcfile, "jl loop%=");
          srcfile << std::endl;

          srcfile << "      //compute the sum" << std::endl;
          for (int m = 0; m < n; ++m) {
            addi(srcfile, "vextractf32x8 ymm0," + r_file[m] + ",0");
            addi(srcfile, "vextractf32x8 ymm1," + r_file[m] + ",1");
            addi(srcfile, "vaddps ymm2,ymm1,ymm0");
            addi(srcfile, "vhaddps ymm3,ymm2,ymm2");
            addi(srcfile, "vhaddps ymm4,ymm3,ymm3");
            addi(srcfile, "vextractf32x4 xmm5,ymm4,0");
            addi(srcfile, "vextractf32x4 xmm6,ymm4,1");
            addi(srcfile, "addss xmm6,xmm5");

            if (line[1] == "IP") {
              addi(srcfile, "xorps xmm0,xmm0");
              addi(srcfile, "subps xmm0,xmm6");
              addi(srcfile, "movss [r11 + " + to_string(m * 4) + "],xmm0");
            }
            else{
              addi(srcfile, "sqrtss xmm0,xmm6");
              addi(srcfile, "movss [r11 + " + to_string(m * 4) + "],xmm0");
			}

            srcfile << std::endl;
          }
        }
        else if (line[2] == "Float") {
          addi(srcfile, "add r8,r8");
          addi(srcfile, "add r8,r8");
          addi(srcfile, "mov rax,0");
          addi(srcfile, "loop%=:");
          srcfile << "      //prefetch the first line" << std::endl;
          addi(srcfile, "mov r13,[r10]");
          addi(srcfile, "prefetchnta [r13 + rax]");
          addi(srcfile, "prefetchnta [r13 + rax + 64]");
          srcfile << "      //load vector" << std::endl;
          addi(srcfile, "vmovups zmm0,ZMMWORD PTR [r9 + rax]");
          addi(srcfile, "vmovups zmm1,ZMMWORD PTR [r9 + rax + 64]");
          for (int m = 0; m < n; ++m) {
            addi(srcfile, "mov r12,r13");

            if (n > 1 && m < n - 1) {
              srcfile << "      //prefetch the next line" << std::endl;
              addi(srcfile, "mov r13,[r10 + " + to_string(8 * m + 8) + "]");
              addi(srcfile, "prefetchnta [r13 + rax]");
              addi(srcfile, "prefetchnta [r13 + rax + 64]");
            }

            srcfile << "      //compute" << std::endl;
            if (line[1] == "IP") {
              addi(srcfile, "vmovups zmm2,ZMMWORD PTR [r12 + rax]");
              addi(srcfile, "vmovups zmm3,ZMMWORD PTR [r12 + rax + 64]");
              addi(srcfile, "vfmadd231ps " + r_file[m] + ",zmm0,zmm2");
              addi(srcfile, "vfmadd231ps " + r_file[m] + ",zmm1,zmm3");
            }
            else {
              addi(srcfile, "vmovups zmm2,ZMMWORD PTR [r12 + rax]");
              addi(srcfile, "vmovups zmm3,ZMMWORD PTR [r12 + rax + 64]");
              addi(srcfile, "vsubps zmm4,zmm2,zmm0");
              addi(srcfile, "vsubps zmm5,zmm3,zmm1");
              addi(srcfile, "vfmadd231ps " + r_file[m] + ",zmm4,zmm4");
              addi(srcfile, "vfmadd231ps " + r_file[m] + ",zmm5,zmm5");
            }
            srcfile << std::endl;
          }
          srcfile << "      //judge the loop" << std::endl;
          addi(srcfile, "add rax,128");
          addi(srcfile, "cmp rax,r8");
          addi(srcfile, "jl loop%=");
          srcfile << std::endl;
          srcfile << "      //compute the sum" << std::endl;
          for (int m = 0; m < n; ++m) {
            addi(srcfile, "vextractf32x8 ymm0," + r_file[m] + ",0");
            addi(srcfile, "vextractf32x8 ymm1," + r_file[m] + ",1");
            addi(srcfile, "vaddps ymm2,ymm1,ymm0");
            addi(srcfile, "vhaddps ymm3,ymm2,ymm2");
            addi(srcfile, "vhaddps ymm4,ymm3,ymm3");
            addi(srcfile, "vextractf32x4 xmm5,ymm4,0");
            addi(srcfile, "vextractf32x4 xmm6,ymm4,1");
            addi(srcfile, "addss xmm6,xmm5");

            if (line[1] == "IP") {
              addi(srcfile, "xorps xmm0,xmm0");
              addi(srcfile, "subps xmm0,xmm6");
              addi(srcfile, "movss [r11 + " + to_string(m * 4) + "],xmm0");
            }
            else{
              addi(srcfile, "sqrtss xmm0,xmm6");
              addi(srcfile, "movss [r11 + " + to_string(m * 4) + "],xmm0");
			}
            srcfile << std::endl;
          }
        }
        else if (line[2] == "Half_Float") {
          addi(srcfile, "add r8,r8");
          addi(srcfile, "add r8,r8");
          addi(srcfile, "mov rax,0");
          addi(srcfile, "mov rbx,0");
          addi(srcfile, "loop%=:");
          srcfile << "      //prefetch the first line" << std::endl;
          addi(srcfile, "mov r13,[r10]");
          addi(srcfile, "prefetchnta [r13 + rbx]");
          srcfile << "      //load vector" << std::endl;
          addi(srcfile, "vmovups zmm0,ZMMWORD PTR [r9 + rax]");
          addi(srcfile, "vmovups zmm1,ZMMWORD PTR [r9 + rax + 64]");
          for (int m = 0; m < n; ++m) {
            addi(srcfile, "mov r12,r13");

            if (n > 1 && m < n - 1) {
              srcfile << "      //prefetch the next line" << std::endl;
              addi(srcfile, "mov r13,[r10 + " + to_string(8 * m + 8) + "]");
              addi(srcfile, "prefetchnta [r13 + rbx]");
            }

            srcfile << "      //compute" << std::endl;
            if (line[1] == "IP") {
              addi(srcfile, "vcvtph2ps zmm2,YMMWORD PTR [r12 + rbx]");
              addi(srcfile, "vcvtph2ps zmm3,YMMWORD PTR [r12 + rbx + 32]");
              addi(srcfile, "vfmadd231ps " + r_file[m] + ",zmm0,zmm2");
              addi(srcfile, "vfmadd231ps " + r_file[m] + ",zmm1,zmm3");
            }
            else {
              addi(srcfile, "vcvtph2ps zmm2,YMMWORD PTR [r12 + rbx]");
              addi(srcfile, "vcvtph2ps zmm3,YMMWORD PTR [r12 + rbx + 32]");
              addi(srcfile, "vsubps zmm4,zmm2,zmm0");
              addi(srcfile, "vsubps zmm5,zmm3,zmm1");
              addi(srcfile, "vfmadd231ps " + r_file[m] + ",zmm4,zmm4");
              addi(srcfile, "vfmadd231ps " + r_file[m] + ",zmm5,zmm5");
            }
            srcfile << std::endl;
          }
          srcfile << "      //judge the loop" << std::endl;
          addi(srcfile, "add rax,128");
          addi(srcfile, "add rbx,64");
          addi(srcfile, "cmp rax,r8");
          addi(srcfile, "jl loop%=");
          srcfile << std::endl;
          srcfile << "      //compute the sum" << std::endl;
          for (int m = 0; m < n; ++m) {
            addi(srcfile, "vextractf32x8 ymm0," + r_file[m] + ",0");
            addi(srcfile, "vextractf32x8 ymm1," + r_file[m] + ",1");
            addi(srcfile, "vaddps ymm2,ymm1,ymm0");
            addi(srcfile, "vhaddps ymm3,ymm2,ymm2");
            addi(srcfile, "vhaddps ymm4,ymm3,ymm3");
            addi(srcfile, "vextractf32x4 xmm5,ymm4,0");
            addi(srcfile, "vextractf32x4 xmm6,ymm4,1");
            addi(srcfile, "addss xmm6,xmm5");

            if (line[1] == "IP") {
              addi(srcfile, "xorps xmm0,xmm0");
              addi(srcfile, "subps xmm0,xmm6");
              addi(srcfile, "movss [r11 + " + to_string(m * 4) + "],xmm0");
            }
            else{
              addi(srcfile, "sqrtss xmm0,xmm6");
              addi(srcfile, "movss [r11 + " + to_string(m * 4) + "],xmm0");
			}

            srcfile << std::endl;
          }
        }
        else {}
      }
      else if (line[3] == "AVX256") {
        for (int k = 16 - N; k < 16; ++k)
          r_file.push_back("ymm" + to_string(k));

        for (int k = 0; k < n; ++k) {
          addi(srcfile, "vxorps " + r_file[k] + "," + r_file[k] + "," + r_file[k]);
        }
        srcfile << std::endl;
        if (line[2] == "Int8") {
          addi(srcfile, "mov rax,0");
          addi(srcfile, "loop%=:");
          srcfile << "      //prefetch the first line" << std::endl;
          addi(srcfile, "mov r13,[r10]");
          addi(srcfile, "prefetchnta [r13 + rax]");

          srcfile << "      //load and convert" << std::endl;
          addi(srcfile, "lddqu xmm2, [r9 + rax]");
          addi(srcfile, "lddqu xmm3, [r9 + rax + 16]");
          addi(srcfile, "lddqu xmm4, [r9 + rax + 32]");
          addi(srcfile, "lddqu xmm5, [r9 + rax + 48]");
          addi(srcfile, "vpmovsxbw ymm2,xmm2");
          addi(srcfile, "vpmovsxbw ymm3,xmm3");
          addi(srcfile, "vpmovsxbw ymm4,xmm4");
          addi(srcfile, "vpmovsxbw ymm5,xmm5");
          srcfile << std::endl;

          for (int m = 0; m < n; ++m) {
            addi(srcfile, "mov r12,r13");

            if (n > 1 && m < n - 1) {
              srcfile << "      //prefetch the next line" << std::endl;
              addi(srcfile, "mov r13,[r10 + " + to_string(8 * m + 8) + "]");
              addi(srcfile, "prefetchnta [r13 + rax]");
            }

            srcfile << "      //compute" << std::endl;
            if (line[1] == "IP") {
              addi(srcfile, "lddqu xmm0, [r12 + rax]");
              addi(srcfile, "lddqu xmm1, [r12 + rax + 16]");
              addi(srcfile, "vpmovsxbw ymm0,xmm0");
              addi(srcfile, "vpmovsxbw ymm1,xmm1");
              addi(srcfile, "vpmaddwd ymm0,ymm2,ymm0");
              addi(srcfile, "vpmaddwd ymm1,ymm3,ymm1");
              addi(srcfile, "vpaddd " + r_file[m] + "," + r_file[m] + ",ymm0");
              addi(srcfile, "vpaddd " + r_file[m] + "," + r_file[m] + ",ymm1");
              addi(srcfile, "lddqu xmm0, [r12 + rax + 32]");
              addi(srcfile, "lddqu xmm1, [r12 + rax + 48]");
              addi(srcfile, "vpmovsxbw ymm0,xmm0");
              addi(srcfile, "vpmovsxbw ymm1,xmm1");
              addi(srcfile, "vpmaddwd ymm0,ymm4,ymm0");
              addi(srcfile, "vpmaddwd ymm1,ymm5,ymm1");
              addi(srcfile, "vpaddd " + r_file[m] + "," + r_file[m] + ",ymm0");
              addi(srcfile, "vpaddd " + r_file[m] + "," + r_file[m] + ",ymm1");
            }
            else {
              addi(srcfile, "lddqu xmm0, [r12 + rax]");
              addi(srcfile, "lddqu xmm1, [r12 + rax + 16]");
              addi(srcfile, "vpmovsxbw ymm0,xmm0");
              addi(srcfile, "vpmovsxbw ymm1,xmm1");
              addi(srcfile, "vpsubw ymm0,ymm2,ymm0");
              addi(srcfile, "vpsubw ymm1,ymm3,ymm1");
              addi(srcfile, "vpmaddwd ymm0,ymm0,ymm0");
              addi(srcfile, "vpmaddwd ymm1,ymm1,ymm1");
              addi(srcfile, "vpaddd " + r_file[m] + "," + r_file[m] + ",ymm0");
              addi(srcfile, "vpaddd " + r_file[m] + "," + r_file[m] + ",ymm1");
              addi(srcfile, "lddqu xmm0, [r12 + rax + 32]");
              addi(srcfile, "lddqu xmm1, [r12 + rax + 48]");
              addi(srcfile, "vpmovsxbw ymm0,xmm0");
              addi(srcfile, "vpmovsxbw ymm1,xmm1");
              addi(srcfile, "vpsubw ymm0,ymm4,ymm0");
              addi(srcfile, "vpsubw ymm1,ymm5,ymm1");
              addi(srcfile, "vpmaddwd ymm0,ymm0,ymm0");
              addi(srcfile, "vpmaddwd ymm1,ymm1,ymm1");
              addi(srcfile, "vpaddd " + r_file[m] + "," + r_file[m] + ",ymm0");
              addi(srcfile, "vpaddd " + r_file[m] + "," + r_file[m] + ",ymm1");
            }
            srcfile << std::endl;
          }
          srcfile << "      //judge the loop" << std::endl;
          addi(srcfile, "add rax,64");
          addi(srcfile, "cmp rax,r8");
          addi(srcfile, "jl loop%=");
          srcfile << std::endl;
          srcfile << "      //compute the sum" << std::endl;
          for (int m = 0; m < n; ++m) {
            addi(srcfile, "vphaddd ymm0," + r_file[m] + "," + r_file[m]);
            addi(srcfile, "vphaddd ymm1,ymm0,ymm0");
            addi(srcfile, "vextractf128 xmm2,ymm1,0");
            addi(srcfile, "vextractf128 xmm3,ymm1,1");
            addi(srcfile, "paddd xmm3,xmm2");

            if (line[1] == "IP") {
              addi(srcfile, "xorps xmm4,xmm4");
              addi(srcfile, "psubd xmm4,xmm3");
			  addi(srcfile, "movss DWORD PTR [r11 + " + to_string(m * 4) + "],xmm4");
			  addi(srcfile, "cvtsi2ss xmm4,DWORD PTR [r11 + " + to_string(m * 4) + "]");
              addi(srcfile, "movss DWORD PTR [r11 + " + to_string(m * 4) + "],xmm4");
            }
            else{
              addi(srcfile, "movss DWORD PTR [r11 + " + to_string(m * 4) + "],xmm3");
			  addi(srcfile, "cvtsi2ss xmm3,DWORD PTR [r11 + " + to_string(m * 4) + "]");
              addi(srcfile, "sqrtss xmm4,xmm3");
              addi(srcfile, "movss DWORD PTR [r11 + " + to_string(m * 4) + "],xmm4");
			}
            srcfile << std::endl;
          }
        }
        else if (line[2] == "Half") {
          addi(srcfile, "add r8,r8");
          addi(srcfile, "mov rax,0");
          addi(srcfile, "loop%=:");
          srcfile << "      //prefetch the first line" << std::endl;
          addi(srcfile, "mov r13,[r10]");
          addi(srcfile, "prefetchnta [r13 + rax]");
          srcfile << "      //load vector and convert" << std::endl;
          addi(srcfile, "vcvtph2ps ymm2,XMMWORD PTR [r9 + rax]");
          addi(srcfile, "vcvtph2ps ymm3,XMMWORD PTR [r9 + rax + 16]");
          addi(srcfile, "vcvtph2ps ymm4,XMMWORD PTR [r9 + rax + 32]");
          addi(srcfile, "vcvtph2ps ymm5,XMMWORD PTR [r9 + rax + 48]");
          srcfile << std::endl;
          for (int m = 0; m < n; ++m) {
            addi(srcfile, "mov r12,r13");

            if (n > 1 && m < n - 1) {
              srcfile << "      //prefetch the next line" << std::endl;
              addi(srcfile, "mov r13,[r10 + " + to_string(8 * m + 8) + "]");
              addi(srcfile, "prefetchnta [r13 + rax]");
            }
            srcfile << "      //compute" << std::endl;
            if (line[1] == "IP") {
              addi(srcfile, "vcvtph2ps ymm0,XMMWORD PTR [r12 + rax]");
              addi(srcfile, "vcvtph2ps ymm1,XMMWORD PTR [r12 + rax + 16]");
              addi(srcfile, "vfmadd231ps " + r_file[m] + ",ymm2,ymm0");
              addi(srcfile, "vfmadd231ps " + r_file[m] + ",ymm3,ymm1");
              addi(srcfile, "vcvtph2ps ymm0,XMMWORD PTR [r12 + rax + 32]");
              addi(srcfile, "vcvtph2ps ymm1,XMMWORD PTR [r12 + rax + 48]");
              addi(srcfile, "vfmadd231ps " + r_file[m] + ",ymm4,ymm0");
              addi(srcfile, "vfmadd231ps " + r_file[m] + ",ymm5,ymm1");
            }
            else {
              addi(srcfile, "vcvtph2ps ymm0,XMMWORD PTR [r12 + rax]");
              addi(srcfile, "vcvtph2ps ymm1,XMMWORD PTR [r12 + rax + 16]");
              addi(srcfile, "vsubps ymm0,ymm2,ymm0");
              addi(srcfile, "vsubps ymm1,ymm3,ymm1");
              addi(srcfile, "vfmadd231ps " + r_file[m] + ",ymm0,ymm0");
              addi(srcfile, "vfmadd231ps " + r_file[m] + ",ymm1,ymm1");
              addi(srcfile, "vcvtph2ps ymm0,XMMWORD PTR [r12 + rax + 32]");
              addi(srcfile, "vcvtph2ps ymm1,XMMWORD PTR [r12 + rax + 48]");
              addi(srcfile, "vsubps ymm0,ymm4,ymm0");
              addi(srcfile, "vsubps ymm1,ymm5,ymm1");
              addi(srcfile, "vfmadd231ps " + r_file[m] + ",ymm0,ymm0");
              addi(srcfile, "vfmadd231ps " + r_file[m] + ",ymm1,ymm1");
            }
            srcfile << std::endl;
          }
          srcfile << "      //judge the loop" << std::endl;
          addi(srcfile, "add rax,64");
          addi(srcfile, "cmp rax,r8");
          addi(srcfile, "jl loop%=");
          srcfile << std::endl;
          srcfile << "      //compute the sum" << std::endl;
          for (int m = 0; m < n; ++m) {
            addi(srcfile, "vhaddps ymm0," + r_file[m] + "," + r_file[m]);
            addi(srcfile, "vhaddps ymm1,ymm0,ymm0");
            addi(srcfile, "vextractf128 xmm2,ymm1,0");
            addi(srcfile, "vextractf128 xmm3,ymm1,1");
            addi(srcfile, "addss xmm3,xmm2");

            if (line[1] == "IP") {
              addi(srcfile, "xorps xmm4,xmm4");
              addi(srcfile, "subps xmm4,xmm3");
              addi(srcfile, "movss [r11 + " + to_string(m * 4) + "],xmm4");
            }
            else{
              addi(srcfile, "sqrtss xmm4,xmm3");
              addi(srcfile, "movss [r11 + " + to_string(m * 4) + "],xmm4");
            }
            srcfile << std::endl;
          }
        }
        else if (line[2] == "Float") {
          addi(srcfile, "add r8,r8");
          addi(srcfile, "add r8,r8");
          addi(srcfile, "mov rax,0");
          addi(srcfile, "loop%=:");
          srcfile << "      //prefetch the first line" << std::endl;
          addi(srcfile, "mov r13,[r10]");
          addi(srcfile, "prefetchnta [r13 + rax]");
          addi(srcfile, "prefetchnta [r13 + rax + 64]");
          srcfile << "      //load vector" << std::endl;
          addi(srcfile, "vmovaps ymm2,YMMWORD PTR [r9 + rax]");
          addi(srcfile, "vmovaps ymm3,YMMWORD PTR [r9 + rax + 32]");
          addi(srcfile, "vmovaps ymm4,YMMWORD PTR [r9 + rax + 64]");
          addi(srcfile, "vmovaps ymm5,YMMWORD PTR [r9 + rax + 96]");
          srcfile << std::endl;
          for (int m = 0; m < n; ++m) {
            addi(srcfile, "mov r12,r13");

            if (n > 1 && m < n - 1) {
              srcfile << "      //prefetch the next line" << std::endl;
              addi(srcfile, "mov r13,[r10 + " + to_string(8 * m + 8) + "]");
              addi(srcfile, "prefetchnta [r13 + rax]");
              addi(srcfile, "prefetchnta [r13 + rax + 64]");
            }
            srcfile << "      //compute" << std::endl;
            if (line[1] == "IP") {
              addi(srcfile, "vmovaps ymm0,YMMWORD PTR [r12 + rax]");
              addi(srcfile, "vmovaps ymm1,YMMWORD PTR [r12 + rax + 32]");
              addi(srcfile, "vfmadd231ps " + r_file[m] + ",ymm2,ymm0");
              addi(srcfile, "vfmadd231ps " + r_file[m] + ",ymm3,ymm1");

              addi(srcfile, "vmovaps ymm0,YMMWORD PTR [r12 + rax + 64]");
              addi(srcfile, "vmovaps ymm1,YMMWORD PTR [r12 + rax + 96]");
              addi(srcfile, "vfmadd231ps " + r_file[m] + ",ymm4,ymm0");
              addi(srcfile, "vfmadd231ps " + r_file[m] + ",ymm5,ymm1");
            }
            else {
              addi(srcfile, "vmovaps ymm0,YMMWORD PTR [r12 + rax]");
              addi(srcfile, "vmovaps ymm1,YMMWORD PTR [r12 + rax + 32]");
              addi(srcfile, "vsubps ymm0,ymm2,ymm0");
              addi(srcfile, "vsubps ymm1,ymm3,ymm1");
              addi(srcfile, "vfmadd231ps " + r_file[m] + ",ymm0,ymm0");
              addi(srcfile, "vfmadd231ps " + r_file[m] + ",ymm1,ymm1");

              addi(srcfile, "vmovaps ymm0,YMMWORD PTR [r12 + rax + 64]");
              addi(srcfile, "vmovaps ymm1,YMMWORD PTR [r12 + rax + 96]");
              addi(srcfile, "vsubps ymm0,ymm4,ymm0");
              addi(srcfile, "vsubps ymm1,ymm5,ymm1");
              addi(srcfile, "vfmadd231ps " + r_file[m] + ",ymm0,ymm0");
              addi(srcfile, "vfmadd231ps " + r_file[m] + ",ymm1,ymm1");
            }
            srcfile << std::endl;
          }
          srcfile << "      //judege the loop" << std::endl;
          addi(srcfile, "add rax,128");
          addi(srcfile, "cmp rax,r8");
          addi(srcfile, "jl loop%=");
          srcfile << std::endl;
          srcfile << "      //compute the sum" << std::endl;
          for (int m = 0; m < n; ++m) {
            addi(srcfile, "vhaddps ymm0," + r_file[m] + "," + r_file[m]);
            addi(srcfile, "vhaddps ymm1,ymm0,ymm0");
            addi(srcfile, "vextractf128 xmm2,ymm1,0");
            addi(srcfile, "vextractf128 xmm3,ymm1,1");
            addi(srcfile, "addss xmm3,xmm2");

            if (line[1] == "IP") {
              addi(srcfile, "xorps xmm4,xmm4");
              addi(srcfile, "subps xmm4,xmm3");
              addi(srcfile, "movss [r11 + " + to_string(m * 4) + "],xmm4");
            }
            else{
              addi(srcfile, "sqrtss xmm4,xmm3");
              addi(srcfile, "movss [r11 + " + to_string(m * 4) + "],xmm4");
			}
            srcfile << std::endl;
          }
        }
        else if (line[2] == "Half_Float") {
          addi(srcfile, "add r8,r8");
          addi(srcfile, "add r8,r8");
          addi(srcfile, "mov rax,0");
          addi(srcfile, "mov rbx,0");
          addi(srcfile, "loop%=:");
          srcfile << "      //prefetch the first line" << std::endl;
          addi(srcfile, "mov r13,[r10]");
          addi(srcfile, "prefetchnta [r13 + rbx]");
          srcfile << "      //load vector" << std::endl;
          addi(srcfile, "vmovaps ymm2,YMMWORD PTR [r9 + rax]");
          addi(srcfile, "vmovaps ymm3,YMMWORD PTR [r9 + rax + 32]");
          addi(srcfile, "vmovaps ymm4,YMMWORD PTR [r9 + rax + 64]");
          addi(srcfile, "vmovaps ymm5,YMMWORD PTR [r9 + rax + 96]");
          srcfile << std::endl;
          for (int m = 0; m < n; ++m) {
            addi(srcfile, "mov r12,r13");

            if (n > 1 && m < n - 1) {
              srcfile << "      //prefetch the next line" << std::endl;
              addi(srcfile, "mov r13,[r10 + " + to_string(8 * m + 8) + "]");
              addi(srcfile, "prefetchnta [r13 + rbx]");
            }
            srcfile << "      //compute" << std::endl;
            if (line[1] == "IP") {
              addi(srcfile, "vcvtph2ps ymm0,XMMWORD PTR [r12 + rbx]");
              addi(srcfile, "vcvtph2ps ymm1,XMMWORD PTR [r12 + rbx + 16]");
              addi(srcfile, "vfmadd231ps " + r_file[m] + ",ymm0,ymm2");
              addi(srcfile, "vfmadd231ps " + r_file[m] + ",ymm1,ymm3");
              addi(srcfile, "vcvtph2ps ymm0,XMMWORD PTR [r12 + rbx + 32]");
              addi(srcfile, "vcvtph2ps ymm1,XMMWORD PTR [r12 + rbx + 48]");
              addi(srcfile, "vfmadd231ps " + r_file[m] + ",ymm0,ymm4");
              addi(srcfile, "vfmadd231ps " + r_file[m] + ",ymm1,ymm5");
            }
            else {
              addi(srcfile, "vcvtph2ps ymm0,XMMWORD PTR [r12 + rbx]");
              addi(srcfile, "vcvtph2ps ymm1,XMMWORD PTR [r12 + rbx + 16]");
              addi(srcfile, "vsubps ymm0,ymm2,ymm0");
              addi(srcfile, "vsubps ymm1,ymm3,ymm1");
              addi(srcfile, "vfmadd231ps " + r_file[m] + ",ymm0,ymm0");
              addi(srcfile, "vfmadd231ps " + r_file[m] + ",ymm1,ymm1");
              addi(srcfile, "vcvtph2ps ymm0,XMMWORD PTR [r12 + rbx + 32]");
              addi(srcfile, "vcvtph2ps ymm1,XMMWORD PTR [r12 + rbx + 48]");
              addi(srcfile, "vsubps ymm0,ymm4,ymm0");
              addi(srcfile, "vsubps ymm1,ymm5,ymm1");
              addi(srcfile, "vfmadd231ps " + r_file[m] + ",ymm0,ymm0");
              addi(srcfile, "vfmadd231ps " + r_file[m] + ",ymm1,ymm1");
            }
            srcfile << std::endl;
          }
          srcfile << "      //judge the loop" << std::endl;
          addi(srcfile, "add rax,128");
          addi(srcfile, "add rbx,64");
          addi(srcfile, "cmp rax,r8");
          addi(srcfile, "jl loop%=");
          srcfile << std::endl;
          srcfile << "      //compute the sum" << std::endl;
          for (int m = 0; m < n; ++m) {
            addi(srcfile, "vhaddps ymm0," + r_file[m] + "," + r_file[m]);
            addi(srcfile, "vhaddps ymm1,ymm0,ymm0");
            addi(srcfile, "vextractf128 xmm2,ymm1,0");
            addi(srcfile, "vextractf128 xmm3,ymm1,1");
            addi(srcfile, "addss xmm3,xmm2");

            if (line[1] == "IP") {
              addi(srcfile, "xorps xmm4,xmm4");
              addi(srcfile, "subps xmm4,xmm3");
              addi(srcfile, "movss [r11 + " + to_string(m * 4) + "],xmm4");
            }
            else{
              addi(srcfile, "sqrtss xmm4,xmm3");
              addi(srcfile, "movss [r11 + " + to_string(m * 4) + "],xmm4");
			}

            srcfile << std::endl;
          }
        }
        else {}
      }
      else {}

      srcfile
        << "    :\n"
        << "    :[sp] \"rm\"(sp)\n"
        << "    : \"r8\",\n      \"r9\",\n      \"r10\",\n"
        "      \"r11\",\n      \"r12\",\n      \"r13\",\n"
        "      \"r15\",\n      \"rax\",\n      \"rbx\",\n"
        "      \"memory\");\n";
      srcfile << "}\n\n";
    }
	hinfo += "    }},\n";
  }
  hinfo += "  };\npublic:\n  const std::vector<GEMV>& operator[](std::string name){\n    return GemvPtr[name];\n  }\n};\n";
  hfile << std::endl << hinfo;
  hfile << std::endl << "}" << std::endl;
  srcfile << std::endl << "}" << std::endl;
  hfile.close();
  srcfile.close();
  return 0;
}
