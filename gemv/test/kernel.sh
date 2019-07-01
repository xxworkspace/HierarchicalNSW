g++ -std=c++11 codegen_kernel.cpp -o codegen_kernel
./codegen_kernel
g++ -std=c++11 -masm=intel -c AVXKernel.cc -o AVXKernel.o
g++ -std=c++11 -c cpuinfo.cc -o cpuinfo.o
g++ -std=c++11 -c util.cpp -o util.o
g++ -std=c++11 -c kernel.cc -o kernel.o
g++ -std=c++11 avxkernel_test.cpp AVXKernel.o kernel.o cpuinfo.o -o avxkernel
