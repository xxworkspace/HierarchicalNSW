g++ -std=c++11 codegen_gemv_25_10.cpp -o codegen_gemv_25_10
./codegen_gemv_25_10
#g++ -std=c++11 codegen_gemv_24_8.cpp -o codegen_gemv_24_8
#./codegen_gemv_24_8
g++ -std=c++11 -masm=intel -c GemvKernel.cc -o GemvKernel.o
g++ -std=c++11 -c cpuinfo.cc -o cpuinfo.o
g++ -std=c++11 -c gemv.cc -o gemv.o
g++ -std=c++11 -c util.cpp -o util.o
g++ -std=c++11 avxgemv_test.cpp util.o GemvKernel.o gemv.o cpuinfo.o -o avxgemv_test 
