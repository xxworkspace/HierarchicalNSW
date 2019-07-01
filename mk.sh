echo "cd gemv"
cd gemv
echo "mkdir bin"
mkdir bin
echo "g++ -std=c++11 codegen/codegen_gemv_25_10.cpp -o bin/codegen_gemv_25_10"
g++ -std=c++11 codegen/codegen_gemv_25_10.cpp -o bin/codegen_gemv_25_10
echo "g++ -std=c++11 codegen/codegen_kernel.cpp -o bin/codegen_kernel"
g++ -std=c++11 codegen/codegen_kernel.cpp -o bin/codegen_kernel
echo "./bin/codegen_gemv_25_10"
./bin/codegen_gemv_25_10
echo "./bin/codegen_kernel"
./bin/codegen_kernel
echo "mv GemvKernel.h ./include"
mv GemvKernel.h ./include
echo "mv AVXKernel.h ./include"
mv AVXKernel.h ./include
echo "mkdir obj"
mkdir obj
echo "compile *.o"
g++ -std=c++11 -masm=intel -fPIC  -I./include -c GemvKernel.cc -o obj/GemvKernel.o
g++ -std=c++11 -masm=intel -fPIC  -I./include -c AVXKernel.cc -o obj/AVXKernel.o
g++ -std=c++11 -fPIC -I./include -c src/cpuinfo.cc -o obj/cpuinfo.o
g++ -std=c++11 -fPIC -I./include -c src/kernel.cc -o obj/kernel.o
g++ -std=c++11 -fPIC -I./include -c src/gemv.cc -o obj/gemv.o
g++ -std=c++11 -fPIC -I./include -c src/int8.cc -o obj/int8.o
g++ -std=c++11 -fPIC -I./include -c src/util.cpp -o obj/util.o
echo "build *.so"
cd ../
mkdir lib
g++ -O3 -shared -std=c++11 -I/usr/local/include/python2.7 -I./gemv/include -I/usr/include/python2.7 -fPIC `python -m pybind11 --includes` gemv/obj/GemvKernel.o gemv/obj/AVXKernel.o gemv/obj/gemv.o gemv/obj/kernel.o gemv/obj/int8.o gemv/obj/cpuinfo.o gemv/obj/util.o pybind.cpp -o lib/hnswxx.so
g++ -O3 -shared -std=c++11 -I/usr/local/include/python3.6 -I./gemv/include -I/usr/include/python3.6 -fPIC `python3 -m pybind11 --includes` gemv/obj/GemvKernel.o gemv/obj/AVXKernel.o gemv/obj/gemv.o gemv/obj/kernel.o gemv/obj/int8.o gemv/obj/cpuinfo.o gemv/obj/util.o pybind.cpp -o lib/hnswxx`python3-config --extension-suffix`
echo "clean ..."
cd gemv
rm -r obj
rm -r bin
rm *.cc
rm include/*Kernel*
