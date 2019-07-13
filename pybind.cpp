
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <iostream>
#include <thread>
#include "hnsw.h"
#include "gemv.h"
#include "int8.h"

namespace py = pybind11;
using int8 = gemv::int8;
using half = gemv::half;
using LABEL = hnsw::LABEL;
using uint8 = gemv::uint8;
/*
 * replacement for the openmp '#pragma omp parallel for' directive
 * only handles a subset of functionality (no reductions etc)
 * Process ids from start (inclusive) to end (EXCLUSIVE)
 *
 * The method is borrowed from nmslib
 */
template<class Function>
inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
  if (numThreads <= 0) {
    numThreads = std::thread::hardware_concurrency();
  }

  if (numThreads == 1) {
    for (size_t id = start; id < end; id++) {
      fn(id, 0);
    }
  }
  else {
    std::vector<std::thread> threads;
    std::atomic<size_t> current(start);
    std::exception_ptr lastException = nullptr;
    std::mutex lastExceptMutex;

    for (size_t threadId = 0; threadId < numThreads; ++threadId) {
      threads.push_back(std::thread([&, threadId] {
        while (true) {
          size_t id = current.fetch_add(1);
          if ((id >= end)) {
            break;
          }

          try {
            fn(id, threadId);
          }
          catch (...) {
            std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
            lastException = std::current_exception();
            /*
             * This will work even when current is the largest value that
             * size_t can fit, because fetch_add returns the previous value
             * before the increment (what will result in overflow
             * and produce 0 instead of current + 1).
             */
            current = end;
            break;
          }
        }
      }));
    }
    for (auto &thread : threads) {
      thread.join();
    }
    if (lastException) {
      std::rethrow_exception(lastException);
    }
  }
}

class GraphIndex {
  //graph parameter
  unsigned M;
  unsigned dim;
  gemv::DisT dist;
  unsigned query_ef;
  unsigned construction_ef;
  unsigned max_elements;
  bool normalize;
  unsigned num_threads_default;
  //float->int8 quatizaiton
  bool quatization;
  float max_value;
  //the graph pointer
  std::string space;
  std::string dtype;

  hnsw::HierarchicalNSW<int8, int8, float> * INT8;
  hnsw::HierarchicalNSW<half, half, float> * HALF;
  hnsw::HierarchicalNSW<half, half, float> * FLOAT;
  hnsw::HierarchicalNSW<uint8, uint8, float> * UINT8;
public:
  GraphIndex(
    unsigned _m = 60,unsigned _dim = 128,unsigned _max_elements = 3600000,
    unsigned _query_ef = 120,unsigned _construction_ef = 360,
    std::string _space = "l2",std::string _dtype = "float",
    bool _quatization = false,float _max_value  = 0.0):
    M(_m),dim(_dim),max_elements(_max_elements),
    query_ef(_query_ef),construction_ef(_construction_ef),
    space(_space),dtype(_dtype),
	quatization(_quatization),max_value(_max_value),
	normalize(false),INT8(NULL),HALF(NULL),FLOAT(NULL),UINT8(NULL){
    num_threads_default = std::thread::hardware_concurrency();
    if (_space == "ip" || _space == "IP") dist = gemv::DisT::IP;
    else if (_space == "L2" || _space == "l2") dist = gemv::DisT::L2;
    else if (_space == "cosine" || _space == "Cosine") {
      normalize = true;
      dist = gemv::DisT::IP;
    }else
      throw std::runtime_error("space is not supported");

    if (dtype == "int8")
      INT8  = new hnsw::HierarchicalNSW<int8, int8, float>(max_elements, M, dim, query_ef, construction_ef, dist);
    else if (dtype == "half")
      HALF  = new hnsw::HierarchicalNSW<half, half, float>(max_elements, M, dim, query_ef, construction_ef, dist);
    else if (dtype == "float")
      FLOAT = new hnsw::HierarchicalNSW<half, half, float>(max_elements, M, dim, query_ef, construction_ef, dist);
    else if (dtype == "uint8")
      UINT8 = new hnsw::HierarchicalNSW<uint8, uint8, float>(max_elements, M, dim, query_ef, construction_ef, dist);
    else
      throw std::runtime_error("data type is not supported");
  }
  ~GraphIndex() {
    if(INT8) delete INT8;
	if(HALF) delete HALF;
	if(FLOAT)delete FLOAT;
    if(UINT8) delete UINT8;
  }

  void saveIndex(const std::string &path) {
    if (dtype == "int8")
      INT8->writeOut(path);
    else if (dtype == "half")
      HALF->writeOut(path);
    else if (dtype == "float")
      FLOAT->writeOut(path);
    else if (dtype == "uint8")
      UINT8->writeOut(path);
  }

  void loadIndex(const std::string &path) {
    if (dtype == "int8")
      INT8->readIn(path);
    else if (dtype == "half")
      HALF->readIn(path);
    else if (dtype == "float")
      FLOAT->readIn(path);
    else if (dtype == "uint8")
      UINT8->readIn(path);
  }

#define GET_SHAPE(rows,cols,buffer)                        \
  if (buffer.ndim != 2 && buffer.ndim != 1)                \
    throw std::runtime_error("data shape must be 1D/2D");  \
  if (buffer.ndim == 2) {                                  \
    rows = buffer.shape[0];cols = buffer.shape[1];         \
  }else{                                                   \
    rows = 1;cols = buffer.shape[0];}

  void addItems(py::object data, py::object idx) {
    size_t rows = 0, cols = 0;
    std::vector<void*> inputs;
    if (dtype == "int8") {
      if(!quatization){
        py::array_t<int8, py::array::c_style | py::array::forcecast > items(data);
        auto buffer = items.request();

        GET_SHAPE(rows,cols,buffer)
        inputs.reserve(rows);
        if (cols != dim) throw std::runtime_error("wrong dimensionality of the vectors");
        for (uint32_t i = 0; i < rows; ++i){
          void* tmp = aligned_alloc(64,sizeof(int8)*cols);
          memcpy(tmp,items.data(i),sizeof(int8)*cols);
          inputs.push_back(tmp);
      	}
      }else{
        py::array_t<float, py::array::c_style | py::array::forcecast > items(data);
        auto buffer = items.request();

        GET_SHAPE(rows,cols,buffer)
        inputs.reserve(rows);
        if (cols != dim) throw std::runtime_error("wrong dimensionality of the vectors");
        for (uint32_t i = 0; i < rows; ++i){
          char* tmp = (char*)aligned_alloc(64,sizeof(int8)*cols);
          floats2int8(tmp,(float*)items.data(i),dim,max_value);
          inputs.push_back(tmp);
    	}
      }
    }
    else if (dtype == "half") {
      py::array_t<half, py::array::c_style | py::array::forcecast > items(data);
      auto buffer = items.request();

      GET_SHAPE(rows, cols, buffer)
      inputs.reserve(rows);
      if (cols != dim) throw std::runtime_error("wrong dimensionality of the vectors");
      for (uint32_t i = 0; i < rows; ++i){
        void* tmp = aligned_alloc(64,sizeof(half)*cols);
        memcpy(tmp,items.data(i),sizeof(half)*cols);
        inputs.push_back(tmp);
      }
    }
    else if (dtype == "float") {
      py::array_t<float, py::array::c_style | py::array::forcecast > items(data);
      auto buffer = items.request();

      GET_SHAPE(rows, cols, buffer)
      inputs.reserve(rows);
      if (cols != dim) throw std::runtime_error("wrong dimensionality of the vectors");
      for (uint32_t i = 0; i < rows; ++i){
        void* tmp = aligned_alloc(64,sizeof(float)*cols);
        memcpy(tmp,items.data(i),sizeof(float)*cols);
        inputs.push_back(tmp);
      }
    }else if(dtype == "uint8"){
      py::array_t<uint8, py::array::c_style | py::array::forcecast > items(data);
      auto buffer = items.request();

      GET_SHAPE(rows,cols,buffer)
      inputs.reserve(rows);
      if (cols != dim) throw std::runtime_error("wrong dimensionality of the vectors");
      for (uint32_t i = 0; i < rows; ++i){
        void* tmp = aligned_alloc(64,sizeof(uint8)*cols);
        memcpy(tmp,items.data(i),sizeof(uint8)*cols);
        inputs.push_back(tmp);
      }
    }

    std::vector<LABEL> ids;
    {
      py::array_t <LABEL, py::array::c_style | py::array::forcecast > items(idx);
      auto ids_numpy = items.request();
      if (ids_numpy.ndim == 1 && ids_numpy.shape[0] == rows) {
        std::vector<LABEL> ids1(ids_numpy.shape[0]);
        for (uint32_t i = 0; i < ids1.size(); i++)
          ids1[i] = items.data()[i];
        ids.swap(ids1);
      }
      else if (ids_numpy.ndim == 0 && rows == 1)
        ids.push_back(*items.data());
      else
        throw std::runtime_error("wrong dimensionality of the labels");
    }

    if (normalize && dtype == "half") {
      gemv::KParams<half, half> kp;
      kp.dim = dim;
      for (uint32_t i = 0; i < rows; i++) {
        kp.src.push_back((half*)inputs[i]);
        kp.dst.push_back((half*)inputs[i]);
      }
      gemv::L2NormAVX<half, half>(kp);
    }
    else if (dtype == "float") {
      gemv::KParams<float, half> kp;
      kp.dim = dim;
      for (uint32_t i = 0; i < rows; i++) {
        kp.src.push_back((float*)inputs[i]);
        inputs[i] = aligned_alloc(64,sizeof(half)*cols);
        kp.dst.push_back((half*)inputs[i]);
      }
      if(normalize)
        gemv::L2NormAVX<float, half>(kp);
      else
        gemv::Float2HalfAVX(kp);
      
      for(auto tmp : kp.src)
        free(tmp);
    }else if(normalize && dtype == "int8")
      throw std::runtime_error("int8 do not support normalization");

    if (ids.size() < num_threads_default * 4) {
      for (uint32_t row = 0; row < rows; ++row){
        if (dtype == "int8")
          INT8->insertPoint((int8*)inputs[row], ids[row]);
        else if (dtype == "half")
          HALF->insertPoint((half*)inputs[row], ids[row]);
        else if (dtype == "float")
          FLOAT->insertPoint((half*)inputs[row], ids[row]);
        else if (dtype == "uint8")
          UINT8->insertPoint((uint8*)inputs[row], ids[row]);
      }
    }
    else {
      ParallelFor(0, rows, num_threads_default, [&](size_t row, size_t threadId) {
        if (dtype == "int8")
          INT8->insertPoint((int8*)inputs[row], ids[row]);
        else if (dtype == "half")
          HALF->insertPoint((half*)inputs[row], ids[row]);
        else if (dtype == "float")
          FLOAT->insertPoint((half*)inputs[row], ids[row]);
        else if (dtype == "uint8")
          UINT8->insertPoint((uint8*)inputs[row], ids[row]);
      });
    }

    for(auto tmp : inputs)
      free(tmp);
  }

  py::object knnQuery_return_numpy(py::object data, size_t k = 1) {
    size_t rows = 0,cols = 0;
    std::vector<void*> inputs;
    if (dtype == "int8") {
      if(!quatization){
        py::array_t<int8, py::array::c_style | py::array::forcecast> items(data);
        auto buffer = items.request();

        GET_SHAPE(rows, cols, buffer)
        if (cols != dim) throw std::runtime_error("wrong dimensionality of the vectors");
        for (uint32_t i = 0; i < rows; ++i){
          void* tmp = aligned_alloc(64,sizeof(int8)*cols);
          memcpy(tmp,items.data(i),sizeof(int8)*cols);
          inputs.push_back(tmp);
     	}
      }else{
        py::array_t<float, py::array::c_style | py::array::forcecast> items(data);
        auto buffer = items.request();

        GET_SHAPE(rows, cols, buffer)
        if (cols != dim) throw std::runtime_error("wrong dimensionality of the vectors");
        for (uint32_t i = 0; i < rows; ++i){
          char* tmp = (char*)aligned_alloc(64,sizeof(char)*cols);
          floats2int8(tmp,(float*)items.data(i),dim,max_value);
          inputs.push_back(tmp);
    	}
      }
    }
    else if (dtype == "half") {
      py::array_t<half, py::array::c_style | py::array::forcecast> items(data);
      auto buffer = items.request();

      GET_SHAPE(rows, cols, buffer)
      if (cols != dim) throw std::runtime_error("wrong dimensionality of the vectors");
      for (uint32_t i = 0; i < rows; ++i){
        void* tmp = aligned_alloc(64,sizeof(half)*cols);
        memcpy(tmp,items.data(i),sizeof(half)*cols);
        inputs.push_back(tmp);
      }
    }
    else if (dtype == "float") {
      py::array_t<float, py::array::c_style | py::array::forcecast> items(data);
      auto buffer = items.request();

      GET_SHAPE(rows, cols, buffer)
      if (cols != dim) throw std::runtime_error("wrong dimensionality of the vectors");
      for (uint32_t i = 0; i < rows; ++i){
        void* tmp = aligned_alloc(64,sizeof(float)*cols);
        memcpy(tmp,items.data(i),sizeof(float)*cols);
        inputs.push_back(tmp);
      }
    }else if(dtype == "uint8"){
      py::array_t<uint8, py::array::c_style | py::array::forcecast> items(data);
      auto buffer = items.request();

      GET_SHAPE(rows, cols, buffer)
      if (cols != dim) throw std::runtime_error("wrong dimensionality of the vectors");
      for (uint32_t i = 0; i < rows; ++i){
        void* tmp = aligned_alloc(64,sizeof(uint8)*cols);
        memcpy(tmp,items.data(i),sizeof(uint8)*cols);
        inputs.push_back(tmp);
      }
    }

    if (normalize && dtype == "half") {
      gemv::KParams<half, half> kp;
      kp.dim = dim;
      for (uint32_t i = 0; i < rows; i++) {
        kp.src.push_back((half*)inputs[i]);
        kp.dst.push_back((half*)inputs[i]);
      }
      gemv::L2NormAVX<half, half>(kp);
    }
    else if (dtype == "float") {
      gemv::KParams<float, half> kp;
      kp.dim = dim;
      for (uint32_t i = 0; i < rows; i++) {
        kp.src.push_back((float*)inputs[i]);
        kp.dst.push_back((half*)inputs[i]);
      }
      if(normalize)
        gemv::L2NormAVX<float, half>(kp);
      else
        gemv::Float2HalfAVX(kp);
    }else if(normalize && dtype == "int8")
      throw std::runtime_error("int8 do not support normalization");

    LABEL* data_numpy_l = new LABEL[rows*k];
    float* data_numpy_d = new float[rows*k];

    if (inputs.size() < num_threads_default * 4) {
      for (uint32_t i = 0; i < inputs.size(); ++i) {
        std::priority_queue<std::pair<float, LABEL>> result;
        if (dtype == "int8")
          result = INT8->searchKnn((int8*)inputs[i], k);
        if (dtype == "half")
          result = HALF->searchKnn((half*)inputs[i], k);
        else if (dtype == "float")
          result = FLOAT->searchKnn((half*)inputs[i], k);
        else if (dtype == "uint8")
          result = UINT8->searchKnn((uint8*)inputs[i], k);

        uint32_t count = i * k + result.size();
        for(int j = count - 1 ; result.size() ; --j){
          data_numpy_d[j] = result.top().first;
          data_numpy_l[j] = result.top().second;
          result.pop();
        }
        for( ; count <  (i + 1) * k ; ++ count){
          data_numpy_d[count] = 1e30;
          data_numpy_l[count] = 0;
        }
      }
    }
    else {
      ParallelFor(0, rows, num_threads_default, [&](size_t row, size_t threadId) {
        std::priority_queue<std::pair<float, LABEL>> result;
        if (dtype == "int8")
          result = INT8->searchKnn((int8*)inputs[row], k);
        if (dtype == "half")
          result = HALF->searchKnn((half*)inputs[row], k);
        else if (dtype == "float")
          result = FLOAT->searchKnn((half*)inputs[row], k);
        else if (dtype == "uint8")
          result = UINT8->searchKnn((uint8*)inputs[row], k);

        uint32_t count = row * k + result.size();
        for(int j = count - 1 ; result.size() ; --j){
          data_numpy_d[j] = result.top().first;
          data_numpy_l[j] = result.top().second;
          result.pop();
        }
        for( ; count <  (row + 1) * k ; ++ count){
          data_numpy_d[count] = 1e30;
          data_numpy_l[count] = 0;
        }
      });
    }

    for(auto tmp : inputs)
      free(tmp);

    py::capsule free_when_done_l(data_numpy_l, [](void *f) {
      delete (LABEL*)f;
    });
    py::capsule free_when_done_d(data_numpy_d, [](void *f) {
      delete (float*)f;
    });

    return py::make_tuple(
      py::array_t<LABEL>(
        { rows, k },              // shape
        { k * sizeof(LABEL),
         sizeof(LABEL) },        // C-style contiguous strides for double
        data_numpy_l,           // the data pointer
        free_when_done_l),
      py::array_t<float>(
        { rows, k },             // shape
        { k * sizeof(float),
        sizeof(float) },        // C-style contiguous strides for double
        data_numpy_d,          // the data pointer
        free_when_done_d));
  }
};

PYBIND11_MODULE(hnswxx,m) {
  py::class_<GraphIndex>(m, "hnswxx")
    .def(py::init<unsigned,unsigned,signed,unsigned,unsigned,std::string,std::string,bool,float>(),
         py::arg("M") = 60,py::arg("dim") = 128,py::arg("max_elements") = 3600000,
         py::arg("query_ef") = 120, py::arg("construction_ef") = 360,
         py::arg("space") = u"l2",py::arg("dtype") = u"float",
         py::arg("quatization") = false,py::arg("max_value") = 0.0)
    .def("add_items",  &GraphIndex::addItems, py::arg("data"), py::arg("idx"))
    .def("knn_query",  &GraphIndex::knnQuery_return_numpy, py::arg("data"), py::arg("k") = 1)
    .def("save_index", &GraphIndex::saveIndex, py::arg("path"))
    .def("load_index", &GraphIndex::loadIndex, py::arg("path"));
}
