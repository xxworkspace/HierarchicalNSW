/*
 * Copyright (c) BIGO, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <map>
#include <mutex>
#include <queue>
#include <vector>
#include <random>
#include <atomic>
#include <sstream>
#include <fstream>
#include <malloc.h>
#include <stdint.h>
#include <stdlib.h>
#include <iostream>

#include <unordered_set>
#include <unordered_map>

#include "util.h"
#include "gemv.h"
#include "kernel.h"

namespace hnsw {
  typedef unsigned int Index;

  template<class T>
  struct DataT {
    uint32_t dim;
    uint32_t dsize;
    std::mutex mtx;
    std::vector<T*> data;
    DataT() :dsize(0) {
    }
    ~DataT() {
      for (auto t : data)
        free(t);
      data.clear();
    }

    void setMaxElements(uint32_t _max_elements) {
      data.resize(_max_elements);
    }

    void setDim(uint32_t _dim) {
      dim = _dim;
    }
    const uint32_t size(){
      return dsize;
    }

    void addPoint(T * p, uint32_t& pointid) {
      T* _p = (T*)aligned_alloc(64,dim * sizeof(T));
      memcpy(_p, p, sizeof(T)*dim);
      std::lock_guard<std::mutex> lock(mtx);
      pointid = dsize++;
      data[pointid] = _p;
    }

    inline T* getPoint(Index id) {
      return data[id];
    }

    inline T* operator[](Index id) {
      return data[id];
    }

    const uint32_t getDimension() {
      return dim;
    }

    const uint64_t getMemoryOccupy(){
      return sizeof(dsize) + dsize*dim*sizeof(T);
    }

    void dump(char*serial){
      uint64_t pos = 0;
      memcpy(serial,&dsize,sizeof(uint32_t));
      pos += sizeof(uint32_t);
      for(auto ptr : data){
        if(ptr){
          memcpy(&serial[pos],ptr,sizeof(T)*dim);
          pos += sizeof(T)*dim;
        }
      }
    }

    void load(char*serial){
      uint64_t pos = 0;
      memcpy(&dsize,serial,sizeof(uint32_t));
      pos += sizeof(uint32_t);
      for(int i = 0 ; i < dsize ; ++i){
        auto ptr = (T*)aligned_alloc(64,sizeof(T)*dim);
        memcpy(ptr,&serial[pos],sizeof(T)*dim);
        pos += sizeof(T)*dim;
      }
    }

    void writeOut(std::string filename) {
      std::ofstream ofs(filename, std::ios::binary);
      uint32_t tmp = data.size();
      ofs.write((char*)&tmp, sizeof(uint32_t));
      ofs.write((char*)&dim, sizeof(uint32_t));
      tmp = sizeof(T);
      ofs.write((char*)&tmp, sizeof(uint32_t));
      tmp = 0;
      for (auto pt : data) {
        if (pt == nullptr)
          ofs.write((char*)&tmp, sizeof(uint32_t));
        else {
          ofs.write((char*)&dim, sizeof(uint32_t));
          ofs.write((char*)pt, sizeof(T)*dim);
        }
      }
      ofs.close();
    }

    void readIn(std::string filename) {
      std::ifstream ifs(filename, std::ios::binary);
      if (ifs.fail()) {
        std::cout << "Data File Open Fail!" << std::endl;
        exit(0);
      }

      uint32_t tmp;
      ifs.read((char*)&tmp, sizeof(uint32_t));
      data.resize(tmp);
      ifs.read((char*)&dim, sizeof(uint32_t));
      ifs.read((char*)&tmp, sizeof(uint32_t));
      if (tmp != sizeof(T)) {
        std::cout << "Data Type Error!" << std::endl;
        exit(0);
      }
      for (uint32_t i = 0; i < data.size(); i++) {
        ifs.read((char*)&tmp, sizeof(uint32_t));
        if (tmp == 0)
          continue;
        data[i] = (T*)malloc(dim * sizeof(T));
        ifs.read((char*)data[i], dim * sizeof(T));
      }
      ifs.close();
    }
  };

  typedef uint64_t LABEL;
  struct NodeId {
    std::vector<LABEL> ids;
    void setMaxElements(uint32_t _max_elements) {
      ids.resize(_max_elements);
    }

    void insertId(uint32_t pid, LABEL id) {
      ids[pid] = id;
    }

    LABEL getId(uint32_t pid) {
      return ids[pid];
    }

    LABEL operator[](uint32_t pid) {
      return ids[pid];
    }

    const uint64_t getMemoryOccupy(){
      return ids.size()*sizeof(uint32_t);
    }

    void dump(char*serial){
      memcpy(serial,ids.data(),sizeof(LABEL)*ids.size());
    }
    void load(char*serial){
      memcpy(ids.data(),serial,sizeof(LABEL)*ids.size());
    }

    void writeOut(const std::string& filename) {
      std::ofstream ofs(filename, std::ios::binary);
      ofs.write((char*)&ids[0], ids.size() * sizeof(LABEL));
      ofs.close();
    }

    void readIn(const std::string& filename) {
      std::ifstream ifs(filename, std::ios::binary);
      if (ifs.fail()) {
        std::cout << "Data File Open Fail!" << std::endl;
        exit(0);
      }
      ifs.read((char*)&ids[0], ids.size() * sizeof(LABEL));
      ifs.close();
    }
  };

  struct HierarchicalGraphIndex {
    //typedef std::vector<Index> PointIndex;
    //typedef std::vector<PointIndex> HierarchicalPointIndex;
    typedef std::vector<Index*> HierarchicalGraph;

    uint32_t M;
    HierarchicalGraph graphIndex;
    ~HierarchicalGraphIndex() {
      for (auto pindex : graphIndex) {
        if (!pindex) free(pindex);
      }
    }
    void setMaxElements(uint32_t _max_elements, uint32_t m) {
      graphIndex.resize(_max_elements);
      M = m;
    }

    const uint64_t getMemoryOccupy(){
      uint64_t size = sizeof(uint32_t);
      for(auto tmp : graphIndex){
        uint32_t level = 0;
        if(tmp)
          level = *tmp;
        size += sizeof(Index)*(2 * M + level * M + (1 + level) + 1);
      }
      return size;
    }

    void dump(char* serial){
      uint32_t index = 0;
      uint64_t pos = sizeof(uint32_t);
      for(auto ptr : graphIndex){
        if(ptr){
          uint32_t level = *ptr;
          uint32_t size = sizeof(Index)*(2 * M + level * M + (1 + level) + 1);
          memcpy(&serial[pos],ptr,size);
          pos += size;
          ++ index;
        }
      }
      memcpy(serial,&index,sizeof(uint32_t));
    }
    void load(char* serial){
      uint32_t dsize = 0;
      uint64_t pos = sizeof(uint32_t);
      memcpy(&dsize,serial,sizeof(uint32_t));
      for(int i = 0 ; i < dsize ; ++i){
        uint32_t level = 0;
        memcpy(&level,&serial[pos],sizeof(uint32_t));
        uint32_t size = sizeof(Index)*(2 * M + level * M + (1 + level) + 1);
        Index* tmp = (Index*)malloc(size);
        memcpy(tmp,&serial[pos],size);
        pos += size;
      }
    }

    void insertPointIndex(uint32_t level, uint32_t pointid) {
      Index* tmp = (Index*)malloc(sizeof(Index)*(2 * M + level * M + (1 + level) + 1));
      memset(tmp, 0, sizeof(Index)*(2 * M + level * M + (1 + level) + 1));
      *tmp = level;
      graphIndex[pointid] = tmp;
    }

    inline Index* getPointIndex(uint32_t level, Index pointid) {
      return level ? (graphIndex[pointid] + (2 * M + 1 + (level - 1)*(M + 1))) + 1 : graphIndex[pointid] + 1;//graphIndex[id][level];
    }

    void writeOut(std::string filename) {
      std::ofstream ofs(filename, std::ios::binary);
      uint32_t len = graphIndex.size();
      ofs.write((char*)&len, sizeof(uint32_t));
      ofs.write((char*)&M, sizeof(uint32_t));
      for (auto pindex : graphIndex)
        ofs.write((char*)pindex, sizeof(Index)*(2 * M + (*pindex) * M + (1 + (*pindex)) + 1));
      ofs.close();
    }

    void readIn(std::string filename) {
      std::ifstream ifs(filename, std::ios::binary);
      if (ifs.fail()) {
        std::cout << "Data File Open Fail!" << std::endl;
        exit(0);
      }
      uint32_t tmp;
      ifs.read((char*)&tmp, sizeof(uint32_t));
      ifs.read((char*)&M, sizeof(uint32_t));
      graphIndex.resize(tmp);
      for (uint32_t i = 0; i < graphIndex.size(); i++) {
        ifs.read((char*)&tmp, sizeof(uint32_t));
        graphIndex[i] = (Index*)malloc(sizeof(Index)*(2 * M + tmp * M + (1 + tmp) + 1));
        graphIndex[i][0] = tmp;
        ifs.read((char*)&graphIndex[i][1], sizeof(Index)*(2 * M + tmp * M + (1 + tmp)));
      }
      ifs.close();
    }
  };

  struct Visited {
    uint64_t flag;
    std::vector<uint64_t>  visited;
    Visited() :flag(0) {
    }

    void setMaxElements(uint32_t max_elements) {
      visited.resize(max_elements);
    }

    inline bool isVisited(Index id) {
      return (flag == visited[id]);
    }
    inline void visit(Index id) {
      visited[id] = flag;
    }
    void reset() {
      ++flag;
    }
  };

  struct VisitedPool {
    std::vector<Visited> visited;
    std::vector<bool> usable;
    uint32_t max_element;
    std::mutex mtx;

    void setMaxElements(uint32_t _max_elements) {
      max_element = _max_elements;
      usable.reserve(256);
      visited.reserve(256);
      for (int i = 0; i < 128; i++) {
        usable.push_back(true);
        visited.emplace_back();
        visited.back().setMaxElements(max_element);
      }
    }

    void getVisited(uint32_t &id, Visited* & ptr) {
      std::lock_guard<std::mutex> lock(mtx);
      for (uint32_t i = 0; i < usable.size(); i++) {
        if (usable[i] == true) {
          id = i;
          usable[i] = false;
          ptr = &visited[id];
          return;
        }
      }
      visited.emplace_back();
      usable.push_back(false);
      id = usable.size() - 1;
      ptr = &visited[id];
      ptr->setMaxElements(max_element);
    }

    void release(uint32_t id) {
      std::lock_guard<std::mutex> lock(mtx);
      usable[id] = true;
    }
  };

  template<class T1, class T2, class T3 = float>
  class HierarchicalNSW {
  private:
    unsigned int max_link;
    unsigned int max0_link;
    unsigned int max_level;

    uint32_t M;
    uint32_t Dim;
    gemv::DisT disT;
    uint32_t query_ef;
    uint32_t construct_ef;
    Index globalEnterPoint;
    std::default_random_engine level_generator_;

    NodeId nodeIds;
    std::mutex mtx;
    DataT<T1> dataT;
    uint32_t max_elements;
    std::vector<std::mutex> mtxs;
    HierarchicalGraphIndex index;

    VisitedPool visitedpool;
  public:
    HierarchicalNSW(uint32_t _max_elements, uint32_t m, uint32_t dim, uint32_t q_ef, uint32_t c_ef, gemv::DisT dist = gemv::DisT::IP)
      :M(m), Dim(dim),
      query_ef(q_ef), construct_ef(c_ef),
      disT(dist),
      max_link(m), max0_link(2 * m),
      max_level(0),
      max_elements(_max_elements),
      mtxs(_max_elements), globalEnterPoint(0) {
      dataT.setDim(dim);
      dataT.setMaxElements(max_elements);
      index.setMaxElements(max_elements, m);
      nodeIds.setMaxElements(max_elements);
      visitedpool.setMaxElements(max_elements);
      level_generator_.seed(time(0));
    }

    const uint32_t size(){
      return dataT.size();
	}

    struct CompareByFirst {
      constexpr bool operator()(std::pair<T3, Index> const&a,
        std::pair<T3, Index> const&b)const noexcept {
        return a.first < b.first;
      }
    };

    uint32_t getRandomLevel() {
      float reverse_size = 1.0 / logf(1.0 * max_link);
      std::uniform_real_distribution<double> distribution(0, 1.0);
      double r = -log(distribution(level_generator_)) * reverse_size;
      return (int)r;
    }

    //#define TEST(gp,dim) \
        std::cout<<__LINE__<<std::endl; \
        for(int i = 0 ; i < gp.neighbour.size() ; ++i) \
        std::cout<<gp.distance[i]<<" "<<_L2(gp.neighbour[i],gp.query,dim)<<std::endl;

    typedef std::priority_queue<std::pair<T3, Index>> PriorityQueue;
    typedef std::vector<std::pair<T3, Index>> List;
    typedef std::priority_queue<std::pair<T3, Index>, std::vector<std::pair<T3, Index>>, CompareByFirst> CandidateSet;
    template<class T>
    CandidateSet searchBaseLayer(Index enterpoint, T* data_point,
      uint32_t level, uint32_t ef) {
      CandidateSet top_candidates;
      CandidateSet candidateSet;

      gemv::Params<T1, T, T3> gp;
      gp.dim = Dim;
      gp.dist = disT;
      gp.neighbour.clear();
      gp.query = data_point;
      gp.neighbour.push_back(dataT.getPoint(enterpoint));

      gemv::GemvAVX<T1, T, T3>(gp);
      //first is smallest
      candidateSet.emplace(-gp.distance[0], enterpoint);
      //first is largest
      top_candidates.emplace(gp.distance[0], enterpoint);
      uint32_t vid;
      Visited* visited;
      visitedpool.getVisited(vid, visited);
      visited->reset();
      visited->visit(enterpoint);

      std::vector<Index> candidate_ids;
      candidate_ids.reserve(ef);
      while (candidateSet.size()) {
        auto cur = candidateSet.top();
        candidateSet.pop();
        if (-cur.first > top_candidates.top().first && top_candidates.size() >= ef)
          break;
        Index cur_enterpoint = cur.second;
        std::lock_guard<std::mutex> lock(mtxs[cur_enterpoint]);
        Index* pindex = index.getPointIndex(level, cur_enterpoint);
        uint32_t count = *pindex;
        ++pindex;

        candidate_ids.clear();
        gp.neighbour.clear();
        for (uint32_t i = 0; i < count; ++i) {
          Index id = pindex[i];
          if (!visited->isVisited(id)) {
            gp.neighbour.push_back(dataT.getPoint(id));
            candidate_ids.push_back(id);
            visited->visit(id);
          }
        }
        gemv::GemvAVX<T1, T, T3>(gp);
        for (uint32_t i = 0; i < candidate_ids.size(); ++i) {
          if (top_candidates.top().first > gp.distance[i]
            || top_candidates.size() < ef) {
            candidateSet.emplace(-gp.distance[i], candidate_ids[i]);
            top_candidates.emplace(gp.distance[i], candidate_ids[i]);
            if (top_candidates.size() > ef)
              top_candidates.pop();
          }
        }
      }
      visitedpool.release(vid);
      return top_candidates;
    }

    void getNeighborsByHeuristic(CandidateSet& top_candidates, Index MM) {
      if (top_candidates.size() <= MM)
        return;

      List tmplist;
      PriorityQueue closest;
      while (top_candidates.size() > 0) {
        closest.emplace(-top_candidates.top().first, top_candidates.top().second);
        top_candidates.pop();
      }

      gemv::Params<T1, T1, T3> gp;
      gp.dim = Dim;
      gp.dist = disT;
      gp.neighbour.resize(1);
      while (closest.size()) {
        if (tmplist.size() >= MM) break;
        auto cur = closest.top();
        T3 bound = -cur.first;
        closest.pop();

        bool good = true;
        gp.query = dataT.getPoint(cur.second);
        for (auto tmp : tmplist) {
          gp.neighbour[0] = dataT.getPoint(tmp.second);
          gemv::GemvAVX<T1, T1, T3>(gp);
          if (gp.distance[0] < bound) {
            good = false;
            break;
          }
        }
        if (good) {
          tmplist.push_back(cur);
        }
      }

      for (auto tmp : tmplist) {
        top_candidates.emplace(-tmp.first, tmp.second);
      }
    }

    void mutuallyConnectNewElement(T1* data_point, Index cur_id, CandidateSet& top_candidates, uint32_t level) {
      Index Mmax = level ? max_link : max0_link;
      getNeighborsByHeuristic(top_candidates, M);

      Index* cur_pindex = index.getPointIndex(level, cur_id);
      uint32_t& countx = *cur_pindex;
      ++cur_pindex;

      std::vector<std::pair<T3, Index>> selectedNeighbors;
      selectedNeighbors.reserve(M);
      while (top_candidates.size() > 0) {
        selectedNeighbors.push_back(top_candidates.top());
        cur_pindex[countx++] = top_candidates.top().second;
        top_candidates.pop();
      }

      gemv::Params<T1, T1, T3> gp;
      gp.dim = Dim;
      gp.dist = disT;
      for (auto idx : selectedNeighbors) {
        std::lock_guard<std::mutex> lock(mtxs[idx.second]);
        Index* pindex = index.getPointIndex(level, idx.second);
        uint32_t& count = *pindex;
        ++pindex;
        if (count < Mmax)
          pindex[count++] = cur_id;
        else {
          CandidateSet candidates;
          candidates.push(idx);

          gp.neighbour.clear();
          gp.query = dataT[idx.second];
          for (uint32_t i = 0; i < count; ++i)
            gp.neighbour.push_back(dataT[pindex[i]]);
          gemv::GemvAVX<T1, T1, T3>(gp);
          for (uint32_t i = 0; i < count; i++)
            candidates.emplace(gp.distance[i], pindex[i]);

          getNeighborsByHeuristic(candidates, Mmax);
          count = 0;
          while (candidates.size()) {
            pindex[count++] = candidates.top().second;
            candidates.pop();
          }
        }
      }
    }

    void insertPoint(T1 *point_data, LABEL id) {
      mtx.lock();
      uint32_t curlevel = getRandomLevel();
      if (curlevel <= max_level)
        mtx.unlock();
      Index pointid;
      Index cur_enterpoint = globalEnterPoint;
      dataT.addPoint(point_data, pointid);
      nodeIds.insertId(pointid, id);
      index.insertPointIndex(curlevel, pointid);

      if (curlevel < max_level) {
        gemv::Params<T1, T1, T3> gp;
        gp.dim = Dim;
        gp.dist = disT;
        gp.query = dataT.getPoint(pointid);
        gp.neighbour.push_back(dataT.getPoint(cur_enterpoint));
        gemv::GemvAVX<T1, T1, T3>(gp);
        T3 bound = gp.distance[0];

        for (uint32_t level = max_level; level > curlevel; --level) {
          while (true) {
            //auto pindex = index.getPointIndex(level, cur_enterpoint);
            Index* pindex = index.getPointIndex(level, cur_enterpoint);
            uint32_t count = *pindex;
            ++pindex;

            gp.neighbour.clear();
            for (uint32_t i = 0; i < count; ++i)
              gp.neighbour.push_back(dataT.getPoint(pindex[i]));
            gemv::GemvAVX<T1, T1, T3>(gp);
            bool flag = false;
            for (uint32_t i = 0; i < count; i++) {
              if (gp.distance[i] < bound) {
                cur_enterpoint = pindex[i];
                bound = gp.distance[i];
                flag = true;
              }
            }
            if (!flag)break;
          }
        }
      }

      for (int level = std::min(curlevel, max_level); level >= 0 && pointid; --level) {
        CandidateSet top_candidates = searchBaseLayer(cur_enterpoint, dataT.getPoint(pointid), level, construct_ef);
        mutuallyConnectNewElement(dataT.getPoint(pointid), pointid, top_candidates, level);
      }

      if (curlevel > max_level) {
        max_level = curlevel;
        globalEnterPoint = pointid;
        mtx.unlock();
      }
    }

    typedef std::priority_queue<std::pair<T3, LABEL>> PriQueue;
    PriQueue searchKnn(T2* query_data, uint32_t k) {
      Index cur_enterpoint = globalEnterPoint;
      gemv::Params<T1, T2, T3> gp;
      gp.dim = Dim;
      gp.dist = disT;
      gp.query = query_data;
      gp.neighbour.push_back(dataT.getPoint(cur_enterpoint));

      gemv::GemvAVX<T1, T2, T3>(gp);
      T3 bound = gp.distance[0];
      for (int level = max_level; level > 0; --level) {
        while (true) {
          Index* pindex = index.getPointIndex(level, cur_enterpoint);
          uint32_t count = *pindex;
          ++pindex;

          gp.neighbour.clear();
          for (uint32_t i = 0; i < count; i++)
            gp.neighbour.push_back(dataT.getPoint(pindex[i]));
          gemv::GemvAVX<T1, T2, T3>(gp);
          bool flag = false;
          for (uint32_t i = 0; i < count; ++i) {
            if (gp.distance[i] < bound) {
              cur_enterpoint = pindex[i];
              bound = gp.distance[i];
              flag = true;
            }
          }
          if (!flag)break;
        }
      }

      CandidateSet top_candidates = searchBaseLayer(cur_enterpoint, query_data, 0, query_ef);
      while (top_candidates.size() > k) top_candidates.pop();
      PriQueue tmpList;
      while (top_candidates.size()) {
        auto tmp = top_candidates.top();
        tmpList.emplace(tmp.first,nodeIds[tmp.second]);
        top_candidates.pop();
      }
      return tmpList;
    }
    /*
    PriorityQueue bruteForceKnn(T2* query_data, uint32_t k) {
      gemv::Params<T1, T2, T3> gp;
      gp.dim = Dim;
      gp.dist = disT;
      gp.query = query_data;
      gp.neighbour = dataT.data;
      gemv::GemvAVX(gp);

      PriorityQueue tmplist;
      for (int i = 0; i < k; i++) {
      T3 value = 1e8;
      Index index = -1;
      for (int j = 0; j < gp.distance.size(); j++) {
      if (gp.distance[j] < value) {
      value = gp.distance[j];
      index = j;
      }
      }
      if (index == -1) break;
      tmplist.emplace(value, nodeIds[index]);
      gp.distance[index] = 1e8;
      }
      return tmplist;
    }*/
    const std::vector<char> dump(){
      uint32_t s0 = index.getMemoryOccupy();
      uint32_t s1 = dataT.getMemoryOccupy();
      uint32_t s2 = nodeIds.getMemoryOccupy();
      std::vector<char> serial;
      serial.resize(s0 + s1 + s2);
      index.dump(&serial[0]);
      dataT.dump(&serial[s0]);
      nodeIds.dump(&serial[s0 + s1]);
      return serial;
    }
	
    void load(std::vector<char> serial){
      index.load(&serial[0]);
      uint32_t s0 = index.getMemoryOccupy();
      dataT.load(&serial[s0]);
      uint32_t s1 = dataT.getMemoryOccupy();
      nodeIds.load(&serial[s0 + s1]);
    }

    void writeOut(std::string filename) {
      index.writeOut(filename + "_index.bin");
      dataT.writeOut(filename + "_data.bin");
      nodeIds.writeOut(filename + "_idx.bin");
    }
    void readIn(const std::string filename) {
      index.readIn(filename + "_index.bin");
      dataT.readIn(filename + "_data.bin");
      nodeIds.readIn(filename + "_idx.bin");
    }
  };
}
