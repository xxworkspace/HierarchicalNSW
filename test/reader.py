
from __future__ import print_function

import os
import struct
import numpy as np


def bvecs(file_path,len):
  fp = open(file_path,"rb")
  dim = struct.unpack('i',fp.read(4))
  fp.seek(0,2)
  size = fp.tell()
  fp.seek(0,0)
  
  num = size/((dim[0] + 4))
  if len > -1:
    num = min(num,len)
  data_list = []
  for i in range(num):
    dim = struct.unpack('i',fp.read(4))
    data = struct.unpack('b'*dim[0],fp.read(dim[0]))
    data = np.array(data).astype("float32")
    data_list.append(data)
  fp.close()
  return np.concatenate(data_list).reshape(-1,dim[0])

def fvecs(file_path,len):
  fp = open(file_path,"rb")
  dim = struct.unpack('i',fp.read(4))
  fp.seek(0,2)
  size = fp.tell()
  fp.seek(0,0)
  
  num = size/((dim[0]*4 + 4))
  if len > -1:
    num = min(num,len)
  data_list = []
  for i in range(num):
    dim = struct.unpack('i',fp.read(4))
    data = struct.unpack('f'*dim[0],fp.read(dim[0] * 4))
    data = np.array(data).astype("float32")
    data_list.append(data)
  fp.close()
  return np.concatenate(data_list).reshape(-1,dim[0])

def ivecs(file_path,len):
  fp = open(file_path,"rb")
  dim = struct.unpack('i',fp.read(4))
  fp.seek(0,2)
  size = fp.tell()
  fp.seek(0,0)

  num = size/((dim[0]*4 + 4))
  if len > -1:
    num = min(num,len)
  data_list = []
  for i in range(num):
    dim = struct.unpack('i',fp.read(4))
    data = struct.unpack('i'*dim[0],fp.read(dim[0] * 4))
    data = np.array(data).astype("float32")
    data_list.append(data)
  fp.close()
  return np.concatenate(data_list).reshape(-1,dim[0])

def reader(file_path,len = -1):
  if not os.path.exists(file_path):
    print("file open fail!")
    return None

  (filepath,tempfilename) = os.path.split(file_path)
  (filename,extension) = os.path.splitext(tempfilename)
  
  if extension == ".bvecs":
    return bvecs(file_path,len)
  elif extension == ".fvecs":
    return fvecs(file_path,len)
  elif extension == ".ivecs":
    return ivecs(file_path,len)
  else:
    print("file format is not supprt!")
    return None