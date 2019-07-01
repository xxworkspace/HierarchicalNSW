
from reader import reader
import numpy as np
import hnswxx
import time

hnsw = hnswxx.hnswxx(_space="l2",_dtype="float")

data = reader("sift_base.fvecs")
query= reader("sift_query.fvecs")
label= np.arange(data.shape[0])

st = time.time()
hnsw.add_items(data,label)
end = time.time()
print('time cost',end - st,'s')

st = time.time()
rs = hnsw.knn_query(query,10)
end = time.time()
print('time cost',end - st,'s')
