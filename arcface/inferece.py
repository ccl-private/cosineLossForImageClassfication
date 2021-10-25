import datetime
import os
import pickle
import argparse
from io import BytesIO

import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from scipy import interpolate
import mindspore as ms
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore import context

from src.iresnet import iresnet100

import cv2

image_size = (112, 112)
time0 = datetime.datetime.now()
model = iresnet100()
param_dict = load_checkpoint("/home/ccl/Documents/codes/python/minds/models/research/cv/arcface/output2/CKP-25_2694.ckpt")
load_param_into_net(model, param_dict)
time_now = datetime.datetime.now()
diff = time_now - time0
print('model loading time', diff.total_seconds())

img1 = cv2.imread("/data/ccl/RP2K_rp2k_dataset/all/test_new/三得利啤酒超纯7.5度500罐啤组合装/1022_14621.jpg")
# img2 = cv2.imread("/data/ccl/RP2K_rp2k_dataset/all/test_new/555（双冰）/0205_25873.jpg")
img2 = cv2.imread("/data/ccl/RP2K_rp2k_dataset/all/test_new/三得利啤酒超纯7.5度500罐啤组合装/1022_57320.jpg")
# img2 = cv2.imread("/data/ccl/RP2K_rp2k_dataset/all/test_new/555（经典）/截屏2019-12-10下午5.15.14.png")
# img2 = cv2.imread("/data/ccl/RP2K_rp2k_dataset/all/test_new/Seasir丁香料酒/0717_42914.jpg")

img1 = cv2.resize(img1, image_size) / 255
img2 = cv2.resize(img2, image_size) / 255
img1 = np.transpose(img1, (2, 0, 1))
img2 = np.transpose(img2, (2, 0, 1))

img1 = np.expand_dims(img1, axis=0)
img2 = np.expand_dims(img2, axis=0)

time0 = datetime.datetime.now()
net_out = model(ms.Tensor(img1, ms.float32))
_embeddings1 = net_out.asnumpy()
time_now = datetime.datetime.now()
diff = time_now - time0
print("inferece time1", diff.total_seconds())

time1 = datetime.datetime.now()
net_out = model(ms.Tensor(img2, ms.float32))
_embeddings2 = net_out.asnumpy()
time_now = datetime.datetime.now()
diff = time_now - time1
print("inferece time2", diff.total_seconds())

from numpy import dot
from numpy.linalg import norm

a = _embeddings1.reshape(-1)
b = _embeddings2.reshape(-1)

cos_sim = dot(a, b)/(norm(a)*norm(b))

print("cos_sim", cos_sim)
