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
param_dict = load_checkpoint("./device1/device1/CKP-25_9308.ckpt")
load_param_into_net(model, param_dict)
time_now = datetime.datetime.now()
diff = time_now - time0
print('model loading time', diff.total_seconds())

from mindspore import context
context.set_context(device_id=3)

img1 = cv2.imread("/data/ccl/RP2K_rp2k_dataset/all/test_new/黄果树（蓝佳品）/截屏2019-12-10下午6.09.53.png")
img1 = cv2.resize(img1, image_size) / 255
img1 = np.transpose(img1, (2, 0, 1))

img1 = np.expand_dims(img1, axis=0)

time0 = datetime.datetime.now()
net_out = model(ms.Tensor(img1, ms.float32))
_embeddings1 = net_out.asnumpy()
time_now = datetime.datetime.now()
diff = time_now - time0
print("inferece time1", diff.total_seconds())

import json
path_new = '/data/ccl/RP2K_rp2k_dataset/all/texts/'

with open(os.path.join(path_new, 'results.json'), 'r') as result_file:
    save_dict = json.load(result_file)

key_list = list(save_dict.keys())
_embeddings_list = list()
for key in key_list:
    t = np.array(save_dict[key]).reshape(-1)
    _embeddings_list.append(t)

_embeddings2 = np.array(_embeddings_list)
print(_embeddings2.shape)

from numpy import dot
from numpy.linalg import norm


def cos_sim_mat(Matrix, B):
    """¼ÆËã¾ØÕóÖÐÃ¿¸öÐÐÏòÁ¿ÓëÁíÒ»¸öÐÐÏòÁ¿µÄÓàÏÒÏàËÆ¶È
    ×¢²áÌØÕ÷ÖÐÖ»ÓÐÒ»ÕÅÍ¼Æ¬µÄ512Î¬ÌØÕ÷£¬»á´æÔÚÎÊÌâ£¬ÐèÒªreshapeº¯Êý±£Ö¤ÌØÕ÷ÊÇ(1x512)"""
    num = np.dot(Matrix, B.T).reshape(-1)
    denom = norm(Matrix.reshape(-1, 512), axis=1, keepdims=True) * norm(B)
    denom = denom.reshape(-1)
    cos_val = num / denom
    sim = 0.5 + 0.5 * cos_val
    return sim


a = _embeddings1.reshape(-1, 512)
b = _embeddings2

cos_sim = cos_sim_mat(b, a)

print(key_list)
print("cos_sim", cos_sim)
print(np.argmax(cos_sim), key_list[np.argmax(cos_sim)], cos_sim[np.argmax(cos_sim)])
print(np.sum(cos_sim > 0.8))
