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
param_dict = load_checkpoint("/home/ccl/Documents/codes/python/minds/models/research/cv/arcface/device1/device1/CKP-25_9308.ckpt")
load_param_into_net(model, param_dict)
time_now = datetime.datetime.now()
diff = time_now - time0
print('model loading time', diff.total_seconds())

from mindspore import context
context.set_context(device_id=3)

img1 = cv2.imread("/data/ccl/RP2K_rp2k_dataset/all/test_new/Seasir虾油露/0716_7768.jpg")
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

path = '/data/ccl/RP2K_rp2k_dataset/all/test_new/'
desired_size = 112

classes = dict()

walk_ = os.walk(path)

count = 0
count_right = 0
count_wrong = 0
for root, dirs, files in walk_:
    if dirs == []:
        pass
    # print(root, dirs, files)
    count += 1
    # if count > 10:
    #     break

    for file in files:
        print()
        print(count)
        print(os.path.join(root, file))
        im = cv2.imread(os.path.join(root, file))
        old_size = im.shape[:2]  # old_size is in (height, width) format

        new_im = im
        if old_size != image_size:

            ratio = float(desired_size) / max(old_size)
            new_size = tuple([int(x * ratio) for x in old_size])

            # new_size should be in (width, height) format

            im = cv2.resize(im, (new_size[1], new_size[0]))

            delta_w = desired_size - new_size[1]
            delta_h = desired_size - new_size[0]
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left, right = delta_w // 2, delta_w - (delta_w // 2)

            color = [0, 0, 0]
            new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                        value=color)

        new_im = new_im / 255

        new_im = np.transpose(new_im, (2, 0, 1))
        new_im = np.expand_dims(new_im, axis=0)
        time0 = datetime.datetime.now()
        net_out = model(ms.Tensor(new_im, ms.float32))
        embeddings = net_out.asnumpy()
        time1 = datetime.datetime.now()
        print(embeddings.shape, time1 - time0)

        a = embeddings.reshape(-1, 512)
        b = _embeddings2

        cos_sim = cos_sim_mat(b, a)
        print(np.argmax(cos_sim), key_list[np.argmax(cos_sim)], cos_sim[np.argmax(cos_sim)])
        print(np.sum(cos_sim > 0.8))

        class_name = root.split('/')[-1]
        print(class_name, key_list[np.argmax(cos_sim)])
        if class_name == key_list[np.argmax(cos_sim)]:
            count_right += 1
        else:
            count_wrong += 1

print("count_right", count_right)
print("count_wrong", count_wrong)
