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
import json

from numpy import dot
from numpy.linalg import norm


def cos_sim(_embeddings1, _embeddings2):

    a = _embeddings1.reshape(-1)
    b = _embeddings2.reshape(-1)

    cos_sim_v = dot(a, b) / (norm(a)*norm(b))

    print("cos_sim", cos_sim_v)
    return cos_sim_v


image_size = (112, 112)
time0 = datetime.datetime.now()
model = iresnet100()
param_dict = load_checkpoint(
    "/home/ccl/Documents/codes/python/minds/models/research/cv/arcface/device1/device1/CKP-25_9308.ckpt")
load_param_into_net(model, param_dict)
time_now = datetime.datetime.now()
diff = time_now - time0
print('model loading time', diff.total_seconds())

from mindspore import context
context.set_context(device_id=3)

# -------------------------------------------------------------------------------

path = '/data/ccl/RP2K_rp2k_dataset/all/train_augmented/'
desired_size = 112

classes = dict()

walk_ = os.walk(path)

count = 0
for root, dirs, files in walk_:
    if dirs == []:
        pass
    print(root, dirs, files)
    count += 1
    # if count > 10:
    #     break

    for file in files:
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
        print(embeddings.shape)

        class_name = root.split('/')[-1]

        if class_name not in classes.keys():
            classes[class_name] = list()
        classes[class_name].append(embeddings)

        # if len(classes[class_name]) > 2:
        #     break

        # path_new = '/data/ccl/RP2K_rp2k_dataset/all/texts/'
        # np.savetxt(os.path.join(path_new, class_name + ".txt"), embeddings)


dict_class_embeddings = dict()
for key in classes.keys():
    print(key)
    list_numpy = np.array(classes[key])
    print(list_numpy.shape)
    avg_ = np.mean(list_numpy, axis=0) / len(classes[key])
    print(avg_.shape)
    dict_class_embeddings[key] = avg_.tolist()

path_new = '/data/ccl/RP2K_rp2k_dataset/all/texts/'
with open(os.path.join(path_new, 'results.json'), 'w', encoding='utf-8') as result_file:
    json.dump(dict_class_embeddings, result_file)

with open(os.path.join(path_new, 'results.json'), 'r') as result_file:
    save_dict = json.load(result_file)

print(save_dict[str(list(classes.keys())[0])])  # Kelsey

# img1 = cv2.imread("/data/ccl/RP2K_rp2k_dataset/all/test_new/三得利啤酒超纯7.5度500罐啤组合装/1022_14621.jpg")
# # img2 = cv2.imread("/data/ccl/RP2K_rp2k_dataset/all/test_new/555（双冰）/0205_25873.jpg")
# img2 = cv2.imread("/data/ccl/RP2K_rp2k_dataset/all/test_new/三得利啤酒超纯7.5度500罐啤组合装/1022_57320.jpg")
# # img2 = cv2.imread("/data/ccl/RP2K_rp2k_dataset/all/test_new/555（经典）/截屏2019-12-10下午5.15.14.png")
# # img2 = cv2.imread("/data/ccl/RP2K_rp2k_dataset/all/test_new/Seasir丁香料酒/0717_42914.jpg")
#
# img1 = cv2.resize(img1, image_size) / 255
# img2 = cv2.resize(img2, image_size) / 255
# img1 = np.transpose(img1, (2, 0, 1))
# img2 = np.transpose(img2, (2, 0, 1))
#
# img1 = np.expand_dims(img1, axis=0)
# img2 = np.expand_dims(img2, axis=0)
#
# time0 = datetime.datetime.now()
# net_out = model(ms.Tensor(img1, ms.float32))
# _embeddings1 = net_out.asnumpy()
# time_now = datetime.datetime.now()
# diff = time_now - time0
# print("inferece time1", diff.total_seconds())
#
# time1 = datetime.datetime.now()
# net_out = model(ms.Tensor(img2, ms.float32))
# _embeddings2 = net_out.asnumpy()
# time_now = datetime.datetime.now()
# diff = time_now - time1
# print("inferece time2", diff.total_seconds())
#
