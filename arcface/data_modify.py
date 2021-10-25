import cv2
import os

import os

path = '/data/ccl/RP2K_rp2k_dataset/all/test'
path_new = '/data/ccl/RP2K_rp2k_dataset/all/test_new'
desired_size = 112

classes = list()
for root, dirs, files in os.walk(path):
    if dirs == []:
        pass
    else:
        classes = dirs
        for dir_ in dirs:
            if not os.path.exists(os.path.join(path_new, dir_)):
                os.makedirs(os.path.join(path_new, dir_))
        continue
    print(root, dirs, files)

    for file in files:
        im = cv2.imread(os.path.join(root, file))
        old_size = im.shape[:2]  # old_size is in (height, width) format

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
        path_ = root.split('/')
        cv2.imwrite(os.path.join(path_new, path_[-1], file), new_im)

print(classes)


