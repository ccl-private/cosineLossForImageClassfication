import numpy as np
import Augmentor
from Augmentor.Pipeline import Operation
import cv2
from PIL import Image
import os


# Create your new operation by inheriting from the Operation superclass:
class ResizeCenterPadding(Operation):
    # Here you can accept as many custom parameters as required:
    def __init__(self, probability, desired_size):
        # Call the superclass's constructor (meaning you must
        # supply a probability value):
        Operation.__init__(self, probability)
        # Set your custom operation's member variables here as required:
        self.desired_size = desired_size

    # Your class must implement the perform_operation method:
    def perform_operation(self, images):
        # Start of code to perform custom image operation.
        for i in range(len(images)):
            image = images[i]
            image_numpy = np.array(image).astype('uint8')
            # im = cv2.cvtColor(image_numpy, cv2.COLOR_RGB2BGR)
            im = image_numpy
            old_size = im.shape[:2]  # old_size is in (height, width) format

            ratio = float(self.desired_size) / max(old_size)
            new_size = tuple([int(x * ratio) for x in old_size])

            # new_size should be in (width, height) format

            im = cv2.resize(im, (new_size[1], new_size[0]))

            delta_w = self.desired_size - new_size[1]
            delta_h = self.desired_size - new_size[0]
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left, right = delta_w // 2, delta_w - (delta_w // 2)

            color = [0, 0, 0]
            new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                        value=color)
            image = Image.fromarray(new_im)
            images[i] = image
        # End of code to perform custom image operation.

        # Return the image so that it can further processed in the pipeline:
        return images


path = '/data/ccl/RP2K_rp2k_dataset/all/train_new'
path_new = '/data/ccl/RP2K_rp2k_dataset/all/train_augmented'
desired_size = 112
pics_num = 500

count = 0
empty = 0
file_num_list = list()
classes = list()
classes_new = list()

for root, dirs, files in os.walk(path_new):
    if dirs == []:
        pass
    else:
        classes_new= dirs
        # for dir_ in dirs:
        #     if not os.path.exists(os.path.join(path_new, dir_)):
        #         os.makedirs(os.path.join(path_new, dir_))
        continue
for root, dirs, files in os.walk(path):
    if dirs == []:
        pass
    else:
        classes = dirs
        # for dir_ in dirs:
        #     if not os.path.exists(os.path.join(path_new, dir_)):
        #         os.makedirs(os.path.join(path_new, dir_))
        continue
    # print(count, "_________", root, dirs, files)
    file_num_list.append(len(files))
    count += 1

    name = root.split('/')[-1]
    if name in classes_new:
        print(name, "was done already. ")
        continue
    print(count, name)
    print(root, os.path.join(path_new, name))

    p = Augmentor.Pipeline(source_directory=root,
                           output_directory=os.path.join(path_new, name))

    p.rotate_without_crop(probability=0.5, max_left_rotation=25, max_right_rotation=25)
    p.flip_left_right(probability=0.5)
    # p.random_erasing(probability=0.5, rectangle_area=0.5)
    p.add_operation(ResizeCenterPadding(probability=1, desired_size=112))
    try:
        p.sample(int(pics_num/2))
    except Exception as e:
        print(e)
        empty += 1

    p.skew_tilt(probability=0.5, magnitude=0.2)
    p.skew_corner(probability=0.5, magnitude=0.2)
    p.flip_left_right(probability=0.5)
    p.add_operation(ResizeCenterPadding(probability=1, desired_size=112))
    try:
        p.sample(int(pics_num / 2))
    except Exception as e:
        print(e)
        empty += 1

print("classes", classes)
print(empty, count)
