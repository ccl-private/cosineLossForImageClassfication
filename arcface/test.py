import Augmentor
from Augmentor.Pipeline import Operation
import cv2
import numpy as np
from PIL import Image


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


p = Augmentor.Pipeline(source_directory="/data/ccl/RP2K_rp2k_dataset/all/train_new/贝纳颂经典系列250ml蓝山",
                       output_directory="/data/ccl/RP2K_rp2k_dataset/all/train_augmented_small")

# p.rotate_without_crop(probability=0.5, max_left_rotation=25, max_right_rotation=25)
p.random_erasing(probability=1, rectangle_area=0.5)
# p.add_operation(ResizeCenterPadding(probability=1, desired_size=112))
p.sample(333)

# p.skew_tilt(probability=0.5, magnitude=1)
# p.skew_corner(probability=0.5, magnitude=1)
# p.add_operation(ResizeCenterPadding(probability=1, desired_size=112))
# p.sample(333)
