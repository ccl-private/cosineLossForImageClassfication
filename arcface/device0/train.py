# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
python train.py
"""
import argparse
import logging
import os
import numpy as np

import mindspore.nn as nn
from mindspore import context, Tensor
import mindspore.ops as ops
from mindspore.context import ParallelMode
from mindspore.train.model import Model
from mindspore import dtype as mstype
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor, Callback
from mindspore.communication.management import init
from mindspore.communication import management as MutiDev
from mindspore.parallel import _cost_model_context as cost_model_context
from mindspore.parallel import set_algo_parameters

# from src.dataset import create_dataset
from src.iresnet import iresnet100
from src.loss import PartialFC

import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as CV
import mindspore.dataset.transforms.c_transforms as C
import time

import math
from mindspore.ops import functional as F


def create_dataset(dataset_path, batch_size=24, repeat_num=1, do_train=True, target="GPU"):
    """定义数据集"""
    data_set = ds.ImageFolderDataset(dataset_path, num_parallel_workers=20, shuffle=True)

    image_size = 112
    mean = [0.5 * 255, 0.5 * 255, 0.5 * 255]
    std = [0.5 * 255, 0.5 * 255, 0.5 * 255]

    # define map operations
    if do_train:
        trans = [
            # C.RandomCropDecodeResize(image_size, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
            CV.Decode(),
            CV.Resize((112, 112)),
            # CV.CenterCrop(image_size),
            CV.RandomHorizontalFlip(prob=0.5),
            CV.Normalize(mean=mean, std=std),
            CV.HWC2CHW()
        ]
    else:
        trans = [
            CV.Decode(),
            CV.Resize((112, 112)),
            # CV.CenterCrop(image_size),
            CV.Normalize(mean=mean, std=std),
            CV.HWC2CHW()
        ]

    type_cast_op = C.TypeCast(mstype.int32)

    data_set = data_set.map(input_columns="image",
                num_parallel_workers=1, operations=trans)
    data_set = data_set.map(input_columns="label", num_parallel_workers=1,
                operations=type_cast_op)

    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=True)

    # apply dataset repeat operation
    data_set = data_set.repeat(repeat_num)

    return data_set


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Datasets
parser.add_argument('--train_url', default='.', type=str,
                    help='output path')
parser.add_argument('--data_url', default='/data/ccl/RP2K_rp2k_dataset/all/train_augmented/', type=str)
# Optimization options
parser.add_argument('--epochs', default=25, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--num_classes', default=2388, type=int, metavar='N',
                    help='num of classes')
parser.add_argument('--batch_size', default=128, type=int, metavar='N',
                    help='train batchsize (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.08, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--schedule', type=int, nargs='+', default=[10, 16, 21],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Device options
parser.add_argument('--device_target', type=str,
                    default='GPU', choices=['GPU', 'Ascend', 'CPU'])
parser.add_argument('--device_num', type=int, default=2)
parser.add_argument('--device_id', type=int, default=0)
parser.add_argument('--modelarts', type=bool, default=False)

args = parser.parse_args()


def lr_generator(lr_init, total_epochs, steps_per_epoch):
    '''lr_generator
    '''
    lr_each_step = []
    for i in range(total_epochs):
        if i in args.schedule:
            lr_init *= args.gamma
        for _ in range(steps_per_epoch):
            lr_each_step.append(lr_init)
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    return Tensor(lr_each_step)


class MyNetWithLoss(nn.Cell):
    '''
    WithLossCell
    '''
    def __init__(self, backbone, cfg):
        super(MyNetWithLoss, self).__init__(auto_prefix=False)
        self._backbone = backbone.to_float(mstype.float16)
        self._loss_fn = PartialFC(num_classes=cfg.num_classes,
                                  world_size=cfg.device_num).to_float(mstype.float32)
        self.L2Norm = ops.L2Normalize(axis=1)

    def construct(self, data, label):
        out = self._backbone(data)
        loss = self._loss_fn(out, label)
        return loss


if __name__ == "__main__":
    train_epoch = args.epochs
    target = args.device_target
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=target, save_graphs=False)
    if args.device_num > 1:
        device_id = int(os.getenv('DEVICE_ID'))
        print(device_id)
        context.set_context(device_id=device_id)
    else:
        # context.set_context(device_id=args.device_id)
        context.set_context(device_target="GPU")
    if args.device_num > 1:
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True,
                                          )
        context.set_auto_parallel_context(parameter_broadcast=True)
        cost_model_context.set_cost_model_context(device_memory_capacity=32.0 * 1024.0 * 1024.0 * 1024.0,
                                                  costmodel_gamma=0.001,
                                                  costmodel_beta=280.0)
        set_algo_parameters(elementwise_op_strategy_follow=True)
        init()

    if args.modelarts:
        import moxing as mox

        mox.file.copy_parallel(
            src_url=args.data_url, dst_url='/cache/data_path_' + os.getenv('DEVICE_ID'))
        zip_command = "unzip -o -q /cache/data_path_" + os.getenv('DEVICE_ID') \
                      + "/MS1M.zip -d /cache/data_path_" + \
            os.getenv('DEVICE_ID')
        os.system(zip_command)
        train_dataset = create_dataset(dataset_path='/cache/data_path_' + os.getenv('DEVICE_ID') + '/MS1M/',
                                       do_train=True,
                                       repeat_num=1, batch_size=args.batch_size, target=target)
    else:
        train_dataset = create_dataset(dataset_path=args.data_url, do_train=True,
                                       repeat_num=1, batch_size=args.batch_size, target=target)
    step = train_dataset.get_dataset_size()
    lr = lr_generator(args.lr, train_epoch, steps_per_epoch=step)
    net = iresnet100()
    train_net = MyNetWithLoss(net, args)

    from mindspore import load_checkpoint, load_param_into_net

    # 将模型参数存入parameter的字典中
    param_dict = load_checkpoint("/home/ccl/Documents/codes/python/minds/models/research/cv/arcface/backup/CKP-12_6765.ckpt")
    # 将参数加载到网络中
    load_param_into_net(train_net, param_dict)

    optimizer = nn.SGD(params=train_net.trainable_params(), learning_rate=args.lr / 512 * args.batch_size * args.device_num,
                       momentum=args.momentum, weight_decay=args.weight_decay)

    model = Model(train_net, optimizer=optimizer)

    time_cb = TimeMonitor(data_size=train_dataset.get_dataset_size())
    loss_cb = LossMonitor()
    cb = [time_cb, loss_cb]
    config_ck = CheckpointConfig(
        save_checkpoint_steps=60, keep_checkpoint_max=5)
    # if args.modelarts:
    #     ckpt_cb = ModelCheckpoint(prefix="ArcFace-", config=config_ck,
    #                               directory='/cache/train_output/')
    #     cb.append(ckpt_cb)
    # else:
    #     if args.device_num == 8 and MutiDev.get_rank() % 8 == 0:
    #         ckpt_cb = ModelCheckpoint(prefix="ArcFace-", config=config_ck,
    #                                   directory=args.train_url)
    #         cb.append(ckpt_cb)
    #     if args.device_num == 1:
    #         ckpt_cb = ModelCheckpoint(prefix="ArcFace-", config=config_ck,
    #                                   directory=args.train_url)
    #         cb.append(ckpt_cb)


    class StopAtTime(Callback):
        import logging
        def __init__(self, run_time):
            super(StopAtTime, self).__init__()
            self.run_time = run_time * 60

        def begin(self, run_context):
            cb_params = run_context.original_args()
            cb_params.init_time = time.time()

        def step_end(self, run_context):
            cb_params = run_context.original_args()
            epoch_num = cb_params.cur_epoch_num
            step_num = cb_params.cur_step_num
            loss = cb_params.net_outputs

            cur_time = time.time()
            logging.log(level=31, msg="epoch: " + str(epoch_num) + " step: " + str(step_num) + " loss: " + str(loss))
            # run_context.request_stop()


    def learning_rate_function(lr, cur_epoch_num):
        if cur_epoch_num in args.schedule:
            lr *= args.gamma
        return lr


    class LearningRateScheduler(Callback):
        """
        Change the learning_rate during training.
        Note:
            This class is not supported on CPU.
        Args:
            learning_rate_function (Function): The function about how to change the learning rate during training.
        Examples:
            >>> #from _lr_scheduler_callback import LearningRateScheduler
            >>> import mindspore.nn as nn
            >>> from mindspore.train import Model
            ...
            >>> def learning_rate_function(lr, cur_step_num):
            ...     if cur_step_num%1000 == 0:
            ...         lr = lr*0.1
            ...     return lr
            ...
            >>> lr = 0.1
            >>> momentum = 0.9
            >>> net = Net()
            >>> loss = nn.SoftmaxCrossEntropyWithLogits()
            >>> optim = nn.Momentum(net.trainable_params(), learning_rate=lr, momentum=momentum)
            >>> model = Model(net, loss_fn=loss, optimizer=optim)
            ...
            >>> dataset = create_custom_dataset("custom_dataset_path")
            >>> model.train(1, dataset, callbacks=[LearningRateScheduler(learning_rate_function)],
            ...             dataset_sink_mode=False)
        """

        def __init__(self, learning_rate_function):
            super(LearningRateScheduler, self).__init__()
            self.learning_rate_function = learning_rate_function

        def step_end(self, run_context):
            cb_params = run_context.original_args()
            arr_lr = cb_params.optimizer.learning_rate.asnumpy()
            lr = float(np.array2string(arr_lr))
            logging.log(level=31, msg="lr: " + str(lr))
            new_lr = self.learning_rate_function(lr, cb_params.cur_epoch_num)
            if not math.isclose(lr, new_lr, rel_tol=1e-10):
                F.assign(cb_params.optimizer.learning_rate, Tensor(new_lr, mstype.float32))
                logging.log(level=31, msg=f'At step {cb_params.cur_step_num}, learning_rate change to {new_lr}')


    stop_cb = StopAtTime(run_time=10)

    ckp_cb = ModelCheckpoint(directory=args.train_url)

    lr_cd = LearningRateScheduler(learning_rate_function)

    model.train(train_epoch, train_dataset, callbacks=[stop_cb, ckp_cb, lr_cd], dataset_sink_mode=True)
    if args.modelarts:
        mox.file.copy_parallel(
            src_url='/cache/train_output', dst_url=args.train_url)
