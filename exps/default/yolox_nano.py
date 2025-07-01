# #!/usr/bin/env python3
# # -*- coding:utf-8 -*-
# # Copyright (c) Megvii, Inc. and its affiliates.

# import os

# import torch.nn as nn

# from yolox.exp import Exp as MyExp


# class Exp(MyExp):
#     def __init__(self):
#         super(Exp, self).__init__()
#         self.depth = 0.33
#         self.width = 0.25
#         self.input_size = (416, 416)
#         self.random_size = (10, 20)
#         self.mosaic_scale = (0.5, 1.5)
#         self.test_size = (416, 416)
#         self.mosaic_prob = 0.5
#         self.enable_mixup = False
#         self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

#     def get_model(self, sublinear=False):

#         def init_yolo(M):
#             for m in M.modules():
#                 if isinstance(m, nn.BatchNorm2d):
#                     m.eps = 1e-3
#                     m.momentum = 0.03
#         if "model" not in self.__dict__:
#             from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
#             in_channels = [256, 512, 1024]
#             # NANO model use depthwise = True, which is main difference.
#             backbone = YOLOPAFPN(
#                 self.depth, self.width, in_channels=in_channels,
#                 act=self.act, depthwise=True,
#             )
#             head = YOLOXHead(
#                 self.num_classes, self.width, in_channels=in_channels,
#                 act=self.act, depthwise=True
#             )
#             self.model = YOLOX(backbone, head)

#         self.model.apply(init_yolo)
#         self.model.head.initialize_biases(1e-2)
#         return self.model




# New one for nano
# exps/custom_human_detection.py
# Custom experiment config that matches your trained model

import os
import torch
from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        
        # Model architecture settings (matches your checkpoint)
        self.depth = 0.33       # YOLOX-Nano depth multiplier
        self.width = 0.25       # YOLOX-Nano width multiplier  
        
        # CRITICAL: Use regular convolutions, not depthwise separable
        self.act = "silu"
        self.depthwise = False  # This is KEY - disables depthwise separable convs
        
        # Class settings (matches your single-class training)
        self.num_classes = 1    # Only human class
        self.class_names = ["person"]
        
        # Input settings (matches your training)
        self.input_size = (416, 416)  # height, width
        self.test_size = (416, 416)
        self.random_size = (10, 20)  # multiscale training range
        
        # Training settings
        self.max_epoch = 300
        self.warmup_epochs = 5
        self.basic_lr_per_img = 0.01 / 64.0
        self.scheduler = "yoloxwarmcos"
        self.no_aug_epochs = 15
        self.min_lr_ratio = 0.05
        self.ema = True

        # Data augmentation
        self.mosaic_prob = 1.0
        self.mixup_prob = 1.0
        self.hsv_prob = 1.0
        self.flip_prob = 0.5
        self.degrees = 10.0
        self.translate = 0.1
        self.mosaic_scale = (0.1, 2)
        self.enable_mixup = True

        # Evaluation settings
        self.eval_interval = 10
        self.test_conf = 0.01
        self.nmsthre = 0.65

        # Data loader settings
        self.data_num_workers = 4
        self.multiscale_training = True

        # Output directory
        self.output_dir = "./YOLOX_outputs"
        self.print_interval = 10
        self.save_history_ckpt = True

    def get_model(self):
        from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(
                self.depth, 
                self.width, 
                in_channels=in_channels,
                act=self.act,
                depthwise=self.depthwise,  # KEY: Use regular convolutions
            )
            head = YOLOXHead(
                self.num_classes, 
                self.width, 
                in_channels=in_channels,
                act=self.act,
                depthwise=self.depthwise,  # KEY: Use regular convolutions
            )
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        # This would be your custom data loader
        # For demo purposes, we'll use a simple placeholder
        pass

    def get_optimizer(self, batch_size):
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.basic_lr_per_img * batch_size

            pg0, pg1, pg2 = [], [], []  # optimizer parameter groups

            for k, v in self.model.named_modules():
                if hasattr(v, "bias") and isinstance(v.bias, torch.nn.Parameter):
                    pg2.append(v.bias)  # biases
                if isinstance(v, torch.nn.BatchNorm2d) or "bn" in k:
                    pg0.append(v.weight)  # no decay
                elif hasattr(v, "weight") and isinstance(v.weight, torch.nn.Parameter):
                    pg1.append(v.weight)  # apply decay

            optimizer = torch.optim.SGD(
                pg0, lr=lr, momentum=self.momentum, nesterov=True
            )
            optimizer.add_param_group(
                {"params": pg1, "weight_decay": self.weight_decay}
            )  # add pg1 with weight_decay
            optimizer.add_param_group({"params": pg2})
            self.optimizer = optimizer

        return self.optimizer

    def get_lr_scheduler(self, lr, iters_per_epoch):
        from yolox.utils import LRScheduler

        scheduler = LRScheduler(
            self.scheduler,
            lr,
            iters_per_epoch,
            self.max_epoch,
            warmup_epochs=self.warmup_epochs,
            warmup_lr_start=self.warmup_lr,
            no_aug_epochs=self.no_aug_epochs,
            min_lr_ratio=self.min_lr_ratio,
        )
        return scheduler

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        # Evaluation data loader
        pass

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        # Custom evaluator for single class
        from yolox.evaluators import COCOEvaluator
        return COCOEvaluator(
            dataloader=self.get_eval_loader(batch_size, is_distributed, testdev, legacy),
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )