#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : prroi_pool.py
# Author : Jiayuan Mao, Tete Xiao
# Email  : maojiayuan@gmail.com, jasonhsiao97@gmail.com
# Date   : 07/13/2018
#
# This file is part of PreciseRoIPooling.
# Distributed under terms of the MIT license.
# Copyright (c) 2017 Megvii Technology Limited.

import torch.nn as nn

from .functional import prroi_pool2d
import torch
import torch
import torch.nn.functional as F

__all__ = ['PrRoIPool2D']


class PrRoIPool2D(nn.Module):
    def __init__(self, pooled_height, pooled_width, spatial_scale):
        super().__init__()

        self.pooled_height = int(pooled_height)
        self.pooled_width = int(pooled_width)
        self.spatial_scale = float(spatial_scale)


        self.mp = nn.MaxPool2d(2, stride=6)

    def forward(self, features, rois):
        # assert rois.dim() == 2
        # assert rois.size(1) == 5
        # output = []
        # rois = rois.data.float()
        # num_rois = rois.size(0)
        #
        # rois[:, 1:].mul_(self.spatial_scale)
        # rois = rois.long()
        # size = (self.pooled_height, self.pooled_width)
        # for j in range(features.shape[0]):
        #     tem_output = []
        #     for i in range(num_rois):
        #         roi = rois[i]
        #         im_idx = roi[0]
        #         im = features[j].narrow(0, im_idx, 1)[..., roi[2]:(roi[4] + 1), roi[1]:(roi[3] + 1)]
        #         tem_output.append(F.adaptive_max_pool2d(im, size))
        #
        #     tem_output = torch.cat(tem_output, 0)
        # #has_backward = True
        # #if has_backward:
        #     #        output.backward(output.data.clone())
        #     #output.sum().backward()
        #     output.append(tem_output)
        # output.view(rois.shape[0], -1, self.pooled_height, self.pooled_width)
        output = self.mp(features)
        # return prroi_pool2d(features, rois, self.pooled_height, self.pooled_width, self.spatial_scale)
        return output

    def extra_repr(self):
        return 'kernel_size=({pooled_height}, {pooled_width}), spatial_scale={spatial_scale}'.format(**self.__dict__)

