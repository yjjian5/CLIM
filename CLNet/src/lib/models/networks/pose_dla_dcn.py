from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import logging
import numpy as np
from os.path import join

import torch
from torch import einsum
from torch import nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from dcn_v2 import DCN

from einops import rearrange, repeat

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def get_model_url(data='imagenet', name='dla34', hash='ba72cf86'):
    return join('http://dl.yf.io/dla/models', data, '{}-{}.pth'.format(name, hash))


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(Bottleneck, self).__init__()
        expansion = Bottleneck.expansion
        bottle_planes = planes // expansion
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class BottleneckX(nn.Module):
    expansion = 2
    cardinality = 32

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BottleneckX, self).__init__()
        cardinality = BottleneckX.cardinality
        # dim = int(math.floor(planes * (BottleneckV5.expansion / 64.0)))
        # bottle_planes = dim * cardinality
        bottle_planes = planes * cardinality // 32
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation, bias=False,
                               dilation=dilation, groups=cardinality)
        self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1,
            stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,
                 dilation=1, root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride,
                               dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1,
                               dilation=dilation)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size,
                             root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA(nn.Module):
    def __init__(self, levels, channels, num_classes=1000,
                 block=BasicBlock, residual_root=False, linear_root=False):
        super(DLA, self).__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
                      padding=3, bias=False),
            nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True))
        # pre_layer
        # self.pre_img_layer = nn.Sequential(
        #     nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
        #               padding=3, bias=False),
        #     nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
        #     nn.ReLU(inplace=True))
        # self.pre_hm_layer = nn.Sequential(
        #     nn.Conv2d(1, channels[0], kernel_size=7, stride=1,
        #               padding=3, bias=False),
        #     nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
        #     nn.ReLU(inplace=True))

        self.level0 = self._make_conv_level(
            channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,
                           level_root=False,
                           root_residual=residual_root)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,
                           level_root=True, root_residual=residual_root)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2,
                           level_root=True, root_residual=residual_root)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2,
                           level_root=True, root_residual=residual_root)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_level(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.MaxPool2d(stride, stride=stride),
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample))
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(inplanes, planes, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x, pre_img=None):
        # todo: backbone dla
        y = []
        x = self.base_layer(x)
        # if pre_img is not None:
        #     x = x + self.pre_img_layer(pre_img)
        # if pre_hm is not None:
        #     x = x + self.pre_hm_layer(pre_hm)
        for i in range(6):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)

        return y

    def load_pretrained_model(self, data='imagenet', name='dla34', hash='ba72cf86'):
        # fc = self.fc
        if name.endswith('.pth'):
            model_weights = torch.load(data + name)
        else:
            model_url = get_model_url(data, name, hash)
            model_weights = model_zoo.load_url(model_url)
        num_classes = len(model_weights[list(model_weights.keys())[-1]])
        self.fc = nn.Conv2d(
            self.channels[-1], num_classes,
            kernel_size=1, stride=1, padding=0, bias=True)
        # TODO: don't mach the key
        self.load_state_dict(model_weights, False)
        # self.fc = fc


def dla34(pretrained=True, **kwargs):  # DLA-34
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 128, 256, 512],
                block=BasicBlock, **kwargs)
    if pretrained:
        model.load_pretrained_model(data='imagenet', name='dla34', hash='ba72cf86')
    return model


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class DeformConv(nn.Module):
    def __init__(self, chi, cho):
        super(DeformConv, self).__init__()
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.conv = DCN(chi, cho, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deformable_groups=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.actf(x)
        return x


class IDAUp(nn.Module):

    def __init__(self, o, channels, up_f):
        super(IDAUp, self).__init__()
        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])
            proj = DeformConv(c, o)
            node = DeformConv(o, o)

            up = nn.ConvTranspose2d(o, o, f * 2, stride=f,
                                    padding=f // 2, output_padding=0,
                                    groups=o, bias=False)
            fill_up_weights(up)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)

    def forward(self, layers, startp, endp):
        for i in range(startp + 1, endp):
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            node = getattr(self, 'node_' + str(i - startp))
            layers[i] = node(layers[i] + layers[i - 1])


class DLAUp(nn.Module):
    def __init__(self, startp, channels, scales, in_channels=None):
        super(DLAUp, self).__init__()
        self.startp = startp
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self, 'ida_{}'.format(i),
                    IDAUp(channels[j], in_channels[j:],
                          scales[j:] // scales[j]))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        out = [layers[-1]]  # start with 32
        for i in range(len(layers) - self.startp - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            ida(layers, len(layers) - i - 2, len(layers))
            out.insert(0, layers[-1])
        return out


class Interpolate(nn.Module):
    def __init__(self, scale, mode):
        super(Interpolate, self).__init__()
        self.scale = scale
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode=self.mode, align_corners=False)
        return x


# from .transformer import Transformer as transformer_fusion
# from .transformer import Transformer_Myself as transformer_myself
#from .transformer import CrissCrossAttention as CCAtt

# TODO: Fusion


class DLASeg(nn.Module):
    def __init__(self, base_name, heads, pretrained, down_ratio, final_kernel,
                 last_level, head_conv, out_channel=0):
        super(DLASeg, self).__init__()

        # backbone part
        assert down_ratio in [2, 4, 8, 16]
        self.first_level = int(np.log2(down_ratio))
        self.last_level = last_level
        self.base = globals()[base_name](pretrained=pretrained)
        channels = self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        # DLA_UP
        self.dla_up = DLAUp(self.first_level, channels[self.first_level:], scales)

        if out_channel == 0:
            out_channel = channels[self.first_level]

        # IDA_UP
        self.ida_up = IDAUp(out_channel, channels[self.first_level:self.last_level],
                            [2 ** i for i in range(self.last_level - self.first_level)])

        # TODO: *Transformer part
        # self.transformer = transformer_fusion(d_model=256, nhead=1, num_layers=1)
        # self.transformer = transformer_fusion(dim=64)
        self.downsample = nn.Sequential(
            nn.Conv2d(channels[self.first_level], 8,
                      kernel_size=1, bias=True),
            # nn.ReLU(inplace=True),
            # # nn.AvgPool2d(kernel_size=4, stride=4, padding=0),
            # nn.Conv2d(32, 8,
            #           kernel_size=1, bias=True),
            # nn.ReLU(inplace=True)
            # nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
        )
        self.upsample = nn.Sequential(
            # nn.UpsamplingBilinear2d(scale_factor=4),
            # nn.Conv2d(8, 32,
            #           kernel_size=1, bias=True),
            # nn.ReLU(inplace=True),
            nn.Conv2d(8, channels[self.first_level],
                      kernel_size=1, bias=True),
            # nn.ReLU(inplace=True)
            # nn.UpsamplingBilinear2d(scale_factor=2)
        )
        # # self.layers_cc = nn.ModuleList([])
        # # for _ in range(2):
        # self.cc_att = CCAtt(in_dim=64)
        #     # self.layers_cc.append(nn.ModuleList([cc_att]))


        # TODO: timesformer part
        self.timesTs = TSTrans(
            attn_dropout=0.1,
            ff_dropout=0.1
        )
        self.glob_weight = 0.01
        self.front_weight = 0.01

        # head part
        self.heads = heads
        for head in self.heads:
            # get the classes, hm:1, wh:2, reid:128
            classes = self.heads[head]
            # head_conv:256
            if head_conv > 0:
                fc = nn.Sequential(
                    nn.Conv2d(channels[self.first_level], head_conv,
                              kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, classes,
                              kernel_size=final_kernel, stride=1,
                              padding=final_kernel // 2, bias=True))
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(channels[self.first_level], classes,
                               kernel_size=final_kernel, stride=1,
                               padding=final_kernel // 2, bias=True)
                if 'hm' in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)

    # add patch transformer
    def forward(self, x, pre_image=None, pre_hm=None):
        # Original
        # def forward(self, x):
        #     # backbone
        #     x = self.base(x)
        #     x = self.dla_up(x)
        #
        #     y = []
        #     for i in range(self.last_level - self.first_level):
        #         y.append(x[i].clone())
        #
        #     self.ida_up(y, 0, len(y))
        #
        #     # head
        #     z = {}
        #     for head in self.heads:
        #         # head attribution after y[-1]
        #         z[head] = self.__getattr__(head)(y[-1])
        #     return [z]

        # # backbone
        # # x, pre_image: batch,3,608,1088; pre_hm: batch,1,152,272

        # TODO: use pre_image, batch size minus
        # backbone
        x = self.base(x)
        x = self.dla_up(x)

        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())

        self.ida_up(y, 0, len(y))

        last_ft = y[-1]

        # if pre_image is not None:
        #     pre_x = self.base(pre_image)
        #     pre_x = self.dla_up(pre_x)
        #     pre_y = []
        #     for i in range(self.last_level - self.first_level):
        #         pre_y.append(pre_x[i].clone())
        #     self.ida_up(pre_y, 0, len(pre_y))
        #
        #     # todo: Transformer part
        #     # compress the channel from 64 to 8
        #     ft = self.downsample(y[-1])
        #     pre_ft = self.downsample(pre_y[-1])
        #
        #     # read the size of origin tensor
        #     batch_size, feature_dim, feature_h, feature_w = ft.shape
        #
        #     if pre_hm is not None:
        #         pre_hm = pre_hm.expand(batch_size, feature_dim, feature_h, feature_w)
        #         pre_hm_ft = pre_ft * pre_hm
        #
        #     # connect current feature and previous feature
        #     fu_ft_1 = torch.cat((ft.unsqueeze(1), pre_ft.unsqueeze(1)), dim=1)
        #
        #     # do timeS transformer
        #     fu_ft_1 = self.timesTs(fu_ft_1)
        #
        #     # split current feature and previous feature
        #     cur_ft_attn, pre_ft_attn = fu_ft_1.split([1, 1], dim=1)
        #     cur_ft_attn = cur_ft_attn.squeeze(1)
        #     # pre_ft_attn = pre_ft_attn.squeeze(1)
        #
        #     if pre_hm is not None:
        #         fu_ft_2 = torch.cat((ft.unsqueeze(1), pre_hm_ft.unsqueeze(1)), dim=1)
        #         fu_ft_2 = self.timesTs(fu_ft_2)
        #         cur_hm_ft_attn, pre_hm_ft_attn = fu_ft_2.split([1, 1], dim=1)
        #         cur_hm_ft_attn = cur_hm_ft_attn.squeeze(1)
        #         # pre_hm_ft_attn = pre_hm_ft_attn.squeeze(1)
        #
        #     # todo: add self
        #     cur_ft_attn = self.upsample(cur_ft_attn).sigmoid_()
        #     cur_hm_ft_attn = self.upsample(cur_hm_ft_attn).sigmoid_()
        #
        #     # last_ft = y[-1] + self.glob_weight * cur_ft_attn + self.front_weight * cur_hm_ft_attn
        #     # last_ft = ((y[-1] * cur_ft_attn) + (y[-1] * cur_hm_ft_attn)) * 0.001 + y[-1]
        #     last_ft = y[-1] * cur_ft_attn * self.glob_weight + y[-1] * cur_hm_ft_attn * self.front_weight + y[-1]
        #     # last_ft = y[-1] * cur_ft_attn * self.glob_weight + y[-1]
        #     # last_ft = y[-1] * cur_hm_ft_attn * self.front_weight + y[-1]

        # head
        z = {}
        for head in self.heads:
            # head attribution after y[-1]
            z[head] = self.__getattr__(head)(last_ft)
        return [z]


# todo: TimeS Transformer
class TSTrans(nn.Module):
    def __init__(
        self,
        *,
        dim = 512,
        num_frames = 2,
        image_h = 152,
        image_w = 272,
        patch_size = 8,
        channels = 8, # origin 3
        depth = 1, # origin 12
        heads = 8,
        dim_head = 64,
        attn_dropout = 0.,
        ff_dropout = 0.,
    ):
        super().__init__()
        assert image_h % patch_size == 0
        assert image_w % patch_size == 0

        num_patches = (image_h // patch_size) * (image_w // patch_size)
        num_positions = num_frames * num_patches
        patch_dim = channels * patch_size ** 2

        self.heads = heads
        self.patch_size = patch_size
        # embedding 512 -> 512?
        self.to_patch_embedding = nn.Linear(patch_dim, dim)
        # self.cls_token = nn.Parameter(torch.randn(1, dim))

        # no class token
        # self.pos_emb = nn.Embedding(num_positions + 1, dim)
        self.pos_emb = nn.Embedding(num_positions, dim)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            ff = FeedForward(dim, dropout=ff_dropout)
            time_attn = Attention(dim, dim_head=dim_head, heads=heads, dropout=attn_dropout)
            spatial_attn = Attention(dim, dim_head=dim_head, heads=heads, dropout=attn_dropout)

            time_attn, spatial_attn, ff = map(lambda t: PreNorm(dim, t), (time_attn, spatial_attn, ff))

            self.layers.append(nn.ModuleList([time_attn, spatial_attn, ff]))

        self.to_out = nn.Sequential(
            nn.LayerNorm(dim),
            # nn.Linear(dim, num_classes)
        )

    def forward(self, video):
        b, f, _, h, w, *_, device, p = *video.shape, video.device, self.patch_size
        assert h % p == 0 and w % p == 0, f'height {h} and width {w} of video must be divisible by the patch size {p}'

        # calculate num patches in height and width dimension, and number of total patches (n)
        hp, wp = (h // p), (w // p)
        n = hp * wp

        # video to patch embeddings
        video = rearrange(video, 'b f c (h p1) (w p2) -> b (f h w) (p1 p2 c)', p1 = p, p2 = p)
        # todo: whether to do this?
        tokens = self.to_patch_embedding(video)

        x = tokens.clone()

        # positional embedding
        x += self.pos_emb(torch.arange(x.shape[1], device = device))

        # time and space attention
        for (time_attn, spatial_attn, ff) in self.layers:
            x = time_attn(x, 'b (f n) d', '(b n) f d', n = n, mask = None, cls_mask = None, rot_emb = None) + x
            x = spatial_attn(x, 'b (f n) d', '(b f) n d', f = f, cls_mask = None, rot_emb = None) + x
            x = ff(x) + x

        x = self.to_out(x)

        # reshape to origin size
        x = rearrange(x, 'b (f h w) (p1 p2 c) -> b f c (h p1) (w p2)', f = f, h = hp, w = wp, p1 = p, p2 = p)

        return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            # nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)


# attention
def attn(q, k, v):
    sim = einsum('b i d, b j d -> b i j', q, k)
    attn = sim.softmax(dim = -1)
    out = einsum('b i j, b j d -> b i d', attn, v)
    return out


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            # nn.Dropout(dropout)
        )

    def forward(self, x, einops_from, einops_to, mask = None, cls_mask = None, rot_emb = None, **einops_dims):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        q = q * self.scale

        # rearrange across time or space
        q_, k_, v_ = map(lambda t: rearrange(t, f'{einops_from} -> {einops_to}', **einops_dims), (q, k, v))

        # attention
        out = attn(q_, k_, v_)

        # merge back time or space
        out = rearrange(out, f'{einops_to} -> {einops_from}', **einops_dims)

        # merge back the heads
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)

        # combine heads out
        return self.to_out(out)

def get_pose_net(num_layers, heads, head_conv=256, down_ratio=4):
    # create the model(34, heads, 256, 4)
    model = DLASeg('dla{}'.format(num_layers), heads,
                   pretrained=True,
                   down_ratio=down_ratio,
                   final_kernel=1,
                   last_level=5,
                   head_conv=head_conv)
    return model

