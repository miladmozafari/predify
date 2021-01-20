import torch
import torchvision.models as models
import torchvision
import torchvision.transforms as transforms
from   torch.utils.data import DataLoader
from   torch.nn import Sequential, Conv2d, MaxPool2d, BatchNorm2d, ReLU, Flatten, Linear, ConvTranspose2d, Sigmoid, LayerNorm, InstanceNorm2d, Upsample
from   datetime import datetime
import torch.optim as optim
import torch.nn as nn

import os
import sys
import numpy as np
from ..modules import PCoder, PCoderN
from ..utils import to_pair
from . import PNetSameHP, PNetSeparateHP

def get_cifar_resnet18():
    resnet         = models.resnet18()
    resnet.conv1   = Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    resnet.maxpool = MaxPool2d(kernel_size=1, stride=1, ceil_mode=False)
    resnet.fc      = Linear(in_features=512, out_features=100, bias=True)
    return resnet

def get_deep_info(sequence, i, j):
    """
    Returns proper kernel-size,stride,padding for the conv-transpose
    from layer i (higher layer) to j (lower layer) in a sequential.
    """
    P = (0,0)   # deep padding
    R = (1,1)   # deep receptive-field
    D = (1,1)   # deep stride

    in_ch = 0
    out_ch = 0
    for idx in range(i,j,-1):
        if isinstance(sequence[idx], nn.Conv2d) or isinstance(sequence[idx], nn.MaxPool2d):
            if isinstance(sequence[idx], nn.Conv2d) and in_ch == 0:
                in_ch = sequence[idx].out_channels

            s = to_pair(sequence[idx].stride)
            k = to_pair(sequence[idx].kernel_size)
            p = to_pair(sequence[idx].padding)
            P0, P1 = P
            if P0 != 0:
                P0 = k[0] + (P0-1) * s[0] - (k[0] - s[0]) + p[0]
            if P1 != 0:
                P1 = k[1] + (P1-1) * s[1] - (k[1] - s[1]) + p[1]
            P = (P0, P1)
            p = to_pair(sequence[idx].padding)
            P = (max(P[0], p[0]), max(P[1], p[1]))

    for idx in range(j,i):
        if idx >= 0:
            if isinstance(sequence[idx], nn.Conv2d) or isinstance(sequence[idx], nn.MaxPool2d):
                k = sequence[idx].kernel_size if isinstance(sequence[idx].kernel_size, tuple) else (sequence[idx].kernel_size, sequence[idx].kernel_size)
                s = sequence[idx].stride if isinstance(sequence[idx].stride, tuple) else (sequence[idx].stride, sequence[idx].stride)
                R = (R[0] + ((k[0] - 1) * D[0]), R[1] + ((k[1] - 1) * D[1]))
                D = (D[0] * s[0], D[1] * s[1])
    if j == -1:
        out_ch = 3
    else:
        for idx in range(j,-1,-1):
            if isinstance(sequence[idx], nn.Conv2d):
                out_ch = sequence[idx].out_channels
                break

    return in_ch, out_ch, R, D, P

def flatten_resnet(net):
    """
    flattens resnet into a sequence of modules. output of this function will be
    used to compute convtranspose parameters.
    """
    modules = []
    for m in net.children():
        if isinstance(m, nn.Sequential):
            for mm in m:
                modules.extend(list(mm.children()))
        else:
            modules.append(m)
    return modules

class PResNet18V3SameHP(PNetSameHP):
    """
    3x3 conv transpose kernels and upsample
    """
    def __init__(self, resnet, build_graph=False, random_init=True, ff_multiplier: float=0.33, fb_multiplier: float=0.33, er_multiplier: float=0.01):
        super().__init__(resnet, 5, build_graph, random_init, ff_multiplier, fb_multiplier, er_multiplier)

        resnet_seq = flatten_resnet(resnet)
        
        # create the first PCoder
        in_ch, out_ch, r, d, p = get_deep_info(resnet_seq, 3, -1)
        pmodule = Sequential(ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1))
        self.pcoder1 = PCoder(pmodule, True, self.random_init)

        # create the second PCoder
        in_ch, out_ch, r, d, p = get_deep_info(resnet_seq, 13, 3)
        pmodule = Sequential(ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1))
        self.pcoder2 = PCoder(pmodule, True, self.random_init)

        # create the third PCoder
        in_ch, out_ch, r, d, p = get_deep_info(resnet_seq, 24, 13)
        pmodule = Sequential(Upsample(scale_factor=2, mode='bilinear'), ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1))
        self.pcoder3 = PCoder(pmodule, True, self.random_init)

        # create the fourth PCoder
        in_ch, out_ch, r, d, p = get_deep_info(resnet_seq, 35, 24)
        pmodule = Sequential(Upsample(scale_factor=2, mode='bilinear'), ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1))
        self.pcoder4 = PCoder(pmodule, True, self.random_init)

        # create the fifth PCoder
        in_ch, out_ch, r, d, p = get_deep_info(resnet_seq, 46, 35)
        pmodule = Sequential(Upsample(scale_factor=2, mode='bilinear'), ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1))
        self.pcoder5 = PCoder(pmodule, False, self.random_init)

        def fw_hook1(m, m_in, m_out):
            e = self.pcoder1(ff=m_out, fb=self.pcoder2.prd, target=self.input_mem, build_graph=self.build_graph, ffm=self.ffm, fbm=self.fbm, erm=self.erm)
            return e[0]
        self.backbone.conv1.register_forward_hook(fw_hook1)

        def fw_hook2(m, m_in, m_out):
            e = self.pcoder2(ff=m_out, fb=self.pcoder3.prd, target=self.pcoder1.rep, build_graph=self.build_graph, ffm=self.ffm, fbm=self.fbm, erm=self.erm)
            return e[0]
        self.backbone.layer1[1].conv2.register_forward_hook(fw_hook2)

        def fw_hook3(m, m_in, m_out):
            e = self.pcoder3(ff=m_out, fb=self.pcoder4.prd, target=self.pcoder2.rep, build_graph=self.build_graph, ffm=self.ffm, fbm=self.fbm, erm=self.erm)
            return e[0]
        self.backbone.layer2[1].conv2.register_forward_hook(fw_hook3)

        def fw_hook4(m, m_in, m_out):
            e = self.pcoder4(ff=m_out, fb=self.pcoder5.prd, target=self.pcoder3.rep, build_graph=self.build_graph, ffm=self.ffm, fbm=self.fbm, erm=self.erm)
            return e[0]
        self.backbone.layer3[1].conv2.register_forward_hook(fw_hook4)

        def fw_hook5(m, m_in, m_out):
            e = self.pcoder5(ff=m_out, fb=None, target=self.pcoder4.rep, build_graph=self.build_graph, ffm=self.ffm, fbm=self.fbm, erm=self.erm)
            return e[0]
        self.backbone.layer4[1].conv2.register_forward_hook(fw_hook5)

class PResNet18V3SeparateHP(PNetSeparateHP):
    """
    3x3 conv transpose kernels and upsample
    """
    def __init__(self, resnet, build_graph=False, random_init=True, ff_multiplier: float=0.33, fb_multiplier: float=0.33, er_multiplier: float=0.01):
        super().__init__(resnet, 5, build_graph, random_init, ff_multiplier, fb_multiplier, er_multiplier)

        resnet_seq = flatten_resnet(resnet)
        
        # create the first PCoder
        in_ch, out_ch, r, d, p = get_deep_info(resnet_seq, 3, -1)
        pmodule = Sequential(ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1))
        self.pcoder1 = PCoder(pmodule, True, self.random_init)

        # create the second PCoder
        in_ch, out_ch, r, d, p = get_deep_info(resnet_seq, 13, 3)
        pmodule = Sequential(ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1))
        self.pcoder2 = PCoder(pmodule, True, self.random_init)

        # create the third PCoder
        in_ch, out_ch, r, d, p = get_deep_info(resnet_seq, 24, 13)
        pmodule = Sequential(Upsample(scale_factor=2, mode='bilinear'), ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1))
        self.pcoder3 = PCoder(pmodule, True, self.random_init)

        # create the fourth PCoder
        in_ch, out_ch, r, d, p = get_deep_info(resnet_seq, 35, 24)
        pmodule = Sequential(Upsample(scale_factor=2, mode='bilinear'), ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1))
        self.pcoder4 = PCoder(pmodule, True, self.random_init)

        # create the fifth PCoder
        in_ch, out_ch, r, d, p = get_deep_info(resnet_seq, 46, 35)
        pmodule = Sequential(Upsample(scale_factor=2, mode='bilinear'), ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1))
        self.pcoder5 = PCoder(pmodule, False, self.random_init)

        def fw_hook1(m, m_in, m_out):
            e = self.pcoder1(ff=m_out, fb=self.pcoder2.prd, target=self.input_mem, build_graph=self.build_graph, ffm=self.ffm1, fbm=self.fbm1, erm=self.erm1)
            return e[0]
        self.backbone.conv1.register_forward_hook(fw_hook1)

        def fw_hook2(m, m_in, m_out):
            e = self.pcoder2(ff=m_out, fb=self.pcoder3.prd, target=self.pcoder1.rep, build_graph=self.build_graph, ffm=self.ffm2, fbm=self.fbm2, erm=self.erm2)
            return e[0]
        self.backbone.layer1[1].conv2.register_forward_hook(fw_hook2)

        def fw_hook3(m, m_in, m_out):
            e = self.pcoder3(ff=m_out, fb=self.pcoder4.prd, target=self.pcoder2.rep, build_graph=self.build_graph, ffm=self.ffm3, fbm=self.fbm3, erm=self.erm3)
            return e[0]
        self.backbone.layer2[1].conv2.register_forward_hook(fw_hook3)

        def fw_hook4(m, m_in, m_out):
            e = self.pcoder4(ff=m_out, fb=self.pcoder5.prd, target=self.pcoder3.rep, build_graph=self.build_graph, ffm=self.ffm4, fbm=self.fbm4, erm=self.erm4)
            return e[0]
        self.backbone.layer3[1].conv2.register_forward_hook(fw_hook4)

        def fw_hook5(m, m_in, m_out):
            e = self.pcoder5(ff=m_out, fb=None, target=self.pcoder4.rep, build_graph=self.build_graph, ffm=self.ffm5, fbm=self.fbm5, erm=self.erm5)
            return e[0]
        self.backbone.layer4[1].conv2.register_forward_hook(fw_hook5)

class PResNet18V3NSameHP(PNetSameHP):
    """
    3x3 conv transpose kernels and upsample with KC Normalization
    """
    def __init__(self, resnet, build_graph=False, random_init=True, ff_multiplier: float=0.33, fb_multiplier: float=0.33, er_multiplier: float=0.01):
        super().__init__(resnet, 5, build_graph, random_init, ff_multiplier, fb_multiplier, er_multiplier)

        resnet_seq = flatten_resnet(resnet)
        
        # create the first PCoder
        in_ch, out_ch, r, d, p = get_deep_info(resnet_seq, 3, -1)
        pmodule = Sequential(ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1))
        self.pcoder1 = PCoderN(pmodule, True, self.random_init)

        # create the second PCoderN
        in_ch, out_ch, r, d, p = get_deep_info(resnet_seq, 13, 3)
        pmodule = Sequential(ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1))
        self.pcoder2 = PCoderN(pmodule, True, self.random_init)

        # create the third PCoderN
        in_ch, out_ch, r, d, p = get_deep_info(resnet_seq, 24, 13)
        pmodule = Sequential(Upsample(scale_factor=2, mode='bilinear'), ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1))
        self.pcoder3 = PCoderN(pmodule, True, self.random_init)

        # create the fourth PCoderN
        in_ch, out_ch, r, d, p = get_deep_info(resnet_seq, 35, 24)
        pmodule = Sequential(Upsample(scale_factor=2, mode='bilinear'), ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1))
        self.pcoder4 = PCoderN(pmodule, True, self.random_init)

        # create the fifth PCoderN
        in_ch, out_ch, r, d, p = get_deep_info(resnet_seq, 46, 35)
        pmodule = Sequential(Upsample(scale_factor=2, mode='bilinear'), ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1))
        self.pcoder5 = PCoderN(pmodule, False, self.random_init)

        def fw_hook1(m, m_in, m_out):
            e = self.pcoder1(ff=m_out, fb=self.pcoder2.prd, target=self.input_mem, build_graph=self.build_graph, ffm=self.ffm, fbm=self.fbm, erm=self.erm)
            return e[0]
        self.backbone.conv1.register_forward_hook(fw_hook1)

        def fw_hook2(m, m_in, m_out):
            e = self.pcoder2(ff=m_out, fb=self.pcoder3.prd, target=self.pcoder1.rep, build_graph=self.build_graph, ffm=self.ffm, fbm=self.fbm, erm=self.erm)
            return e[0]
        self.backbone.layer1[1].conv2.register_forward_hook(fw_hook2)

        def fw_hook3(m, m_in, m_out):
            e = self.pcoder3(ff=m_out, fb=self.pcoder4.prd, target=self.pcoder2.rep, build_graph=self.build_graph, ffm=self.ffm, fbm=self.fbm, erm=self.erm)
            return e[0]
        self.backbone.layer2[1].conv2.register_forward_hook(fw_hook3)

        def fw_hook4(m, m_in, m_out):
            e = self.pcoder4(ff=m_out, fb=self.pcoder5.prd, target=self.pcoder3.rep, build_graph=self.build_graph, ffm=self.ffm, fbm=self.fbm, erm=self.erm)
            return e[0]
        self.backbone.layer3[1].conv2.register_forward_hook(fw_hook4)

        def fw_hook5(m, m_in, m_out):
            e = self.pcoder5(ff=m_out, fb=None, target=self.pcoder4.rep, build_graph=self.build_graph, ffm=self.ffm, fbm=self.fbm, erm=self.erm)
            return e[0]
        self.backbone.layer4[1].conv2.register_forward_hook(fw_hook5)

class PResNet18V3NSeparateHP(PNetSeparateHP):
    """
    3x3 conv transpose kernels and upsample with KC Normalization
    """
    def __init__(self, resnet, build_graph=False, random_init=True, ff_multiplier: float=0.33, fb_multiplier: float=0.33, er_multiplier: float=0.01):
        super().__init__(resnet, 5, build_graph, random_init, ff_multiplier, fb_multiplier, er_multiplier)

        resnet_seq = flatten_resnet(resnet)
        
        # create the first PCoderN
        in_ch, out_ch, r, d, p = get_deep_info(resnet_seq, 3, -1)
        pmodule = Sequential(ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1))
        self.pcoder1 = PCoderN(pmodule, True, self.random_init)

        # create the second PCoderN
        in_ch, out_ch, r, d, p = get_deep_info(resnet_seq, 13, 3)
        pmodule = Sequential(ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1))
        self.pcoder2 = PCoderN(pmodule, True, self.random_init)

        # create the third PCoderN
        in_ch, out_ch, r, d, p = get_deep_info(resnet_seq, 24, 13)
        pmodule = Sequential(Upsample(scale_factor=2, mode='bilinear'), ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1))
        self.pcoder3 = PCoderN(pmodule, True, self.random_init)

        # create the fourth PCoderN
        in_ch, out_ch, r, d, p = get_deep_info(resnet_seq, 35, 24)
        pmodule = Sequential(Upsample(scale_factor=2, mode='bilinear'), ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1))
        self.pcoder4 = PCoderN(pmodule, True, self.random_init)

        # create the fifth PCoderN
        in_ch, out_ch, r, d, p = get_deep_info(resnet_seq, 46, 35)
        pmodule = Sequential(Upsample(scale_factor=2, mode='bilinear'), ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1))
        self.pcoder5 = PCoderN(pmodule, False, self.random_init)

        def fw_hook1(m, m_in, m_out):
            e = self.pcoder1(ff=m_out, fb=self.pcoder2.prd, target=self.input_mem, build_graph=self.build_graph, ffm=self.ffm1, fbm=self.fbm1, erm=self.erm1)
            return e[0]
        self.backbone.conv1.register_forward_hook(fw_hook1)

        def fw_hook2(m, m_in, m_out):
            e = self.pcoder2(ff=m_out, fb=self.pcoder3.prd, target=self.pcoder1.rep, build_graph=self.build_graph, ffm=self.ffm2, fbm=self.fbm2, erm=self.erm2)
            return e[0]
        self.backbone.layer1[1].conv2.register_forward_hook(fw_hook2)

        def fw_hook3(m, m_in, m_out):
            e = self.pcoder3(ff=m_out, fb=self.pcoder4.prd, target=self.pcoder2.rep, build_graph=self.build_graph, ffm=self.ffm3, fbm=self.fbm3, erm=self.erm3)
            return e[0]
        self.backbone.layer2[1].conv2.register_forward_hook(fw_hook3)

        def fw_hook4(m, m_in, m_out):
            e = self.pcoder4(ff=m_out, fb=self.pcoder5.prd, target=self.pcoder3.rep, build_graph=self.build_graph, ffm=self.ffm4, fbm=self.fbm4, erm=self.erm4)
            return e[0]
        self.backbone.layer3[1].conv2.register_forward_hook(fw_hook4)

        def fw_hook5(m, m_in, m_out):
            e = self.pcoder5(ff=m_out, fb=None, target=self.pcoder4.rep, build_graph=self.build_graph, ffm=self.ffm5, fbm=self.fbm5, erm=self.erm5)
            return e[0]
        self.backbone.layer4[1].conv2.register_forward_hook(fw_hook5)