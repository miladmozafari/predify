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
from ..modules import PCoder
from ..utils import to_pair
from . import PNetSameHP, PNetSeparateHP

def get_cifar_vgg11():
    vgg = models.vgg11()
    vgg.features[2] = MaxPool2d(kernel_size=1, stride=1, padding=0, dilation=1, ceil_mode=False)
    vgg.classifier[6] = Linear(in_features=4096, out_features=100, bias=True)

    return vgg

class PVGG11V3SameHP(PNetSameHP):
    """
    3x3 conv transpose kernels and upsample
    """
    def __init__(self, vgg, build_graph=False, random_init=True, ff_multiplier: float=0.33, fb_multiplier: float=0.33, er_multiplier: float=0.01):
        super().__init__(vgg, 4, build_graph, random_init, ff_multiplier, fb_multiplier, er_multiplier)

        for m in self.backbone.features:
            if isinstance(m, nn.ReLU):
                m.inplace=False

        # create the first PCoder
        pmodule = Sequential(ConvTranspose2d(128, 3, kernel_size=3, stride=1, padding=1))
        self.pcoder1 = PCoder(pmodule, True, self.random_init)

        # create the second PCoder
        pmodule = Sequential(Upsample(scale_factor=2, mode='bilinear'), ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1))
        self.pcoder2 = PCoder(pmodule, True, self.random_init)

        # create the third PCoder
        pmodule = Sequential(Upsample(scale_factor=2, mode='bilinear'), ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1))
        self.pcoder3 = PCoder(pmodule, True, self.random_init)

        # create the fourth PCoder
        pmodule = Sequential(Upsample(scale_factor=2, mode='bilinear'), ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1))
        self.pcoder4 = PCoder(pmodule, False, self.random_init)

        def fw_hook1(m, m_in, m_out):
            e = self.pcoder1(ff=m_out, fb=self.pcoder2.prd, target=self.input_mem, build_graph=self.build_graph, ffm=self.ffm, fbm=self.fbm, erm=self.erm)
            return e[0]
        self.backbone.features[3].register_forward_hook(fw_hook1)

        def fw_hook2(m, m_in, m_out):
            e = self.pcoder2(ff=m_out, fb=self.pcoder3.prd, target=self.pcoder1.rep, build_graph=self.build_graph, ffm=self.ffm, fbm=self.fbm, erm=self.erm)
            return e[0]
        self.backbone.features[8].register_forward_hook(fw_hook2)

        def fw_hook3(m, m_in, m_out):
            e = self.pcoder3(ff=m_out, fb=self.pcoder4.prd, target=self.pcoder2.rep, build_graph=self.build_graph, ffm=self.ffm, fbm=self.fbm, erm=self.erm)
            return e[0]
        self.backbone.features[13].register_forward_hook(fw_hook3)

        def fw_hook4(m, m_in, m_out):
            e = self.pcoder4(ff=m_out, fb=None, target=self.pcoder3.rep, build_graph=self.build_graph, ffm=self.ffm, fbm=self.fbm, erm=self.erm)
            return e[0]
        self.backbone.features[18].register_forward_hook(fw_hook4)

class PVGG11V3SeparateHP(PNetSeparateHP):
    """
    3x3 conv transpose kernels and upsample
    """
    def __init__(self, vgg, build_graph=False, random_init=True, ff_multiplier: float=0.33, fb_multiplier: float=0.33, er_multiplier: float=0.01):
        super().__init__(vgg, 4, build_graph, random_init, ff_multiplier, fb_multiplier, er_multiplier)
        
        for m in self.backbone.features:
            if isinstance(m, nn.ReLU):
                m.inplace=False

        # create the first PCoder
        pmodule = Sequential(ConvTranspose2d(128, 3, kernel_size=3, stride=1, padding=1))
        self.pcoder1 = PCoder(pmodule, True, self.random_init)

        # create the second PCoder
        pmodule = Sequential(Upsample(scale_factor=2, mode='bilinear'), ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1))
        self.pcoder2 = PCoder(pmodule, True, self.random_init)

        # create the third PCoder
        pmodule = Sequential(Upsample(scale_factor=2, mode='bilinear'), ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1))
        self.pcoder3 = PCoder(pmodule, True, self.random_init)

        # create the fourth PCoder
        pmodule = Sequential(Upsample(scale_factor=2, mode='bilinear'), ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1))
        self.pcoder4 = PCoder(pmodule, False, self.random_init)

        def fw_hook1(m, m_in, m_out):
            e = self.pcoder1(ff=m_out, fb=self.pcoder2.prd, target=self.input_mem, build_graph=self.build_graph, ffm=self.ffm1, fbm=self.fbm1, erm=self.erm1)
            return e[0]
        self.backbone.features[3].register_forward_hook(fw_hook1)

        def fw_hook2(m, m_in, m_out):
            e = self.pcoder2(ff=m_out, fb=self.pcoder3.prd, target=self.pcoder1.rep, build_graph=self.build_graph, ffm=self.ffm2, fbm=self.fbm2, erm=self.erm2)
            return e[0]
        self.backbone.features[8].register_forward_hook(fw_hook2)

        def fw_hook3(m, m_in, m_out):
            e = self.pcoder3(ff=m_out, fb=self.pcoder4.prd, target=self.pcoder2.rep, build_graph=self.build_graph, ffm=self.ffm3, fbm=self.fbm3, erm=self.erm3)
            return e[0]
        self.backbone.features[13].register_forward_hook(fw_hook3)

        def fw_hook4(m, m_in, m_out):
            e = self.pcoder4(ff=m_out, fb=None, target=self.pcoder3.rep, build_graph=self.build_graph, ffm=self.ffm4, fbm=self.fbm4, erm=self.erm4)
            return e[0]
        self.backbone.features[18].register_forward_hook(fw_hook4)

class PVGG16SameHP(PNetSameHP):
    """
    """
    def __init__(self, vgg, build_graph=False, random_init=False, ff_multiplier: float=0.33, fb_multiplier: float=0.33, er_multiplier: float=0.01):
        super().__init__(vgg, 5, build_graph, random_init, ff_multiplier, fb_multiplier, er_multiplier)

        for m in self.backbone.features:
            if isinstance(m, nn.ReLU):
                m.inplace=False

        # create the first PCoder
        pmodule = Sequential(ConvTranspose2d(64, 3, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)))
        self.pcoder1 = PCoder(pmodule, True, self.random_init)

        # create the second PCoder
        pmodule = Sequential(ConvTranspose2d(128, 64, kernel_size=(10, 10), stride=(2, 2), padding=(4, 4)), ReLU(inplace=True))
        self.pcoder2 = PCoder(pmodule, True, self.random_init)

        # create the third PCoder
        pmodule = Sequential(ConvTranspose2d(256, 128, kernel_size=(14, 14), stride=(2, 2), padding=(6, 6)), ReLU(inplace=True))
        self.pcoder3 = PCoder(pmodule, True, self.random_init)

        # create the fourth PCoder
        pmodule = Sequential(ConvTranspose2d(512, 256, kernel_size=(14, 14), stride=(2, 2), padding=(6, 6)), ReLU(inplace=True))
        self.pcoder4 = PCoder(pmodule, True, self.random_init)

        # create the fourth PCoder
        pmodule = Sequential(ConvTranspose2d(512, 512, kernel_size=(14, 14), stride=(2, 2), padding=(6, 6)), ReLU(inplace=True))
        self.pcoder5 = PCoder(pmodule, False, self.random_init)

        def fw_hook1(m, m_in, m_out):
            e = self.pcoder1(ff=m_out, fb=self.pcoder2.prd, target=self.input_mem, build_graph=self.build_graph, ffm=self.ffm, fbm=self.fbm, erm=self.erm)
            return e[0]
        self.backbone.features[3].register_forward_hook(fw_hook1)

        def fw_hook2(m, m_in, m_out):
            e = self.pcoder2(ff=m_out, fb=self.pcoder3.prd, target=self.pcoder1.rep, build_graph=self.build_graph, ffm=self.ffm, fbm=self.fbm, erm=self.erm)
            return e[0]
        self.backbone.features[8].register_forward_hook(fw_hook2)

        def fw_hook3(m, m_in, m_out):
            e = self.pcoder3(ff=m_out, fb=self.pcoder4.prd, target=self.pcoder2.rep, build_graph=self.build_graph, ffm=self.ffm, fbm=self.fbm, erm=self.erm)
            return e[0]
        self.backbone.features[15].register_forward_hook(fw_hook3)

        def fw_hook4(m, m_in, m_out):
            e = self.pcoder4(ff=m_out, fb=self.pcoder5.prd, target=self.pcoder3.rep, build_graph=self.build_graph, ffm=self.ffm, fbm=self.fbm, erm=self.erm)
            return e[0]
        self.backbone.features[22].register_forward_hook(fw_hook4)

        def fw_hook5(m, m_in, m_out):
            e = self.pcoder5(ff=m_out, fb=None, target=self.pcoder4.rep, build_graph=self.build_graph, ffm=self.ffm, fbm=self.fbm, erm=self.erm)
            return e[0]
        self.backbone.features[29].register_forward_hook(fw_hook5)

class PVGG16SeparateHP(PNetSeparateHP):
    """
    """
    def __init__(self, vgg, build_graph=False, random_init=False, ff_multiplier: float=0.33, fb_multiplier: float=0.33, er_multiplier: float=0.01):
        super().__init__(vgg, 5, build_graph, random_init, ff_multiplier, fb_multiplier, er_multiplier)

        for m in self.backbone.features:
            if isinstance(m, nn.ReLU):
                m.inplace=False

        # create the first PCoder
        pmodule = Sequential(ConvTranspose2d(64, 3, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)))
        self.pcoder1 = PCoder(pmodule, True, self.random_init)

        # create the second PCoder
        pmodule = Sequential(ConvTranspose2d(128, 64, kernel_size=(10, 10), stride=(2, 2), padding=(4, 4)), ReLU(inplace=True))
        self.pcoder2 = PCoder(pmodule, True, self.random_init)

        # create the third PCoder
        pmodule = Sequential(ConvTranspose2d(256, 128, kernel_size=(14, 14), stride=(2, 2), padding=(6, 6)), ReLU(inplace=True))
        self.pcoder3 = PCoder(pmodule, True, self.random_init)

        # create the fourth PCoder
        pmodule = Sequential(ConvTranspose2d(512, 256, kernel_size=(14, 14), stride=(2, 2), padding=(6, 6)), ReLU(inplace=True))
        self.pcoder4 = PCoder(pmodule, True, self.random_init)

        # create the fourth PCoder
        pmodule = Sequential(ConvTranspose2d(512, 512, kernel_size=(14, 14), stride=(2, 2), padding=(6, 6)), ReLU(inplace=True))
        self.pcoder5 = PCoder(pmodule, False, self.random_init)

        def fw_hook1(m, m_in, m_out):
            e = self.pcoder1(ff=m_out, fb=self.pcoder2.prd, target=self.input_mem, build_graph=self.build_graph, ffm=self.ffm1, fbm=self.fbm1, erm=self.erm1)
            return e[0]
        self.backbone.features[3].register_forward_hook(fw_hook1)

        def fw_hook2(m, m_in, m_out):
            e = self.pcoder2(ff=m_out, fb=self.pcoder3.prd, target=self.pcoder1.rep, build_graph=self.build_graph, ffm=self.ffm2, fbm=self.fbm2, erm=self.erm2)
            return e[0]
        self.backbone.features[8].register_forward_hook(fw_hook2)

        def fw_hook3(m, m_in, m_out):
            e = self.pcoder3(ff=m_out, fb=self.pcoder4.prd, target=self.pcoder2.rep, build_graph=self.build_graph, ffm=self.ffm3, fbm=self.fbm3, erm=self.erm3)
            return e[0]
        self.backbone.features[15].register_forward_hook(fw_hook3)

        def fw_hook4(m, m_in, m_out):
            e = self.pcoder4(ff=m_out, fb=self.pcoder5.prd, target=self.pcoder3.rep, build_graph=self.build_graph, ffm=self.ffm4, fbm=self.fbm4, erm=self.erm4)
            return e[0]
        self.backbone.features[22].register_forward_hook(fw_hook4)

        def fw_hook5(m, m_in, m_out):
            e = self.pcoder5(ff=m_out, fb=None, target=self.pcoder4.rep, build_graph=self.build_graph, ffm=self.ffm5, fbm=self.fbm5, erm=self.erm5)
            return e[0]
        self.backbone.features[29].register_forward_hook(fw_hook5)