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

class ToyNet3(nn.Module):
    def __init__(self, featB, featC, featD, fc):
        super(ToyNet3, self).__init__()

        self.poolingParameter = 2  # important to be the same everywhere

        self.features = Sequential(
            Conv2d(in_channels=3, out_channels=featB, kernel_size=5, stride=1, padding=2),          # 0
            ReLU(),                                                                                 # 1
            MaxPool2d(self.poolingParameter, stride=2),                                             # 2

            Conv2d(in_channels=featB, out_channels=featC, kernel_size=5, stride=1, padding=2),      # 3
            ReLU(),                                                                                 # 4
            MaxPool2d(self.poolingParameter, stride=2),                                             # 5

            Conv2d(in_channels=featC, out_channels=featD, kernel_size=5, stride=1, padding=2),      # 6
            ReLU(),                                                                                 # 7
            MaxPool2d(self.poolingParameter, stride=2))                                             # 8

        self.classifier = Sequential(
            nn.Linear(384, fc),
            ReLU(),
            nn.Linear(fc, 10))

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

class PToyNet3SameHP(PNetSameHP):
    def __init__(self, toynet, build_graph=False, random_init=True, ff_multiplier: float=0.33, fb_multiplier: float=0.33, er_multiplier: float=0.01):
        super().__init__(toynet, 3, build_graph, random_init, ff_multiplier, fb_multiplier, er_multiplier)

        pmodule = Sequential(
            ConvTranspose2d(in_channels=self.backbone.features[0].out_channels, out_channels=self.backbone.features[0].in_channels, kernel_size=5, stride=1, padding=2))
        self.pcoder1 = PCoder(pmodule, True, self.random_init)

        pmodule = Sequential(
            Upsample(scale_factor = self.backbone.poolingParameter, mode='bilinear'),
            ConvTranspose2d(in_channels=self.backbone.features[3].out_channels, out_channels=self.backbone.features[3].in_channels, kernel_size=5, stride=1, padding=2))
        self.pcoder2 = PCoder(pmodule, True, self.random_init)
        
        pmodule = Sequential(
            Upsample(scale_factor = self.backbone.poolingParameter, mode='bilinear'),
            ConvTranspose2d(in_channels=self.backbone.features[6].out_channels, out_channels=self.backbone.features[6].in_channels, kernel_size=5, stride=1, padding=2))
        self.pcoder3 = PCoder(pmodule, False, self.random_init)

        self.handles = []
        def fw_hook1(m, m_in, m_out):
            e = self.pcoder1(ff=m_out, fb=self.pcoder2.prd, target=self.input_mem, build_graph=self.build_graph, ffm=self.ffm, fbm=self.fbm, erm=self.erm)
            return e[0]
        self.handles.append(self.backbone.features[0].register_forward_hook(fw_hook1))

        def fw_hook2(m, m_in, m_out):
            e = self.pcoder2(ff=m_out, fb=self.pcoder3.prd, target=self.pcoder1.rep, build_graph=self.build_graph, ffm=self.ffm, fbm=self.fbm, erm=self.erm)
            return e[0]
        self.handles.append(self.backbone.features[3].register_forward_hook(fw_hook2))

        def fw_hook3(m, m_in, m_out):
            e = self.pcoder3(ff=m_out, fb=None, target=self.pcoder2.rep, build_graph=self.build_graph, ffm=self.ffm, fbm=self.fbm, erm=self.erm)
            return e[0]
        self.handles.append(self.backbone.features[6].register_forward_hook(fw_hook3))

class PToyNet3SeparateHP(PNetSeparateHP):
    def __init__(self, toynet, build_graph=False, random_init=True, ff_multiplier: float=0.33, fb_multiplier: float=0.33, er_multiplier: float=0.01):
        super().__init__(toynet, 3, build_graph, random_init, ff_multiplier, fb_multiplier, er_multiplier)

        pmodule = Sequential(
            ConvTranspose2d(in_channels=self.backbone.features[0].out_channels, out_channels=self.backbone.features[0].in_channels, kernel_size=5, stride=1, padding=2))
        self.pcoder1 = PCoder(pmodule, True, self.random_init)

        pmodule = Sequential(
            Upsample(scale_factor = self.backbone.poolingParameter, mode='bilinear'),
            ConvTranspose2d(in_channels=self.backbone.features[3].out_channels, out_channels=self.backbone.features[3].in_channels, kernel_size=5, stride=1, padding=2))
        self.pcoder2 = PCoder(pmodule, True, self.random_init)
        
        pmodule = Sequential(
            Upsample(scale_factor = self.backbone.poolingParameter, mode='bilinear'),
            ConvTranspose2d(in_channels=self.backbone.features[6].out_channels, out_channels=self.backbone.features[6].in_channels, kernel_size=5, stride=1, padding=2))
        self.pcoder3 = PCoder(pmodule, False, self.random_init)

        self.handles = []
        def fw_hook1(m, m_in, m_out):
            e = self.pcoder1(ff=m_out, fb=self.pcoder2.prd, target=self.input_mem, build_graph=self.build_graph, ffm=self.ffm1, fbm=self.fbm1, erm=self.erm1)
            return e[0]
        self.handles.append(self.backbone.features[0].register_forward_hook(fw_hook1))

        def fw_hook2(m, m_in, m_out):
            e = self.pcoder2(ff=m_out, fb=self.pcoder3.prd, target=self.pcoder1.rep, build_graph=self.build_graph, ffm=self.ffm2, fbm=self.fbm2, erm=self.erm2)
            return e[0]
        self.handles.append(self.backbone.features[3].register_forward_hook(fw_hook2))

        def fw_hook3(m, m_in, m_out):
            e = self.pcoder3(ff=m_out, fb=None, target=self.pcoder2.rep, build_graph=self.build_graph, ffm=self.ffm3, fbm=self.fbm3, erm=self.erm3)
            return e[0]
        self.handles.append(self.backbone.features[6].register_forward_hook(fw_hook3))