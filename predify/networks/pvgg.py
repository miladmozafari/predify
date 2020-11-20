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

def get_cifar_vgg11():
    vgg = models.vgg11()
    vgg.features[2] = MaxPool2d(kernel_size=1, stride=1, padding=0, dilation=1, ceil_mode=False)
    vgg.classifier[6] = Linear(in_features=4096, out_features=100, bias=True)
    return vgg

class PVGG11SameParam(nn.Module):
    """
    Base class for the PVGG with same parameters for all layers
    """
    def __init__(self, vgg=None, build_graph=False, random_init=True, ff_multiplier: float=0.33, fb_multiplier: float=0.33, er_multiplier: float=0.01):
        super(PVGG11SameParam, self).__init__()

        self.build_graph = build_graph
        self.random_init = random_init

        ### PC Parameters (multipliers)
        ffm = torch.tensor(ff_multiplier, dtype=torch.float)
        self.register_buffer(f"ffm", ffm)
        
        fbm = torch.tensor(fb_multiplier,dtype=torch.float)
        self.register_buffer(f"fbm", fbm)

        erm = torch.tensor(er_multiplier, dtype=torch.float)
        self.register_buffer(f"erm", erm)

        # trainable
        ff_part = nn.Parameter(torch.tensor(ff_multiplier))   
        self.register_parameter(f"ff_part", ff_part)

        fb_part = nn.Parameter(torch.tensor(fb_multiplier))
        self.register_parameter(f"fb_part", fb_part)

        mem_part = nn.Parameter(torch.tensor(1.0-ff_multiplier-fb_multiplier))
        self.register_parameter(f"mem_part", mem_part)

        errorm = nn.Parameter(torch.tensor(er_multiplier))
        self.register_parameter(f"errorm", errorm)

        self.input_mem = None

        if vgg is None:
            vgg = get_cifar_vgg11()
        
        for m in vgg.features:
            if isinstance(m, nn.ReLU):
                m.inplace=False

        self.vgg = vgg
        self.vgg.eval()

        self.pcoders = None   # append pcoders here
    
    def forward(self, x=None):
        if x is not None:
            self.reset()
            self.input_mem = x

            if self.random_init:
                self.vgg(self.input_mem) # random initialization

        if not self.build_graph:
            with torch.no_grad():
                output = self.vgg(self.input_mem)
        else:
            output = self.vgg(self.input_mem)
        
        return output
  
    def reset(self):
        self.input_mem = None
        if self.pcoders is None:
            self.pcoders = []
            for m in self.modules():
                if isinstance(m, PCoder):
                    self.pcoders.append(m)
        for pc in self.pcoders:
            pc.reset()

    def get_hyperparameters_values(self):
        return (self.ffm.item(), self.fbm.item(), 1-self.ffm.item()-self.fbm.item(), self.erm.item())

    def get_hyperparameters(self):
        return (self.ff_part, self.fb_part, self.mem_part, self.errorm)

    def update_hyperparameters(self, no_grad=False):
        if no_grad:
            context = torch.no_grad()
        else:
            context = torch.enable_grad()

        with context:
            self.erm = 1 * self.errorm
            a,b,c = torch.sigmoid(self.ff_part), torch.sigmoid(self.fb_part), torch.sigmoid(self.mem_part)
            abc = a+b+c
            self.ffm = a/abc
            self.fbm = b/abc

class PVGG11SameParamV3(PVGG11SameParam):
    """
    3x3 conv transpose kernels and upsample
    """
    def __init__(self, vgg=None, build_graph=False, random_init=True, ff_multiplier: float=0.33, fb_multiplier: float=0.33, er_multiplier: float=0.01):
        super().__init__(vgg, build_graph, random_init, ff_multiplier, fb_multiplier, er_multiplier)

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

        self.hook_handles = []
        def fw_hook1(m, m_in, m_out):
            e = self.pcoder1(ff=m_out, fb=self.pcoder2.prd, target=self.input_mem, build_graph=self.build_graph, ffm=self.ffm, fbm=self.fbm, erm=self.erm)
            return e[0]
        self.hook_handles.append(self.vgg.features[3].register_forward_hook(fw_hook1))

        def fw_hook2(m, m_in, m_out):
            e = self.pcoder2(ff=m_out, fb=self.pcoder3.prd, target=self.pcoder1.rep, build_graph=self.build_graph, ffm=self.ffm, fbm=self.fbm, erm=self.erm)
            return e[0]
        self.hook_handles.append(self.vgg.features[8].register_forward_hook(fw_hook2))

        def fw_hook3(m, m_in, m_out):
            e = self.pcoder3(ff=m_out, fb=self.pcoder4.prd, target=self.pcoder2.rep, build_graph=self.build_graph, ffm=self.ffm, fbm=self.fbm, erm=self.erm)
            return e[0]
        self.hook_handles.append(self.vgg.features[13].register_forward_hook(fw_hook3))

        def fw_hook4(m, m_in, m_out):
            e = self.pcoder4(ff=m_out, fb=None, target=self.pcoder3.rep, build_graph=self.build_graph, ffm=self.ffm, fbm=self.fbm, erm=self.erm)
            return e[0]
        self.hook_handles.append(self.vgg.features[18].register_forward_hook(fw_hook4))

class PVGG11SeparateParam(nn.Module):
    """
    Base class for the PVGG with separate hyper parameters for each layer
    """
    def __init__(self, vgg=None, build_graph=False, random_init=True, number_of_pcoders=4, ff_multiplier: float=0.33, fb_multiplier: float=0.33, er_multiplier: float=0.01):
        super(PVGG11SeparateParam, self).__init__()

        self.build_graph = build_graph
        self.random_init = random_init
        self.number_of_pcoders = number_of_pcoders

        ### PC Parameters (multipliers)
        ffms = [None for i in range(self.number_of_pcoders)]
        fbms = [None for i in range(self.number_of_pcoders)]
        erms = [None for i in range(self.number_of_pcoders)]
        for i in range(self.number_of_pcoders):
            ffms[i] = torch.tensor(ff_multiplier, dtype=torch.float)
            self.register_buffer(f"ffm{i+1}", ffms[i])
            
            fbms[i] = torch.tensor(fb_multiplier,dtype=torch.float)
            self.register_buffer(f"fbm{i+1}", fbms[i])

            erms[i] = torch.tensor(er_multiplier, dtype=torch.float)
            self.register_buffer(f"erm{i+1}", erms[i])

        ff_parts  = [None for i in range(self.number_of_pcoders)]
        fb_parts  = [None for i in range(self.number_of_pcoders)]
        mem_parts = [None for i in range(self.number_of_pcoders)]
        errorms   = [None for i in range(self.number_of_pcoders)]

        # trainable
        for i in range(self.number_of_pcoders):
            ff_parts[i] = nn.Parameter(torch.tensor(ff_multiplier))   
            self.register_parameter(f"ff_part{i+1}", ff_parts[i])

            fb_parts[i] = nn.Parameter(torch.tensor(fb_multiplier))
            self.register_parameter(f"fb_part{i+1}", fb_parts[i])

            mem_parts[i] = nn.Parameter(torch.tensor(1.0-ff_multiplier-fb_multiplier))
            self.register_parameter(f"mem_part{i+1}", mem_parts[i])

            errorms[i] = nn.Parameter(torch.tensor(er_multiplier))
            self.register_parameter(f"errorm{i+1}", errorms[i])

        self.input_mem = None

        if vgg is None:
            vgg = get_cifar_vgg11()
        
        for m in vgg.features:
            if isinstance(m, nn.ReLU):
                m.inplace=False

        self.vgg = vgg
        self.vgg.eval()

        self.pcoders = None   # append pcoders here
    
    def forward(self, x=None):
        if x is not None:
            self.reset()
            self.input_mem = x

            if self.random_init:
                self.vgg(self.input_mem) # random initialization

        if not self.build_graph:
            with torch.no_grad():
                output = self.vgg(self.input_mem)
        else:
            output = self.vgg(self.input_mem)
        
        return output
  
    def reset(self):
        self.input_mem = None
        if self.pcoders is None:
            self.pcoders = []
            for m in self.modules():
                if isinstance(m, PCoder):
                    self.pcoders.append(m)
        for pc in self.pcoders:
            pc.reset()

    def get_hyperparameters_values(self):
        vals = []
        for i in range(self.number_of_pcoders-1):
            vals.append(getattr(self, f"ffm{i+1}").item())
            vals.append(getattr(self, f"fbm{i+1}").item())
            vals.append(1-vals[-1]-vals[-2])
            vals.append(getattr(self, f"erm{i+1}").item())
        
        i = self.number_of_pcoders
        vals.append(getattr(self, f"ffm{i}").item())
        vals.append(0.0)
        vals.append(1-vals[-1]-vals[-2])
        vals.append(getattr(self, f"erm{i}").item())
        
        return vals

    def get_hyperparameters(self):
        pars = []
        for i in range(self.number_of_pcoders):
            pars.append(getattr(self, f"ff_part{i+1}"))
            pars.append(getattr(self, f"fb_part{i+1}"))
            pars.append(getattr(self, f"mem_part{i+1}"))
            pars.append(getattr(self, f"errorm{i+1}"))
        return pars

    def update_hyperparameters(self, no_grad=False):
        if no_grad:
            context = torch.no_grad()
        else:
            context = torch.enable_grad()

        with context:
            for i in range(1,self.number_of_pcoders+1):
                errorm, ff_part, fb_part, mem_part = getattr(self,f"errorm{i}"), getattr(self,f"ff_part{i}"), getattr(self,f"fb_part{i}"), getattr(self,f"mem_part{i}")
                setattr(self, f'erm{i}', 1 * errorm)
                a,b,c = torch.sigmoid(ff_part), torch.sigmoid(fb_part), torch.sigmoid(mem_part)
                abc = a+b+c
                setattr(self, f'ffm{i}', a/abc)
                setattr(self, f'fbm{i}', b/abc)

class PVGG11SeparateParamV3(PVGG11SeparateParam):
    """
    3x3 conv transpose kernels and upsample
    """
    def __init__(self, vgg=None, build_graph=False, random_init=True, number_of_pcoders=4, ff_multiplier: float=0.33, fb_multiplier: float=0.33, er_multiplier: float=0.01):
        super().__init__(vgg, build_graph, random_init, number_of_pcoders, ff_multiplier, fb_multiplier, er_multiplier)

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
        self.vgg.features[3].register_forward_hook(fw_hook1)

        def fw_hook2(m, m_in, m_out):
            e = self.pcoder2(ff=m_out, fb=self.pcoder3.prd, target=self.pcoder1.rep, build_graph=self.build_graph, ffm=self.ffm2, fbm=self.fbm2, erm=self.erm2)
            return e[0]
        self.vgg.features[8].register_forward_hook(fw_hook2)

        def fw_hook3(m, m_in, m_out):
            e = self.pcoder3(ff=m_out, fb=self.pcoder4.prd, target=self.pcoder2.rep, build_graph=self.build_graph, ffm=self.ffm3, fbm=self.fbm3, erm=self.erm3)
            return e[0]
        self.vgg.features[13].register_forward_hook(fw_hook3)

        def fw_hook4(m, m_in, m_out):
            e = self.pcoder4(ff=m_out, fb=None, target=self.pcoder3.rep, build_graph=self.build_graph, ffm=self.ffm4, fbm=self.fbm4, erm=self.erm4)
            return e[0]
        self.vgg.features[18].register_forward_hook(fw_hook4)