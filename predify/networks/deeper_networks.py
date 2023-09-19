import torch.nn as nn

import predify
from predify.networks import PNetSameHP, PNetSeparateHP


class DeepPNetSameHP(PNetSameHP):
    r'''
    A deeper unrolled version of the network. The unrolling with occur for number_of_timesteps timesteps.
    '''

    def __init__(self, backbone: nn.Module, number_of_pcoders: int, number_of_timesteps: int, build_graph: bool = False, random_init: bool = False, ff_multiplier: float = 0.33, fb_multiplier: float = 0.33, er_multiplier: float = 0.01):
        super(DeepPNetSameHP, self).__init__(backbone=backbone, number_of_pcoders=number_of_pcoders, build_graph=build_graph,
                                             random_init=random_init, ff_multiplier=ff_multiplier, fb_multiplier=fb_multiplier, er_multiplier=er_multiplier)
        self.number_of_timesteps = number_of_timesteps

    def forward(self, x):
        assert x is not None
        output = super().forward(x)
        for _ in range(self.number_of_timesteps-1):
            output = super().forward()

        return output


class DeepPNetSeparateHP(PNetSeparateHP):
    r'''
    A deeper unrolled version of the network. The unrolling with occur for number_of_timesteps timesteps.
    '''

    def __init__(self, backbone: nn.Module, number_of_pcoders: int, number_of_timesteps: int, build_graph: bool = False, random_init: bool = False, ff_multiplier: float = 0.33, fb_multiplier: float = 0.33, er_multiplier: float = 0.01):
        super(DeepPNetSeparateHP, self).__init__(backbone=backbone, number_of_pcoders=number_of_pcoders, build_graph=build_graph,
                                                 random_init=random_init, ff_multiplier=ff_multiplier, fb_multiplier=fb_multiplier, er_multiplier=er_multiplier)
        self.number_of_timesteps = number_of_timesteps

    def forward(self, x):
        assert x is not None
        output = super().forward(x)
        for _ in range(self.number_of_timesteps-1):
            output = super().forward()

        return output
