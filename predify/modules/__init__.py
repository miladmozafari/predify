import torch
import torch.nn as nn

class Predictor(nn.Module):
    """
    Implements a predictor including the Prediction Modules
    """
    def __init__(self, pmodule: nn.Module, random_init: bool):
        super(Predictor, self).__init__()
        self.random_init  = random_init     # indicates if in timestep 0 the feedback is random
        
        ### Modules
        self.pmodule = pmodule    # feedback module

        ### Memory
        self.rep = None           # last representation
        self.prd = None           # last prediction

    ### with given hyperparams and dynamics for the first timestep
    def forward(self, ff, build_graph=False):

        if self.rep is None:
            if self.random_init:
                self.rep = torch.randn(ff.size(), device=ff.device)
            else:
                self.rep = ff
        else:
            self.rep = ff

        with torch.enable_grad():
            if not self.rep.requires_grad:
                self.rep.requires_grad = True 
        
            self.prd = self.pmodule(self.rep)
        
            # this should not be done for adversarial attacks
            if not build_graph:
                self.prd = self.prd.detach()

        return self.rep, self.prd

    def reset(self):
        """
        to be called for each new batch of images
        """
        self.rep = None           # last representation
        self.prd = None           # last prediction

class PCoder(Predictor):
    """
    Implements a predictive loop (encoder+predictor) including the Prediction Modules
    """
    def __init__(self, pmodule: nn.Module, has_feedback: bool, random_init: bool):
        super().__init__(pmodule, random_init)

        self.has_feedback = has_feedback    # indicates if the modules receives feedback

        ### Memory
        self.grd = None           # last error gradient

    ### with given hyperparams and dynamics for the first timestep
    def forward(self, ff, fb, target, build_graph=False, ffm=None, fbm=None, erm=None):
        """
        implements the encoder forward pass.
        Args:
            ff: feedforward drive
            fb: feedback drive
            target: target representation to compare with reconstruction (prediction)
        """

        if self.rep is None:
            if self.random_init:
                self.rep = torch.randn(ff.size(), device=ff.device)
            else:
                self.rep = ff
        else:
            if self.has_feedback:
                self.rep = ffm*ff + fbm*fb + (1-ffm-fbm)*self.rep - erm*self.grd
            else:
                self.rep = ffm*ff + (1-ffm)*self.rep - erm*self.grd

        with torch.enable_grad():
            if not self.rep.requires_grad:
                self.rep.requires_grad = True 
        
            self.prd = self.pmodule(self.rep)
            loss     = nn.functional.mse_loss(self.prd, target)
            self.grd = torch.autograd.grad(loss, self.rep, retain_graph=True)[0]
        
            # this should not be done for adversarial attacks
            if not build_graph:
                self.prd = self.prd.detach()
                self.grd = self.grd.detach()
                self.rep = self.rep.detach()
        return self.rep, self.prd

    def reset(self):
        """
        to be called for each new batch of images
        """
        super().reset()
        self.grd = None           # last error gradient