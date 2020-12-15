import torch
import torch.nn as nn

class Predictor(nn.Module):
    r"""
    Implements a predictor module. Predictors generate predictions via the given `pmodule` from input representations.
    A Predictor contains three main attributes:

    * :attr:`pmodule` is a callable module inherited from `torch.nn.Module` with single output.
    * :attr:`rep` is the output representation of the Predictor. It is equal to the input representation.
    * :attr:`prd` is the prediction of the Predictor (e.g. the output of the `pmodule`).

    Args:
        pmodule (torch.nn.Module): Prediction module. It requires to be a callable module inherited from `torch.nn.Module` with single output.
        random_init (boolean): Indicates whether the module starts from a random representation (if `True`) or the given feedforward one (if `False`)
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
    def forward(self, ff: torch.Tensor, build_graph: bool=False):
        r"""
        Computes the prediction from the given representation.

        Args:
            ff (torch.Tensor): Input representation.
            build_graph (boolean): Indicates whether the computation graph should be built (set it to `True` during the training phase)

        Returns:
            Tuple: the output representation and prediction
        """
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
        r"""
        To be called for each new batch of images.
        """
        self.rep = None           # last representation
        self.prd = None           # last prediction

class PCoder(Predictor):
    r"""
    Implements a predictive coding module. PCoders generate predictions via the given `pmodule` from input representations while correcting their own representation
    with the following equation:
    
    .. math::
        \beta(feedforward) + \lambda(feedback) + (1-\beta-\lambda)(memory) - \alpha(gradient)
    
    where :math:`0\leq \beta + \lambda \leq 1`.

    A PCoder contains the following attributes:

    * :attr:`pmodule` is a callable module inherited from `torch.nn.Module` with single output.
    * :attr:`rep` is the output representation of the Predictor which updates based on the equation above.
    * :attr:`prd` is the prediction of the Predictor (e.g. the output of the `pmodule`).
    * :attr:`grd` is the gradient of the prediction error.

    Args:
        pmodule (torch.nn.Module): Prediction module. It requires to be a callable module inherited from `torch.nn.Module` with single output.
        has_feedback (boolean): Indicates whether the module receives feedback or not.
        random_init (boolean): Indicates whether the module starts from a random representation (if `True`) or the given feedforward one (if `False`)
    """
    def __init__(self, pmodule: nn.Module, has_feedback: bool, random_init: bool):
        super().__init__(pmodule, random_init)

        self.has_feedback = has_feedback    # indicates if the modules receives feedback

        ### Memory
        self.grd = None           # last error gradient

    ### with given hyperparams and dynamics for the first timestep
    def forward(self, ff: torch.Tensor, fb: torch.Tensor, target: torch.Tensor, build_graph: bool=False, ffm: float=None, fbm: float=None, erm: float=None):
        r"""
        Updates PCoder for one timestep.

        Args:
            ff (torch.Tensor): Feedforward drive.
            fb (torch.Tensor): Feedback drive (`None` if `has_feedback` is `False`).
            target (torch.Tensor): Target representation to compare with the prediction.
            build_graph (boolean): Indicates whether the computation graph should be built (set it to `True` during the training phase)
            ffm (float): The value of :math:`\beta`.
            fbm (float): The value of :math:`\lambda`.
            erm (float): The value of :math:`\alphas`.
        
        Returns:
            Tuple: the output representation and prediction
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
        r"""
        To be called for each new batch of images.
        """
        super().reset()
        self.grd = None           # last error gradient