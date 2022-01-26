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
    Implements a predictive coding module. PCoders generate predictions via the given `pmodule` from input representations while generating their own representation
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
            erm (float): The value of :math:`\alpha`.
        
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
            self.prediction_error =  nn.functional.mse_loss(self.prd, target)
            self.grd = torch.autograd.grad(self.prediction_error, self.rep, retain_graph=True)[0]
            
            if not build_graph:
                self.prd = self.prd.detach()
                self.grd = self.grd.detach()
                self.rep = self.rep.detach()
                self.prediction_error = self.prediction_error.detach()
        return self.rep, self.prd

    def reset(self):
        r"""
        To be called for each new batch of images.
        """
        super().reset()
        self.grd = None           # last error gradient

class PCoderN(PCoder):
    r"""
    Implements a predictive coding module with scaled gradient term.
    PCoderNs generate predictions via the given `pmodule` from input representations while correcting their own representation
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

    Note:
        K-C normalization is used for the gradient term, where K is the size of the prediction tensor and C is the effective window size of a predictor cell.
        C is computed in this way:
            - generate random input X
            - pass it through the prediction module and get the output P
            - repeat 10 times:
                - make a copy of X as X' and change the central cell randomly
                - pass it through the prediction module and compute the output P'
                - count number of different cells between P and P'
            - use the averge "difference" as the C
    """
    def __init__(self, pmodule: nn.Module, has_feedback: bool, random_init: bool):
        super().__init__(pmodule, has_feedback, random_init)
        self.register_buffer('C_sqrt', torch.tensor(-1, dtype=torch.float))
        
    
    def compute_C_sqrt(self, target):
        r"""
        Computes `C` and returns its square root.
        `target` is the tensor to compare the prediction with
        """
        if self.rep is None:
            raise Exception("PCoder's representation cannot be `None` while executing this function.")

        x = self.rep.detach().clone()
        x.requires_grad = True
        with torch.enable_grad():
            xpred = self.pmodule(x)
            # xloss = nn.functional.mse_loss(xpred, target)
            # xgrad = torch.autograd.grad(xloss, x, retain_graph=True)[0]

        xpred_orig = xpred.detach().clone()

        cnt = 0
        for repeat in range(10):
            x = self.rep.detach().clone()
            
            x[:,x.shape[1]//2,x.shape[2]//2,x.shape[3]//2] = torch.randint(-10000,10000,(x.shape[0],), device=x.device).float()
            x.requires_grad = True
            with torch.enable_grad():
                xpred = self.pmodule(x)
                # xloss = nn.functional.mse_loss(xpred, target)
                # xgrad = torch.autograd.grad(xloss, x, retain_graph=True)[0]
            
            xpred_rand = xpred.detach().clone()
            with torch.no_grad():
                
                diff = xpred_orig - xpred_rand

                cnt += (xpred_orig != xpred_rand).sum().float() / diff.shape[0]   # divided by the batch size
        cnt = cnt / 10.0
        self.C_sqrt = torch.sqrt(cnt)

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
            erm (float): The value of :math:`\alpha`.
        
        Returns:
            Tuple: the output representation and prediction
        """

        if self.rep is None:
            if self.random_init:
                self.rep = torch.randn(ff.size(), device=ff.device)
            else:
                self.rep = ff
        else:
            error_scale = self.prd.numel()/self.C_sqrt
            if self.has_feedback:
                self.rep = ffm*ff + fbm*fb + (1-ffm-fbm)*self.rep - erm*error_scale*self.grd
            else:
                self.rep = ffm*ff + (1-ffm)*self.rep - erm*error_scale*self.grd

        if self.C_sqrt == -1:
            self.compute_C_sqrt(target)
            # print(self.C_sqrt * self.C_sqrt)

        with torch.enable_grad():
            if not self.rep.requires_grad:
                self.rep.requires_grad = True 
        
            self.prd = self.pmodule(self.rep)
            self.prediction_error  = nn.functional.mse_loss(self.prd, target)
            self.grd = torch.autograd.grad(self.prediction_error, self.rep, retain_graph=True)[0]
            
            if not build_graph:
                self.prd = self.prd.detach()
                self.grd = self.grd.detach()
                self.rep = self.rep.detach()
                self.prediction_error = self.prediction_error.detach()

        return self.rep, self.prd
