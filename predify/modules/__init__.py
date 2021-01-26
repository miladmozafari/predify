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
            
            # grads = []
            # for i in range(target.shape[0]):
            #     loss  = nn.functional.mse_loss(self.prd[i], target[i])
            #     grads.append(torch.autograd.grad(loss, self.rep, retain_graph=True)[0][i])
            # self.grd = torch.stack(grads)
        
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

# class PCoder(Predictor):
#     r"""
#     Implements a predictive coding module. PCoders generate predictions via the given `pmodule` from input representations while correcting their own representation
#     with the following equation:
    
#     .. math::
#         \beta(feedforward) + \lambda(feedback) + (1-\beta-\lambda)(memory) - \alpha(gradient)
    
#     where :math:`0\leq \beta + \lambda \leq 1`.

#     A PCoder contains the following attributes:

#     * :attr:`pmodule` is a callable module inherited from `torch.nn.Module` with single output.
#     * :attr:`rep` is the output representation of the Predictor which updates based on the equation above.
#     * :attr:`prd` is the prediction of the Predictor (e.g. the output of the `pmodule`).
#     * :attr:`grd` is the gradient of the prediction error.

#     Args:
#         pmodule (torch.nn.Module): Prediction module. It requires to be a callable module inherited from `torch.nn.Module` with single output.
#         has_feedback (boolean): Indicates whether the module receives feedback or not.
#         random_init (boolean): Indicates whether the module starts from a random representation (if `True`) or the given feedforward one (if `False`)
#     """
#     def __init__(self, pmodule: nn.Module, has_feedback: bool, random_init: bool):
#         super().__init__(pmodule, random_init)

#         self.has_feedback = has_feedback    # indicates if the modules receives feedback

#         ### Memory
#         self.grd = None           # last error gradient

#         ### memory of the target of prediction
#         self.trg = None

#         ### Pred and Error function
#         def predict_and_return_the_error(x):
#             """
#             x should be a tensor with single value
#             """
#             rep = torch.zeros_like(self.rep, requires_grad=True)[0:1]
#             rep[0,rep.shape[1]//2,rep.shape[2]//2,rep.shape[3]//2] = x

#             prd = self.pmodule(rep)

#             return prd
#         self.pred_n_error = predict_and_return_the_error

#     ### with given hyperparams and dynamics for the first timestep
#     def forward(self, ff: torch.Tensor, fb: torch.Tensor, target: torch.Tensor, build_graph: bool=False, ffm: float=None, fbm: float=None, erm: float=None):
#         r"""
#         Updates PCoder for one timestep.

#         Args:
#             ff (torch.Tensor): Feedforward drive.
#             fb (torch.Tensor): Feedback drive (`None` if `has_feedback` is `False`).
#             target (torch.Tensor): Target representation to compare with the prediction.
#             build_graph (boolean): Indicates whether the computation graph should be built (set it to `True` during the training phase)
#             ffm (float): The value of :math:`\beta`.
#             fbm (float): The value of :math:`\lambda`.
#             erm (float): The value of :math:`\alphas`.
        
#         Returns:
#             Tuple: the output representation and prediction
#         """
#         self.trg = target
#         if self.rep is None:
#             if self.random_init:
#                 self.rep = torch.randn(ff.size(), device=ff.device)
#             else:
#                 self.rep = ff
#         else:
#             if self.has_feedback:
#                 self.rep = ffm*ff + fbm*fb + (1-ffm-fbm)*self.rep - erm*self.grd
#             else:
#                 self.rep = ffm*ff + (1-ffm)*self.rep - erm*self.grd

#         if self.grd is None:
#             jac = torch.autograd.functional.jacobian(self.pred_n_error, torch.tensor([1.0],requires_grad=True), strict=True)
        
#         with torch.enable_grad():
#             if not self.rep.requires_grad:
#                 self.rep.requires_grad = True 

#             # self.grd = torch.diagonal(jac).permute(3,0,1,2)

#             self.prd = self.pmodule(self.rep)
#             loss     = nn.functional.mse_loss(self.prd, target)
#             self.grd = torch.autograd.grad(loss, self.rep, retain_graph=True)[0]
#             # this should not be done for adversarial attacks
#             if not build_graph:
#                 self.prd = self.prd.detach()
#                 self.grd = self.grd.detach()
#                 self.rep = self.rep.detach()
#         return self.rep, self.prd

#     def reset(self):
#         r"""
#         To be called for each new batch of images.
#         """
#         super().reset()
#         self.grd = None           # last error gradient

class PCoderN(PCoder):
    r"""
    Implements a predictive coding module with normalization of the gradient term.
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
    """
    def __init__(self, pmodule: nn.Module, has_feedback: bool, random_init: bool):
        super().__init__(pmodule, has_feedback, random_init)
        self.register_buffer('C_sqrt', torch.tensor(-1, dtype=torch.float))
        # self.register_buffer('C_sqrt', None)

        ### Pred and Error function
        def receptive_filed_helper(x):
            r"""
            `x` should be a tensor with a single value. This function should not be called when `self.rep` is `None`
            """ 
            # with torch.enable_grad():
            rep = torch.zeros_like(self.rep, requires_grad=True)[0:1]
            rep[0,rep.shape[1]//2,rep.shape[2]//2,rep.shape[3]//2] = x

            prd = self.pmodule(rep)

            return prd
        self.rf_helper = receptive_filed_helper
    
    def compute_C_sqrt(self):
        r"""
        Computes `C` and returns its square root.
        """
        if self.rep is None:
            raise Exception("PCoder's representation cannot be `None` while executing this function.")
        # if not self.rep.requires_grad:
        #     self.rep.requires_grad = True
        jac = torch.autograd.functional.jacobian(self.rf_helper, torch.tensor([1.0],requires_grad=True, device=self.rep.device), strict=True)
        self.C_sqrt = torch.sqrt((jac!=0).sum().float())


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
            error_scale = self.prd.numel()/self.C_sqrt
            if self.has_feedback:
                self.rep = ffm*ff + fbm*fb + (1-ffm-fbm)*self.rep - erm*error_scale*self.grd
            else:
                self.rep = ffm*ff + (1-ffm)*self.rep - erm*error_scale*self.grd

        if self.C_sqrt == -1:
            with torch.enable_grad():
                self.compute_C_sqrt()
            print(self.C_sqrt * self.C_sqrt)

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