from typing import Sequence
import torch
import torch.nn as nn
import copy
from ..modules import PCoder

class PNetSameHP(nn.Module):
    r"""
    Implements the base class for adding Predicitive Coding Dynamics to an existing network with shared hyperparameters for all PCoders.
    Assume that there are :math:`n` PCoders. Let :math:`pc_{i}^{r}(t)` be the output representation of PCoder :math:`i` at timestep :math:`t` based on which :math:`pc_{i}^{p}(t)` will be calculated as its prediction (:math:`i \in {1,...,n}` and :math:`t \in {0,...,T}`). Also, let :math:`ff_{i}(t)` be the feedforward drive that enters the PCoder :math:`i` at timestep :math:`t`. Then, the update dynamics will be as follows. 
    
    * Initialization (:math:`t=0`):
    
    :math:`pc_{i}^{r}(0)` will be initialized by the feedforward drive (:math:`ff_{i}(0)`) or a random representation drawn from the standard normal.

    * Iterations (:math:`0< t \leq T`):

    .. math::
        pc_{i}^{r}(t) = \beta(ff_{i}(t)) + \lambda(pc_{i+1}^{p}(t-1)) + (1-\beta-\lambda)(pc_{i}^{r}(t-1)) - \alpha(\frac{\partial \epsilon_{i}(t-1)}{\partial pc_{i}^{r}(t-1)}),

    where, :math:`\epsilon_{i}(t-1)` is the prediction error computed between :math:`pc_{i}^{p}(t-1)` and :math:`pc_{i-1}^{r}(t-1)`.

    .. note::
        * :math:`pc_{0}^{r}(t)` = :math:`pc_{0}^{r}(0)` = Static input
        * For the last PCoder, there will be no feedback drive, thus there will be no :math:`lambda` as well.

    Args:
        backbone (torch.nn.Module): A callable module inherited from `torch.nn.Module`.
        numbre_of_pcoders (int): Number of PCoders
        build_graph (boolean): Indicates whether the computation graph should be built (set it to `True` during the training phase)
        random_init (boolean): Indicates whether the PCoders starts from a random representation (if `True`) or the given feedforward one (if `False`)
        ff_multiplier (float, optional): Value of :math:`\beta`. Default: 0.33.
        fb_multiplier (float, optional): Value of :math:`\lambda`. Default: 0.33.
        er_multiplier (float, optional): Value of :math:`\alpha`. Default: 0.01.
    """
    def __init__(self, backbone: nn.Module, numbre_of_pcoders: int, build_graph: bool=False, random_init: bool=True, ff_multiplier: float=0.33, fb_multiplier: float=0.33, er_multiplier: float=0.01):
        super(PNetSameHP, self).__init__()

        self.build_graph = build_graph
        self.random_init = random_init
        self.numbre_of_pcoders = numbre_of_pcoders

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

        self.compute_hp_parameters_from_values()

        self.input_mem = None

        self.backbone = copy.deepcopy(backbone)
        self.backbone.eval()

        for m in self.backbone.modules():
            if hasattr(m, 'inplace'):
                m.inplace = False

        self.pcoders = None   # pcoders will be (should be) appended here
    
    def forward(self, x=None):
        if x is not None:
            self.reset()
            self.input_mem = x

            if self.random_init:
                self.backbone(self.input_mem) # random initialization

        if not self.build_graph:
            with torch.no_grad():
                output = self.backbone(self.input_mem)
        else:
            output = self.backbone(self.input_mem)
        
        return output
  
    def reset(self):
        r"""
        To be called for each new batch of images.
        """
        self.input_mem = None
        if self.pcoders is None:
            self.pcoders = []
            for m in self.modules():
                if isinstance(m, PCoder):
                    self.pcoders.append(m)

        for pc in self.pcoders:
            pc.reset()

    def get_hyperparameters_values(self):
        r"""
        Returns the values of hyperparameters (:math:`\beta, \lambda, (1-\beta-\lambda), \alpha`).
        """
        return (self.ffm.item(), self.fbm.item(), 1-self.ffm.item()-self.fbm.item(), self.erm.item())

    def get_hyperparameters(self):
        r"""
        Returns the hyperparameters.
        """
        return (self.ff_part, self.fb_part, self.mem_part, self.errorm)

    def update_hyperparameters(self, no_grad: bool=False):
        r"""
        Updates the values of hyperparameters based on their auxiliary parameters. To be called after updating hyperparameters.

        Args:
            no_grad (boolean): `True` to disable making the computational graph.
        """

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

    def compute_hp_parameters_from_values(self):
        r"""
        Computes auxiliary parameters based on the values of hyperparameters. To be used for the initialization of hyperparameters.
        """

        with torch.no_grad():
            self.errorm.copy_(self.erm)
            self.ff_part.copy_(-torch.log((1-self.ffm)/self.ffm))
            self.fb_part.copy_(-torch.log((1-self.fbm)/self.fbm))
            fmm = 1-self.ffm-self.fbm
            self.mem_part.copy_(-torch.log((1-fmm)/fmm))

    def set_hyperparameters(self, hps: dict):
        r"""
        Sets the values of hyperparameters.

        Args:
            hps (dict): a dictionary in the form of `{'ffm' : float, 'fbm': float, 'erm': float}`
        """
        self.ffm = torch.tensor(hps['ffm'], dtype=torch.float)
        self.ffb = torch.tensor(hps['fbm'], dtype=torch.float)
        self.erm = torch.tensor(hps['erm'], dtype=torch.float)
        self.compute_hp_parameters_from_values()


class PNetSeparateHP(nn.Module):
    r"""
    Implements the base class for adding Predicitive Coding Dynamics to an existing network with separate hyperparameters for each PCoder.
    Assume that there are :math:`n` PCoders. Let :math:`pc_{i}^{r}(t)` be the output representation of PCoder :math:`i` at timestep :math:`t` based on which :math:`pc_{i}^{p}(t)` will be calculated as its prediction (:math:`i \in {1,...,n}` and :math:`t \in {0,...,T}`). Also, let :math:`ff_{i}(t)` be the feedforward drive that enters the PCoder :math:`i` at timestep :math:`t`. Then, the update dynamics will be as follows. 
    
    * Initialization (:math:`t=0`):
    
    :math:`pc_{i}^{r}(0)` will be initialized by the feedforward drive (:math:`ff_{i}(0)`) or a random representation drawn from the standard normal.

    * Iterations (:math:`0< t \leq T`):

    .. math::
        pc_{i}^{r}(t) = \beta_{i}(ff_{i}(t)) + \lambda_{i}(pc_{i+1}^{p}(t-1)) + (1-\beta_{i}-\lambda_{i})(pc_{i}^{r}(t-1)) - \alpha(\frac{\partial \epsilon_{i}(t-1)}{\partial pc_{i}^{r}(t-1)}),

    where, :math:`\epsilon_{i}(t-1)` is the prediction error computed between :math:`pc_{i}^{p}(t-1)` and :math:`pc_{i-1}^{r}(t-1)`.

    .. note::
        * :math:`pc_{0}^{r}(t)` = :math:`pc_{0}^{r}(0)` = Static input
        * For the last PCoder, there will be no feedback drive, thus there will be no :math:`lambda_{n}` as well.

    Args:
        backbone (torch.nn.Module): A callable module inherited from `torch.nn.Module`.
        numbre_of_pcoders (int): Number of PCoders
        build_graph (boolean): Indicates whether the computation graph should be built (set it to `True` during the training phase)
        random_init (boolean): Indicates whether the PCoders starts from a random representation (if `True`) or the given feedforward one (if `False`)
        ff_multiplier (sequence of float, float, optional): Values of :math:`\beta_{i}`. Default: 0.33.
        fb_multiplier (sequence of float, float, optional): Values of :math:`\lambda_{i}`. Default: 0.33.
        er_multiplier (sequence of float, float, optional): Values of :math:`\alpha_{i}`. Default: 0.01.

    .. note::
        If a single value is given for each of the hyperparameters, it will be duplicated for all PCoders.
    """

    def __init__(self, backbone: nn.Module, number_of_pcoders: int, build_graph: bool=False, random_init: bool=True, ff_multiplier: float=0.33, fb_multiplier: float=0.33, er_multiplier: float=0.01):
        r"""
        each of the hyperparameters can be a single floating point value or a list of values. in case of a single value, it will be repeated for all layers.
        """
        if isinstance(ff_multiplier, float):
            ff_multiplier = [ff_multiplier for i in range(number_of_pcoders)]
        if isinstance(fb_multiplier, float):
            fb_multiplier = [fb_multiplier for i in range(number_of_pcoders)]
        if isinstance(er_multiplier, float):
            er_multiplier = [er_multiplier for i in range(number_of_pcoders)]

        super(PNetSeparateHP, self).__init__()


        self.build_graph = build_graph
        self.random_init = random_init
        self.number_of_pcoders = number_of_pcoders

        ### PC Parameters (multipliers)
        ffms = [None for i in range(self.number_of_pcoders)]
        fbms = [None for i in range(self.number_of_pcoders)]
        erms = [None for i in range(self.number_of_pcoders)]
        for i in range(self.number_of_pcoders):
            ffms[i] = torch.tensor(ff_multiplier[i], dtype=torch.float)
            self.register_buffer(f"ffm{i+1}", ffms[i])
            
            fbms[i] = torch.tensor(fb_multiplier[i], dtype=torch.float)
            self.register_buffer(f"fbm{i+1}", fbms[i])

            erms[i] = torch.tensor(er_multiplier[i], dtype=torch.float)
            self.register_buffer(f"erm{i+1}", erms[i])

        ff_parts  = [None for i in range(self.number_of_pcoders)]
        fb_parts  = [None for i in range(self.number_of_pcoders)]
        mem_parts = [None for i in range(self.number_of_pcoders)]
        errorms   = [None for i in range(self.number_of_pcoders)]

        # trainable
        for i in range(self.number_of_pcoders):
            ff_parts[i] = nn.Parameter(torch.tensor(ff_multiplier[i]))   
            self.register_parameter(f"ff_part{i+1}", ff_parts[i])

            fb_parts[i] = nn.Parameter(torch.tensor(fb_multiplier[i]))
            self.register_parameter(f"fb_part{i+1}", fb_parts[i])

            mem_parts[i] = nn.Parameter(torch.tensor(1.0-ff_multiplier[i]-fb_multiplier[i]))
            self.register_parameter(f"mem_part{i+1}", mem_parts[i])

            errorms[i] = nn.Parameter(torch.tensor(er_multiplier[i]))
            self.register_parameter(f"errorm{i+1}", errorms[i])

        self.compute_hp_parameters_from_values()

        self.input_mem = None

        self.backbone = copy.deepcopy(backbone)
        self.backbone.eval()

        for m in self.backbone.modules():
            if hasattr(m, 'inplace'):
                m.inplace = False

        self.pcoders = None   # append pcoders here
    
    def forward(self, x=None):
        if x is not None:
            self.reset()
            self.input_mem = x

            if self.random_init:
                self.backbone(self.input_mem) # random initialization

        if not self.build_graph:
            with torch.no_grad():
                output = self.backbone(self.input_mem)
        else:
            output = self.backbone(self.input_mem)
        
        return output
  
    def reset(self):
        r"""
        To be called for each new batch of images.
        """
        self.input_mem = None
        if self.pcoders is None:
            self.pcoders = []
            for m in self.modules():
                if isinstance(m, PCoder):
                    self.pcoders.append(m)
        for pc in self.pcoders:
            pc.reset()

    def get_hyperparameters_values(self):
        r"""
        Returns the values of hyperparameters (:math:`\beta_{i}, \lambda_{i}, (1-\beta_{i}-\lambda_{i}), \alpha_{i}` for :math:`i \in {1,...,n}`).
        """
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
        r"""
        Returns the hyperparameters.
        """
        pars = []
        for i in range(self.number_of_pcoders):
            pars.append(getattr(self, f"ff_part{i+1}"))
            pars.append(getattr(self, f"fb_part{i+1}"))
            pars.append(getattr(self, f"mem_part{i+1}"))
            pars.append(getattr(self, f"errorm{i+1}"))
        return pars

    def update_hyperparameters(self, no_grad=False):
        r"""
        Updates the values of hyperparameters based on their auxiliary parameters. To be called after updating hyperparameters.

        Args:
            no_grad (boolean): `True` to disable making the computational graph.
        """

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

    def compute_hp_parameters_from_values(self):
        r"""
        Computes auxiliary parameters based on the values of hyperparameters. To be used for the initialization of hyperparameters.
        """
        with torch.no_grad():
            for i in range(1,self.number_of_pcoders+1):
                erm, ffm, fbm = getattr(self,f"erm{i}"), getattr(self,f"ffm{i}"), getattr(self,f"fbm{i}")
                fmm = 1-ffm-fbm
                
                errorm, ff_part, fb_part, mem_part = getattr(self,f"errorm{i}"), getattr(self,f"ff_part{i}"), getattr(self,f"fb_part{i}"), getattr(self,f"mem_part{i}")
                errorm.copy_(erm)
                ff_part.copy_(-torch.log((1-ffm)/ffm))
                fb_part.copy_(-torch.log((1-fbm)/fbm))
                mem_part.copy_(-torch.log((1-fmm)/fmm))

    def set_hyperparameters(self, hps: Sequence):
        r"""
        Sets the values of hyperparameters.

        Args:
            hps (Sequence: dict): a sequence of dictionaries in the form of `{'ffm' : float, 'fbm': float, 'erm': float}`
            defining hyperparameter values of each PCoder
        """
        for i in range(self.number_of_pcoders):
            setattr(self, f"ffm{i+1}", torch.tensor(hps[i]['ffm'], dtype=torch.float))
            setattr(self, f"fbm{i+1}", torch.tensor(hps[i]['fbm'], dtype=torch.float))
            setattr(self, f"erm{i+1}", torch.tensor(hps[i]['erm'], dtype=torch.float))

        self.compute_hp_parameters_from_values()

