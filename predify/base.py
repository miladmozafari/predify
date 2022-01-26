import torch
import os
import toml
import warnings

def imports_str(is_3d=False, p_imports=None, same_hp=False, gscale=False, indent=""):
    if gscale:
        imports =  indent + "from predify.modules import PCoderN" + os.linesep
    else:
        imports =  indent + "from predify.modules import PCoder" + os.linesep
    if same_hp:
        imports += indent + "from predify.networks import PNetSameHP" + os.linesep
    else:
        imports += indent + "from predify.networks import PNetSeparateHP" + os.linesep
    if p_imports is None:
        if is_3d:
            imports += indent + "from torch.nn import Sequential, ConvTranspose3d, Upsample" + os.linesep
        else:
            imports += indent + "from torch.nn import Sequential, ConvTranspose2d, Upsample" + os.linesep
    else:
        for p_imp in p_imports:
            imports += indent + p_imp + os.linesep
    return imports

def pcoder_str(pcoder_idx, module_name, src_shape, target_shape, has_feedback, same_param, gscale, module=None, indent=""):
    s = indent + f"# PCoder number {pcoder_idx}" + os.linesep
    
    is_3d = len(src_shape) == 5

    if module is None:
        upsample = ""
        if is_3d:
            if src_shape[-3:] != target_shape[-3:]:
                scale_factor = (target_shape[-3]/src_shape[-3], target_shape[-2]/src_shape[-2], target_shape[-1]/src_shape[-1])
                upsample = f"Upsample(scale_factor={scale_factor}, mode='bilinear', align_corners=False),"
            convt = f"ConvTranspose3d({src_shape[-4]}, {target_shape[-4]}, kernel_size=3, stride=1, padding=1)"
        else:
            if src_shape[-2:] != target_shape[-2:]:
                scale_factor = (target_shape[-2]/src_shape[-2], target_shape[-1]/src_shape[-1])
                upsample = f"Upsample(scale_factor={scale_factor}, mode='bilinear', align_corners=False),"
            convt = f"ConvTranspose2d({src_shape[-3]}, {target_shape[-3]}, kernel_size=3, stride=1, padding=1)"
        pmodule = f"pmodule = Sequential({upsample}{convt})"
    else:
        pmodule = f"pmodule = {module}"
    
    s += indent + pmodule + os.linesep
    if gscale:
        s += indent + f"self.pcoder{pcoder_idx} = PCoderN(pmodule, {has_feedback}, self.random_init)" + os.linesep
    else:
        s += indent + f"self.pcoder{pcoder_idx} = PCoder(pmodule, {has_feedback}, self.random_init)" + os.linesep

    fb = f"self.pcoder{pcoder_idx+1}.prd" if has_feedback else "None"
    target = f"self.pcoder{pcoder_idx-1}.rep" if pcoder_idx > 1 else "self.input_mem"
    ffm = "self.ffm" if same_param else f"self.ffm{pcoder_idx}"
    fbm = "self.fbm" if same_param else f"self.fbm{pcoder_idx}"
    erm = "self.erm" if same_param else f"self.erm{pcoder_idx}"

    s += indent + f"def fw_hook{pcoder_idx}(m, m_in, m_out):" + os.linesep
    s += indent + f"    e = self.pcoder{pcoder_idx}(ff=m_out, fb={fb}, target={target}, build_graph=self.build_graph, ffm={ffm}, fbm={fbm}, erm={erm})" + os.linesep
    s += indent + f"    return e[0]" + os.linesep
    s += indent + f"self.backbone.{module_name}.register_forward_hook(fw_hook{pcoder_idx})" + os.linesep
    return s

def header_str(number_of_pcoders, class_name, same_param, hyperparameters, indent=""):
    if same_param:
        inherit = "SameHP(PNetSameHP)"
        ff = f"{float(hyperparameters['feedforward'])}"
        fb = f"{float(hyperparameters['feedback'])}"
        er = f"{float(hyperparameters['pc'])}"
    else:
        inherit = "SeparateHP(PNetSeparateHP)"
        ff, fb, er = "(", "(", "("
        for hp in hyperparameters:
            ff += f"{float(hp['feedforward'])},"
            fb += f"{float(hp['feedback'])},"
            er += f"{float(hp['pc'])},"
        ff = ff[:-1] + ")"
        fb = fb[:-1] + ")"
        er = er[:-1] + ")"
    s  =  indent + f"class {class_name}{inherit}:" + os.linesep
    s +=  indent + f"    def __init__(self, backbone, build_graph=False, random_init=False, ff_multiplier={ff}, fb_multiplier={fb}, er_multiplier={er}):" + os.linesep
    s +=  indent + f"        super().__init__(backbone, {number_of_pcoders}, build_graph, random_init, ff_multiplier, fb_multiplier, er_multiplier)" + os.linesep
    return s

def parse_toml(config_file):
    config = toml.load(config_file)

    name = config.get("name", "network")
    if name == "Net":
        raise Exception("Config File Error! 'name' cannot have the value 'Net'. Please select other names.")
    if not name.isidentifier():
        raise Exception("Config File Error! 'name' should be a valid Python identifier.")
    
    input_size = config.get("input_size", None)
    if input_size is None:
        raise Exception("Config File Error! 'input_size' is not specified.")
    if not isinstance(input_size, list) or len(input_size) <= 2 or len(input_size) >= 5:
        raise Exception("Config File Error! 'input_size' should be defined as a list of [channels, height, width] for 2D inputs or [channels, depth, height, width] for 3D inputs.")
    if len(input_size) == 4:
        warnings.warn("Input dimension is set to 3D since the input size is defined as list of length 4: [channels, depth, height, width]. If you want to use 2D inputs, please specify it as [channels, height, width]", stacklevel=2)
    input_size = tuple(input_size)

    pcoders = config.get("pcoders", None)
    layers = []
    pmodules = []
    hyperparams = []

    if pcoders is None:
        raise Exception("Config File Error! 'pcoders' is not specified.")
    if not isinstance(pcoders, list) or len(pcoders) == 0:
        raise Exception("Config File Error! 'pcoders' should be defined as a non-empty list.")
    for pc_idx, pc in enumerate(pcoders):
        if 'module' not in pc:
            raise Exception(f"Config File Error! 'module' is not specified for PCoder number {pc_idx + 1} (starting from 1).")
        layer = pc['module']
        layers.append(layer)

        predictor = pc.get("predictor", None)
        if predictor is None:
            warnings.warn(f"'predictor' is not defined for PCoder number {pc_idx}. A default module will be inferred based on the other PCoders.", stacklevel=2)
        pmodules.append(predictor)
        
        if "hyperparameters" not in pc:
            warnings.warn(f"'hyperparameters' is not defined for PCoder number {pc_idx}. Default values of feedforward=0.3, feedback=0.3, pc=0.01 will be set.", stacklevel=2)
        hps = pc.get("hyperparameters", {"feedforward":0.3, "feedback":0.3, "pc":0.01})
        if not ("feedforward" in hps and "feedback" in hps and "pc" in hps):
            raise Exception(f"Config File Error! 'hyperparameters' for PCoder number {pc_idx + 1} should be a disctionary that contains 'feedforward', 'feedback', 'pc' as the keys.")
        hyperparams.append(hps)

    p_imports = config.get("imports", None)
    if p_imports is not None:
        if not isinstance(p_imports, list):
            raise Exception(f"Config File Error! 'imports' should be defined as a list of strings.")

    gscale = config.get("gradient_scaling", None)
    if gscale is None:
        warnings.warn(f"'gradient_scaling' is not defined. It will be set to False by default.", stacklevel=2)
        gscale = False

    same_hp = config.get("shared_hyperparameters", None)
    if same_hp is None:
        warnings.warn(f"'shared_hyperparameters' is not defined. It will be set to False by default.", stacklevel=2)
        same_hp = False
    if same_hp is True:
        warnings.warn(f"'shared_hyperparameters' is True. It will overwrite the values defined for each PCoder. It uses the default value or the values provided for the first PCoder.", stacklevel=2)

    return name, layers, pmodules, hyperparams, p_imports, input_size, gscale, same_hp
        
def predify(net, config_file, output_address=None):
    """
    Generates a python script that defines the predified version of the given network.
    
    Args:
        net: a PyTorch compatible network object.
        config_file: address to the .toml file that contains predify-required info.
        output_address: address of the output script file. If `None`, the name of the network will be used.
    """
    config = parse_toml(config_file)
    predify_core(net, *config, output_address)

def predify_core(net, name, layers, pmodules, hps, p_imports, input_size, gscale, same_hp, output_address):
    net.eval()

    layer_sizes = {}
    def get_size(m, m_in, m_out):
        layer_sizes[m] = m_out.shape

    modules = []
    for layer in layers:
        modules.append(eval(f"net.{layer}"))
        modules[-1].register_forward_hook(get_size)

    x = torch.ones(1, *input_size)
    with torch.no_grad():
        net(x)

    is_3d = len(input_size) == 5

    if output_address is None:
        output_address = f"{name.lower()}.py"
    with open(output_address, "w") as f:
        f.write(imports_str(is_3d, p_imports=p_imports, same_hp=same_hp, gscale=gscale))
        f.write(os.linesep)

        if same_hp:
            hps = hps[0]
        f.write(header_str(len(layers), f"{name}", same_hp, hyperparameters=hps, indent=""))
        f.write(os.linesep)

        indent = " " * 8

        target_shape = input_size
        for idx, (module_name, module) in enumerate(zip(layers, modules), 1):
            src_shape = layer_sizes[module]
            s = pcoder_str(idx, module_name, src_shape, target_shape, idx < len(layers), same_hp, gscale, module=pmodules[idx-1], indent=indent)
            f.write(s)
            f.write(os.linesep)
            target_shape = src_shape
        f.write(os.linesep)