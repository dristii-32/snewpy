from inspect import isclass
import os
import numpy as np
import importlib

def expand_dimensions_to(a:np.ndarray, ndim:int)->np.ndarray:
    """Expand the dimensions of the array, adding dimensions of len=1 to the right,
    so total dimensions equal to `ndim`"""
    new_shape = (list(a.shape)+[1]*ndim)[:ndim]
    return a.reshape(new_shape)
    
def strip_extensions(filename):
    strip_extensions = ['.dat', '.txt', '.fits', '.h5', '.tar', '.gz', '.bz2', '.npz', '.npy']
    # Split the filename into the base and the extension
    while True:
        filename, ext = os.path.splitext(filename)
        if ext.lower() not in strip_extensions:
            filename += ext
            break    
    return filename

def get_transformation(flavor_transformation: str):
    """Identify the flavor transformation from a string

    Parameters
    ---------
    flavor_transformation : str
        Name of the flavor transformation

    Returns
    -------
    FlavorTransformation object initialized with default parameters
    """

    IMO_mix_params = MixingParameters(MassHierarchy.INVERTED)
    NMO_mix_params = MixingParameters(MassHierarchy.NORMAL) 

    # Choose flavor transformation. Use dict to associate the transformation name with its class.
    # The default mixing paramaters are the normal hierarchy values
    flavor_transformation_dict = {'NoTransformation': NoTransformation(), 
                                  'CompleteExchange': CompleteExchange(),                                
                                  'AdiabaticMSW_NMO': AdiabaticMSW(NMO_mix_params), 
                                  'AdiabaticMSW_IMO': AdiabaticMSW(IMO_mix_params), 
                                  'NonAdiabaticMSWH_NMO': NonAdiabaticMSWH(NMO_mix_params), 
                                  'NonAdiabaticMSWH_IMO': NonAdiabaticMSWH(IMO_mix_params), 
                                  'TwoFlavorDecoherence': TwoFlavorDecoherence(NMO_mix_params), 
                                  'TwoFlavorDecoherence_NMO': TwoFlavorDecoherence(NMO_mix_params), 
                                  'TwoFlavorDecoherence_IMO': TwoFlavorDecoherence(IMO_mix_params), 
                                  'ThreeFlavorDecoherence': ThreeFlavorDecoherence(NMO_mix_params),
                                  'NeutrinoDecay_NMO': NeutrinoDecay(NMO_mix_params), 
                                  'NeutrinoDecay_IMO': NeutrinoDecay(IMO_mix_params), 
                                  'QuantumDecoherence_NMO': QuantumDecoherence(NMO_mix_params), 
                                  'QuantumDecoherence_IMO': QuantumDecoherence(IMO_mix_params),
                                  }

    try:
        return flavor_transformation_dict[flavor_transformation]
    except KeyError:
        raise ValueError(f"Flavor transformation '{flavor_transformation}' not found.")
    
def get_model_class(model_type: str):
    """Look up model class corresponding to the given model name.
    """    
    models_dict = {}
    modules_list = ["snewpy.models.ccsn", "snewpy.models.presn"]
    for module_name in modules_list:
        module = importlib.import_module(module_name)
        models_dict.update({k:v for k,v in vars(module).items() if isclass(v)})

    try:
        return models_dict[model_type]
    except KeyError:
        raise ValueError(f"Model '{model_type}' not found.")
