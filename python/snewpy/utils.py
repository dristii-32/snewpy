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
