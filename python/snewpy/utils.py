import os
import numpy as np

def expand_dimensions_to(a:np.ndarray, ndim:int)->np.ndarray:
    """Expand the dimensions of the array, adding dimensions of len=1 to the right,
    so total dimensions equal to `ndim`"""
    new_shape = (list(a.shape)+[1]*ndim)[:ndim]
    return a.reshape(new_shape)
    
def strip_extensions(filename):
    strip_extensions = ['.dat', '.txt', '.fits', '.h5', '.tar', '.gz', '.bz2']
    # Split the filename into the base and the extension
    while True:
        filename, ext = os.path.splitext(filename)
        print(filename,ext)
        if ext.lower() not in strip_extensions:
            filename += ext
            break    
    return filename
    
