## Module for storing integration routines
import xarray as xr
import numpy as np

def forward_euler(C, diffusion, boundary, deriv, parameters):
    """_summary_

    Args:
        C (Quantity): Concentration C(x, y, t)
        diffusion (XYfunc): Diffusion coefficient D(x, y)
        boundary (callable): Function that imposes BCs on a Quantity object by modifying in-place
        deriv (callable): Function that takes in a Quantity object and returns the derivative at
        interior points. 
        parameters (dict): Dictionary containing other simulation parameters

    Returns:
        DataSet: xarray dataset containing simulation attributes and results
    """
    
    return ds 