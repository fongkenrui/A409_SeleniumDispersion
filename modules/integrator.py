## Module for storing integration routines
import xarray as xr
import numpy as np
from .functions import forward_euler, zero_dirichlet, set_initial_condition_2D


def forward_euler_final(C, diffusion, initial_condition):
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

    set_initial_condition_2D(C, initial_condition)
    
    for t in np.arange(1, C.n_time):

        forward_euler(C, diffusion)
        zero_dirichlet(C) #figure out the boundary condtions 

        C.store_timestep(t)
        C.shift()
    
    X,Y = np.meshgrid(C.xcoords, C.ycoords, indexing = 'ij')
    ds = xr.Dataset(
        data_vars = dict(
            concentration = (['x', 'y', 't'], C.value),
            diffusion = (['x','y'], diffusion(X,Y)),
        ),
        
        coords={
            'x': C.xcoords,
            'y': C.ycoords,
            't': C.tcoords,
        },
        attrs={
            'dx': C.dx,
            'dy': C.dy,
            'dt': C.dt,
            'n_grid': C.n_grid,
            'n_time': C.n_time,
            #'sources': sources, if we can figure it out LOLZ
            'metadata': 'Generated by forward_euler_final',
        },
    )
    
    return ds 