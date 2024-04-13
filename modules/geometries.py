import numpy as np 
from modules.classes import Interpolate

def create_soil_geometry(conc, pattern_func, args, D_background):
    """Create soil geometry baased on a seed soil matrix"""
    
    seed_matrix = pattern_func(*args)
    n_grid_pt = len(seed_matrix[0])
    
    if seed_matrix.shape[0] != seed_matrix.shape[1]:
        raise('A seed soil matrix must be a square matrix.')
        
    elif seed_matrix.shape[1] != n_grid_pt:
        # if the seed matrix is smaller than a n_grid_pt-square matrix, adjust the size
        
        background_matrix = np.full((n_grid_pt, n_grid_pt), D_background)
        idx = n_grid_pt // 2 - seed_matrix.shape[0] // 2
        background_matrix[idx:-idx, idx:-idx] = seed_matrix
        seed_matrix = background_matrix

    xint = np.linspace(-0.5 * conc.n_grid * conc.dx, 0.5 * conc.n_grid * conc.dx, n_grid_pt)
    yint = np.linspace(-0.5 * conc.n_grid * conc.dy, 0.5 * conc.n_grid * conc.dy, n_grid_pt)
    diffusion = Interpolate(seed_matrix, xint, yint, s=0)
    
    return diffusion


def plot_soil_geometry(conc, pattern_func, args, D_background):
    """Display soil geometry"""

    diffusion = create_soil_geometry(conc, pattern_func, args, D_background)
    fig, ax = diffusion.plot_color_map(func='func')
    
    return fig, ax


def triangle_2(a,b):
    """Return a soil matrix of a triangle pattern"""
    
    matrix = np.array(        
       [[a,a,a,a,a,a,a,a,a,a],
        [a,a,a,a,a,a,a,a,a,a],
        [a,a,a,a,a,a,a,a,a,a],
        [a,a,a,a,a,a,a,a,a,a],
        [a,a,a,a,a,a,a,a,a,b],
        [a,a,a,a,a,a,a,a,b,b],
        [a,a,a,a,a,a,a,b,b,b],
        [a,a,a,a,a,a,b,b,b,b],
        [a,a,a,a,a,b,b,b,b,b],
        [a,a,a,a,b,b,b,b,b,b]])

    return matrix


def square_upper_corner(a,b):
    """Return a soil matrix of a square pattern with an upper corner"""
    
    matrix = a*np.ones((10, 10))
    matrix[:5,:5] = b
    return matrix


def square_vertical_2layers(a,b):
    """Return a soil matrix of a square pattern with vertical division"""
    
    matrix = a*np.ones((10, 10))
    matrix[:5,:] = b
    return matrix


def square_horizontal_2layers(a,b):
    """Return a soil matrix of a square pattern with horizontal division"""
    
    matrix = a*np.ones((10, 10))
    matrix[:,:5] = b
    return matrix


def circular(a,b):
    """Return a soil matrix of a radiative pattern with two types of soils"""
    
    matrix = a*np.ones((10, 10))
    matrix[2:-2,2:-2] = np.array([
        [a,a,a,a,a,a],
        [a,a,b,b,a,a],
        [a,b,b,b,b,a],
        [a,b,b,b,b,a],
        [a,a,b,b,a,a],
        [a,a,a,a,a,a]])
    return matrix


def square_layers(a,b,c): 
    """Return a soil matrix of a squared layer pattern"""
    
    matrix = np.array([
        [a,a,a,a,a,a,a,a,a,a],
        [a,a,a,a,a,a,a,a,a,a],
        [a,a,a,a,a,a,a,a,a,a],
        [a,a,a,b,b,b,b,a,a,a],
        [a,a,a,b,c,c,b,a,a,a],
        [a,a,a,b,c,c,b,a,a,a],
        [a,a,a,b,b,b,b,a,a,a],
        [a,a,a,a,a,a,a,a,a,a],
        [a,a,a,a,a,a,a,a,a,a],
        [a,a,a,a,a,a,a,a,a,a],
    ])
    return matrix


def square_horizontal_3layers(a,b,c):
    """Return a soil matrix of a square pattern with three horizontal layers"""
    
    matrix = a*np.ones((6, 6))
    matrix[:, 2:4] = b
    matrix[:, 4:] = c
    return matrix


def square_vertical_3layers(a,b,c):
    """Return a soil matrix of a square pattern with three vertical layers"""
    
    matrix = a*np.ones((6, 6))
    matrix[2:4, :] = b
    matrix[4:, :] = c
    return matrix
    

def circular_layers(a,b,c):
    """Return a soil matrix of a radiative pattern with three types of soils"""
    
    matrix = np.array([
        [a,a,a,a,a,a],
        [a,a,b,b,a,a],
        [a,b,c,c,b,a],
        [a,b,c,c,b,a],
        [a,a,b,b,a,a],
        [a,a,a,a,a,a]])
    return matrix
