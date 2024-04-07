## Module for 1-D Crank Nicholson Solver

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import copy
from .classes import Quantity1D, Quantity2D
from .functions import set_initial_condition_1D
from scipy.linalg import solve_banded
from scipy.linalg.lapack import cgtsv 

def generate_left_matrix(C, diffusion):
    """Generates the Crank-Nicholson matrix A to be inverted on the LHS of the matrix equation $AC_{i+1} = B{C_i}$

    Args:
        C (_type_): _description_
        diffusion (_type_): _description_
    """
    dx = C.dx 
    dt = C.dt
    n_grid = C.n_grid
    xcoords = C.xcoords
    # Define coefficient clusters
    def D1(x):
        return - (diffusion(x, 0) * dt)/(2 * dx**2) + (diffusion.partial_x(x, 0) * dt)/(4 * dx)
    
    def D2(x):
        return (diffusion(x, 0)* dt)/(dx**2) + 1

    def D3(x):
        return - (diffusion(x, 0) * dt)/(2 * dx**2) - (diffusion.partial_x(x, 0) * dt)/(4 * dx)

    matrix = np.zeros((n_grid, n_grid))

    # Generate the tridiagonal matrix
    for i in range(1, n_grid-1):
        x = xcoords[i]
        matrix[i, i-1:i+2] = np.array([D1(x), D2(x), D3(x)])

    # Set open neumann boundary conditions; C1 - C0 = C2 - C1, CN-1 - CN-2 = CN-2 - CN-3
    matrix[0, 0:2] = np.array([-1, 1])
    matrix[-1, -2:] = np.array([-1, 1])
    '''
    u = np.diag(matrix, k=1)
    d = np.diag(matrix)
    l = np.diag(matrix, k=-1)
    u = np.append(u, 0)
    l = np.insert(l, 0, 0)
    ab = np.array([u, d, l])
    '''
    return matrix

def generate_diagonal_banded_form(C, diffusion):
    """Generates the Crank-Nicholson matrix A to be inverted on the LHS of the matrix equation $AC_{i+1} = B{C_i}$.
    Matrix is in diagonal-banded form for input into a tridiagonal solver.

    Args:
        C (_type_): _description_
        diffusion (_type_): _description_
    """
    dx = C.dx 
    dt = C.dt
    n_grid = C.n_grid
    # Define coefficient clusters
    def D1(x):
        return - (diffusion(x, 0) * dt)/(2 * dx**2) + (diffusion.partial_x(x, 0) * dt)/(4 * dx)
    
    def D2(x):
        return (diffusion(x, 0)* dt)/(dx**2) + 1

    def D3(x):
        return - (diffusion(x, 0) * dt)/(2 * dx**2) - (diffusion.partial_x(x, 0) * dt)/(4 * dx)

    x_coords = C.xcoords
    u = np.zeros_like(x_coords)
    d = np.zeros_like(x_coords)
    l = np.zeros_like(x_coords)
    # 1-D vectors corresponding to upper, lower and middle diagonals
    u[2: n_grid] = D3(x_coords[2: n_grid])
    d[1: n_grid-1] = D2(x_coords[1: n_grid-1])
    l[0: n_grid-2] = D1(x_coords[0: n_grid-2])
    # Set boundary conditions
    u[1] = 1
    d[0] = -1
    d[-1] = 1
    l[-2] = -1
    # Stack into a banded form
    ab = np.stack((u, d, l))
    return ab

def generate_right_matrix(C, diffusion):
    """Generates the matrix B which operates on C_i in the matrix equation $A{C_i+1} = B{C_i}$
    --------------------------------------------------------------------------------------------------------------
    In order to maintain a tridiagonal matrix structure, the Neumann boundary condition is modified such that 
    C(t+dt, x1) - C(t+dt, x0) = C(t, x2) - C(t, x1) rather than C(t+dt, x1) - C(t+dt, x0) = C(t+dt, x2) - C(t+dt, x1).
    The coefficients are encoded in the left and right matrices which operate on C(t+dt) and C(t) respectively.
    -------------------------------------------------------------------------------------------------------------

    Args:
        C (_type_): _description_
        diffusion (_type_): _description_
    """

    dx = C.dx 
    dt = C.dt
    n_grid = C.n_grid
    xcoords = C.xcoords
    # Define coefficient clusters
    def D1(x):
        return (diffusion(x, 0) * dt)/(2 * dx**2) - (diffusion.partial_x(x, 0) * dt)/(4 * dx)
    
    def D2(x):
        return -(diffusion(x, 0) * dt)/(dx**2) + 1

    def D3(x):
        return (diffusion(x, 0) * dt)/(2 * dx**2) + (diffusion.partial_x(x, 0) * dt)/(4 * dx)

    matrix = np.zeros((n_grid, n_grid))
    
    # Generate the tridiagonal matrix
    for i in range(1, n_grid-1):
        x = xcoords[i]
        matrix[i, i-1:i+2] = np.array([D1(x), D2(x), D3(x)])

    # Set open neumann boundary conditions; C1 - C0 = C2 - C1
    matrix[0, 1:3] = np.array([-1, 1])
    matrix[-1, -3:-1] = np.array([-1, 1])
    
    return matrix


def crank_nicholson_1D(
    C, # Quantity1D object
    diffusion,
    initial_condition, # Vector of same shape as C sliced at a specific timestep
    sources = [], # Source/Sink array for constant sources/sinks
    sinks = [], # Sinks need to be modeled differently; linear sink where draw rate = k/2 (C(t+dt) + C(t))
    # Give up for now, sinks are a massive pain to model
):
    """Main routine for running the 1-D Crank-Nicholson method.

    Args:
        C (Quantity1D): Quantity1D object
        diffusion (XYfunc): XYfunc object, class holds 2-D function but can be defined to hold 1-D functions by calling with y=0

    """
    n_time = C.n_time
    n_grid = C.n_grid
    set_initial_condition_1D(C, initial_condition)

    ab = generate_diagonal_banded_form(C, diffusion)
    A = generate_left_matrix(C, diffusion)
    B = generate_right_matrix(C, diffusion)

    for timestep in range(1, n_time):
        C_i = C.now
        b = B@C_i
        # Add source term if applicable
        if len(sources) > 0:
            b = b + sources
        # numpy solve_banded implements cgtsv if a tridiagonal matrix is given
        C.next = solve_banded((1, 1), ab, b)
        C.store_timestep(timestep)
        C.shift()

    # Cast C to an xarray format
    ds = xr.DataArray(
        data=C.value,
        data_vars=dict(
            concentration=(['x', 't'], concentration),
            diffusion=(['x',], diffusion),
        ),
        coords={
            'x': C.xcoords,
            't': C.tcoords,
        },
        attrs={
            'dx': C.dx,
            'dt': C.dt,
            'n_grid': n_grid,
            'n_time': n_time,
            'initial_condition': initial_condition,
            'sources': sources,
            'metadata': 'Generated by crank_nicholson_1D',
        },
    )
    ds['concentration'] = C.value
    ds['diffusion'] = diffusion(C.xcoords, 0)

    return ds