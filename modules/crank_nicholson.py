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
    # Define coefficient clusters
    def D1(x):
        partial_x = diffusion.partial_x()
        return - (diffusion(x, 0) * dt)/(2 * dx**2) - (partial_x(x, 0) * dt)/(4 * dx)
    
    def D2(x):
        return (2*diffusion(x, 0))/(2 * dx**2) + 1

    def D3(x):
        partial_x = diffusion.partial_x()
        return - (diffusion(x, 0) * dt)/(2 * dx**2) + (partial_x(x, 0) * dt)/(4 * dx)

    matrix = np.zeros((n_grid, n_grid))

    # Generate the tridiagonal matrix
    for i in range(1, n_grid-1):
        x = i*dx
        matrix[i-1:i+2] = np.array([D1(x), D2(x), D3(x)])

    # Set open neumann boundary conditions; C1 - C0 = C2 - C1
    matrix[0, 0:2] = np.array([-1, 1])
    matrix[-1, -3:-1] = np.array([-1, 1])

    return matrix

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
    # Define coefficient clusters
    def D1(x):
        partial_x = diffusion.partial_x()
        return (diffusion(x, 0) * dt)/(2 * dx**2) + (partial_x(x, 0) * dt)/(4 * dx)
    
    def D2(x):
        return -(2*diffusion(x, 0))/(2 * dx**2) + 1

    def D3(x):
        partial_x = diffusion.partial_x()
        return (diffusion(x, 0) * dt)/(2 * dx**2) - (partial_x(x, 0) * dt)/(4 * dx)

    matrix = np.zeros((n_grid, n_grid))

    # Generate the tridiagonal matrix
    for i in range(1, n_grid-1):
        x = i*dx
        matrix[i-1:i+2] = np.array([D1(x), D2(x), D3(x)])

    # Set open neumann boundary conditions; C1 - C0 = C2 - C1
    matrix[0, 1:3] = np.array([-1, 1])
    matrix[-1, -4:-2] = np.array([-1, 1])
    
    return matrix


def crank_nicholson_1D(
    C, # Quantity1D object
    diffusion,
    initial_condition, # Vector of same shape as C sliced at a specific timestep
    sources = 0, # Array of sources and sinks
):
    """Main routine for running the 1-D Crank-Nicholson method.

    Args:
        C (Quantity1D): Quantity1D object
        diffusion (XYfunc): XYfunc object, class holds 2-D function but can be defined to hold 1-D functions by calling with y=0

    """
    n_time = C.n_time
    n_grid = C.n_grid
    set_initial_condition_1D(C, initial_condition)

    for timestep in range(1, n_time):
        C_i = C.now
        A = generate_left_matrix(C, diffusion)
        B = generate_right_matrix(C, diffusion)
        b = B.dot(C_i) + sources
        # numpy solve_banded implements cgtsv if a tridiagonal matrix is given
        C.next = solve_banded((1, 1), A, b, overwrite_ab=True, overwrite_b=True)
        C.store_timestep(timestep)
        C.shift()

    x_coords = np.arange(n_grid)*C.dx
    t_coords = np.arange(n_time)*C.dt
    # Cast C to an xarray format
    ds = xr.DataArray(
        data=C.value,
        coords={
            'x': x_coords,
            't': t_coords,
        },
        name='concentration',
        attrs={
            'dx': C.dx,
            'dt': C.dt,
            'n_grid': n_grid,
            'n_time': n_time,
            'initial_condition': initial_condition,
            'sources': sources,
            'diffusion_coefficients': diffusion(x_coords),
            'metadata': 'Generated by crank_nicholson_1D'
        },
    )
    return ds