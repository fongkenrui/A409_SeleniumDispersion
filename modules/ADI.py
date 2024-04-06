## Module for implementing 2-D Crank-Nicholson with Alternating-Direction Implicit Method

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import copy
from .classes import Quantity1D, Quantity2D
from .functions import set_initial_condition_2D
from scipy.linalg import solve_banded
from scipy.linalg.lapack import cgtsv 

## Matrix Functions

def generate_left_matrix_x(C, diffusion, y, BC): #TODO: Memoization tradeoff? Will have n_grid matrices to store
    """Generates the Crank-Nicholson matrix A to be inverted on the LHS of the matrix equation $AC_{i+1} = B{C_i}$.
    Matrix is in diagonal-banded form for input into a tridiagonal solver.
    --------------------------------------------------------------------------------------------------------------
    In order to maintain a tridiagonal matrix structure, the Neumann boundary condition is modified such that 
    C(t+dt, x1) - C(t+dt, x0) = C(t, x2) - C(t, x1) rather than C(t+dt, x1) - C(t+dt, x0) = C(t+dt, x2) - C(t+dt, x1).
    The coefficients are encoded in the left and right matrices which operate on C(t+dt) and C(t) respectively.
    -------------------------------------------------------------------------------------------------------------

    Args:
        C (_type_): _description_
        diffusion (_type_): _description_
        y (float): current y-coordinate held fixed
    """
    dx = C.dx 
    dt = C.dt/2
    n_grid = C.n_grid
    # Define coefficient clusters
    def D1(x, y):
        return - (diffusion(x, y) * dt)/(2 * dx**2) + (diffusion.partial_x(x, y) * dt)/(4 * dx)
    
    def D2(x, y):
        return (diffusion(x, y)* dt)/(dx**2) + 1

    def D3(x, y):
        return - (diffusion(x, y) * dt)/(2 * dx**2) - (diffusion.partial_x(x, y) * dt)/(4 * dx)

    x_coords = C.xcoords
    u = np.zeros_like(x_coords)
    d = np.zeros_like(x_coords)
    l = np.zeros_like(x_coords)
    # 1-D vectors corresponding to upper, lower and middle diagonals
    u[2: n_grid] = D3(x_coords[2: n_grid], y)
    d[1: n_grid-1] = D2(x_coords[1: n_grid-1], y)
    l[0: n_grid-2] = D1(x_coords[0: n_grid-2], y)
    # Set boundary conditions
    if BC == 'neumann':
        u[1] = 1
        d[0] = -1
        d[-1] = 1
        l[-2] = -1
    # Stack into a banded form
    ab = np.stack((u, d, l))
    return ab

def generate_left_matrix_y(C, diffusion, x, BC): 
    """Generates the Crank-Nicholson matrix A to be inverted on the LHS of the matrix equation $AC_{i+1} = B{C_i}$.
    Matrix is in diagonal-banded form for input into a tridiagonal solver.
    --------------------------------------------------------------------------------------------------------------
    In order to maintain a tridiagonal matrix structure, the Neumann boundary condition is modified such that 
    C(t+dt, x1) - C(t+dt, x0) = C(t, x2) - C(t, x1) rather than C(t+dt, x1) - C(t+dt, x0) = C(t+dt, x2) - C(t+dt, x1).
    The coefficients are encoded in the left and right matrices which operate on C(t+dt) and C(t) respectively.
    -------------------------------------------------------------------------------------------------------------

    Args:
        C (_type_): _description_
        diffusion (_type_): _description_
        x (float): current x-coordinate held fixed
    """
    dy = C.dy 
    dt = C.dt/2
    n_grid = C.n_grid
    # Define coefficient clusters
    def D1(x, y):
        return - (diffusion(x, y) * dt)/(2 * dy**2) + (diffusion.partial_y(x, y) * dt)/(4 * dy)
    
    def D2(x, y):
        return (diffusion(x, y)* dt)/(dy**2) + 1

    def D3(x, y):
        return - (diffusion(x, y) * dt)/(2 * dy**2) - (diffusion.partial_y(x, y) * dt)/(4 * dy)

    y_coords = C.ycoords
    u = np.zeros_like(y_coords)
    d = np.zeros_like(y_coords)
    l = np.zeros_like(y_coords)
    # 1-D vectors corresponding to upper, lower and middle diagonals
    u[2: n_grid] = D3(x, y_coords[2: n_grid])
    d[1: n_grid-1] = D2(x, y_coords[1: n_grid-1])
    l[0: n_grid-2] = D1(x, y_coords[0: n_grid-2])
    # Set boundary conditions
    if BC == 'neumann':
        u[1] = 1
        d[0] = -1
        d[-1] = 1
        l[-2] = -1
    # Stack into a banded form
    ab = np.stack((u, d, l))
    return ab


def generate_right_matrix_x(C, diffusion, y, BC): 
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
    dt = C.dt/2
    n_grid = C.n_grid
    xcoords = C.xcoords
    # Define coefficient clusters
    def D1(x, y):
        return (diffusion(x, y) * dt)/(2 * dx**2) - (diffusion.partial_x(x, y) * dt)/(4 * dx)
    
    def D2(x, y):
        return -(diffusion(x, y) * dt)/(dx**2) + 1

    def D3(x, y):
        return (diffusion(x, y) * dt)/(2 * dx**2) + (diffusion.partial_x(x, y) * dt)/(4 * dx)

    matrix = np.zeros((n_grid, n_grid))
    
    # Generate the tridiagonal matrix
    for i in range(1, n_grid-1):
        x = xcoords[i]
        matrix[i, i-1:i+2] = np.array([D1(x, y), D2(x, y), D3(x, y)])

    # Set neumann boundary conditions; C1 - C0 = 0 by leaving 1st and last row empty
    if BC == 'Neumann':
        return matrix

    else:
        matrix[0, 1:3] = np.array([-1, 1])
        matrix[-1, -3:-1] = np.array([-1, 1])
        return matrix

def generate_right_matrix_y(C, diffusion, x, BC): 
    """Generates the matrix B which operates on C_i in the matrix equation $A{C_i+1} = B{C_i} + d$
    --------------------------------------------------------------------------------------------------------------
    In order to maintain a tridiagonal matrix structure, the Neumann boundary condition is modified such that 
    C(t+dt, x1) - C(t+dt, x0) = C(t, x2) - C(t, x1) rather than C(t+dt, x1) - C(t+dt, x0) = C(t+dt, x2) - C(t+dt, x1).
    The coefficients are encoded in the left and right matrices which operate on C(t+dt) and C(t) respectively.
    -------------------------------------------------------------------------------------------------------------

    Args:
        C (_type_): _description_
        diffusion (_type_): _description_
    """

    dy = C.dy
    dt = C.dt/2
    n_grid = C.n_grid
    ycoords = C.ycoords
    # Define coefficient clusters
    def D1(x, y):
        return (diffusion(x, y) * dt)/(2 * dy**2) - (diffusion.partial_y(x, y) * dt)/(4 * dy)
    
    def D2(x, y):
        return -(diffusion(x, y) * dt)/(dy**2) + 1

    def D3(x, y):
        return (diffusion(x, y) * dt)/(2 * dy**2) + (diffusion.partial_x(x, y) * dt)/(4 * dy)

    matrix = np.zeros((n_grid, n_grid))
    
    # Generate the tridiagonal matrix
    for i in range(1, n_grid-1):
        y = ycoords[i]
        matrix[i, i-1:i+2] = np.array([D1(x, y), D2(x, y), D3(x, y)])

    # Set open neumann boundary conditions; C1 - C0 = C2 - C1
    if BC == 'neumann':
        return matrix
    
    else:
        matrix[0, 1:3] = np.array([-1, 1])
        matrix[-1, -3:-1] = np.array([-1, 1])
        
        return matrix

def generate_explicit_comp_x(C, diffusion, j, BC):
    """Generates the vector d in the matrix equation $A{C_i+1} = B{C_i} + d$

    Args:
        C (_type_): Concentration Quantity object
        diffusion (_type_): diffusion coefficient object
        j (_type_): j-index of current y value

    Returns:
        ndarray: 1-D ndarray
    """
    x_coords = C.xcoords
    n_grid = C.n_grid
    y = C.ycoords[j]
    dy = C.dy
    dt = C.dt/2
    diff_vec = diffusion(x_coords, y) # diffusion needs to be a vectorized function
    grad_diff_vec = diffusion.partial_y(x_coords, y)
    C_jp1 = C.now[:, j+1] 
    C_j = C.now[:, j]
    C_jm1 = C.now[:, j-1]

    term1 = (dt/dy**2) * (diff_vec * (C_jp1 - 2*C_j + C_jm1))
    term2 = dt/(2*dy) * (grad_diff_vec * (C_jp1 - C_jm1))
    b = term1 + term2
    if BC == 'neumann':
        b[0] = 0
        b[-1] = 0
        return b
    else:
        return b

def generate_explicit_comp_y(C, diffusion, i, BC): 
    """Generates the vector d in the matrix equation $A{C_i+1} = B{C_i} + d$

    Args:
        C (_type_): Concentration Quantity object
        diffusion (_type_): diffusion coefficient object
        j (_type_): i-index of current i value

    Returns:
        ndarray: 1-D ndarray
    """
    y_coords = C.ycoords
    n_grid = C.n_grid
    x = C.xcoords[i]
    dx = C.dx
    dt = C.dt/2
    diff_vec = diffusion(x, y_coords) # diffusion needs to be a vectorized function
    grad_diff_vec = diffusion.partial_x(x, y_coords)
    C_ip1 = C.now[i+1,:] 
    C_i = C.now[i,:]
    C_im1 = C.now[i-1,:]

    term1 = (dt/dx**2) * (diff_vec * (C_ip1 - 2*C_i + C_im1))
    term2 = dt/(2*dx) * (grad_diff_vec * (C_ip1 - C_im1))
    b = term1 + term2
    if BC == 'neumann':
        b[0] = 0
        b[-1] = 0
        return b
    else:
        return b


def ADI(
    C,
    diffusion,
    initial_condition,
    BC = 'neumann',
):
    """Main routine for running 2D Crank-Nicholson with the alternating-direction implicit method.

    Args:
        C (_type_): _description_
        diffusion (_type_): _description_
        initial_condition (_type_): _description_
    """
    n_time = C.n_time
    n_grid = C.n_grid
    dx = C.dx
    dy = C.dy
    set_initial_condition_2D(C, initial_condition)
    xcoords = C.xcoords
    ycoords = C.ycoords

    # Generate matrices beforehand? Its alot of matrices to store...
    # Maybe store in a dictionary? 

    for timestep in range(1, n_time-1):
        # Perform x-direction implicit
        for j in range(1, n_grid - 1): #half-timestep
            y = ycoords[j]
            C_i = C.now[:, j] # Slice the array
            ab = generate_left_matrix_x(C, diffusion, y, BC=BC)
            B = generate_right_matrix_x(C, diffusion, y, BC=BC)
            # Explicit component
            d = generate_explicit_comp_x(C, diffusion, j, BC=BC)
            b = B@C_i + d
            try:
                C_next = solve_banded((1,1), ab, b)
            except ValueError as e:
                print("Timestep:", timestep, "x-dir")
                print(B)
                print(d)
                raise ValueError
            C.next[:, j] = C_next
        # Handle explicit direction boundaries
        if BC == 'neumann':
            C.next[:, -1] = C.next[:, -2]
            C.next[:, 0] = C.next[:, 1]
        # Skip over saving the half-timestep results
        C.shift()

        # Perform y-direction implicit
        for i in range(1, n_grid-1):
            x = xcoords[i]
            C_j = C.now[i, :] # Slice the array
            ab = generate_left_matrix_y(C, diffusion, x, BC=BC)
            B = generate_right_matrix_y(C, diffusion, x, BC=BC)
            # Explicit component
            d = generate_explicit_comp_y(C, diffusion, i, BC=BC)
            b = B@C_j + d
            try:
                C_next = solve_banded((1,1), ab, b)
            except ValueError as e:
                print("Timestep:", timestep, "x-dir,", e)
                print(B)
                print(d)
            C.next[i, :] = C_next
        # Handle explicit direction boundaries
        if BC == 'neumann':
            C.next[-1, :] = C.next[-2, :]
            C.next[0, :] = C.next[1, :]

        C.store_timestep(timestep)
        C.shift()

    # Cast C to an xarray format
    X, Y = np.meshgrid(xcoords, ycoords)

    ds = xr.DataArray(
        data=C.value,
        coords={
            'x': C.xcoords,
            'y': C.ycoords,
            't': C.tcoords,
        },
        name='concentration',
        attrs={
            'dx': C.dx,
            'dy': C.dy,
            'dt': C.dt,
            'n_grid': n_grid,
            'n_time': n_time,
            'initial_condition': initial_condition,
            'diffusion_coefficient': diffusion(X, Y),
            'metadata': 'Generated by crank_nicholson_1D',
        },
    )

    return ds