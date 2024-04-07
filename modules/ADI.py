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
    def r1(x, y):
        return (diffusion(x, y) * dt)/(2 * dx**2)

    def r2(x, y):
        return (diffusion.partial_x(x, y) * dt)/(4 * dx)

    def D1(x, y):
        return - r1(x, y) + r2(x, y)

    def D2(x, y):
        return 2*r1(x, y) + 1

    def D3(x, y):
        return - r1(x, y) - r2(x, y)
    x_coords = C.xcoords
    u = np.zeros_like(x_coords)
    d = np.zeros_like(x_coords)
    l = np.zeros_like(x_coords)
    # 1-D vectors corresponding to upper, lower and middle diagonals
    u[2: n_grid] = D3(x_coords[1: n_grid-1], y)
    d[1: n_grid-1] = D2(x_coords[1: n_grid-1], y)
    l[0: n_grid-2] = D1(x_coords[1: n_grid-1], y)
    # Set boundary conditions
    if BC == 'neumann':
        u[1] = -2*r2(x_coords[0], y)
        d[0] = - D2(x_coords[0], y)
        d[-1] = D2(x_coords[-1], y)
        l[-2] = -2*r2(x_coords[-1], y)
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
    def r1(x, y):
        return (diffusion(x, y) * dt)/(2 * dy**2)

    def r2(x, y):
        return (diffusion.partial_y(x, y) * dt)/(4 * dy)

    def D1(x, y):
        return - r1(x, y) + r2(x, y)

    def D2(x, y):
        return 2*r1(x, y) + 1

    def D3(x, y):
        return - r1(x, y) - r2(x, y)

    y_coords = C.ycoords
    u = np.zeros_like(y_coords)
    d = np.zeros_like(y_coords)
    l = np.zeros_like(y_coords)
    # 1-D vectors corresponding to upper, lower and middle diagonals
    u[2: n_grid] = D3(x, y_coords[1: n_grid-1])
    d[1: n_grid-1] = D2(x, y_coords[1: n_grid-1])
    l[0: n_grid-2] = D1(x, y_coords[1: n_grid-1])
    # Set boundary conditions
    if BC == 'neumann':
        u[1] = -2*r2(x, y_coords[0])
        d[0] = - D2(x, y_coords[0])
        d[-1] = D2(x, y_coords[-1])
        l[-2] = -2*r2(x, y_coords[-1])
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

    def r1(x, y):
        return (diffusion(x, y) * dt)/(2 * dx**2)

    def r2(x, y):
        return (diffusion.partial_x(x, y) * dt)/(4 * dx)

    def D1(x, y):
        return r1(x, y) - r2(x, y)
    
    def D2(x, y):
        return -2*r1(x, y) + 1

    def D3(x, y):
        return r1(x, y) + r2(x, y)

    matrix = np.zeros((n_grid, n_grid))
    
    # Generate the tridiagonal matrix
    for i in range(1, n_grid-1):
        x = xcoords[i]
        matrix[i, i-1:i+2] = np.array([D1(x, y), D2(x, y), D3(x, y)])

    # Set neumann boundary conditions; C1 - C0 = 0 by leaving 1st and last row empty
    if BC == 'Neumann':
        matrix[0, 0] = D2(xcoords[0], y)
        matrix[0, 1] = 2*r1(xcoords[0], y)
        matrix[-1, -2] = 2*r1(xcoords[-1], y)
        matrix[-1, -1] = D2(xcoords[-1], y)

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

    def r1(x, y):
        return (diffusion(x, y) * dt)/(2 * dy**2)

    def r2(x, y):
        return (diffusion.partial_y(x, y) * dt)/(4 * dy)

    def D1(x, y):
        return r1(x, y) - r2(x, y)
    
    def D2(x, y):
        return -2*r1(x, y) + 1

    def D3(x, y):
        return r1(x, y) + r2(x, y)

    matrix = np.zeros((n_grid, n_grid))
    
    # Generate the tridiagonal matrix
    for i in range(1, n_grid-1):
        y = ycoords[i]
        matrix[i, i-1:i+2] = np.array([D1(x, y), D2(x, y), D3(x, y)])

    # Set open neumann boundary conditions; C1 - C0 = C2 - C1
    if BC == 'neumann':
        matrix[0, 0] = D2(x, ycoords[0])
        matrix[0, 1] = 2*r1(x, ycoords[0])
        matrix[-1, -2] = 2*r1(x, ycoords[-1])
        matrix[-1, -1] = D2(x, ycoords[-1])
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

    # Handle explicit direction boundary conditions
    if j == 0:
        if BC == 'neumann':
            C_jp1 = C.now[:, j+1]
            C_j = C.now[:, j]
            return (dt/dy**2) * (diff_vec * (2*C_jp1 - 2*C_j))

    elif j == (n_grid - 1):
        if BC == 'neumann':
            C_j = C.now[:, j]
            C_jm1 = C.now[:, j-1]
            return (dt/dy**2) * (diff_vec * (2*C_jm1 - 2*C_j))

    else:
        C_jp1 = C.now[:, j+1] 
        C_j = C.now[:, j]
        C_jm1 = C.now[:, j-1]

        term1 = (dt/dy**2) * (diff_vec * (C_jp1 - 2*C_j + C_jm1))
        term2 = dt/(2*dy) * (grad_diff_vec * (C_jp1 - C_jm1))
        b = term1 + term2
        
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

    # Handle explicit direction boundary conditions
    if i == 0:
        if BC == 'neumann':
            C_ip1 = C.now[i+1,:] 
            C_i = C.now[i,:]
            return (dt/dx**2) * (diff_vec * (2*C_ip1 - 2*C_i))

    elif i == (n_grid - 1):
        if BC == 'neumann':
            C_i = C.now[i,:]
            C_im1 = C.now[i-1,:]
            return (dt/dx**2) * (diff_vec * (2*C_im1 - 2*C_i))

    else:
        C_ip1 = C.now[i+1,:] 
        C_i = C.now[i,:]
        C_im1 = C.now[i-1,:]

        term1 = (dt/dx**2) * (diff_vec * (C_ip1 - 2*C_i + C_im1))
        term2 = dt/(2*dx) * (grad_diff_vec * (C_ip1 - C_im1))
        b = term1 + term2
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

    for timestep in range(1, n_time):
        # Perform x-direction implicit
        for j in range(0, n_grid): #half-timestep
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
        # Skip over saving the half-timestep results
        C.shift()

        # Perform y-direction implicit
        for i in range(0, n_grids):
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

        C.store_timestep(timestep)
        C.shift()

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