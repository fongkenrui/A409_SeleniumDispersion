## Module for 1-D Crank Nicholson Solver

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import copy
from classes import Quantity1D, Quantity2D


def generate_left_matrix(C, diffusion):
    """Generates the Crank-Nicholson matrix A to be inverted on the LHS of the matrix equation $AC_{i+1} = B{C_i}$

    Args:
        C (_type_): _description_
        diffusion (_type_): _description_
    """
    dx = C.dx 
    dt = C.dt
    return

def generate_right_matrix(C, diffusion):
    """Generates the matrix B which operates on C_i in the matrix equation $A{C_i+1} = B{C_i}$

    Args:
        C (_type_): _description_
        diffusion (_type_): _description_
    """
    return


def crank_nicholson_1D(
    C, # Quantity1D object
    diffusion,
    initial_condition, # Vector of same shape as C sliced at a specific timestep
):
    """Main routine for running the 1-D Crank-Nicholson method.

    Args:
        C (Quantity1D): Quantity1D object
        diffusion (XYfunc): XYfunc object, class holds 2-D function but can be defined to hold 1-D functions by calling with y=0

    """
    n_time = C.n_time

    for timestep in range(n_time):
        # d
        do 

    return