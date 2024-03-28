## Module for storing functions defining derivatives, initial conditions and boundary conditions

def centred_diff_2D(C, diffusion):
    """Centred difference scheme for 2-D heat equation with varying diffusion coefficient

    Args:
        C (Quantity): A Quantity object representing the concentration C(x, y, t)
        diffusion (XYfunc): An XYfunc object representing the diffusion coefficient D(x, y)

    returns:
        dC_dt (ArrayLike): A numpy ndarray for the interior points of the grid (size (N-2) by (N-2))
    """
    return

def zero_dirichlet(C): 
    """Performs in-place modification of arrays stored in the C Quantity object to enforce boundary conditions.
    Zero dirichlet implies a boundary in equilibrium with a large material 'bath' with zero pollutans. This is 
    approximately true for large enough boundaries. 

    Args:
        C (Quantity): A Quantity object representing the concentration C(x, y, t)
    """
    return


def neumann(C): # In hindsight, this kind of neumann conditions should be more accurate
    """Performs in-place modification of arrays stored in the C Quantity object to enforce boundary conditions.
    Neumann BCs are enforced by setting boundary fluxes to be equal to that of neighbouring interior points.

    Args:
        C (Quantity): A Quantity object representing the concentration C(x, y, t)
    """
    return


def set_initial_condition_2D(C, initial_condition):
    """Generically sets the initial condition at C.now for a 2D spatial grid. For different time-stepping schemes custom initial_condition
    functions may need to be defined.

    Args:
        C (_type_): _description_
    """
    # Array broadcasting rules checks that array sizes are identical
    C.now[:,:] = initial_condition[:,:] 
    C.store_timestep(0)

def set_initial_condition_1D(C, initial_condition):
    """Generically sets the initial condition at C.now for a 1D spatial grid. For different time-stepping schemes custom initial_condition
    functions may need to be defined.

    Args:
        C (_type_): _description_
    """
    C.now[:] = initial_condition[:]
    C.store_timestep(0)

