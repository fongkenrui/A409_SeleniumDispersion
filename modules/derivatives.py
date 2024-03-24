## Module for storing derivative functions and boundary conditions

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


