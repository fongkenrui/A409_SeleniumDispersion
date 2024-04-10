## Module for storing functions defining derivatives, initial conditions and boundary conditions
import numpy as np 

def forward_euler(C, diffusion, sources):
    """Centred difference scheme for 2-D heat equation with varying diffusion coefficient

    Args:
        C (Quantity): A Quantity object representing the concentration C(x, y, t)
        diffusion (XYfunc): An XYfunc object representing the diffusion coefficient D(x, y)
 
    returns:
        dC_dt (ArrayLike): A numpy ndarray for the interior points of the grid (size (N-2) by (N-2))
    """
    
    array_x, array_y = np.meshgrid(C.xcoords, C.ycoords, indexing='ij')
    array_x, array_y = array_x[1:-1, 1:-1], array_y[1:-1, 1:-1]

    deriv2 = (C.now[2:, 1:-1] - 2*C.now[1:-1, 1:-1] + C.now[0:-2,1:-1])/C.dx**2 + (C.now[1:-1, 2:] - 2*C.now[1:-1, 1:-1] + C.now[1:-1, 0:-2])/C.dy**2
    deriv1 = diffusion.partial_x(array_x,array_y)*(C.now[2:, 1:-1] - C.now[0:-2,1:-1])/(2*C.dx) + diffusion.partial_y(array_x,array_y)*(C.now[1:-1, 2:] - C.now[1:-1, 0:-2 ])/(2*C.dy)

    C.next[1:-1,1:-1] = C.now[1:-1, 1:-1 ] + C.dt*(diffusion(array_x, array_y)*(deriv2) + (deriv1) + sources[1:-1,1:-1])



def zero_dirichlet(C): 
    """Performs in-place modification of arrays stored in the C Quantity object to enforce boundary conditions.
    Zero dirichlet implies a boundary in equilibrium with a large material 'bath' with zero pollutans. This is 
    approximately true for large enough boundaries. 

    Args:
        C (Quantity): A Quantity object representing the concentration C(x, y, t)
    """

    C.next[0, : ] = 0 
    C.next[-1, : ] = 0
    C.next[ : ,0] = 0
    C.next[ : ,-1] = 0
    



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
    C.store_timestep(0, "now")

def set_initial_condition_1D(C, initial_condition):
    """Generically sets the initial condition at C.now for a 1D spatial grid. For different time-stepping schemes custom initial_condition
    functions may need to be defined.

    Args:
        C (_type_): _description_
    """
    C.now[:] = initial_condition[:]
    C.store_timestep(0, "now")

def animate(ds, vmin=None, vmax=None):
    if not vmin:
        vmin = ds.min()
    if not vmax:
        vmax = ds.max()

    n_time = ds.attrs['n_time']
    fig, ax = plt.subplots()
    x = ds.coords['x']
    y = ds.coords['y']
    z = ds['concentration'].values
    def animate(t):
        ax.clear()
        ax.pcolormesh(x, y, z[t], vmin=vmin, vmax=vmax, cmap='seismic')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    
    anim = animation.FuncAnimation(fig, animate, frames = n_time, interval=1, repeat=True)
    plt.show()
    writergif = animation.PillowWriter(fps=30)
    return anim, writergif
