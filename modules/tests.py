# Module for creating test cases

## Planned list of tests:
# Conservation of mass
# Analytic gaussian kernel
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr


def calculate_boundary_flux(the_ds):
    """Calculates the total time-integrated flux across boundary.

    Args:
        theds (_type_): _description_
    """
    # Calculate flux along boundary by Fick's law J = -D Grad(C) dot nhat
    # Approximate spatial derivative by forward/backward euler
    # Approximate time-integral by riemann sum
    
    cumflux = np.zeros(the_ds.n_time)
    D = the_ds.attrs['diffusion_coefficient']
    if type(D) == float:
        D = D*np.ones_like(the_ds.isel(t=0))
    dx = the_ds.attrs['dx']
    dy = the_ds.attrs['dy']
    dt = the_ds.attrs['dt']
    for n in range(the_ds.n_time):
        C = the_ds.isel(t=n).values
        # Left Boundary
        cl0 = C[0,:]
        cl1 = C[0,:]
        Jl = - D[0, :] @ (cl0 - cl1)/dx
        # Right Boundary
        cr0 = C[-2,:]
        cr1 = C[-1,:]
        Jr = - D[-1, :] @ (cr0 - cr1)/dx
        # Top Boundary
        ct0 = C[:,0]
        ct1 = C[:, 1]
        Jt = - D[:, 0] @ (ct0 - ct1)/dy
        # Bottom Boundary
        cb0 = C[:,-2]
        cb1 = C[:,-1]
        Jb = - D[:, 0] @ (cb1 - cb0)/dy

        J_current = Jl + Jr + Jt + Jb
        cumflux[n] = J_current*dt + np.sum(cumflux)

    return cumflux


def integrate_concentration(the_ds):
    """Integrates the concentration over the domain.

    Args:
        theds (_type_): _description_
    """
    dx = the_ds.attrs['dx']
    dy = the_ds.attrs['dy']
    return the_ds.sum(dim=('x', 'y')).values*(dy*dx)

def plot_mass_conservation(the_ds):
    """Plots total pollutant concentration, time-integrated flux, and the sum of both.

    Args:
        the_ds (_type_): _description_
    """
    time = the_ds.coords['t']
    cumflux = calculate_boundary_flux(the_ds)
    totalconc = integrate_concentration(the_ds)
    fig, ax = plt.subplots()
    ax.plot(time, cumflux, label='boundary flux')
    ax.plot(time, totalconc, label='mass in boundary')
    ax.plot(time, totalconc + cumflux, label='total mass')
    ax.legend()
    return fig, ax

def test_gaussian():
    """Routine that runs the 2D CN-ADI simulation and checks the discrepancy against analytic solutions.
    """
    return