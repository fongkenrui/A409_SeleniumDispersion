# Module for creating test cases

## Planned list of tests:
# Conservation of mass
# Analytic gaussian kernel
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from .ADI import ADI
from .classes import Quantity2D, Analytic, Interpolate
from matplotlib import animation


def calculate_boundary_flux(the_ds):
    """Calculates the total time-integrated flux across boundary.

    Args:
        theds (_type_): _description_
    """
    # Calculate flux along boundary by Fick's law J = -D Grad(C) dot nhat
    # Approximate spatial derivative by forward/backward euler
    # Approximate time-integral by riemann sum
    
    cumflux = np.zeros(the_ds.n_time)
    D = the_ds['diffusion'].values
    if type(D) == float:
        D = D*np.ones_like(the_ds.isel(t=0))
    dx = the_ds.attrs['dx']
    dy = the_ds.attrs['dy']
    dt = the_ds.attrs['dt']
    for n in range(the_ds.n_time):
        C = the_ds['concentration'].isel(t=n).values
        # Left Boundary
        cl0 = C[0,:]
        cl1 = C[1,:]
        Jl = D[0, :] @ (cl1 - cl0)*(dy/dx)
        # Right Boundary
        cr0 = C[-2,:]
        cr1 = C[-1,:]
        Jr = D[-1, :] @ (cr0 - cr1)*(dy/dx)
        # Top Boundary
        ct0 = C[:,0]
        ct1 = C[:, 1]
        Jt = D[:, 0] @ (ct1 - ct0)*(dx/dy)
        # Bottom Boundary
        cb0 = C[:,-2]
        cb1 = C[:,-1]
        Jb = D[:, 0] @ (cb0 - cb1)*(dx/dy)

        J_current = Jl + Jr + Jt + Jb
        if n == 0:
            cumflux[n] = J_current*dt
        else:
            cumflux[n] = J_current*dt + cumflux[n-1]

    return cumflux


def integrate_concentration(the_ds):
    """Integrates the concentration over the domain.

    Args:
        theds (_type_): _description_
    """
    dx = the_ds.attrs['dx']
    dy = the_ds.attrs['dy']
    return the_ds['concentration'].sum(dim=('x', 'y')).values*(dy*dx)

def integrate_sources(the_ds):
    dx = the_ds.attrs['dx']
    dy = the_ds.attrs['dy']
    time = the_ds.coords['t'].values
    return the_ds['sources'].sum(dim=('x', 'y')).values*(dy*dx) * time

def plot_mass_conservation(the_ds):
    """Plots total pollutant concentration, time-integrated boundary flux, source influx, and conserved mass
    normalized by the sum of conserved mass at t=0 + inflow. Conserved mass = total concentration + time-integrated
    boundary flux.

    Args:
        the_ds (_type_): _description_
    """
    time = the_ds.coords['t']
    cumflux = calculate_boundary_flux(the_ds)
    totalconc = integrate_concentration(the_ds)
    inflow = integrate_sources(the_ds)
    conserved_mass = totalconc + cumflux 
    fig, ax = plt.subplots()
    ax.plot(time, cumflux/(conserved_mass[0] + inflow), label='cumulative boundary flux')
    ax.plot(time, totalconc/(conserved_mass[0] + inflow), label='mass in boundary')
    ax.plot(time, inflow/(conserved_mass[0] + inflow), label='cumulative inflow')
    ax.plot(time, conserved_mass/(conserved_mass[0] + inflow), label='total mass')
    ax.set_xlabel("time")
    ax.set_ylabel("fractional mass")
    ax.legend()
    ax.set_title("Mass fraction scaled by initial conserved mass + inflow")
    return fig, ax

def test_gaussian(simfunc):
    """Routine that runs the 2D CN-ADI simulation and checks the discrepancy against analytic solutions.
    """
    # Define the domain
    xrange = (-10, 10)
    yrange = (-10, 10)
    trange=(0, 1)
    n_grid = 50
    n_time = 500
    conc = Quantity2D(
        n_grid,
        n_time,
        xrange,
        yrange,
        trange,
    )

    xcoords = conc.xcoords
    ycoords = conc.ycoords
    tcoords = conc.tcoords
    X, Y = np.meshgrid(xcoords, ycoords, indexing='ij')
    initial_condition =  (1/(4*np.pi))*np.exp(- (X**2 + Y**2)/4)

    diffusion = Interpolate(np.ones_like(X), xcoords, ycoords)

    def kernel(x, y, t):
        t0 = -1
        return (1/(4*np.pi*(t-t0)))*np.exp(-(x**2 + y**2)/(4*(t-t0)))

    xg, yg, tg = np.meshgrid(xcoords, ycoords, tcoords, indexing='ij')
    analytic = kernel(xg, yg, tg)
    result_ds = simfunc(conc, diffusion, initial_condition)['concentration']

    ads = xr.DataArray(
        data=analytic,
        coords={
            'x': xcoords,
            'y': ycoords,
            't': tcoords,
        },
        name='concentration',
        attrs={
            'n_grid': n_grid,
            'n_time': n_time,
            'initial_condition': initial_condition,
        },
    )
    diff = (result_ds - ads)/ads
    diff.rename('relative error')
    return diff

