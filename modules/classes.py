
## Module for storing classes

import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator, bisplrep, bisplev


class XYfunc(object):
    """Root class for defining coefficient functions f(x, y) and their gradients. The stored functions
    need to be vectorized.
    """

    def __init__(self):
        """
        To be overridden in a child class 
        """
        self.func = None
        self.partial_x = None
        self.partial_y = None
        raise Error("Method needs to be overriden in a child class!")

    def __call__(self, *args):
        """Call the function stored in the class. Inherited by all child classes.
        """
        return self.func(*args)

    def partial_x(self):
        """
        Partial derivative of function with respect to x
        """
        raise Error("Method needs to be overriden in a child class!") 

    def partial_y(self):
        """
        Partial derivative of function with respect to y
        """
        raise Error("Method needs to be overriden in a child class!")

    def plot_2D(self):
        """
        Convenience function to visualize the function on a 2-D plot
        """
        return


class Interpolate(XYfunc):
    """Class for holding a 2-D spline function interpolated from an array 
    of values, as well its gradient functions.
    """
    def __init__(self, array, xcoords, ycoords, s=0):
        """Takes in an array of values f(x, y) with vectors of x and y values,
        and performs 2-D spline interpolation Scipy.RectBivariateSpline

        Args:
            array (ArrayLike): f(x,y) evaluated at discrete (x,y) gridpoints
            xcoords (ArrayLike): 1-D vector of x-values
            ycoords (ArrayLike): 1-D vector of y-values 
            s (float): Smoothing parameter for interpolation
        """

        # Attribute storing the actual spline function, needed for plot_2D parent method
        spline = RectBivariateSpline(xcoords, ycoords, array, s=s)
        self.func = lambda x, y: spline(x, y, grid=False)
        # Store spline derivatives so it does not get repeatedly constructed
        self.partial_x_store = spline.partial_derivative(dx=1, dy=0)
        self.partial_y_store = spline.partial_derivative(dx=0, dy=1)
        self.partial_x = lambda x, y: self.partial_x_store(x, y, grid=False)
        self.partial_y = lambda x, y: self.partial_y_store(x, y, grid=False)
        # Useful for defining the relevant domain for visualizing the function
        # Create a finer mesh based on the bounds given
        self.xcoords = xcoords
        self.ycoords = ycoords
        # Finite differences needed for regular grid interpolator
        self.dx = min(0.01, np.diff(self.xcoords)[0])
        self.dy = min(0.01, np.diff(self.ycoords)[0])

    def partial_x(self, x, y):
        """Partial derivative of function with respect to x
        """
        return self.partial_x(x, y)
        #return 0.5*self.dx*(self.func(x+self.dx, y) - self.func(x-self.dx, y))

    def partial_y(self, x, y):
        """Partial derivative of function with respect to y
        """
        return self.partial_y(x, y)
        #return 0.5*self.dy*(self.func(x, y+self.dy) - self.func(x, y-self.dy))

    def interpolate_with_func(self, func='func'):
        """
        Return interpolated values using func
        """
        X, Y = np.meshgrid(self.xcoords, self.ycoords)
        if func == 'func':
            Z = self.func(X, Y)
        elif func == 'partial_x':
            Z = self.partial_x(X, Y)
        elif func == 'partial_y':
            Z = self.partial_y(X, Y)
        else:
            raise ValueError("func must be one of 'func', 'partial_x', 'partial_y.")
            
        return X, Y, Z
        
    def plot_2D(self, func='func'):
        """
        Convenience function to visualize the function on a 2-D plot
        """
        X, Y, Z = self.interpolate_with_func(func)
        
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_surface(X, Y, Z)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        return fig, ax
    
    def plot_color_map(self, func='func'):
        """
        Display a clour map of diffusion coefficients created by interpolation
        """
        X, Y, Z = self.interpolate_with_func(func)

        fig, ax = plt.subplots(figsize=(5,5), subplot_kw = {'aspect':1})
        ax.contourf(X,Y,Z)
        ax.set_xlabel('Distance from the source in x direction (m)')
        ax.set_ylabel('Distance from the source in y direction (m)')
        cf = ax.contourf(X,Y,Z, cmap=plt.cm.get_cmap('gist_earth_r'))
        cbar = fig.colorbar(cf, ax=ax)
        cbar.formatter.set_powerlimits((0, 0))
        cbar.set_label('Diffusion coefficient (m$^2$ s$^{-1}$)', rotation=270, labelpad=15)
        return fig, ax


class Analytic(XYfunc):
    """Class for defining an analytic coefficient function f(x, y). The stored
    functions need to be vectorized.

    Args:
        xyfunc (_type_): _description_
    """

    def __init__(self, func, dx=0.01, dy=0.01):
        self.func = func
        self.partial_x = None
        self.partial_y = None
        # Parameters for finite difference approx of gradients
        self.dx = dx
        self.dy = dy

    def set_partial_x(self, func):
        self.partial_x = func

    def set_partial_y(self, func):
        self.partial_y = func

    def partial_x(self):
        if self.partial_x:
            return self.partial_x
        else:
            # Use a finite difference approximation to construct a function
            dx = self.dx
            partial_x = lambda x, y: (self.func(x + dx, y) - self.func(x - dx, y))/(2*dx)
            return partial_x

    def partial_y(self):
        if self.partial_y:
            return self.partial_y
        else:
            # Use a finite difference approximation to construct a function
            dy = self.dy
            partial_y = lambda x, y: (self.func(x, y + dy) - self.func(x, y - dy))/(2*dy)
            return partial_y
 
# More or less the same as Lab 7 Quantity object
class Quantity2D(object):
    """Object to hold arrays and attributes for the function to be solved f(x, y, t).

    Args:
        object (_type_): _description_
    """

    def __init__(self, 
        n_grid, 
        n_time,
        xrange,
        yrange,
        trange,
        ):
        """Initialize with the following parameters:

        Args:
            n_grid (_type_): _description_
            n_time (_type_): _description_
            xrange (tup): Tuple defining the x-domain interval [x0, xn]
            yrange (tup): Tuple defining the y-domain interval [y0, yn]
            trange (tup): Tuple defining the time interval [t0, tn]
        """
        self.n_grid = n_grid
        self.n_time = n_time
        self.xrange = xrange
        self.yrange = yrange
        self.trange = trange

        self.dx = (xrange[1] - xrange[0])/(n_grid - 1)
        self.dy = (yrange[1] - yrange[0])/(n_grid - 1)
        self.dt = (trange[1] - trange[0])/(n_time - 1)

        self.xcoords = np.linspace(xrange[0], xrange[1], n_grid)
        self.ycoords = np.linspace(yrange[0], yrange[1], n_grid)
        self.tcoords = np.linspace(trange[0], trange[1], n_time)
        self.prev = np.empty((n_grid, n_grid))
        self.now = np.empty((n_grid, n_grid))
        self.next = np.empty((n_grid, n_grid))

        self.store = np.empty((n_grid, n_grid, n_time))

    @property
    def value(self):
        """Property-like variable to access stored array values without
        risk of modifying the original copy.

        Returns:
            _type_: _description_
        """
        return copy.copy(self.store)
    
    def store_timestep(self, time_step, attr='next'):
        """Copy the values for the specified time step to the storage
        array.

        The `attr` argument is the name of the attribute array (prev,
        now, or next) that we are going to store.  Assigning the value
        'next' to it in the function def statement makes that the
        default, chosen because that is the most common use (in the
        time step loop).
        """
        # The __getattribute__ method let us access the attribute
        # using its name in string form;
        # i.e. x.__getattribute__('foo') is the same as x.foo, but the
        # former lets us change the name of the attribute to operate
        # on at runtime.
        self.store[:, :, time_step] = self.__getattribute__(attr)

    def shift(self):
        """Copy the .now values to .prev, and the .next values to .new.

        This reduces the storage requirements of the model to 3 n_grid
        long arrays for each quantity, which becomes important as the
        domain size and model complexity increase.  It is possible to
        reduce the storage required to 2 arrays per quantity.
        """
        # Note the use of the copy() method from the copy module in
        # the standard library here to get a copy of the array, not a
        # copy of the reference to it.  This is an important and
        # subtle aspect of the Python data model.
        self.prev = copy.copy(self.now)
        self.now = copy.copy(self.next)


class Quantity1D(object):
    """Object to hold arrays and attributes for the function to be solved f(x, t).
       
    Args:
        object (_type_): _description_
    """

    def __init__(self, 
        n_grid, 
        n_time,
        xrange,
        trange,
        ):
        """Initialize with the following parameters:

        Args:
            n_grid (_type_): _description_
            n_time (_type_): _description_
            xrange (tup): Tuple defining the x-domain interval [x0, xn]
            trange (tup): Tuple defining the time interval [t0, tn]
        """
        self.n_grid = n_grid
        self.n_time = n_time
        self.xrange = xrange
        self.trange = trange

        self.dx = (xrange[1] - xrange[0])/(n_grid - 1)
        self.dt = (trange[1] - trange[0])/(n_time - 1)

        self.xcoords = np.linspace(xrange[0], xrange[1], n_grid)
        self.tcoords = np.linspace(trange[0], trange[1], n_time)

        self.prev = np.empty(n_grid)
        self.now = np.empty(n_grid)
        self.next = np.empty(n_grid)

        self.store = np.empty((n_grid, n_time))

    @property
    def value(self):
        """Property-like variable to access stored array values without
        risk of modifying the original copy.

        Returns:
            _type_: _description_
        """
        return copy.copy(self.store)
    
    def store_timestep(self, time_step, attr='next'):
        """Copy the values for the specified time step to the storage
        array.

        The `attr` argument is the name of the attribute array (prev,
        now, or next) that we are going to store.  Assigning the value
        'next' to it in the function def statement makes that the
        default, chosen because that is the most common use (in the
        time step loop).
        """
        # The __getattribute__ method let us access the attribute
        # using its name in string form;
        # i.e. x.__getattribute__('foo') is the same as x.foo, but the
        # former lets us change the name of the attribute to operate
        # on at runtime.
        self.store[:, time_step] = self.__getattribute__(attr)

    def shift(self):
        """Copy the .now values to .prev, and the .next values to .new.

        This reduces the storage requirements of the model to 3 n_grid
        long arrays for each quantity, which becomes important as the
        domain size and model complexity increase.  It is possible to
        reduce the storage required to 2 arrays per quantity.
        """
        # Note the use of the copy() method from the copy module in
        # the standard library here to get a copy of the array, not a
        # copy of the reference to it.  This is an important and
        # subtle aspect of the Python data model.
        self.prev = copy.copy(self.now)
        self.now = copy.copy(self.next)