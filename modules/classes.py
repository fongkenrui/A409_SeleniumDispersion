
## Module for storing classes

import numpy as np
import matplotlib.pyplot as plt
import copy

class XYfunc(object):
    """Root class for defining coefficient functions f(x, y) and their gradients.
    """

    def __init__(self):
        """
        To be overridden in a child class 
        """
        self.func = None
        raise Error("Method needs to be overriden in a child class!")

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


class Interpolate(xyfunc):
    """Class for holding a 2-D spline function interpolated from an array 
    of values, as well its gradient functions.
    """
    def __init__(self, array, xcoords, ycoords):
        """Takes in an array of values f(x, y) with vectors of x and y values,
        and performs 2-D spline interpolation Scipy.RectBivariateSpline

        Args:
            array (ArrayLike): f(x,y) evaluated at discrete (x,y) gridpoints
            xcoords (ArrayLike): 1-D vector of x-values
            ycoords (ArrayLike): 1-D vector of y-values 
        """

        # Attribute storing the actual spline function, needed for plot_2D parent method
        self.func = None
        # Useful for defining the relevant domain for visualizing the function 
        self.xcoords = xcoords
        self.ycoords = ycoords
        # Gradient of spline function should be generated at initialization
        return

    def partial_x(self):
        """Partial derivative of function with respect to x
        """
        return partial_x

    def partial_y(self):
        """Partial derivative of function with respect to y
        """
        return partial_y

    def plot_2D(self):
        """
        Convenience function to visualize the function on a 2-D plot
        """
        return

# More or less the same as Lab 7 Quantity object
class Quantity(object):
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
        self.dt = (trange[1] - trange[0])/(n_time = 1)

        self.prev = np.empty((n_grid, n_grid))
        self.now = np.empty((n_grid, n_grid))
        self.next = np.empty((n_grid, n_grid))

        self.store = np.empty((n_grid, n_grid, n_time))
    
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