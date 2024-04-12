import numpy as np 

def triangle_2(a,b):
    """Return a soil matrix of a triangle pattern"""
    
    a = np.full((10, 10), 0.02)  
    b = np.tril(a)
    matrix = a + b
    return matrix


def square_upper_corner(a,b):
    """Return a soil matrix of a square pattern with an upper corner"""
    
    matrix = a*np.ones((10, 10))
    matrix[:5,:5] = b
    return matrix


def square_vertical_2layers(a,b):
    """Return a soil matrix of a square pattern with vertical division"""
    
    matrix = a*np.ones((10, 10))
    matrix[:5,:] = b
    return matrix


def square_horizontal_2layers(a,b):
    """Return a soil matrix of a square pattern with horizontal division"""
    
    matrix = a*np.ones((10, 10))
    matrix[:,:5] = b
    return matrix


def circular(a,b):
    """Return a soil matrix of a radiative pattern with two types of soils"""
    
    matrix = a*np.ones((10, 10))
    matrix[2:-2,2:-2] = np.array([
        [a,a,a,a,a,a],
        [a,a,b,b,a,a],
        [a,b,b,b,b,a],
        [a,b,b,b,b,a],
        [a,a,b,b,a,a],
        [a,a,a,a,a,a]])
    return matrix


def square_layers(a,b,c): 
    """Return a soil matrix of a squared layer pattern"""
    
    matrix = np.array([[a,a,a,a,a,a],
                    [a,b,b,b,b,a],
                    [a,b,c,c,b,a],
                    [a,b,c,c,b,a],
                    [a,b,b,b,b,a],
                    [a,a,a,a,a,a]])
    return matrix


def square_horizontal_3layers(a,b,c):
    """Return a soil matrix of a square pattern with three horizontal layers"""
    
    matrix = a*np.ones((6, 6))
    matrix[:, 2:4] = b
    matrix[:, 4:] = c
    return matrix


def square_vertical_3layers(a,b,c):
    """Return a soil matrix of a square pattern with three vertical layers"""
    
    matrix = a*np.ones((6, 6))
    matrix[2:4, :] = b
    matrix[4:, :] = c
    return matrix
    

def circular_layers(a,b,c):
    """Return a soil matrix of a radiative pattern with three types of soils"""
    
    matrix = np.array([
        [a,a,a,a,a,a],
        [a,a,b,b,a,a],
        [a,b,c,c,b,a],
        [a,b,c,c,b,a],
        [a,a,b,b,b,a],
        [a,a,a,a,a,a]])
    return matrix
