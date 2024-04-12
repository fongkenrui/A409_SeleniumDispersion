import numpy as np 

"""Soil Geometries for two types of soil. a and b are soil types"""

#triangular matrix 

def traingle_2(a,b): 
    
    a = np.full((10, 10), 0.02)  
    b = np.tril(a)
    matrix = a+b
    
    print(matrix)


#Square matrices for two types of soil

#upper corner 
def square_2a(a,b):
    
    diffarr = a*np.ones((10, 10))
    diffarr[:5,:5] = b # Vary along x; partial_x should be constant
    print(diffarr)

#horizontal division 
def square_2b(a,b):
    
    diffarr = a*np.ones((10, 10))
    diffarr[:5,:] = b # Vary along x; partial_x should be constant
    print(diffarr)

#vertical division 
def square_2c(a,b):
    
    diffarr = a*np.ones((10, 10))
    diffarr[:,:5] = b # Vary along x; partial_x should be constant
    print(diffarr)
    
    


# circular symmetry 

def circular(a,b):
    
    arr = np.array([[a,a,a,a,a,a,a],
           [a,a,b,b,b,a,a],
           [a,b,b,b,b,b,a],
           [a,b,b,b,b,b,a],
           [a,a,b,b,b,a,a],
           [a,a,a,a,a,a,a]])
    
    print(arr)


circular(2,0)


'matrices for 3 soil types: a,b,c'

#square matrices 

def square_layers(a,b,c): 
    
    arr = np.array([[a,a,a,a,a,a],
                    [a,b,b,b,b,a],
                    [a,b,c,c,b,a],
                    [a,b,c,c,b,a],
                    [a,b,b,b,b,a],
                    [a,a,a,a,a,a]])
    print(arr)

def square_3a(a,b,c):
    
    diffarr = a*np.ones((6, 6))
    diffarr[:, 2:4] = b
    diffarr[:, 4:] = c
    
    print(diffarr)
    


def square_3b(a,b,c):
    
    diffarr = a*np.ones((6, 6))
    diffarr[2:4, :] = b
    diffarr[4:, :] = c
    
    print(diffarr)
    

def circular_layers(a,b,c):
    
        arr = np.array([[a,a,a,a,a,a],
                        [a,a,b,b,a,a],
                        [a,b,c,c,b,a],
                        [a,b,c,c,b,a],
                        [a,a,b,b,b,a],
                        [a,a,a,a,a,a]])
        print(arr)
 
circular_layers(1,2,3)