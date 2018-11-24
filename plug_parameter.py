#plug_parameters.py

import numpy as np

def standard_velocities(rock_type, filename):
    
    data = np.loadtxt(filename,dtype=bytes,delimiter=';',skiprows=1).astype(str)
    rocktypes = data[:,0].tolist()
    
    if rock_type not in rocktypes:
        return None
    
    velocities = np.vectorize(lambda x: x.replace(",","."))(data[:,1:]).astype(float)
    index = rocktypes.index(rock_type)

    return velocities[index]

def density(length, diameter, weight):

    volume = np.pi*(diameter/2)**2*length

    return weight/volume


def velocity(time, length):
    
    return length/time

