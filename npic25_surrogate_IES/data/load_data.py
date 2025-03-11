# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 20:22:09 2024

@author: Majdi Radaideh
"""

import numpy as np
import pandas as pd

#----------------------------------------------------
#Input data
#------------------------------------------------
#input data is a 3D tensor with shape
# (1024, 46, 4)
# (samples, time_steps, features)

# There are 1024 simulations/samples, each simulation is specified with 3 input parameters + time index (total 4 features).
# The features in order are: ['time','massflow','shape_factor', 'porosity']
# Note that "time_steps" axis specifies the value of the input as a function of time for 46 time steps.
# All inputs except "massflow" are constant with time. Mass flow rate has different values for the 46 time steps. 

x=np.load('inputs.npy')

# Take a look at the example below for the first input sample in the 3D tensor
x_sample1 = pd.DataFrame(x[0,:,:], columns=['time','massflow','shape_factor','porosity'])
print(x_sample1)

print()

#----------------------------------------------------
# Output data
#----------------------------------------------------
#output data is a 3D tensor with shape 
# (1024, 13, 6)
# (samples, sensor, time_slice)
# There are 1024 simulations/samples, each simulation yields temperature 
# measurments from 13 sensors for six specific time points (4000s, 6000s, 8000s, 10000s, 12000s, 14000s)
y=np.load('outputs.npy')

#take a look at the example below for the first output sample
y_sample1 = pd.DataFrame(y[0,:,:], columns=['t4000','t6000','t8000','t10000','t12000','t14000'],
                         index=['sensor{}'.format(j) for j in range(1,y.shape[1]+1)])
print(y_sample1)