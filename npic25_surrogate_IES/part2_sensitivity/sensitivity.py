# -*- coding: utf-8 -*-

# @author Seydou Sene
# @create date 2024-07-24 11:12:30
# @modify date 2024-07-24 12:22:11
# using code "load_data.py" by Pr. Majdi Radaideh

import numpy as np
import os.path
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd

from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


#import keras_tuner
from keras.utils import set_random_seed
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Input
from keras.callbacks import ModelCheckpoint

from SALib.sample import saltelli, finite_diff, fast_sampler, latin
from SALib.sample import morris as morris_sampler
from SALib.analyze import sobol, dgsm, fast, delta, pawn, hdmr, rsa, morris
from SALib.test_functions import Ishigami
from SALib import ProblemSpec


# To get a runtime idea
start_time = datetime.now()

#----------------------------------------------------
# Input data
#----------------------------------------------------
# input data is a 3D tensor with shape
# (1024, 46, 4)
# (samples, time_steps, features)

# There are 1024 simulations/samples, each simulation is specified with 3 input parameters + time index (total 4 features).
# The features in order are: ['time','massflow','shape_factor', 'porosity']
# Note that "time_steps" axis specifies the value of the input as a function of time for 46 time steps.
# All inputs except "massflow" are constant with time. Mass flow rate has different values for the 46 time steps. 

x=np.load('../data/inputs.npy')

# Take a look at the example below for the first input sample in the 3D tensor
# x_sample1 = pd.DataFrame(x[0,:,:], columns=['time','massflow','shape_factor','porosity'])
# print(x_sample1)
# 
# print()

#----------------------------------------------------
# Output data
#----------------------------------------------------
# output data is a 3D tensor with shape 
# (1024, 13, 6)
# (samples, sensor, time_slice)
# There are 1024 simulations/samples, each simulation yields temperature 
# measurments from 13 sensors for six specific time points (4000s, 6000s, 8000s, 10000s, 12000s, 14000s)

y=np.load('../data/outputs.npy')

# take a look at the example below for the first output sample
# y_sample1 = pd.DataFrame(y[0,:,:], columns=['t4000','t6000','t8000','t10000','t12000','t14000'],
#                          index=['sensor{}'.format(j) for j in range(1,y.shape[1]+1)])
# print(y_sample1)

# Flatten y into a 2D array with 13x6 outputs
y_flat = np.reshape(y, (np.size(y,0), -1))

# Flatten x into a 2D array where we put all static values + massflow(t) for all t, 
# T_in being the only non static variable (dropped the time feature, which is just timestep*400)
x_static = x[:, 0, 2:]
x_tins = x[:, :, 1]
x_flat = np.concatenate([x_static, x_tins], axis=1)

nb_outputs = 78 # = 13 * 6 (nb_sensors * nb_output_timesteps)
nb_features = 48 # = 2 (static) + 46 (massflow(t) for all t)
nb_epochs = 100

# Fix random seed for reproducibility
seed = 777
set_random_seed(seed)

# Splitting the data into train and test splits
X_train, X_test, Y_train, Y_test = train_test_split(
    x_flat, y_flat, test_size=0.2, random_state=seed)

# Scale the data
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
X_train_scaled = scaler_x.fit_transform(X_train)
X_test_scaled = scaler_x.transform(X_test)
Y_train_scaled = scaler_y.fit_transform(Y_train)
Y_test = Y_test

def fnn(): #based on hp_tuning results
    # Create model
    model = Sequential()
    model.add(Input(shape=(nb_features,)))
    
    model.add(Dense(704, activation= "relu"))
    model.add(Dense(64, activation= "relu"))
    model.add(Dense(128, activation= "relu"))
    model.add(Dense(416, activation= "relu"))
    model.add(Dense(864, activation= "relu"))
            
    model.add(Dense(nb_outputs, activation= "linear" ))
    
    # Compile model
    learning_rate = 0.00195
    model.compile(loss= "mean_squared_error" , optimizer= Adam(learning_rate), metrics=['mse'])
    return model

model = fnn()

# checkpointing
filepath= "weights_postprocessing_seed%d.keras" % seed
checkpoint = ModelCheckpoint(filepath, monitor= "val_mse", verbose=0, save_best_only=True, mode= "min")
callbacks_list = [checkpoint]

# Train the model
history = model.fit(X_train_scaled, Y_train_scaled, validation_split = 0.15, epochs=nb_epochs, callbacks=callbacks_list, verbose=1)

# Make predictions and return back to initial values
Y_pred_scaled = model(X_test_scaled).numpy() # not using model.predict because it is faster here (and is safe)
Y_pred = scaler_y.inverse_transform(Y_pred_scaled)

if not os.path.exists('nn_results'):
    os.makedirs('nn_results')

# Calculate and plot metrics
plt.figure()
plt.title('MAPE per output')
plt.scatter(range(nb_outputs), mean_absolute_percentage_error(Y_test, Y_pred, multioutput="raw_values"))
plt.xlabel('Output (k * time_pos + sensor_nb)')
plt.ylabel('MAPE')
plt.savefig('./nn_results/mape.png', dpi=300, bbox_inches='tight')

plt.figure()
plt.title('R2 score per output')
plt.scatter(range(nb_outputs), r2_score(Y_test, Y_pred, multioutput="raw_values"))
plt.xlabel('Output (k * time_pos + sensor_nb)')
plt.ylabel('R2 score')
plt.savefig('./nn_results/r2.png', dpi=300, bbox_inches='tight')

plt.figure()
plt.title('Loss per epoch')
plt.plot(range(nb_epochs), history.history['mse'], label='Training')
plt.plot(range(nb_epochs), history.history['val_mse'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('./nn_results/nn_loss.png', dpi=300, bbox_inches='tight')


print(np.shape(Y_test))
plt.figure()
plt.title('Predictions vs Target values, output 72')
plt.plot(range(len(Y_test[72])), Y_test[72], label='Target')
plt.plot(range(len(Y_test[72])), Y_pred[72], label='Prediction')
plt.xlabel('Observation')
plt.ylabel('Value')
plt.legend()
plt.savefig('./nn_results/pred_target_random_output.png', dpi=300, bbox_inches='tight')

print("Runtime :", datetime.now() - start_time)
print("Seed :", seed)

r2_y=r2_score(Y_test, Y_pred, multioutput="raw_values")
mape_y=mean_absolute_error(Y_test, Y_pred, multioutput="raw_values")

plt.close()


#--------------------------
#for sensitivity plotting
#--------------------------

def plot_SA_bar(SA_cluster, ylabel=None, figname=None, figtitle=None):

    # Define the first DataFrame
    df1 = SA_cluster[0][['ST', 'ST_conf']]
    df2 = SA_cluster[1][['ST', 'ST_conf']]
    df3 = SA_cluster[2][['ST', 'ST_conf']]
    df4 = SA_cluster[3][['ST', 'ST_conf']]
    df5 = SA_cluster[4][['ST', 'ST_conf']]
    df6 = SA_cluster[5][['ST', 'ST_conf']]
    
    
    # Define the bar width and spacing
    bar_width = 0.1
    spacing = 0.05
    group_spacing = 0.1
    indices1 = np.arange(len(df1.index))*0.15
    indices2 = indices1 + len(df1.index)*0.15 + group_spacing
    indices3 = indices2 + len(df1.index)*0.15 + group_spacing
    indices4 = indices3 + len(df1.index)*0.15 + group_spacing
    indices5 = indices4 + len(df1.index)*0.15 + group_spacing
    indices6 = indices5 + len(df1.index)*0.15 + group_spacing
    
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plotting ST bars for both DataFrames
    ax.bar(indices1  + spacing/2, df1['ST'], bar_width, yerr=df1['ST_conf'], label='t=4000s', capsize=5)
    ax.bar(indices2  + spacing/2, df2['ST'], bar_width, yerr=df2['ST_conf'], label='t=6000s', capsize=5)
    ax.bar(indices3  + spacing/2, df3['ST'], bar_width, yerr=df3['ST_conf'], label='t=8000s', capsize=5)
    ax.bar(indices4  + spacing/2, df4['ST'], bar_width, yerr=df4['ST_conf'], label='t=10,000s', capsize=5)
    ax.bar(indices5  + spacing/2, df5['ST'], bar_width, yerr=df5['ST_conf'], label='t=12,000s', capsize=5)
    ax.bar(indices6  + spacing/2, df6['ST'], bar_width, yerr=df6['ST_conf'], label='t=14,000s', capsize=5)
    
    # Customizing the plot
    ax.set_title(figtitle)
    ax.set_xlabel('Parameters')
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    ax.set_xticks(np.concatenate((indices1, indices2, indices3, indices4, indices5, indices6)))
    ax.set_xticklabels(list(df1.index) + list(df2.index) + list(df3.index) + list(df4.index) +  
                       list(df5.index) + list(df6.index), rotation='vertical')
    ax.legend()
    
    # Display the plot
    plt.tight_layout()
    plt.savefig(figname, dpi=300, bbox_inches='tight')
    plt.close()
    
#----------------------------------
#sobol sensitivity
#----------------------------------

def gen_sobol(variable_names,upper,lower,N):

    # Extract the number of variables from the DataFrame
    num_vars = len(variable_names)
    # Create a list of lower and upper bounds for each variable
    low_up = [[lower[i], upper[i]] for i in range(num_vars)]
    
    problem = {
        'num_vars': num_vars,
        'names': variable_names,
        'bounds': low_up
    }
    

    # Generate Sobol sequences
    param_values = pd.DataFrame(saltelli.sample(problem, N), columns=variable_names) 
    
    #print('Meredith=', param_values)
    
    return param_values

n_sob=2100
ny=y_flat.shape[1]
upper = [x[:,:,2].max(axis=0).max(), x[:,:,3].max(axis=0).max()] + list(x[:,:,1].max(axis=0)) 
lower = [x[:,:,2].min(axis=0).min(), x[:,:,3].min(axis=0).min()] + list(x[:,:,1].min(axis=0)) 

for k,item in enumerate(upper):
    if item == 0:
        upper[k] += 0.0001    #just to escape error in SAlib of having lower and upper bounds being equal
        
x_names = ['sf', 'porosity'] + ['m{}'.format(i) for i in range(x.shape[1])]
nx = len(x_names)          #number of input variables

Xsob = gen_sobol(variable_names = x_names,upper=upper,lower=lower, N=n_sob)   #can you 
Xsob_scaled= scaler_x.transform(Xsob)
Ysob = model.predict(Xsob_scaled)

# Create a list of lower and upper bounds for each variable
low_up = [[lower[i], upper[i]] for i in range(nx)]

problem = {
    'num_vars': nx,
    'names': x_names,
    'bounds': low_up,
    'groups': ['Group_{num}'.format(num=k) for k in range(1,nx+1)],
}

sensor_indices={}
first_index=0
ntimes=6

if not os.path.exists('sobol_results'):
    os.makedirs('sobol_results')

for k in range(1,14):
    sensor_indices[k]= [i for i in range(first_index,first_index+6)]
    first_index+=ntimes

for sens, sens_indices in sensor_indices.items():

    Sobol_cluster = []
    for k in range(ntimes):  # Rows
        Si = sobol.analyze(problem, Ysob[:,sens_indices[k]])
        
        Sobol_Si = pd.DataFrame([Si['S1'], Si['S1_conf'], Si['ST'], Si['ST_conf']], columns=x_names, index=['S1', 'S1_conf', 'ST', 'ST_conf']).T
        main_Sobol = Sobol_Si.sort_values(['ST'], axis='index', ascending=False).head()
        Sobol_cluster.append(main_Sobol)
        
    plot_SA_bar(Sobol_cluster, ylabel='Sobol Total Index ($S_T$)',
                        figname='./sobol_results/sobol_sensor{}_teds.png'.format(sens),
                        figtitle='Sobol Indices for Sensor {} as a function of time'.format(sens))
    
    print('Sobol sensor {} is processed'.format(sens))

#-----------------------------------------------------
# Fourier Amplitude Sensitivity Test (FAST) 

if not os.path.exists('fast_results'):
    os.makedirs('fast_results')
    
Xfast=pd.DataFrame(fast_sampler.sample(problem, 4200), columns=x_names)
Xfast_scaled= scaler_x.transform(Xfast)
Yfast = model.predict(Xfast_scaled)

for sens, sens_indices in sensor_indices.items():

    FAST_cluster = []
    for k in range(ntimes):  # Rows
        Si = fast.analyze(problem, Yfast[:,sens_indices[k]], num_resamples=500, print_to_console=False)
        FAST_Si = pd.DataFrame([Si['S1'], Si['S1_conf'], Si['ST'], Si['ST_conf']], columns=x_names, index=['S1', 'S1_conf', 'ST', 'ST_conf']).T
        main_fast = FAST_Si.sort_values(['ST'], axis='index', ascending=False).head()
        FAST_cluster.append(main_fast)
        
    plot_SA_bar(FAST_cluster, ylabel='FAST Total Index ($S_T$)',
                        figname='./fast_results/fast_sensor{}_teds.png'.format(sens),
                        figtitle='FAST Indices for Sensor {} as a function of time'.format(sens))
    
    print('FAST sensor {} is processed'.format(sens))


