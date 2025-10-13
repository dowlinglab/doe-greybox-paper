from pyomo.common.dependencies import numpy as np, pathlib

from pyomo.contrib.doe import DesignOfExperiments

import pyomo.environ as pyo

from TC_Lab_parameter_estimation import TC_Lab_parmest

import matplotlib.pyplot as plt
import pandas as pd
import json
import sys
import os

import tclab
from tclab import TCLab, clock, Historian, Plotter



def test_TC_Lab_experiment():
    # Visually test the device
    lab = TCLab()
    lab.LED(50)
    lab.close()
    
    # Run diagnostics to ensure
    # device is running properly
    print("Version = ", tclab.__version__)
    tclab.diagnose()
    

def run_TC_Lab_experiment(Q1=None, Q2=None, time_points=None, t_jump=0.2):
    # Make sure inputs are the right lengths
    if Q1 is None and Q2 is None:
        raise ValueError("Either Q1 or Q1 and Q2 must be defined to run a TC Lab experiment in this format. Please provide a list of power percentage values from 0 to 100 in the same length as the time_periods input.")
    
    # Open an instance of TC Lab
    with TCLab() as lab:
        # Set up historian to track values
        h = Historian(lab.sources)
        
        # Set heaters to inital values
        lab.Q1(Q1[0])
        if Q2 is not None:
            lab.Q2(Q2[0])
        print(f"Set Heater 1 to {lab.Q1():.1f} %")
        print(f"Set Heater 2 to {lab.Q2():.1f} %")

        # Hold an index for time_points
        time_ind = 0
        
        # Adjust heater setting to the control setting
        # Increment by 0.2 seconds to avoid missing any
        # time points.
        for t in clock(time_points[-1], 0.2):
            print(f"{t:5.1f} sec:  T1 = {lab.T1:4.1f} C   T2 = {lab.T2:4.1f} C")
            
            if t > time_points[time_ind]:
                # Set heater temps
                lab.Q1(Q1[0])
                if Q2 is not None:
                    lab.Q2(Q2[0])
                
                # Increment time index
                time_ind += 1
                h.update(t)
    
    # Write the historian to a csv and
    # return a pandas dataframe with
    # the data.
    h.to_csv("TCLab_temp_data.csv")
    
    tclab_df = pd.read_csv("TCLab_temp_data.csv")
    
    # Remove temp_file
    os.remove("TCLab_temp_data.csv")
    
    return tclab_df


def automated_TC_Lab_model_identification(initial_datasets=None, n=2):
    # Add initial data sets if specified
    if initial_datasets:
        data_sets = initial_datasets
    else:
        data_sets = []
    
    initial_data_len = len(data_sets)
    
    # TODO: Use standard naming
    
    for i in range(n):
        if i != 0:
            # Run the TC Lab
            
            # Save the data somewhere
            
            # Read the data for parameter estimation
            DATA_DIR = pathlib.Path(__file__).parent
            new_data_file_path = DATA_DIR / "experimental_values_1.json"
            
            data_sets.append(new_data_file_path)
        
        # Perform parameter estimation with all existing data
        theta_current = TC_Lab_parameter_estimation(data_files=data_sets, generate_Th=False)
        
        # Gather FIM priors to perform estimation (use constant
        # scaling for number of experiments?)
        prior_FIM = np.zeros((4, 4))
        
        # Compute current constant scaling
        curr_const_scaling = 1 + i + initial_data_len
        
        for prev_file in data_files:
            # Read data
            df = pd.read_csv(prev_file)
            
            # Make data object
            tc_data = TC_Lab_data(
                name="Specified Profile for Heater 1",
                time=df['Time'].values[::skip],
                T1=df['T1'].values[::skip],
                u1=df['Q1'].values[::skip],
                P1=200,
                TS1_data=None,
                T2=df['T2'].values[::skip],
                u2=df['Q2'].values[::skip],
                P2=200,
                TS2_data=None,
                Tamb=df['T1'].values[0],
            )
            
            # Make experiment object
            experiment = TC_Lab_experiment(
                data=tc_data,
                theta_initial=theta_current,
                number_of_states=number_tclab_states,
                include_Th=True,
            )
            
            # Make DoE object
            TC_Lab_compute_FIM = DesignOfExperiments(
                experiment=experiment,
                step=1e-2,
                scale_constant_value=curr_const_scaling,
                scale_nominal_param_value=True,
                tee=True,
            )
            
            # Compute FIM
            FIM = TC_Lab_compute_FIM.compute_FIM(method='sequential')
            
            # Add to prior
            prior_FIM += FIM
        
        # Make new experiment object
        experiment_DoE = TC_Lab_experiment(
                data=tc_data,
                theta_initial=theta_current,
                number_of_states=number_tclab_states,
                include_Th=True,
        )
        
        # Make new DoE object for design
        TC_Lab_DoE = DesignOfExperiments(
            experiment=experiment_DoE,
            step=1e-2,
            scale_constant_value=curr_const_scaling,
            scale_nominal_param_value=True,
            tee=True,
            prior_FIM=prior_FIM,
            objective_option="determinant",
        )
        
        # Perform DoE
        TC_Lab_DoE.run_doe()
        
        FIM_current = TC_Lab_DoE.get_FIM()
        
        # Rescale FIM???
        
        # Grab result from optimal DoE
        
        # Set control profile from DoE result
        
        # Allow system to cool down before running again
    
    return theta_current, FIM_current


if __name__ == "__main__":
    # Specifiy initial data
    data_sets = []
    
    # Run automated DoE
    # theta, FIM = automated_TC_Lab_model_identification(initial_datasets=data_sets, n=2)
    
    Q1 = [100, 100, 100, 50, 50, 50, 0, 0, 0, 50, 50, 50,]
    time_points = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, ]
    data = run_TC_Lab_experiment(Q1=Q1, time_points=time_points)
    
    print(data)
    
    # Print and plot the theta values and posteriors?
    
