from pyomo.common.dependencies import numpy as np, pathlib

import pyomo.environ as pyo

from pyomo.contrib.parmest import parmest

from TC_Lab_experiment import TC_Lab_experiment, extract_plot_results

from TC_Lab_data_helper import TC_Lab_data

import pandas as pd


def TC_Lab_parmest(data_files, generate_Th=False, number_of_states=2):
    """
    Function to estimate the parameters from
    TC Lab data.

    Arguments
    ---------
    data_files: str or path, paths to the data files
    generate_Th: bool, option to generate noised Th data
                 from simulation, default False
    number_of_states: int, number of states in the model
                      either 2 or 4, default 2

    """

    # Iterate through data files to build experiment
    # objects for each experiment.
    experiments = []
    tc_datas = []
    for file in data_files:
        # Read the data with pandas
        df = pd.read_csv(file)

        # Create the data object
        tc_data = TC_Lab_data(
            name="Sine Wave Test for Heater 1",
            time=df['Time'].values[:],
            T1=df['T1'].values[:],
            u1=df['Q1'].values[:],
            P1=200,
            TS1_data=None,
            T2=df['T2'].values[:],
            u2=df['Q2'].values[:],
            P2=200,
            TS2_data=None,
            Tamb=df['T1'].values[0],
        )

        tc_datas.append(tc_data)

        # Create experiment objects
        experiment = TC_Lab_experiment(
            data=tc_data, number_of_states=number_of_states, include_Th=False
        )
        experiment.get_labeled_model()

        # If generating Th with noise, add noise
        std = 0.25**2  # Assuming 0.25 degree error

        # Generate the noise
        for k, v in experiment.model.experiment_outputs.items():
            if "Th" in k.name:
                noise = np.random.normal(0, std, 1)  # Generate noise
                experiment.model.experiment_outputs[k] = (
                    k() + noise[0]
                )  # Set value from noise

        # Add experiment to list
        experiments.append(experiment)

    # Add solver options
    solver_options = {}
    solver_options["linear_solver"] = "ma57"
    
    # Perform estimation
    pest = parmest.Estimator(experiments, obj_function='SSE', tee=True, solver_options=solver_options)

    obj, theta = pest.theta_est()

    parmest_regression_results = extract_plot_results(
        tc_datas[0], pest.ef_instance.Scenario0
    )

    parmest_regression_results = extract_plot_results(
        tc_datas[1], pest.ef_instance.Scenario1
    )

    return theta
