from pyomo.common.dependencies import numpy as np, pathlib

import pyomo.environ as pyo

from pyomo.contrib.parmest import parmest

from TC_Lab_experiment import TC_Lab_experiment, extract_plot_results, recover_standard_params

from TC_Lab_data_helper import TC_Lab_data

import pandas as pd

import matplotlib.pyplot as plt


# Plotting options
SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('lines', linewidth=3)


def TC_Lab_parmest(data_files, generate_Th=False, number_of_states=2, reparam=False, CpS_CpH_ratio=None, plot_results=True):
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
    reparam: bool, option to reparametrize the model

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
            data=tc_data, number_of_states=number_of_states, include_Th=False, reparam=reparam, CpS_CpH_ratio=CpS_CpH_ratio,
        )
        experiment.get_labeled_model()

        # If generating T_h with noise, add noise
        std = 0.25**2  # Assuming 0.25 degree error

        # Generate the noise
        # for k, v in experiment.model.experiment_outputs.items():
        #     if "Th" in k.name:
        #         noise = np.random.normal(0, std, 1)  # Generate noise
        #         experiment.model.experiment_outputs[k] = (
        #             k() + noise[0]
        #         )  # Set value from noise

        # Add experiment to list
        experiments.append(experiment)

    # Add solver options
    solver_options = {"linear_solver": "ma57"}

    # Perform estimation
    pest = parmest.Estimator(experiments, obj_function='SSE', tee=True, solver_options=solver_options)

    obj, theta = pest.theta_est()

    if plot_results:
        parmest_regression_results = extract_plot_results(
            tc_datas[0], pest.ef_instance.Scenario0, reparam=reparam, save_plot=True, file_name="sine-wave-experiment-tclab.png"
        )

        parmest_regression_results = extract_plot_results(
            tc_datas[1], pest.ef_instance.Scenario1, reparam=reparam, save_plot=True, file_name="step-test-experiment-tclab.png"
        )

    if reparam:
        theta = recover_standard_params(theta_vals=theta, alpha=pest.ef_instance.Scenario0.alpha, P1=pest.ef_instance.Scenario0.P1)

    return theta


if __name__ == "__main__":
    # Read in the data
    DATA_DIR = pathlib.Path(__file__).parent
    file_path = DATA_DIR / "data" / "tclab_sine_test_5min_period.csv"
    file_path_2 = DATA_DIR / "data" / "tclab_step_test.csv"

    # Set up the data
    from TC_Lab_data_helper import TC_Lab_data, helper

    df = pd.read_csv(file_path)
    df2 = pd.read_csv(file_path_2)

    # We want to use the 2-state model
    number_tclab_states = 2

    # Here, we will induce a step size of 6 seconds, as to not give too many
    # degrees of freedom for experimental design.
    skip = 30

    # Create the data object considering the new control points every 10 seconds
    tc_data = TC_Lab_data(
        name="Sine Wave Test for Heater 1",
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

    # Create the data object considering the new control points every 10 seconds
    tc_data2 = TC_Lab_data(
        name="Step Test for Heater 1",
        time=df2['Time'].values[::skip],
        T1=df2['T1'].values[::skip],
        u1=df2['Q1'].values[::skip],
        P1=200,
        TS1_data=None,
        T2=df2['T2'].values[::skip],
        u2=df2['Q2'].values[::skip],
        P2=200,
        TS2_data=None,
        Tamb=df2['T1'].values[0],
    )

    theta_values = TC_Lab_parmest([file_path, file_path_2], generate_Th=False, reparam=True)