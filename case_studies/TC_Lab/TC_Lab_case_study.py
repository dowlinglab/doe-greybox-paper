from pyomo.common.dependencies import numpy as np, pathlib

from TC_Lab_experiment import (
    TC_Lab_experiment,
    extract_results,
    extract_plot_results,
    results_summary,
)

from TC_Lab_data_helper import TC_Lab_data, helper

from TC_Lab_parameter_estimation import TC_Lab_parmest

from pyomo.contrib.doe import DesignOfExperiments

import pyomo.environ as pyo

import matplotlib.pyplot as plt
import json
import pandas as pd
import sys

# If needing the solvers
try:
    import idaes
except:
    pass


def run_single_TC_Lab_experiment():
    # Read in the data
    DATA_DIR = pathlib.Path(__file__).parent
    # file_path = DATA_DIR / "data" / "validation_experiment_env_2_step_50_run_1.csv"
    file_path = DATA_DIR / "data" / "tclab_sine_test_5min_period.csv"
    file_path_2 = DATA_DIR / "data" / "tclab_step_test.csv"

    df = pd.read_csv(file_path)
    df2 = pd.read_csv(file_path_2)

    # We want to use the 2-state model
    number_tclab_states = 2

    # Here, we will induce a step size of 6 seconds, as to not give too many
    # degrees of freedom for experimental design.
    skip = 15

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

    # TODO: Perform parameter estimation on the fly
    theta_values = {
        'Ua': 0.0417051733576387,
        'Ub': 0.009440714239773074,
        'inv_CpH': 0.1659093525658045,
        'inv_CpS': 5.8357556063605465,
    }

    theta_values = TC_Lab_parmest([file_path, file_path_2], generate_Th=False)

    # Create initial experiment
    experiment = TC_Lab_experiment(
        data=tc_data,
        theta_initial=theta_values,
        number_of_states=number_tclab_states,
        include_Th=True,
    )

    # solver = pyo.SolverFactory("ipoptv2")

    TC_Lab_DoE = DesignOfExperiments(
        experiment=experiment,
        step=1e-2,
        scale_constant_value=1,
        scale_nominal_param_value=True,
        tee=True,
    )

    # Analyze initial FIM for prior information
    FIM = TC_Lab_DoE.compute_FIM(method='sequential')

    # Create initial experiment
    experiment2 = TC_Lab_experiment(
        data=tc_data,
        theta_initial=theta_values,
        number_of_states=number_tclab_states,
        include_Th=True,
    )

    # solver = pyo.SolverFactory("ipoptv2")

    TC_Lab_DoE2 = DesignOfExperiments(
        experiment=experiment2,
        step=1e-2,
        scale_constant_value=1,
        scale_nominal_param_value=True,
        tee=True,
    )

    # Analyze initial FIM for prior information
    FIM2 = TC_Lab_DoE.compute_FIM(method='sequential')

    results_summary(FIM2)

    # New experiment to perform experimental design
    doe_experiment = TC_Lab_experiment(
        data=tc_data,
        theta_initial=theta_values,
        number_of_states=number_tclab_states,
        include_Th=True,
    )

    # Create the design of experiments object using our experiment instance from above
    TC_Lab_DoE_D = DesignOfExperiments(
        experiment=doe_experiment,
        step=1e-2,
        scale_constant_value=1,
        scale_nominal_param_value=True,
        objective_option="determinant",  # Now we specify a type of objective, D-opt = "determinant"
        prior_FIM=FIM
        + FIM2,  # We use the prior information from the existing experiment!
        tee=True,
    )

    # Run the experimental design
    TC_Lab_DoE_D.run_doe()

    # Extract the results
    dopt_pyomo_doe_results = extract_plot_results(
        None, TC_Lab_DoE_D.model.scenario_blocks[0]
    )

    # Print results summary
    results_summary(TC_Lab_DoE_D.results['FIM'])

    ###################
    # End optimal DoE


if __name__ == "__main__":
    run_single_TC_Lab_experiment()
