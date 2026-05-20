from pyomo.common.dependencies import numpy as np, pathlib

from TC_Lab_experiment import (
    TC_Lab_experiment,
    extract_results,
    extract_plot_results,
    results_summary,
)

from TC_Lab_data_helper import (
    TC_Lab_data,
    helper,
    plot_pairwise_uncertainties,
    plot_correlation_matrix,
)

from TC_Lab_parameter_estimation import TC_Lab_parmest

from pyomo.contrib.doe import DesignOfExperiments

import pyomo.environ as pyo

import matplotlib.pyplot as plt
import json
import pandas as pd
import sys

import copy

# If needing the solvers
try:
    import idaes
except:
    pass


def run_sense_TC_Lab_experiment(
    include_Th=False,
    reparam=False,
    objective_option="determinant",
    save_plot=False,
    file_name=None,
):
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
    skip = 30

    # Constant scaling
    constant_scaling = 1e0

    # Nominal Param Scaling
    scale_nominal_param_value = True

    # step size for FD derivatives
    step_size = 1e-3

    # CpS:CpH ratio
    CpS_to_CpH_ratio = 1 / 8

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

    theta_values = TC_Lab_parmest(
        [file_path],
        generate_Th=False,
        reparam=reparam,
        CpS_CpH_ratio=CpS_to_CpH_ratio,
        plot_results=False,
    )

    theta_values["inv_CpS"] = 1 / 0.17

    # Grab unscaled FIM
    experiment = TC_Lab_experiment(
        data=tc_data,
        theta_initial=theta_values,
        number_of_states=number_tclab_states,
        include_Th=include_Th,
        reparam=reparam,
        CpS_CpH_ratio=CpS_to_CpH_ratio,
    )

    TC_Lab_DoE_exp1 = DesignOfExperiments(
        experiment=experiment,
        step=step_size,
        scale_constant_value=1,
        scale_nominal_param_value=False,
        tee=True,
    )

    # Analyze initial FIM for prior information
    FIM = TC_Lab_DoE_exp1.compute_FIM(method='sequential')

    # Create initial experiment
    experiment2 = TC_Lab_experiment(
        data=tc_data,
        theta_initial=theta_values,
        number_of_states=number_tclab_states,
        include_Th=include_Th,
        reparam=reparam,
        CpS_CpH_ratio=CpS_to_CpH_ratio,
    )

    TC_Lab_DoE_exp2 = DesignOfExperiments(
        experiment=experiment2,
        step=step_size,
        scale_constant_value=constant_scaling,
        scale_nominal_param_value=scale_nominal_param_value,
        tee=True,
    )

    FIM_sc = TC_Lab_DoE_exp2.compute_FIM(method='sequential')

    rng = np.random.default_rng()

    theta_vals = [0] * 50
    FIM_vals = [0] * 50
    optimal_profiles = [0] * 50

    mean = theta_values.values[0:3]
    cov = np.linalg.pinv(FIM)
    samples = rng.multivariate_normal(mean, cov, 50)

    for i in range(50):
        # Sample the uncertainty distribution
        sample = samples[i]

        theta_values["Ua"] = sample[0]
        theta_values["Ub"] = max(sample[1], 0.005)
        theta_values["inv_CpH"] = sample[2]
        theta_values["inv_CpS"] = 1 / 0.17

        # New experiment to perform experimental design
        doe_experiment = TC_Lab_experiment(
            data=tc_data,
            theta_initial=theta_values,
            number_of_states=number_tclab_states,
            include_Th=include_Th,
            reparam=reparam,
            CpS_CpH_ratio=CpS_to_CpH_ratio,
        )

        # Create the design of experiments object using our experiment instance from above
        TC_Lab_DoE = DesignOfExperiments(
            experiment=doe_experiment,
            step=step_size,
            use_grey_box_objective=True,  # Comment out if normal
            scale_constant_value=constant_scaling,
            scale_nominal_param_value=scale_nominal_param_value,
            objective_option=objective_option,  # Now we specify a type of objective, D-opt = "determinant"
            prior_FIM=FIM_sc,  # We use the prior information from the existing experiment!
            tee=True,
            grey_box_tee=True,
        )

        # Run the experimental design
        TC_Lab_DoE.run_doe()

        theta_values_normal_param = theta_values

        # Store the results
        theta_vals[i] = copy.deepcopy(theta_values)
        FIM_vals[i] = TC_Lab_DoE.results["FIM"]
        optimal_profiles[i] = TC_Lab_DoE.results["Experiment Design"]

    print(samples)
    print(theta_vals)

    return theta_vals, FIM_vals, optimal_profiles

    ###################
    # End optimal DoE


if __name__ == "__main__":
    default_file_name = "optimal_profile_using_{}.png"
    objective_options = [
        "determinant",
        "trace",
        "minimum_eigenvalue",
        "condition_number",
    ]

    theta_vals, FIM_sens, opt_profiles = run_sense_TC_Lab_experiment(
        reparam=False,
        objective_option="minimum_eigenvalue",
        save_plot=True,
        file_name=default_file_name.format("minimum_eigenvalue"),
    )

    print("\n\n\nTheta values for each run\n\n\n")
    for i in range(50):
        print(theta_vals[i])

    print("\n\n\nFIMs for each run\n\n\n")
    for i in range(50):
        print(FIM_sens[i])

    print("\n\n\nOptimal profiles for each run\n\n\n")
    for i in range(50):
        print("[", end="")
        for j in range(len(opt_profiles[i])):
            print(f"{opt_profiles[i][j]:.3f}", end=", ")
        print("]")

    # Computing some stats
    opt_prof_means = []
    opt_prof_std_err = []
    for i in range(31):
        curr_vals = []
        for j in range(50):
            curr_vals.append(opt_profiles[j][i])
        opt_prof_means.append(np.mean(np.asarray(curr_vals)))
        opt_prof_std_err.append(np.std(np.asarray(curr_vals)))
        print(opt_prof_means[i], opt_prof_std_err[i])

    opt_prof_E_nominal = [
        np.float64(50.00000049499998),
        np.float64(99.99995076623004),
        np.float64(99.99996954750448),
        np.float64(99.9999731385892),
        np.float64(99.99997183664357),
        np.float64(99.99996408451622),
        np.float64(99.99992067281262),
        np.float64(9.925020423261664e-05),
        np.float64(4.562362636961466e-05),
        np.float64(4.4645508409188064e-05),
        np.float64(8.644313877356185e-05),
        np.float64(99.99989740359239),
        np.float64(99.99995631818321),
        np.float64(99.99996319715403),
        np.float64(99.99995706939951),
        np.float64(99.99990410493784),
        np.float64(9.357468944186229e-05),
        np.float64(4.575015084613644e-05),
        np.float64(4.394098514398443e-05),
        np.float64(6.902534876320119e-05),
        np.float64(95.46980485936068),
        np.float64(99.9968625249621),
        np.float64(6.748994347939746e-05),
        np.float64(3.949317788224626e-05),
        np.float64(3.3527631748510024e-05),
        np.float64(3.411864210364031e-05),
        np.float64(4.2830527592526946e-05),
        np.float64(9.96447377038543e-05),
        np.float64(99.99990561042165),
        np.float64(99.99994868349131),
        np.float64(99.99993181171732),
    ]

    plt.figure(figsize=(12, 5))
    plt.errorbar(
        np.linspace(0, 900, 31),
        opt_prof_means,
        color="gold",
        yerr=opt_prof_std_err,
        fmt='-o',
        capsize=3,
        label='Average Control Profile',
    )
    plt.plot(
        np.linspace(0, 900, 31),
        opt_prof_E_nominal,
        ls="--",
        color="purple",
        label='E-optimal Profile',
        zorder=3,
    )
    plt.legend(bbox_to_anchor=(1.05, 1.05))
    plt.xlabel("Time (s)")
    plt.ylabel("Heater Power (%)")
    plt.tight_layout()
    plt.savefig("E-opt_sensitivity_to_model_unc.png", format="png", dpi=750)
    plt.show()
