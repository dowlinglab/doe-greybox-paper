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

# If needing the solvers
try:
    import idaes
except:
    pass


def run_single_TC_Lab_experiment(
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

    if CpS_to_CpH_ratio is not None:
        # Add dummy value for the data object to hold
        theta_values["inv_CpS"] = 1 / 0.17

    # Create initial experiment
    experiment = TC_Lab_experiment(
        data=tc_data2,
        theta_initial=theta_values,
        number_of_states=number_tclab_states,
        include_Th=include_Th,
        reparam=reparam,
        CpS_CpH_ratio=CpS_to_CpH_ratio,
    )

    TC_Lab_DoE_exp1 = DesignOfExperiments(
        experiment=experiment,
        step=step_size,
        scale_constant_value=constant_scaling,
        scale_nominal_param_value=scale_nominal_param_value,
        tee=True,
    )

    # Analyze initial FIM for prior information (Step test)
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

    # Analyze initial FIM for prior information (Sine Wave Test)
    FIM2 = TC_Lab_DoE_exp2.compute_FIM(method='sequential')

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
        prior_FIM=FIM2,  # We use the prior information from the existing experiment!
        tee=True,
        grey_box_tee=True,
    )

    # Run the experimental design
    TC_Lab_DoE.run_doe()

    # Extract the results
    dopt_pyomo_doe_results = extract_plot_results(
        None,
        TC_Lab_DoE.model.scenario_blocks[0],
        save_plot=save_plot,
        file_name=file_name,
        reparam=reparam,
    )

    # Plot the pairwise uncertainties before and after
    if objective_option == "determinant":
        theta_values_normal_param = {
            "Ua": 0.0408,
            "Ub": 0.0303,
            "inv_CpH": 1 / 5.47,
            "inv_CpS": 1 / 0.588,
        }

        theta_values_normal_param = theta_values

        # Gather new design FIM
        new_FIM = TC_Lab_DoE.results["FIM"]

        # Optionally print some information
        verbose = False
        if verbose:
            print("\n\n\n\n\n\n\n\n\n\n")
            print(FIM2)
            print("\n\n\n\n\n\n\n\n\n\n")

            print("\n\n\n\n\n\n\n\n\n\n")
            print(FIM + FIM2)
            print("\n\n\n\n\n\n\n\n\n\n")

            print("\n\n\n\n\n\n\n\n\n\n")
            print(new_FIM)
            print("\n\n\n\n\n\n\n\n\n\n")

        theta_labels = [
            r"$\boldsymbol{U_a}$",
            r"$\boldsymbol{U_b}$",
            r"$\boldsymbol{Cp_H^{-1}}$",
        ]  # r"$\boldsymbol{Cp_S^{-1}}$"]

        if "inv_CpS" in theta_values_normal_param.keys():
            theta_values_normal_param.pop("inv_CpS")

        # Plot the Pairwise Uncertainties
        plot_pairwise_uncertainties(
            [FIM2, new_FIM],
            theta_labels,
            list(theta_values_normal_param.values),
            n_std=1,
            add_legend=True,
        )
        plt.savefig("uncertainty_reduction_TC_Lab.png")
        plt.show()

        plot_pairwise_uncertainties(
            [FIM2], theta_labels, list(theta_values_normal_param.values), n_std=1
        )
        plt.savefig("only_prior_uncertainty_comparison_TC_Lab.png")
        plt.show()

        plot_pairwise_uncertainties(
            [FIM2, new_FIM, FIM + FIM2],
            theta_labels,
            list(theta_values_normal_param.values),
            n_std=1,
            add_legend=True,
        )
        plt.savefig("uncertainty_reduction_TC_Lab_comparison_with_normal_exp.png")
        plt.show()

    return TC_Lab_DoE.results['FIM'], FIM2, FIM + FIM2

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
    # objective_options = ["determinant", ]
    FIM_results_dict = {i: 0 for i in objective_options}

    for objective_option in objective_options:
        # Run the condition
        temp_FIM, prior2, prior12 = run_single_TC_Lab_experiment(
            reparam=False,
            objective_option=objective_option,
            save_plot=True,
            file_name=default_file_name.format(objective_option),
        )

        # Save the results
        FIM_results_dict[objective_option] = temp_FIM

    # Save the FIM data to a json file
    with open('FIM_results_optimality_conditions.json', 'w') as f:
        json.dump(FIM_results_dict, f)

    overall_df = pd.DataFrame(columns=objective_options, index=objective_options)

    # Labels for correlation matrices
    theta_labels = [
        r"$\boldsymbol{U_a}$",
        r"$\boldsymbol{U_b}$",
        r"$\boldsymbol{Cp_H^{-1}}$",
    ]

    # Report data and save optimality conditions to a file
    for objective_option in objective_options:
        results_summary(FIM_results_dict[objective_option])

        result = FIM_results_dict[objective_option]

        eigenvalues, eigenvectors = np.linalg.eig(result)
        min_eig = min(eigenvalues)

        A_opt = np.log10(np.trace(np.linalg.inv(result)))
        D_opt = np.log10(np.linalg.det(result))
        E_opt = np.log10(min_eig)
        ME_opt = np.log10(np.linalg.cond(result))

        overall_df.loc[[objective_option], ["trace"]] = A_opt
        overall_df.loc[[objective_option], ["determinant"]] = D_opt
        overall_df.loc[[objective_option], ["minimum_eigenvalue"]] = E_opt
        overall_df.loc[[objective_option], ["condition_number"]] = ME_opt

        # Plot correlation matrix for each experiment
        plot_correlation_matrix(result, theta_labels)
        plt.savefig(
            "correlation_matrix_after_optimal_experiment_{}.png".format(
                objective_option
            )
        )
        plt.clf()
        plt.close()

    overall_df.to_csv("TC_Lab_case_study_optimality_conditions_summary_results.csv")
    print(overall_df)

    # Generate Correlation Matrices for priors
    plt.clf()
    plt.close()

    # Plotting first the prior
    plot_correlation_matrix(prior2, theta_labels)
    plt.savefig("correlation_matrix_only_step_test.png")
    plt.clf()
    plt.close()

    # Then plot the model-free experiment
    plot_correlation_matrix(prior12, theta_labels)
    plt.savefig("correlation_matrix_step_and_sine_test.png")
    plt.clf()
    plt.close()
