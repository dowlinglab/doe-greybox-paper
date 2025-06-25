from pyomo.common.dependencies import numpy as np, pathlib

from pyomo.contrib.doe.examples.rooney_biegler_experiment import (
    RooneyBieglerExperimentDoE,
)
from pyomo.contrib.doe import DesignOfExperiments

import pyomo.contrib.parmest.parmest as parmest

import pyomo.environ as pyo

import matplotlib.pyplot as plt
import json
import sys

from rooney_biegler_comparison import rooney_biegler_sensitivity, rooney_biegler_parameter_estimation


def run_greybox_optimization():
    # Gather preliminary estimate of coefficients with n experiments
    theta = rooney_biegler_parameter_estimation()

    # Create a RooneyBiegler Experiment, pass the theta estimate there
    experiment = RooneyBieglerExperimentDoE(data={'hour': 10, 'y': 22}, theta=theta)

    # Use a central difference, with step size 1e-3
    fd_formula = "central"
    step_size = 1e-3

    # Use the determinant objective with scaled sensitivity matrix
    objective_option = "determinant"
    scale_nominal_param_value = True

    data = [[1, 8.3], [7, 19.8], [2, 10.3], [5, 15.6], [3, 19.0], [4, 16.0]]
    FIM_prior = np.zeros((2, 2))
    # Calculate prior using existing experiments
    for i in range(len(data)):
        if i >= int(sys.argv[1]):
            break
        prev_experiment = RooneyBieglerExperimentDoE(
            data={'hour': data[i][0], 'y': data[i][1]}
        )
        doe_obj = DesignOfExperiments(
            prev_experiment,
            fd_formula=fd_formula,
            step=step_size,
            objective_option=objective_option,
            scale_nominal_param_value=scale_nominal_param_value,
            prior_FIM=None,
            tee=False,
        )

        FIM_prior += doe_obj.compute_FIM(method='sequential')

    if sys.argv[1] == 0:
        FIM_prior[0][0] += 1e-6
        FIM_prior[1][1] += 1e-6

    # Compute new FIM calculation for a range of time values
    experiment = RooneyBieglerExperimentDoE(data={'hour': 5, 'y': 15})

    objective_options = ["determinant",
                         "trace",
                         "minimum_eigenvalue",
                         "condition_number"]

    optimal_points = []
    optimal_objective_value = 0

    for objective_option in objective_options:
        doe_obj_gb = DesignOfExperiments(
            experiment,
            fd_formula=fd_formula,
            step=step_size,
            objective_option=objective_option,
            use_grey_box_objective=True,
            scale_constant_value=1,
            scale_nominal_param_value=scale_nominal_param_value,
            prior_FIM=FIM_prior,
            jac_initial=None,
            fim_initial=None,
            L_diagonal_lower_bound=1e-7,
            solver=None,
            tee=False,
            get_labeled_model_args=None,
            _Cholesky_option=True,
            _only_compute_fim_lower=True,
        )

        doe_obj_gb.run_doe()

        if objective_option == "determinant":
            optimal_objective_value = np.log(np.linalg.det(doe_obj_gb.results["FIM"]))
        elif objective_option == "trace":
            optimal_objective_value = np.trace(np.linalg.inv(doe_obj_gb.results["FIM"]))
        elif objective_option == "minimum_eigenvalue":
            eig, _ = np.linalg.eig(doe_obj_gb.results["FIM"])
            min_eig = np.min(eig)
            optimal_objective_value = min_eig
        elif objective_option == "condition_number":
            optimal_objective_value = np.linalg.cond(doe_obj_gb.results["FIM"])
        optimal_points.append([float(doe_obj_gb.results["Experiment Design"][0]), float(optimal_objective_value)])

    ax = rooney_biegler_sensitivity()

    ax[0, 0].plot(optimal_points[0][0], optimal_points[0][1], marker='*', color='gold', ms=20)
    ax[0, 1].plot(optimal_points[2][0], optimal_points[2][1], marker='*', color='gold', ms=20)
    ax[1, 0].plot(optimal_points[3][0], optimal_points[3][1], marker='*', color='gold', ms=20)
    ax[1, 1].plot(optimal_points[1][0], optimal_points[1][1], marker='*', color='gold', ms=20)
    plt.show()


if __name__ == "__main__":
    run_greybox_optimization()