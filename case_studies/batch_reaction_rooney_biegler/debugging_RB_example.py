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

from rooney_biegler_comparison import (
    rooney_biegler_sensitivity,
    rooney_biegler_parameter_estimation,
)


def run_greybox_optimization(ub_dec=None, lb_obj=None):
    # Decide flags based on inputs
    if ub_dec is not None:
        ub_dec_  flag = True
    if lb_obj is not None:
        lb_obj_flag = True
    
    # Gather preliminary estimate of coefficients with n experiments
    theta = rooney_biegler_parameter_estimation()

    # Create a RooneyBiegler Experiment, pass the theta estimate there
    experiment = RooneyBieglerExperimentDoE(data={'hour': 1.78, 'y': 15}, theta=theta)

    # Use a central difference, with step size 1e-3
    fd_formula = "central"
    step_size = 1e-3

    # Use the determinant objective with scaled sensitivity matrix
    objective_option = "determinant"
    scale_nominal_param_value = True

    data = [[1, 8.3], [7, 19.8], [2, 10.3], [5, 15.6], [3, 19.0], [4, 16.0]]
    FIM_prior = np.zeros((2, 2))
    # Calculate prior using existing experiments (Only the first two)
    for i in range(2):
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

    # Compute new FIM calculation for a range of time values
    objective_options = ["determinant"]
    # "trace",
    # "minimum_eigenvalue",
    # "condition_number"]

    optimal_points = {}
    optimal_objective_value = 0

    # Define our own grey box solver
    grey_box_solver = pyo.SolverFactory("cyipopt")
    # grey_box_solver.config.options["hessian_approximation"] = "limited-memory"
    grey_box_solver.config.options["linear_solver"] = "ma27"
    grey_box_solver.config.options['mu_strategy'] = "monotone"

    try_things = {
        "determinant": 1.78,  # 1.78 for optimal
        "trace": 1.32,
        "minimum_eigenvalue": 1.32,
        "condition_number": 0.88,
    }

    for objective_option in objective_options:
        experiment = RooneyBieglerExperimentDoE(
            data={'hour': try_things[objective_option], 'y': 15}
        )

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
            grey_box_solver=grey_box_solver,
            grey_box_tee=True,
            get_labeled_model_args=None,
            _Cholesky_option=True,
            _only_compute_fim_lower=True,
            ub_dec_flag=ub_dec_flag,
            ub_dec=ub_dec,
            lb_obj_flag=lb_obj_flag,
            lb_obj=lb_obj,
        )

        doe_obj_gb.run_doe()

        if objective_option == "determinant":
            doe_obj_nongb = DesignOfExperiments(
                experiment,
                fd_formula=fd_formula,
                step=step_size,
                objective_option=objective_option,
                scale_constant_value=1,
                scale_nominal_param_value=scale_nominal_param_value,
                prior_FIM=FIM_prior,
                jac_initial=None,
                fim_initial=None,
                L_diagonal_lower_bound=1e-7,
                solver=None,
                tee=True,
                get_labeled_model_args=None,
                _Cholesky_option=True,
                _only_compute_fim_lower=True,
                ub_dec_flag=ub_dec_flag,
                ub_dec=ub_dec,
                lb_obj_flag=lb_obj_flag,
                lb_obj=lb_obj,
            )
            doe_obj_nongb.run_doe()
            
            optimal_points['Grey Box'] = np.log(np.linalg.det(doe_obj_gb.results["FIM"]))
            optimal_points['Regular DoE'] = np.log(np.linalg.det(doe_obj_nongb.results["FIM"]))

    return optimal_points


if __name__ == "__main__":
    # Standard run
    vanilla_points = run_greybox_optimization()
    
    # UB beyond optimal (should converge optimally but will not)
    slight_off_points = run_greybox_optimization(ub_dec=1.85)
    
    # UB at optimal
    should_be_optimal = run_greybox_optimization(ub_dec=1.78)
    
    # LB of obj at 10.00 (below optima)
    should_converge = run_greybox_optimization(lb_obj=10.00)
    