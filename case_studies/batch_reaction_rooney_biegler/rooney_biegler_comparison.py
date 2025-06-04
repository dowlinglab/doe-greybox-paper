#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
"""
Rooney Biegler model, based on Rooney, W. C. and Biegler, L. T. (2001). Design for
model parameter uncertainty using nonlinear confidence regions. AIChE Journal,
47(8), 1794-1804.
"""
from pyomo.common.dependencies import numpy as np, pathlib

from pyomo.contrib.doe.examples.rooney_biegler_experiment import (
    RooneyBieglerExperimentDoE,
)
from pyomo.contrib.doe import DesignOfExperiments

import pyomo.environ as pyo

import matplotlib.pyplot as plt
import json
import sys


def rooney_biegler_sensitivity():
    # Create a RooneyBiegler Experiment
    experiment = RooneyBieglerExperimentDoE(data={'hour': 10, 'y': 22})

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
        if i > int(sys.argv[1]):
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

    # Run FIM calculation for a range of time values
    time_vals = np.linspace(0, 10, 101)
    det_vals = []
    min_eig_vals = []
    max_eig_vals = []
    cond_vals = []
    A_opt_vals = []
    trace_FIM_vals = []
    y_vals = []
    for t in time_vals:
        experiment = RooneyBieglerExperimentDoE(data={'hour': t, 'y': 20})

        doe_obj = DesignOfExperiments(
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
            tee=False,
            get_labeled_model_args=None,
            _Cholesky_option=True,
            _only_compute_fim_lower=True,
        )

        FIM = doe_obj.compute_FIM(method='sequential')
        eig, _ = np.linalg.eig(FIM)
        min_eig = np.min(eig)

        A_opt_vals.append(np.trace(np.linalg.inv(FIM)))
        trace_FIM_vals.append(np.trace(FIM))
        min_eig_vals.append(min_eig)
        max_eig_vals.append(np.max(eig))
        det_vals.append(np.log(np.linalg.det(FIM)))
        cond_vals.append(np.linalg.cond(FIM))

    fig, ax = plt.subplots(2, 3, figsize=(9, 8))

    i_tot = 0

    def get_row_col(i_val, fig_row, fig_col):
        return i_val // fig_col, i_val % fig_col

    j, i = get_row_col(i_tot, 2, 3)
    ax[j, i].plot(time_vals, det_vals)
    ax[j, i].set_title("Log Determinant")
    ax[j, i].set_xlabel("Sample Time (days)")
    ax[j, i].grid()
    i_tot += 1

    j, i = get_row_col(i_tot, 2, 3)
    ax[j, i].plot(time_vals, min_eig_vals)
    ax[j, i].set_title("Minimum eigenvalue")
    ax[j, i].set_xlabel("Sample Time (days)")
    ax[j, i].grid()
    i_tot += 1

    j, i = get_row_col(i_tot, 2, 3)
    ax[j, i].plot(time_vals, max_eig_vals)
    ax[j, i].set_title("Maximum eigenvalue")
    ax[j, i].set_xlabel("Sample Time (days)")
    ax[j, i].grid()
    i_tot += 1

    j, i = get_row_col(i_tot, 2, 3)
    ax[j, i].plot(time_vals, cond_vals)
    ax[j, i].set_title("Condition Number")
    ax[j, i].set_xlabel("Sample Time (days)")
    ax[j, i].grid()
    i_tot += 1

    j, i = get_row_col(i_tot, 2, 3)
    ax[j, i].plot(time_vals, A_opt_vals)
    ax[j, i].set_title("A-optimality")
    ax[j, i].set_xlabel("Sample Time (days)")
    ax[j, i].grid()
    i_tot += 1

    j, i = get_row_col(i_tot, 2, 3)
    ax[j, i].plot(time_vals, trace_FIM_vals)
    ax[j, i].set_title("Trace of FIM")
    ax[j, i].set_xlabel("Sample Time (days)")
    ax[j, i].grid()
    i_tot += 1

    plt.tight_layout()
    # plt.show()
    plt.savefig("Rooney_Biegler_Comparison.png", format="png", dpi=450)


if __name__ == "__main__":
    # run_rooney_biegler_doe()
    rooney_biegler_sensitivity()
