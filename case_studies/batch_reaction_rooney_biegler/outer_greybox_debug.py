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

from pyomo.common.dependencies import (
    numpy as np,
    numpy_available,
    scipy,
    scipy_available,
)

from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxBlock

from enum import Enum
import itertools
import logging

if scipy_available and numpy_available:
    from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxModel

import pyomo.environ as pyo


class OuterGreyBox(
    ExternalGreyBoxModel if (scipy_available and numpy_available) else object
):
    def __init__(self, gb=True):
        """
        Grey box model for metrics on the FIM. This methodology reduces
        numerical complexity for the computation of FIM metrics related
        to eigenvalue decomposition.

        Parameters
        ----------
        doe_object:
           Design of Experiments object that contains a built model
           (with sensitivity matrix, Q, and fisher information matrix, FIM).
           The external grey box model will utilize elements of the
           `doe_object` model to build the FIM metric with consistent naming.
        obj_option:
           String representation of the objective option. Current available
           options are: ``determinant`` (D-optimality), ``trace`` (A-optimality),
           ``minimum_eigenvalue`` (E-optimality), ``condition_number``
           (modified E-optimality).
           default: ``determinant``
        logger_level:
           logging level to be specified if different from doe_object's logging level.
           default: None, or equivalently, use the logging level of doe_object.
                    NOTE: Use logging.DEBUG for all messages.
        """
        self.gb = gb

        # Grab parameter list from the doe_object model
        self._n_params = 2

        self.sense = -1

    def _get_FIM(self, t=None):
        # Grabs the current FIM subject
        # to the input values.
        # This function currently assumes
        # that we use a lower triangular
        # FIM.
        if t is None:
            t = self._input_values[0]

        if self.gb:
            doe_object = run_outer_greybox_gb(t=t)
        else:
            doe_object = run_outer_greybox_normal(t=t)
        current_FIM = np.asarray(doe_object.get_FIM())
        return current_FIM

    def input_names(self):
        # Cartesian product gives us matrix indices flattened in row-first format
        # Can use itertools.combinations(self._param_names, 2) with added
        # diagonal elements, or do double for loops if we switch to upper triangular
        return ["hour"]

    def has_objective(self):
        return True

    def equality_constraint_names(self):
        # TODO: Are there any objectives that will have constraints?
        return []

    def evaluate_objective(self):
        # Evaluates the objective value for the specified
        # ObjectiveLib type.
        current_FIM = self._get_FIM()

        M = np.asarray(current_FIM, dtype=np.float64).reshape(
            self._n_params, self._n_params
        )

        (sign, logdet) = np.linalg.slogdet(M)
        obj_value = logdet

        return obj_value * self.sense

    def set_input_values(self, input_values):
        # Set initial values to be flattened initial FIM (aligns with input names)
        self._input_values = np.array([1.0])
        np.copyto(self._input_values, input_values)

    def evaluate_equality_constraints(self):
        # TODO: are there any objectives that will have constraints?
        return None

    def finalize_block_construction(self, pyomo_block):
        # Set bounds on the inputs/outputs
        # Set initial values of the inputs/outputs
        # This will depend on the objective used
        pass

    def evaluate_jacobian_equality_constraints(self):
        # TODO: Do any objectives require constraints?

        # Returns coo_matrix of the correct shape
        return None

    def evaluate_grad_objective(self):
        # Compute the jacobian of the objective function with
        # respect to the fisher information matrix. Then return
        # a coo_matrix that aligns with what IPOPT will expect.
        current_FIM = self._get_FIM(t=self._input_values[0])

        M = np.asarray(current_FIM, dtype=np.float64).reshape(
            self._n_params, self._n_params
        )

        (sign, logdet) = np.linalg.slogdet(M)
        obj_value = logdet

        perturbed_FIM = self._get_FIM(t=self._input_values[0] * 1.001)

        M_hat = np.asarray(perturbed_FIM, dtype=np.float64).reshape(
            self._n_params, self._n_params
        )

        (sign, logdet_hat) = np.linalg.slogdet(M_hat)
        obj_value_hat = logdet_hat

        # Evaluate FD
        jac_M = (obj_value_hat - obj_value) / 1e-3

        return np.asarray([jac_M]) * self.sense


def build_outer_greybox_model(gb=True):
    # Builds a simple greybox model with
    # one input (sample time) and an
    # objective function.
    model = pyo.ConcreteModel()
    model.t = pyo.Var(initialize=1.0)

    model.obj_cons = pyo.Block()

    # Create FIM External Grey Box object
    grey_box_model = OuterGreyBox(gb=gb)

    # Attach External Grey Box Model
    # to the model as an External
    # Grey Box Block
    model.obj_cons.egb_block = ExternalGreyBoxBlock(external_model=grey_box_model)

    # Add the FIM and External Grey
    # Box inputs constraints
    model.obj_cons.equality_con = pyo.Constraint(
        expr=model.t == model.obj_cons.egb_block.inputs["hour"]
    )

    return model


def run_outer_greybox_gb(t=1.0):
    # Gather preliminary estimate of coefficients with n experiments
    theta = rooney_biegler_parameter_estimation()

    # Create a RooneyBiegler Experiment, pass the theta estimate there
    experiment = RooneyBieglerExperimentDoE(data={'hour': t, 'y': 10}, theta=theta)

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
        if i >= int(2):
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

    # Compute new FIM calculation for a range of time values
    objective_options = ["determinant"]

    optimal_points = []
    optimal_objective_value = 0

    # Define our own grey box solver
    grey_box_solver = pyo.SolverFactory("cyipopt")
    grey_box_solver.config.options["hessian_approximation"] = "limited-memory"
    grey_box_solver.config.options["linear_solver"] = "ma27"
    grey_box_solver.config.options['mu_strategy'] = "monotone"

    experiment = RooneyBieglerExperimentDoE(data={'hour': t, 'y': 10})

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
        grey_box_tee=False,
        get_labeled_model_args=None,
        _Cholesky_option=True,
        _only_compute_fim_lower=True,
    )

    doe_obj_gb.run_doe()

    return doe_obj_gb


def run_outer_greybox_normal(t=1.0):
    # Gather preliminary estimate of coefficients with n experiments
    theta = rooney_biegler_parameter_estimation()

    # Create a RooneyBiegler Experiment, pass the theta estimate there
    experiment = RooneyBieglerExperimentDoE(data={'hour': t, 'y': 10}, theta=theta)

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
        if i >= int(2):
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

    experiment = RooneyBieglerExperimentDoE(data={'hour': t, 'y': 10})

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
        tee=False,
        get_labeled_model_args=None,
        _Cholesky_option=True,
        _only_compute_fim_lower=True,
    )

    doe_obj_nongb.run_doe()

    return doe_obj_nongb


if __name__ == "__main__":
    grey_box_solver = pyo.SolverFactory("cyipopt")
    grey_box_solver.config.options["hessian_approximation"] = "limited-memory"
    grey_box_solver.config.options["linear_solver"] = "ma27"
    grey_box_solver.config.options['mu_strategy'] = "monotone"

    gb_model = build_outer_greybox_model(gb=True)
    normal_model = build_outer_greybox_model(gb=False)

    grey_box_solver.solve(gb_model, tee=True)
    grey_box_solver.solve(normal_model, tee=True)

    gb_model.pprint()
    normal_model.pprint()
