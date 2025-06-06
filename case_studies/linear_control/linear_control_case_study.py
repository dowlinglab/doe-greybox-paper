from pyomo.common.dependencies import numpy as np, pathlib

from linear_control_experiment import (
    LinearControlExperiment,
)

from pyomo.contrib.doe import DesignOfExperiments

import pyomo.environ as pyo

import matplotlib.pyplot as plt
import json
import sys


def run_linear_control_experiment():
    # Read in the data
    DATA_DIR = pathlib.Path(__file__).parent
    file_path = DATA_DIR / "experimental_values_1.json"

    with open(file_path) as f:
        data_ex = json.load(f)

    # Put temperature control time points into correct format for reactor experiment
    data_ex["control_points"] = {
        float(k): v for k, v in data_ex["control_points"].items()
    }

    # Create the experiment
    experiment = LinearControlExperiment(data=data_ex, nfe=10, ncp=3)

    model = experiment.get_labeled_model()

    solver = pyo.SolverFactory("ipopt")
    result = solver.solve(model, tee=True)

    # Grab state data
    x1_vals = [model.x2[t]() for t in model.t]
    x2_vals = [model.x1[t]() for t in model.t]
    t_vals = [t for t in model.t]

    plt.plot(t_vals, x1_vals, label="x1")
    plt.plot(t_vals, x2_vals, label="x2")

    plt.legend()
    plt.show()

    # Use a central difference, with step size 1e-3
    fd_formula = "central"
    step_size = 1e-3

    # Use the determinant objective with scaled sensitivity matrix
    objective_option = "determinant"
    scale_nominal_param_value = True

    # Set up DoE object
    doe_obj = DesignOfExperiments(
        experiment,
        fd_formula=fd_formula,
        step=step_size,
        objective_option=objective_option,
        scale_constant_value=1,
        scale_nominal_param_value=scale_nominal_param_value,
        prior_FIM=None,
        jac_initial=None,
        fim_initial=None,
        L_diagonal_lower_bound=1e-7,
        solver=None,
        tee=True,
        get_labeled_model_args=None,
        _Cholesky_option=True,
        _only_compute_fim_lower=True,
    )

    FIM = doe_obj.compute_FIM(method="sequential")

    print(FIM)

    # Grab second experiment
    file_path_2 = DATA_DIR / "experimental_values_2.json"

    with open(file_path_2) as f:
        data_ex_2 = json.load(f)

    # Put temperature control time points into correct format for reactor experiment
    data_ex_2["control_points"] = {
        float(k): v for k, v in data_ex_2["control_points"].items()
    }

    experiment_2 = LinearControlExperiment(data=data_ex_2, nfe=10, ncp=3)

    doe_obj_2 = DesignOfExperiments(
        experiment_2,
        fd_formula=fd_formula,
        step=step_size,
        objective_option=objective_option,
        scale_constant_value=1,
        scale_nominal_param_value=scale_nominal_param_value,
        prior_FIM=FIM,
        jac_initial=None,
        fim_initial=None,
        L_diagonal_lower_bound=1e-7,
        solver=None,
        tee=False,
        get_labeled_model_args=None,
        _Cholesky_option=True,
        _only_compute_fim_lower=True,
    )
    doe_obj_2.run_doe()

    # Print out a results summary
    print("Optimal experiment values: ")
    print(
        "\tInitial x1 value: {:.2f}".format(
            doe_obj_2.results["Experiment Design"][0]
        )
    )
    print("Optimal experiment values: ")
    print(
        "\tInitial x2 value: {:.2f}".format(
            doe_obj_2.results["Experiment Design"][1]
        )
    )
    print(
        ("\tTemperature values: [" + "{:.2f}, " * 8 + "{:.2f}]").format(
            *doe_obj_2.results["Experiment Design"][2:]
        )
    )
    print("FIM at optimal design:\n {}".format(np.array(doe_obj_2.results["FIM"])))
    print(
        "Objective value at optimal design: {:.2f}".format(
            pyo.value(doe_obj_2.model.objective)
        )
    )

    print(doe_obj_2.results["Experiment Design Names"])

    ###################
    # End optimal DoE


if __name__ == "__main__":
    run_linear_control_experiment()