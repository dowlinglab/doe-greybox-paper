# from ms_contactor_experiment import MSContactorExperiment
from medium_fidelity.membrane_experiment import MembraneExperiment

import pyomo.environ as pyo
from pyomo.contrib.doe import DesignOfExperiments

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os.path

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

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

def plot_pairwise_uncertainties(FIMs, theta_labels, theta_hat, n_std):
    cov_mat_before = np.linalg.inv(FIMs[0])

    n = len(theta_labels)

    fig, ax = plt.subplots(ncols=n-1, nrows=n-1, figsize=(n*4, n*2.5))

    for ind1, i in enumerate(range(0, n - 1)):
        # Loop over columns -- subdiagonal
        for ind2, j in enumerate(range(1, n)):
            curr_subplot = ind1 + (n - 1) * ind2 + 1
            if ind1 > ind2:
                plt.subplot(n - 1, n - 1, curr_subplot).remove()
                continue
            # Create subplots below the diagonal
            plt.subplot(n - 1, n - 1, curr_subplot)

            # Plot theta estimate
            plt.scatter(theta_hat[i], theta_hat[j], s=10)
            plt.xlabel(theta_labels[i], fontweight='bold', fontsize=25)
            plt.ylabel(theta_labels[j], fontweight='bold', fontsize=25)

            # Fix ticks
            plt.tick_params(direction="in", top=True, right=True)

            max_scale_x = 0
            max_scale_y = 0

            # Select rows from cov
            rows = cov_mat_before[(i, j), :]

            # Select columns from FIM
            cov = rows[:, (i, j)]

            # Draw non-dimensionalized
            pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
            ell_radius_x = np.sqrt(1 + pearson)
            ell_radius_y = np.sqrt(1 - pearson)
            ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                              edgecolor='k', lw=3, facecolor=(0.5, 0.8, 0.9), alpha=0.5)

            print(i, j, pearson)

            # Calculating the standard deviation of x from
            # the squareroot of the variance and multiplying
            # with the given number of standard deviations.
            scale_x = np.sqrt(cov[0, 0]) * n_std

            # calculating the standard deviation of y
            scale_y = np.sqrt(cov[1, 1]) * n_std

            # transforming ellipse
            transf = transforms.Affine2D() \
                .rotate_deg(45) \
                .scale(scale_x, scale_y) \
                .translate(theta_hat[i], theta_hat[j])

            # Plot ellipse
            ax = plt.gca()
            ellipse.set_transform(transf + ax.transData)
            ax.add_patch(ellipse)

            # Keep track of limits
            max_scale_x = np.max([scale_x, max_scale_x])
            max_scale_y = np.max([scale_y, max_scale_y])

            # Adjust plot limits
            plt.xlim([theta_hat[i] - max_scale_x, theta_hat[i] + max_scale_x])
            plt.ylim([theta_hat[j] - max_scale_y, theta_hat[j] + max_scale_y])

    if len(FIMs) > 1:
        cov_mat_after = np.linalg.pinv(FIMs[1])
        for ind1, i in enumerate(range(0, n - 1)):
            # Loop over columns -- subdiagonal
            for ind2, j in enumerate(range(1, n)):
                curr_subplot = ind1 + (n - 1) * ind2 + 1
                if ind1 > ind2:
                    plt.subplot(n - 1, n - 1, curr_subplot).remove()
                    continue
                # Create subplots below the diagonal
                plt.subplot(n - 1, n - 1, curr_subplot)

                # Plot theta estimate
                plt.scatter(theta_hat[i], theta_hat[j], s=10)
                plt.xlabel(theta_labels[i], fontweight='bold', fontsize=25)
                plt.ylabel(theta_labels[j], fontweight='bold', fontsize=25)

                # Fix ticks
                plt.tick_params(direction="in", top=True, right=True)

                max_scale_x = 0
                max_scale_y = 0

                # Select rows from cov
                rows = cov_mat_after[(i, j), :]

                # Select columns from FIM
                cov = rows[:, (i, j)]

                # Draw non-dimensionalized
                pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
                ell_radius_x = np.sqrt(1 + pearson)
                ell_radius_y = np.sqrt(1 - pearson)
                ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                                  edgecolor='k', lw=3, facecolor=(0.4, 0.4, 0.4), alpha=0.7)

                print(i, j, pearson)

                # Plot theta estimate
                plt.scatter(theta_hat[i], theta_hat[j], color='k', s=20)

                # Calculating the standard deviation of x from
                # the squareroot of the variance and multiplying
                # with the given number of standard deviations.
                scale_x = np.sqrt(cov[0, 0]) * n_std

                # calculating the standard deviation of y
                scale_y = np.sqrt(cov[1, 1]) * n_std

                # transforming ellipse
                transf = transforms.Affine2D() \
                    .rotate_deg(45) \
                    .scale(scale_x, scale_y) \
                    .translate(theta_hat[i], theta_hat[j])

                # Plot ellipse
                ax = plt.gca()
                ellipse.set_transform(transf + ax.transData)
                ax.add_patch(ellipse)

                # Keep track of limits
                max_scale_x = np.max([scale_x, max_scale_x]) * 2
                max_scale_y = np.max([scale_y, max_scale_y]) * 2

                # Adjust plot limits
                plt.xlim([theta_hat[i] - max_scale_x, theta_hat[i] + max_scale_x])
                plt.ylim([theta_hat[j] - max_scale_y, theta_hat[j] + max_scale_y])

    if len(FIMs) == 3:
        for i in range(0, n):
            # Loop over columns -- subdiagonal
            for j in range(0, n):
                curr_subplot = i + n * j + 1
                if i == j or j < i:
                    plt.subplot(n, n, curr_subplot).remove()
                    continue
                # Create subplots below the diagonal
                plt.subplot(n, n, curr_subplot)

                # Plot theta estimate
                plt.scatter(theta_hat[i], theta_hat[j], s=10)
                plt.xlabel(theta_labels[i], fontweight='bold')
                plt.ylabel(theta_labels[j], fontweight='bold')

                # Fix ticks
                plt.tick_params(direction="in", top=True, right=True)

                max_scale_x = 0
                max_scale_y = 0

                # Select rows from cov
                rows = cov_mat_after[(i, j), :]

                # Select columns from FIM
                cov = rows[:, (i, j)]

                # Draw non-dimensionalized
                pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
                ell_radius_x = np.sqrt(1 + pearson)
                ell_radius_y = np.sqrt(1 - pearson)
                ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                                  edgecolor='k', lw=3, facecolor=(1, 1, 1), alpha=0.7)

                # Plot theta estimate
                plt.scatter(theta_hat[i], theta_hat[j], color='k', s=20)

                # Calculating the standard deviation of x from
                # the squareroot of the variance and multiplying
                # with the given number of standard deviations.
                scale_x = np.sqrt(cov[0, 0]) * n_std

                # calculating the standard deviation of y
                scale_y = np.sqrt(cov[1, 1]) * n_std

                # transforming ellipse
                transf = transforms.Affine2D() \
                    .rotate_deg(45) \
                    .scale(scale_x, scale_y) \
                    .translate(theta_hat[i], theta_hat[j])

                # Plot ellipse
                ax = plt.gca()
                ellipse.set_transform(transf + ax.transData)
                ax.add_patch(ellipse)

    plt.tight_layout()


# Function for extracting FIM metrics
def get_FIM_metrics(result):
    eigenvalues, eigenvectors = np.linalg.eig(result)
    min_eig = min(eigenvalues)
    print(eigenvalues)

    A_opt = np.trace(np.linalg.pinv(result))
    D_opt = np.log10(np.linalg.det(result))
    E_opt = min_eig
    ME_opt = np.log10(np.linalg.cond(result))

    return A_opt, D_opt, E_opt, ME_opt


# Set the design ranges for the decision vars
design_ranges = {
    "feed_flow": [90, 94, 98, 102, 108, 110],
    "diafiltrate_flow": [33, 32, 31, 30, 29, 28, 27],
}

# Data from parmest estimate:
theta_hat = {"fs.Lp":2.998e-7,
             "fs.constant_sieving_coeff[Li]":1.001,
             "fs.constant_sieving_coeff[Co]":0.3962,
             "fs.ionic_strength_coeff[Li]":5.024e-4,
             "fs.ionic_strength_coeff[Co]":9.25e-5
}

# Default design
membrane_design = {}
membrane_design["Q_feed (m^3/hr)"] = 90.0
membrane_design["C_Li_feed (kg/m^3)"] = 1.7
membrane_design["C_Co_feed (kg/m^3)"] = 17
membrane_design["Q_diaf (m^3/hr)"] = 27.0
membrane_design["Q_Li_prdt (m^3/hr)"] = 86.28
membrane_design["Q_Co_prdt (m^3/hr)"] = 36.04
membrane_design["C_Co_Co_prdt (kg/m^3)"] = 38.57
membrane_design["C_Li_Co_prdt (kg/m^3)"] = 0.692
membrane_design["C_Li_Li_prdt (kg/m^3)"] = 1.740
membrane_design["C_Co_Li_prdt (kg/m^3)"] = 3.087

prior_data_df = pd.read_csv(os.path.join("medium_fidelity", "membrane_cascade_data.csv"))

# Make prior
for i in range(4):
    FIM_prior = np.zeros((5, 5))

    temp_membrane_design = prior_data_df.iloc[i, :]

    doe_experiment = MembraneExperiment(data=temp_membrane_design, theta=theta_hat)

    ms_contactor_DoE = DesignOfExperiments(
        experiment=doe_experiment,
        step=1e-2,
        scale_constant_value=1,
        scale_nominal_param_value=True,
        tee=False,
    )

    FIM_temp = ms_contactor_DoE.compute_FIM(method='sequential')

    FIM_prior += FIM_temp
# Estimate experiments for all conditions
objective_options = [
        "determinant",
        "trace",
        "minimum_eigenvalue",
        "condition_number",
    ]

optimal_points = [0, ] * 4
optimal_FIMs = [0, ] * 4
optimal_FIMs_round_2 = [0, ] * 4
optimal_points_round_2 = [0, ] * 4
optimal_objective_value = 0

# Define our own grey box solver
grey_box_solver = pyo.SolverFactory("cyipopt")
grey_box_solver.config.options["linear_solver"] = "ma57"
grey_box_solver.config.options['mu_strategy'] = "monotone"

for ind, objective_option in enumerate(objective_options):
    # Round 1
    experiment = MembraneExperiment(data=membrane_design, theta=theta_hat)

    ms_contactor_DoE = DesignOfExperiments(
            experiment,
            step=1e-2,
            objective_option=objective_option,
            use_grey_box_objective=True,
            scale_constant_value=1,
            scale_nominal_param_value=True,
            prior_FIM=FIM_prior,
            grey_box_solver=grey_box_solver,
            grey_box_tee=True,
        )

    ms_contactor_DoE.run_doe()

    ms_contactor_DoE.model.scenario_blocks[0].fs.mix2.inlet_1.flow_vol.pprint()

    optimal_FIMs[ind] = ms_contactor_DoE.results["FIM"]
    optimal_points[ind] = ms_contactor_DoE.results["Experiment Design"]

    if objective_option == "minimum_eigenvalue":
        #ms_contactor_DoE.model.scenario_blocks[0].fs.stage3.pprint()
        ms_contactor_DoE.model.scenario_blocks[0].fs.stage1.osmotic_pressure.pprint()
        ms_contactor_DoE.model.scenario_blocks[0].fs.stage2.osmotic_pressure.pprint()
        ms_contactor_DoE.model.scenario_blocks[0].fs.stage3.osmotic_pressure.pprint()

        plt.plot()
        plt.show()
        plt.clf()

    # Round 2
    experiment = MembraneExperiment(data=membrane_design, theta=theta_hat)

    ms_contactor_DoE_2 = DesignOfExperiments(
        experiment,
        step=1e-2,
        objective_option=objective_option,
        use_grey_box_objective=True,
        scale_constant_value=1,
        scale_nominal_param_value=True,
        prior_FIM=np.asarray(optimal_FIMs[ind]),
        grey_box_solver=grey_box_solver,
        grey_box_tee=True,
    )

    ms_contactor_DoE_2.run_doe()

    optimal_FIMs_round_2[ind] = ms_contactor_DoE_2.results["FIM"]
    optimal_points_round_2[ind] = ms_contactor_DoE_2.results["Experiment Design"]

theta_labels = ["$L_p$", "$S_{Li}$", "$S_{Co}$", "$\delta_{Li}$", "$\delta_{Co}$"]
theta_values = [2.998e-7, 1.001, 0.3962, 0.0005024, 9.250e-5]# 0.00000626]
#theta_values = theta_hat.values()

plot_pairwise_uncertainties([FIM_prior, optimal_FIMs[2]], theta_labels, theta_values, n_std=1)
plt.savefig("uncertainty_reduction.png")
plt.show()

plot_pairwise_uncertainties([FIM_prior, ], theta_labels, theta_values, n_std=1)
plt.savefig("only_prior_uncertainty_comparison.png")
plt.show()
#
# design_ranges = {
#     "feed_flow": [9, 9.2, 9.4, 9.6],
#     "diafiltrate_flow": [36, 35, 34, 33],
# }

# Loop through the values
FIM_results = []
data_feed_flow = []
data_diafiltrate_flow = []

count = 0
# Grid search
for diafiltrate_flow in design_ranges["diafiltrate_flow"]:
    for feed_flow in design_ranges["feed_flow"]:
        count += 1
        print("=======Iteration Number: {} =======".format(count))
        print(
            "Design variable values for this iteration: (Feed flow: {}, Diafiltrate flow: {})".format(
                feed_flow, diafiltrate_flow
            )
        )

        data_feed_flow.append(feed_flow)
        data_diafiltrate_flow.append(diafiltrate_flow)

        membrane_design["C_Li_feed (kg/m^3)"] = 2.
        membrane_design["C_Co_feed (kg/m^3)"] = 20.
        membrane_design["Q_feed (m^3/hr)"] = feed_flow
        membrane_design["Q_diaf (m^3/hr)"] = diafiltrate_flow

        print("\n\n\nMAKE EXPERIMENT\n\n\n")

        doe_experiment = MembraneExperiment(data=membrane_design, theta=theta_hat)

        solver = pyo.SolverFactory("ipopt")
        solver.options["linear_solver"] = "ma27"

        # Create the design of experiments object using our experiment instance from above
        ms_contactor_DoE = DesignOfExperiments(
            experiment=doe_experiment,
            step=1e-2,
            scale_constant_value=1,
            scale_nominal_param_value=True,
            tee=False,
        )

        FIM = ms_contactor_DoE.compute_FIM(method='sequential')

        FIM_results.append(FIM + FIM_prior)


# Heatmap Plotting Function
def plot_heatmap(data, title, y_label, x_label, colorbar_label, take_the_log=False):
    # set heatmap x,y ranges
    x_tick_labels = np.sort(np.unique(data[:, 0]))[::-1]
    y_tick_labels = np.sort(np.unique(data[:, 1]))

    # optimality-values
    opt_vals = np.asarray(data[:, 2]).reshape(len(x_tick_labels), len(y_tick_labels))

    if take_the_log:
        opt_vals = np.log(opt_vals)

    print(x_tick_labels)
    print(y_tick_labels)
    print(opt_vals)

    # Plot the colormap
    fig = plt.figure()

    # Plotting options
    ax = fig.add_subplot(111)
    params = {"mathtext.default": "regular"}
    plt.rcParams.update(params)

    # Plotting data
    ax.set_yticks(range(len(y_tick_labels)))
    ax.set_yticklabels(y_tick_labels)
    ax.set_ylabel(y_label)
    ax.set_xticks(range(len(x_tick_labels)))
    ax.set_xticklabels(x_tick_labels)
    ax.set_xlabel(x_label)
    ax.tick_params(axis='x', labelrotation=90)
    im = ax.imshow(opt_vals.T, cmap=plt.cm.hot_r)
    ba = plt.colorbar(im)
    ba.set_label(colorbar_label)
    plt.title(title, fontsize="24")


# Grab the FIM Metrics from the FIM results
FIM_metrics = []

for i in FIM_results:
    FIM_metrics.append(get_FIM_metrics(i))

FIM_metrics_np = np.asarray(FIM_metrics)

# X and Y axis labels
x_label = "Diafiltrate flow [m$^3$/min$^{-1}$]"
y_label = "Feed flow [m$^3$/min$^{-1}$]"

# Draw A-optimality figure
data_A = np.zeros((len(FIM_metrics), 3))
data_A[:, 1] = data_feed_flow
data_A[:, 0] = data_diafiltrate_flow
data_A[:, 2] = FIM_metrics_np[:, 0]
#data_A[:, 2] = np.asarray([i for i in range(len(data_feed_flow))])

plot_heatmap(data_A, "A-optimality", y_label, x_label, r"trace(FIM$^{-1}$)")
plt.tight_layout()
plt.savefig("membrane_A-opt_heatmap.png", format="png", dpi=450)
plt.clf()
plt.close()

# Draw D-optimality figure
data_D = np.zeros((len(FIM_metrics), 3))
data_D[:, 1] = data_feed_flow
data_D[:, 0] = data_diafiltrate_flow
data_D[:, 2] = FIM_metrics_np[:, 1]

plot_heatmap(data_D, "D-optimality", y_label, x_label, "log10(det(FIM))")
plt.tight_layout()
plt.savefig("membrane_D-opt_heatmap.png", format="png", dpi=450)
plt.clf()
plt.close()

# Draw E-optimality figure
data_E = np.zeros((len(FIM_metrics), 3))
data_E[:, 1] = data_feed_flow
data_E[:, 0] = data_diafiltrate_flow
data_E[:, 2] = FIM_metrics_np[:, 2]

plot_heatmap(data_E, "E-optimality", y_label, x_label, "min-eig(FIM)")
plt.tight_layout()
plt.savefig("membrane_E-opt_heatmap.png", format="png", dpi=450)
plt.clf()
plt.close()

# Draw ME-optimality figure
data_ME = np.zeros((len(FIM_metrics), 3))
data_ME[:, 1] = data_feed_flow
data_ME[:, 0] = data_diafiltrate_flow
data_ME[:, 2] = FIM_metrics_np[:, 3]

plot_heatmap(data_ME, "ME-optimality", y_label, x_label, "log10(cond(FIM))")
plt.tight_layout()
plt.savefig("membrane_ME-opt_heatmap.png", format="png", dpi=450)
plt.clf()
plt.close()

print(optimal_points)
print(optimal_FIMs)
print("\n\nROUND 1")
print(np.asarray(optimal_FIMs[2]))
print(np.asarray(FIM_prior))
print("\n\nROUND 2")
print(optimal_points_round_2)
print(optimal_FIMs_round_2)