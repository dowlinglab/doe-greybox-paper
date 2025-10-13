from ms_contactor_experiment import MSContactorExperiment

import pyomo.environ as pyo
from pyomo.contrib.doe import DesignOfExperiments

import numpy as np
import matplotlib.pyplot as plt


SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 24

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('lines', linewidth=3)


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
    "feed_flow": [50, 60, 70, 80, 90],
    "diafiltrate_flow": [30, 25, 20, 15, 10],
}

# Default design
membrane_design = {}
membrane_design["Q_feed"] = 100.0
membrane_design["C_Li_feed"] = 1.7
membrane_design["C_Co_feed"] = 17
membrane_design["Q_diafiltrate"] = 30.0
membrane_design["C_Li_diafiltrate"] = 0.1
membrane_design["C_Co_diafiltrate"] = 0.2

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

        membrane_design["Q_feed"] = feed_flow
        membrane_design["Q_diafiltrate"] = diafiltrate_flow

        print("\n\n\nMAKE EXPERIMENT\n\n\n")

        doe_experiment = MSContactorExperiment(data=membrane_design)

        solver = pyo.SolverFactory("ipopt")
        solver.options["linear_solver"] = "ma27"

        # Create the design of experiments object using our experiment instance from above
        ms_contactor_DoE = DesignOfExperiments(
            experiment=doe_experiment,
            step=1e-2,
            scale_constant_value=1,
            scale_nominal_param_value=True,
            tee=True,
        )

        FIM = ms_contactor_DoE.compute_FIM(method='sequential')

        FIM_results.append(FIM)


# Heatmap Plotting Function
def plot_heatmap(data, title, y_label, x_label, colorbar_label):
    # set heatmap x,y ranges
    x_tick_labels = np.sort(np.unique(data[:, 0]))
    y_tick_labels = np.sort(np.unique(data[:, 1]))[::-1]

    # optimality-values
    opt_vals = np.asarray(data[:, 2]).reshape(len(x_tick_labels), len(y_tick_labels))

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
    im = ax.imshow(opt_vals, cmap=plt.cm.hot_r)
    ba = plt.colorbar(im)
    ba.set_label(colorbar_label)
    plt.title(title, fontsize="24")


# Grab the FIM Metrics from the FIM results
FIM_metrics = []

for i in FIM_results:
    FIM_metrics.append(get_FIM_metrics(i))

FIM_metrics_np = np.asarray(FIM_metrics)

# X and Y axis labels
x_label = "Feed flow [m^3/min]"
y_label = "Diafiltrate flow [m^3/min]"

# Draw A-optimality figure
data_A = np.zeros((len(FIM_metrics), 3))
data_A[:, 0] = data_feed_flow
data_A[:, 1] = data_diafiltrate_flow
data_A[:, 2] = FIM_metrics_np[:, 0]

plot_heatmap(data_A, "A-optimality", y_label, x_label, r"trace(FIM$^{-1}$)")
plt.tight_layout()
plt.savefig("membrane_A-opt_heatmap.png", format="png", dpi=450)
plt.clf()
plt.close()

# Draw D-optimality figure
data_D = np.zeros((len(FIM_metrics), 3))
data_D[:, 0] = data_feed_flow
data_D[:, 1] = data_diafiltrate_flow
data_D[:, 2] = FIM_metrics_np[:, 1]

plot_heatmap(data_D, "D-optimality", y_label, x_label, "log10(det(FIM))")
plt.tight_layout()
plt.savefig("membrane_D-opt_heatmap.png", format="png", dpi=450)
plt.clf()
plt.close()

# Draw E-optimality figure
data_E = np.zeros((len(FIM_metrics), 3))
data_E[:, 0] = data_feed_flow
data_E[:, 1] = data_diafiltrate_flow
data_E[:, 2] = FIM_metrics_np[:, 2]

plot_heatmap(data_E, "E-optimality", y_label, x_label, "min-eig(FIM)")
plt.tight_layout()
plt.savefig("membrane_E-opt_heatmap.png", format="png", dpi=450)
plt.clf()
plt.close()

# Draw ME-optimality figure
data_ME = np.zeros((len(FIM_metrics), 3))
data_ME[:, 0] = data_feed_flow
data_ME[:, 1] = data_diafiltrate_flow
data_ME[:, 2] = FIM_metrics_np[:, 3]

plot_heatmap(data_ME, "ME-optimality", y_label, x_label, "log10(cond(FIM))")
plt.tight_layout()
plt.savefig("membrane_ME-opt_heatmap.png", format="png", dpi=450)
plt.clf()
plt.close()
