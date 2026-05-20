# Utilizing data helping classes and
# functions for the experiment class

# Adapted from code originally available at:
# https://github.com/dowlinglab/pyomo-doe/blob/main/notebooks/tclab_pyomo.py

# Required imports
from dataclasses import dataclass
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib.patches import Ellipse, Patch

from pyomo.common.dependencies import numpy as np, pathlib


@dataclass
class TC_Lab_data:
    """Class for storing data from a TCLab experiment."""

    name: str  # Name of the experiment (optional)
    time: np.array  # Time stamp for measurements, [seconds]
    T1: np.array  # Temperature of heater 1, [degC]
    u1: np.array  # Heater 1 power setting, [0-100]
    P1: float  # Power setting for heater 1, [W]
    TS1_data: np.array  # Setpoint data for temperature of sensor 1, [degC]
    T2: np.array  # Temperature of heater 2, [degC]
    u2: np.array  # Heater 2 power setting, [0-100]
    P2: float  # Power setting for heater 2, [W]
    TS2_data: np.array  # Setpoint data for temperature of sensor 1, [degC]
    Tamb: float  # Ambient temperature, [degC]

    def to_data_frame(self):
        """Convert instance of this class to a pandas DataFrame."""

        df = pd.DataFrame(
            {
                "time": self.time,
                "T1": self.T1,
                "u1": self.u1,
                "P1": self.P1,
                "TS1_data": self.TS1_data,
                "T2": self.T2,
                "u2": self.u2,
                "P2": self.P2,
                "TS2_data": self.TS2_data,
                "Tamb": self.Tamb,
            }
        )

        return df


# Helper function for initializing the model
def helper(my_array, time):
    '''
    Method that builds a dictionary to help initialization.
    Arguments:
        my_array: an array
    Returns:
        data: a dict {time: array_value}
    '''
    # ensure that the dimensions of array and time data match
    assert len(my_array) == len(time), "Dimension mismatch."
    data2 = {}
    for k, t in enumerate(time):
        if my_array[k] is not None:
            data2[t] = my_array[k]
        else:
            # Replace None with 0
            data2[t] = 0
    return data2


def plot_pairwise_uncertainties(FIMs, theta_labels, theta_hat, n_std, add_legend=False):
    cov_mat_before = np.linalg.inv(FIMs[0])

    n = len(theta_labels)

    fig, ax = plt.subplots(ncols=n - 1, nrows=n - 1, figsize=(n * 4, n * 2.5))

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
            ellipse = Ellipse(
                (0, 0),
                width=ell_radius_x * 2,
                height=ell_radius_y * 2,
                edgecolor='k',
                lw=3,
                facecolor=(0.5, 0.8, 0.9),
                alpha=0.5,
            )

            # Calculating the standard deviation of x from
            # the squareroot of the variance and multiplying
            # with the given number of standard deviations.
            scale_x = np.sqrt(cov[0, 0]) * n_std

            # calculating the standard deviation of y
            scale_y = np.sqrt(cov[1, 1]) * n_std

            # transforming ellipse
            transf = (
                transforms.Affine2D()
                .rotate_deg(45)
                .scale(scale_x, scale_y)
                .translate(theta_hat[i], theta_hat[j])
            )

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

    if len(FIMs) == 3:
        cov_mat_after = np.linalg.pinv(FIMs[2])
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
                ellipse = Ellipse(
                    (0, 0),
                    width=ell_radius_x * 2,
                    height=ell_radius_y * 2,
                    edgecolor='red',
                    linestyle='--',
                    hatch='+',
                    lw=3,
                    facecolor=(0.8, 0.8, 0.8),
                    alpha=0.7,
                )

                # Plot theta estimate
                plt.scatter(theta_hat[i], theta_hat[j], color='k', s=20)

                # Calculating the standard deviation of x from
                # the squareroot of the variance and multiplying
                # with the given number of standard deviations.
                scale_x = np.sqrt(cov[0, 0]) * n_std

                # calculating the standard deviation of y
                scale_y = np.sqrt(cov[1, 1]) * n_std

                # transforming ellipse
                transf = (
                    transforms.Affine2D()
                    .rotate_deg(45)
                    .scale(scale_x, scale_y)
                    .translate(theta_hat[i], theta_hat[j])
                )

                # Plot ellipse
                ax = plt.gca()
                ellipse.set_transform(transf + ax.transData)
                ax.add_patch(ellipse)

                max_scale_x = np.max([scale_x, max_scale_x]) * 2
                max_scale_y = np.max([scale_y, max_scale_y]) * 2

                plt.xlim([theta_hat[i] - max_scale_x, theta_hat[i] + max_scale_x])
                plt.ylim([theta_hat[j] - max_scale_y, theta_hat[j] + max_scale_y])

                max_scale_x = max_scale_x / 2
                max_scale_y = max_scale_y / 2

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
                ellipse = Ellipse(
                    (0, 0),
                    width=ell_radius_x * 2,
                    height=ell_radius_y * 2,
                    edgecolor='k',
                    lw=3,
                    facecolor=(0.4, 0.4, 0.4),
                    alpha=0.7,
                )

                # Plot theta estimate
                plt.scatter(theta_hat[i], theta_hat[j], color='k', s=20)

                # Calculating the standard deviation of x from
                # the squareroot of the variance and multiplying
                # with the given number of standard deviations.
                scale_x = np.sqrt(cov[0, 0]) * n_std

                # calculating the standard deviation of y
                scale_y = np.sqrt(cov[1, 1]) * n_std

                # transforming ellipse
                transf = (
                    transforms.Affine2D()
                    .rotate_deg(45)
                    .scale(scale_x, scale_y)
                    .translate(theta_hat[i], theta_hat[j])
                )

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

        if add_legend and len(FIMs) == 2:
            ellipse_legend = [
                Patch(
                    edgecolor='gray',
                    facecolor=(0.5, 0.8, 0.9),
                    alpha=0.5,
                    label='Original Experiments',
                ),
                Patch(
                    edgecolor='k',
                    lw=3,
                    facecolor=(0.4, 0.4, 0.4),
                    alpha=0.7,
                    label='With Optimal Experiment',
                ),
            ]

            fig.legend(
                handles=ellipse_legend,
                loc='upper right',
                bbox_to_anchor=(0.9, 0.9),
                fontsize=24,
            )

    if len(FIMs) == 3:
        if add_legend:
            ellipse_legend = [
                Patch(
                    edgecolor='gray',
                    facecolor=(0.5, 0.8, 0.9),
                    alpha=0.5,
                    label='Original Experiments',
                ),
                Patch(
                    edgecolor='k',
                    lw=3,
                    facecolor=(0.4, 0.4, 0.4),
                    alpha=0.7,
                    label='With Optimal Experiment',
                ),
                Patch(
                    edgecolor='red',
                    linestyle='--',
                    hatch='+',
                    lw=3,
                    facecolor=(0.8, 0.8, 0.8),
                    alpha=0.7,
                    label='With Model-Free Experiment',
                ),
            ]

            fig.legend(
                handles=ellipse_legend,
                loc='upper right',
                bbox_to_anchor=(1.0, 0.95),
                fontsize=24,
            )

    plt.tight_layout()


def plot_correlation_matrix(FIM, theta_labels):
    # Compute the correlation matrix from FIM
    cov_M = np.linalg.inv(FIM)
    corr_M = (
        (np.sqrt(np.diag(1 / np.diag(cov_M))))
        @ cov_M
        @ (np.sqrt(np.diag(1 / np.diag(cov_M))))
    )

    im = plt.imshow(corr_M, cmap="RdBu_r")
    plt.xticks([0, 1, 2], theta_labels)
    plt.yticks([0, 1, 2], theta_labels)
    plt.colorbar(im)
    plt.tight_layout()

    for i in range(3):
        for j in range(3):
            color = "white" if abs(corr_M[i, j]) > 0.65 else "black"
            plt.gca().text(
                j,
                i,
                f"{corr_M[i, j]:.3f}",
                ha="center",
                va="center",
                color=color,
                fontsize=20,
            )
