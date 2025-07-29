# Utilizing data helping classes and 
# functions for the experiment class

# Adapted from code originally available at:
# https://github.com/dowlinglab/pyomo-doe/blob/main/notebooks/tclab_pyomo.py

# Required imports
from dataclasses import dataclass
import pandas as pd

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