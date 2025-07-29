# Adapted from code originally available at:
# https://github.com/dowlinglab/pyomo-doe/blob/main/notebooks/tclab_pyomo.py

# === Required imports ===
import pyomo.environ as pyo
from pyomo.dae import ContinuousSet, DerivativeVar, Simulator

from pyomo.contrib.parmest.experiment import Experiment

from TC_Lab_data_helper import TC_Lab_data, helper

import matplotlib.pyplot as plt
from scipy import interpolate
import numpy as np
import pandas as pd

import shutil
import sys
import os.path
import os
import re

import subprocess

# Experiment object 
class TC_Lab_experiment(Experiment):
    def __init__(self, data, alpha=0.00016, theta_initial=None, number_of_states=2, sine_amplitude=None, sine_period=None, reparam=False, include_Th=False):
        """
        Arguments
        ---------
        data: TC_Lab_Data object
        alpha: float, Conversion factor for TCLab (fixed parameter)
        theta_initial: dictionary, initial guesses for the unknown parameters
        number_of_states: int, number of states in the heat transfer model (must be 2 or 4), default: 2
        sine_amplitude: float, amplitude of the sine wave, default: None (do not use the sine wave)
        sine_period: float, period of the sine wave, default: None (do not use the sine wave)
        
        """
        self.data = data
        
        # If no parameters are provided, offer
        # reasonable guesses from past parameter
        # estimation results.
        if theta_initial is None:
            self.theta_initial={
                "Ua": 0.0535,
                "Ub": 0.0148,
                "inv_CpH": 1 / 6.911,
                "inv_CpS": 1 / 0.318,
                "Uc": 0.001,
            }
        else:
            self.theta_initial = theta_initial

        self.alpha = alpha
        
        # Make sure that the number of states is either 2 or 4
        if number_of_states not in [2, 4, ]:
            raise ValueError("number_of_states must be 2 or 4.")
        self.number_of_states = number_of_states
        
        # Make sure that the sine amplitude and period are reasonable
        if sine_amplitude is not None and sine_period is not None:
            self.sine_period_max = 10  # minutes
            self.sine_period_min = 10 / 60  # minutes

            assert sine_amplitude <= 50, "Sine amplitude must be less than 50."
            assert sine_amplitude >= 0, "Sine amplitude must be greater than 0."

            assert sine_period <= self.sine_period_max, "Sine period must be less than " + str(
                self.sine_period_max
            )
            assert (
                sine_period >= self.sine_period_min
            ), "Sine period must be greater than " + str(self.sine_period_min)
        elif sine_amplitude is not None or sine_period is not None:
            raise ValueError("If sine wave is used, both amplitude and period must be provided.")
        else:
            self.sine_period_max = None
            self.sine_period_min = None
        self.sine_amplitude = sine_amplitude
        self.sine_period = sine_period
        
        self.reparam = reparam
        
        self.include_Th = include_Th
        
        self.model = None
    
    def get_labeled_model(self):
        if self.model is None:
            self.create_model()
            self.finalize_model()
            self.label_experiment()
        return self.model
    
    def create_model(self):
        """
        Method to create an unlabled model of the TC Lab system.
        
        """
        m = self.model = pyo.ConcreteModel()
        
        #########################################
        # Begin model constants definition
        m.Tamb = pyo.Param(initialize=self.data.Tamb)
        m.P1 = pyo.Param(initialize=self.data.P1)
        m.alpha = pyo.Param(initialize=self.alpha)
        m.P2 = pyo.Param(initialize=self.data.P2)
        
        m.Tmax = 85  # Maximum temparture (Deg C)
        
        # End model constants
        #########################################
        
        ################################
        # Defining state variables
        m.t = ContinuousSet(initialize=self.data.time)
        
        # Temperature states for the fins
        m.Th1 = pyo.Var(m.t, bounds=[0, m.Tmax], initialize=m.Tamb.value)
        m.Ts1 = pyo.Var(m.t, bounds=[0, m.Tmax], initialize=m.Tamb.value)

        if self.number_of_states == 4:
            m.Th2 = pyo.Var(m.t, bounds=[0, m.Tmax], initialize=m.Tamb.value)
            m.Ts2 = pyo.Var(m.t, bounds=[0, m.Tmax], initialize=m.Tamb.value)
        
        # Derivatives of the temperature state variables
        m.Th1dot = DerivativeVar(m.Th1, wrt=m.t)
        m.Ts1dot = DerivativeVar(m.Ts1, wrt=m.t)

        if self.number_of_states == 4:
            m.Th2dot = DerivativeVar(m.Th2, wrt=m.t)
            m.Ts2dot = DerivativeVar(m.Ts2, wrt=m.t)
        
        # End state variable definition
        ################################

        ####################################
        # Defining experimental inputs
        
        # Add control variables (experimental design decisions)
        m.U1 = pyo.Var(m.t, bounds=(0, 100), initialize=helper(self.data.u1, self.data.time))
        m.U1.fix()  # Fixed for parameter estimation

        if self.number_of_states == 4:
            m.U2 = pyo.Var(m.t, bounds=(0, 100), initialize=helper(self.data.u2, self.data.time))
            m.U2.fix()  # Fixed for parameter estimation
        
        # End experimental input definition
        ####################################
        
        ####################################
        # Defining unknown model parameters
        # (estimated during parameter estimation)
        
        # Heat transfer coefficients
        if not self.reparam:
            m.Ua = pyo.Var(initialize=self.theta_initial["Ua"], bounds=(0, 1e4))
            m.Ua.fix()
            m.Ub = pyo.Var(initialize=self.theta_initial["Ub"], bounds=(0, 1e4))
            m.Ub.fix()
            
            if self.number_of_states == 4:
                m.Uc = pyo.Var(initialize=self.theta_initial["Uc"], bounds=(0, 1e4))
                m.Uc.fix()
            
            # Inverse of the heat capacity coefficients (1/CpH and 1/CpS)
            m.inv_CpH = pyo.Var(initialize=self.theta_initial["inv_CpH"], bounds=(0, 1e6))
            m.inv_CpH.fix()
            m.inv_CpS = pyo.Var(initialize=self.theta_initial["inv_CpS"], bounds=(0, 1e3))
            m.inv_CpS.fix()
        else:
            # REPARAMETRIZATION
            m.beta_1 = pyo.Var(initialize=self.theta_initial["Ua"] * self.theta_initial["inv_CpH"], bounds=(0, 1e6))
            m.beta_1.fix()
            m.beta_2 = pyo.Var(initialize=self.theta_initial["Ub"] * self.theta_initial["inv_CpH"], bounds=(1e-6, 1e6))
            m.beta_2.fix()
            m.beta_3 = pyo.Var(initialize=self.theta_initial["Ub"] * self.theta_initial["inv_CpS"], bounds=(0, 1e6))
            m.beta_3.fix()
            m.beta_4 = pyo.Var(initialize=self.alpha * pyo.value(m.P1) * self.theta_initial["inv_CpH"], bounds=(0, 1e6))
            m.beta_4.fix()

            if self.number_of_states == 4:
                m.beta_5 = pyo.Var(initialize=self.theta_initial["Uc"] / self.theta_initial["inv_CpH"], bounds=(0, 1e6))
                m.beta_5.fix()
        
        # End unknown parameter definition
        ####################################
        
        ################################
        # Defining model equations
        
        # First fin energy balance
        @m.Constraint(m.t)
        def Th1_ode(m, t):
            if not self.reparam:
                rhs_expr = (m.Ua * (m.Tamb - m.Th1[t]) + m.Ub * (m.Ts1[t] - m.Th1[t]) + m.alpha * m.P1 * m.U1[t]) * m.inv_CpH
            else:
                # REPARAM
                rhs_expr = m.beta_1 * (m.Tamb - m.Th1[t]) + m.beta_2 * (m.Ts1[t] - m.Th1[t]) + m.beta_4 * m.U1[t]
                        
            # If we use the 4-state model, we add heat transfer from sensor 2 to the energy balance on fin 1
            if self.number_of_states == 4:
                if not self.reparam:
                    rhs_expr += (m.Uc * (m.Th2[t] - m.Th1[t])) * m.inv_CpH
                else:
                    # REPARAM
                    rhs_expr += m.beta_5 * (m.Th2[t] - m.Th1[t])
            
            return m.Th1dot[t] == rhs_expr
        
        # First sensor energy balance
        @m.Constraint(m.t)
        def Ts1_ode(m, t):
            if not self.reparam:
                return m.Ts1dot[t] == (m.Ub * (m.Th1[t] - m.Ts1[t])) * m.inv_CpS
            else:
                # REPARAM
                return m.Ts1dot[t] == m.beta_3 * (m.Th1[t] - m.Ts1[t])
        
        # Second fin/sensor (only active for the 4-state model
        if self.number_of_states == 4:
            # Second fin energy balance
            @m.Constraint(m.t)
            def Th2_ode(m, t):
                if not self.reparam:
                    return m.Th2dot[t] == (m.Ua * (m.Tamb - m.Th2[t]) + m.Ub * (m.Ts2[t] - m.Th2[t]) + m.Uc * (m.Th1[t] - m.Th2[t]) + m.alpha * m.P2 * m.U2[t]) * m.inv_CpH
                else:
                    # REPARAM
                    return m.Th2dot[t] == m.beta_1 * (m.Tamb - m.Th2[t]) + m.beta_2 * (m.Ts2[t] - m.Th2[t]) + m.beta_5 * (m.Th1[t] - m.Th2[t]) + m.beta_4 * m.U2[t]
           
            # Second sensor energy balance
            @m.Constraint(m.t)
            def Ts2_ode(m, t):
                if not self.reparam:
                    return m.Ts2dot[t] == (m.Ub * (m.Th2[t] - m.Ts2[t])) * m.inv_CpS
                else:
                    # REPARAM
                    return m.Ts2dot[t] == m.beta_3 * (m.Th2[t] - m.Ts2[t])

        # End model equation definition
        ################################
        
        return m
    
    def finalize_model(self):
        """
        Finalizing the TC Lab model. Here, we will set the 
        experimental conditions and discretize the dae model.
        
        """
        m = self.model
        
        ####################################
        # Set initial conditions
        if self.data.time[0] == 0:
            if self.data.TS1_data is not None and self.data.TS1_data[0] is not None:
                # Initialize with first temperature measurement
                m.Th1[0].fix(self.data.TS1_data[0])
                m.Ts1[0].fix(self.data.TS1_data[0])
            else:
                # Initialize with ambient temperature
                m.Th1[0].fix(m.Tamb)
                m.Ts1[0].fix(m.Tamb)

            if self.number_of_states == 4:
                if self.data.TS2_data is not None and self.data.TS2_data[0] is not None:
                    # Initialize with first temperature measurement
                    m.Th2[0].fix(self.data.TS2_data[0])
                    m.Ts2[0].fix(self.data.TS2_data[0])
                else:
                    # Initialize with ambient temperature
                    m.Th2[0].fix(m.Tamb)
                    m.Ts2[0].fix(m.Tamb)

        # End initial conditions definition
        ####################################
        
        ########################################
        # Defining optional sine wave equations
        # (only when sine wave control is used)
        
        if self.sine_amplitude is not None and self.sine_period is not None:
            # Add measurement control decision variables
            m.u1_period = pyo.Var(
                initialize=self.sine_period, bounds=(self.sine_period_min, self.sine_period_max)
            )  # minutes
            m.u1_amplitude = pyo.Var(initialize=self.sine_amplitude, bounds=(0, 50))  # % power
            
            # Fixed for parameter estimation
            m.u1_period.fix()
            m.u1_amplitude.fix()

            # Add constraint to calculate u1
            @m.Constraint(m.t)
            def u1_constraint(m, t):
                return m.U1[t] == 50 + m.u1_amplitude * sin(2 * np.pi / (m.u1_period * 60) * t)
            
            m.U1.unfix()  # Unfixed for because of above constraints
        
        # TODO: Add second sine wave functionality for 4-state model
        
        # End optional sine wave constraints
        ########################################
        
        #########################################
        # Initialize the model using integration
        m.var_input = pyo.Suffix(direction=pyo.Suffix.LOCAL)

        if self.data.u1 is not None:
            # initialize with data
            m.var_input[m.U1] = helper(self.data.u1, self.data.time)
        else:
            # otherwise initialize control decision of 0
            m.var_input[m.U1] = {0: 0}

        if self.number_of_states == 4:
            if self.data.u2 is not None:
                # initialize with data
                m.var_input[m.U2] = helper(self.data.u2, self.data.time)
            else:
                # otherwise initialize control decision of 0
                m.var_input[m.U2] = {0: 0}

        # Simulate to initialize
        # Makes the solver more efficient
        if self.sine_amplitude is None or self.sine_period is None:
            sim = Simulator(m, package='scipy')
            tsim, profiles = sim.simulate(
                numpoints=100, integrator='vode', varying_inputs=m.var_input
            )
            sim.initialize_model()
        else:
            # sim = Simulator(m, package='casadi')
            # tsim, profiles = sim.simulate(
            #     numpoints=100, integrator='idas', varying_inputs=m.var_input
            # )
            # sim.initialize_model()
            pass
        
        pyo.TransformationFactory('dae.finite_difference').apply_to(
            m, scheme='BACKWARD', nfe=len(self.data.time) - 1
        )
            
        # End dynamic model initialization
        #########################################

    
    def label_experiment(self):
        """
        Annotating (labeling) the model with experimental 
        data, associated measurement error, experimental 
        design decisions, and unknown model parameters.

        """
        m = self.model
        
        #################################
        # Labeling experiment outputs
        # (experiment measurements)
        
        m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        # Add sensor 1 temperature (m.Ts1) to experiment outputs
        m.experiment_outputs.update((m.Ts1[t], self.data.T1[ind]) for ind, t in enumerate(self.data.time))
        if self.include_Th:
            m.experiment_outputs.update((m.Th1[t], self.data.T1[ind]) for ind, t in enumerate(self.data.time))
        if self.number_of_states == 4:
            m.experiment_outputs.update((m.Ts2[t], self.data.T2[ind]) for ind, t in enumerate(self.data.time))
            if self.include_Th:
                m.experiment_outputs.update((m.Th2[t], self.data.T2[ind]) for ind, t in enumerate(self.data.time))
        
        
        # End experiment outputs
        #################################
        
        #################################
        # Labeling unknown parameters
        
        m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        # Add labels to all unknown parameters with nominal value as the value
        if not self.reparam:
            m.unknown_parameters.update((k, k.value) for k in [m.Ua, m.Ub, m.inv_CpH, m.inv_CpS])
            if self.number_of_states == 4:
                m.unknown_parameters[m.Uc] = m.Uc.value
        else:
        # REPARAM
            m.unknown_parameters.update((k, k.value) for k in [m.beta_1, m.beta_2, m.beta_3, m.beta_4])
            if self.number_of_states == 4:
                m.unknown_parameters[m.beta_5] = m.beta_5.value
        
        # End unknown parameters
        #################################
        
        #################################
        # Labeling experiment inputs
        # (experiment design decisions)
        
        m.experiment_inputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        # Add experimental input label for control variable (m.U1)
        if self.sine_amplitude is not None and self.sine_period is not None:
            m.experiment_inputs[m.u1_period] = None
            m.experiment_inputs[m.u1_amplitude] = None
        else:    
            m.experiment_inputs.update((m.U1[t], None) for t in self.data.time)
            if self.number_of_states == 4:
                m.experiment_inputs.update((m.U2[t], None) for t in self.data.time)
        
        # End experiment inputs
        #################################
        
        #################################
        # Labeling measurement error
        # (for experiment outputs)
        
        m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        # Add sensor 1 temperature (m.Ts1) measurement error (assuming constant error of 0.25 deg C)
        m.measurement_error.update((m.Ts1[t], 0.25) for t in self.data.time)
        if self.include_Th:
            m.measurement_error.update((m.Th1[t], 0.25) for t in self.data.time)
        if self.number_of_states == 4:
            m.measurement_error.update((m.Ts2[t], 0.25) for ind, t in enumerate(self.data.time))
            if self.include_Th:
                m.measurement_error.update((m.Th2[t], 0.25) for t in self.data.time)
        
        # End measurement error
        #################################
        

### -------------- Part 5: Extract and visualize results -------------- ###


def extract_results(model, name="Pyomo results", number_of_states=2):
    """Extract results from the Pyomo model"""

    time = np.array([pyo.value(t) for t in model.t])
    Th1 = np.array([pyo.value(model.Th1[t]) for t in model.t])
    Ts1 = np.array([pyo.value(model.Ts1[t]) for t in model.t])
    U1 = np.array([pyo.value(model.U1[t]) for t in model.t])
    P1 = pyo.value(model.P1)
    if not number_of_states == 4:
        Th2 = None
        Ts2 = None
        U2 = None
    else:
        Th2 = np.array([pyo.value(model.Th2[t]) for t in model.t])
        Ts2 = np.array([pyo.value(model.Ts2[t]) for t in model.t])
        U2 = np.array([pyo.value(model.U2[t]) for t in model.t])
    P2 = model.P2
    Tamb = model.Tamb

    return TC_Lab_data(name, time, Th1, U1, P1, Ts1, Th2, U2, P2, Ts2, Tamb)


def extract_plot_results(tc_exp_data, model, number_of_states=2):
    """Extract and plot the results of the Pyomo model

    Arguments:
    ----------
    tc_exp_data: experimental data, TC_Lab_data instance
    model: Pyomo model
    number_of_states: int, number of states, default: 2

    Returns:
    --------
    solution: solution from Pyomo model, extracted and stored in a TC_Lab_data instance

    """

    # For convenience, save in a shorter variable name
    if tc_exp_data is not None:
        exp = tc_exp_data
    else:
        exp = TC_Lab_data(
            None, None, None, None, None, None, None, None, None, None, None
        )

    mod = extract_results(model)

    # create figure
    plt.figure(figsize=(10, 6))

    # subplot 1: temperatures
    plt.subplot(2, 1, 1)

    colors = {
        'T1': 'orange',  # data
        'T2': 'green',  # data
        'Th1': 'red',  # model
        'Ts1': 'blue',  # model
        'Th2': 'purple',  # model
        'Ts2': 'brown',  # model
        'u1_data': 'orange',  # data
        'u2_data': 'green',  # data
        'u1_mod': 'red',  # model
        'u2_mod': 'purple',  # model
    }

    LW = 3.0  # line width

    four_states = (mod.TS2_data is not None) and (mod.T2 is not None)

    if exp.T1 is not None:
        plt.scatter(
            exp.time,
            exp.T1,
            marker='o',
            label="$T_{S,1}$ measured",
            alpha=0.5,
            color=colors["T1"],
        )
    if mod.TS1_data is not None:
        plt.plot(
            mod.time, mod.TS1_data, label="$T_{S,1}$ predicted", color=colors["Ts1"]
        )
    if mod.T1 is not None:
        plt.plot(
            mod.time,
            mod.T1,
            label="$T_{H,1}$ predicted",
            color=colors["Th1"],
            linestyle='--',
        )
    if exp.T2 is not None:
        plt.scatter(
            exp.time,
            exp.T2,
            marker='s',
            label="$T_{S,2}$ measured",
            alpha=0.5,
            color=colors["T2"],
        )
    if mod.TS2_data is not None:
        plt.plot(
            mod.time, mod.TS2_data, label="$T_{S,2}$ predicted", color=colors["Ts2"]
        )
    if mod.T2 is not None:
        plt.plot(
            mod.time,
            mod.T2,
            label="$T_{H,2}$ predicted",
            color=colors["Th2"],
            linestyle='--',
        )

    plt.ylabel('Temperature (°C)')

    if four_states:
        nc = 2
    else:
        nc = 1
    plt.legend(ncol=nc)
    plt.grid(True)

    # subplot 2: control decision
    plt.subplot(2, 1, 2)
    if exp.u1 is not None:
        plt.scatter(
            exp.time,
            exp.u1,
            marker='o',
            label="$u_1$ measured",
            color=colors['u1_data'],
            alpha=0.5,
        )
    if mod.u1 is not None:
        plt.plot(mod.time, mod.u1, label="$u_1$ predicted", color=colors["u1_mod"])
    if exp.u2 is not None:
        plt.scatter(
            exp.time,
            exp.u2,
            marker='s',
            label="$u_2$ measured",
            color=colors["u2_data"],
            alpha=0.5,
        )
    if mod.u2 is not None:
        plt.plot(mod.time, mod.u2, label="$u_2$ predicted", color=colors["u2_mod"])

    plt.ylabel('Heater Power (%)')
    plt.xlabel('Time (s)')
    plt.legend(ncol=nc)
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    print("Model parameters:")
    print("Ua =", round(pyo.value(model.Ua), 4), "Watts/degC")
    print("Ub =", round(pyo.value(model.Ub), 4), "Watts/degC")
    if number_of_states == 4:
        print("Uc =", round(pyo.value(model.Uc), 4), "Watts/degC")
    print("CpH =", round(1 / pyo.value(model.inv_CpH), 4), "Joules/degC")
    print("CpS =", round(1 / pyo.value(model.inv_CpS), 4), "Joules/degC")

    if hasattr(model, 'u1_period'):
        print("u1_period =", round(pyo.value(model.u1_period), 2), "minutes")
    if hasattr(model, 'u1_amplitude'):
        print("u1_amplitude =", round(pyo.value(model.u1_amplitude), 4), "% power")

    print(" ")  # New line

    return mod


def results_summary(result):
    eigenvalues, eigenvectors = np.linalg.eig(result)

    min_eig = min(eigenvalues)

    print("======Results Summary======")
    print("Four design criteria log10() value:")
    print("A-optimality:", np.log10(np.trace(result)))
    print("D-optimality:", np.log10(np.linalg.det(result)))
    print("E-optimality:", np.log10(min_eig))
    print("Modified E-optimality:", np.log10(np.linalg.cond(result)))
    print("\nFIM:\n", result)

    print("\neigenvalues:\n", eigenvalues)

    print("\neigenvectors:\n", eigenvectors)