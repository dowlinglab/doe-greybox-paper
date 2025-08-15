# Pyomo imports
import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.dae import ContinuousSet, DerivativeVar, Simulator
from pyomo.network import Arc

import pyomo.contrib.parmest.parmest as parmest
from pyomo.contrib.parmest.experiment import Experiment

# Modeling needs from IDAES
from idaes.core import (
    Component,
    declare_process_block_class,
    FlowsheetBlock,
    MaterialBalanceType,
    MaterialFlowBasis,
    Phase,
    PhysicalParameterBlock,
    StateBlock,
    StateBlockData,
)

from idaes.core.util.model_statistics import (
    degrees_of_freedom,
    unfixed_variables_set,
    unfixed_variables_generator,
    unfixed_variables_in_activated_equalities_set,
)
from idaes.core.util.initialization import fix_state_vars, propagate_state
from idaes.models.unit_models import (
    MSContactor,
    MSContactorInitializer,
    Mixer,
    MixingType,
    MomentumMixingType,
    MixerInitializer,
)
from pyomo.dae import ContinuousSet, DerivativeVar, Simulator
from pyomo.contrib.parmest.experiment import Experiment
import pyomo.dae as dae

from idaes.core.util import DiagnosticsToolbox

# Other packages
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd

import logging


############################################
# Begin IDAES required model components


# State Block class is used to
# define utility methods that can
# be applied to multiple State
# Block Data instances at one time
class _StateBlock(StateBlock):
    def fix_initialization_states(self):
        """
        Fixes state variables for state blocks.

        Returns:
            None
        """
        fix_state_vars(self)

    def initialization_routine(self):
        pass


# State Block Data class contains
# the actual property variables
# and constraints and will be used
# in every unit model
@declare_process_block_class("LiCoStateBlock", block_class=_StateBlock)
class LiCoStateBlock1Data(StateBlockData):
    def build(self):
        super().build()

        self.flow_vol = pyo.Var(
            units=pyo.units.m**3 / pyo.units.hour,
            bounds=(1e-8, None),
        )
        self.conc_mass_solute = pyo.Var(
            ["Li", "Co"],
            units=pyo.units.kg / pyo.units.m**3,
            bounds=(1e-8, None),
        )

    def get_material_flow_terms(self, p, j):
        if j == "solvent":
            # Assume constant density of pure water
            return self.flow_vol * self.params.dens_H2O
        else:
            return self.flow_vol * self.conc_mass_solute[j]

    def get_material_flow_basis(self):
        return MaterialFlowBasis.mass

    def define_state_vars(self):
        return {
            "flow_vol": self.flow_vol,
            "conc_mass_solute": self.conc_mass_solute,
        }


# This class serves to define the
# property package in the flowsheet
# and a central location for storing
# global parameters related to properties
@declare_process_block_class("LiCoParameters")
class LiCoParameterData(PhysicalParameterBlock):
    def build(self):
        super().build()

        self.phase1 = Phase()

        self.solvent = Component()
        self.Li = Component()
        self.Co = Component()

        self.dens_H2O = pyo.Param(
            default=1000,
            units=pyo.units.kg / pyo.units.m**3,
        )

        self._state_block_class = LiCoStateBlock

    @classmethod
    def define_metadata(cls, obj):
        obj.add_default_units(
            {
                "time": pyo.units.hour,
                "length": pyo.units.m,
                "mass": pyo.units.kg,
                "amount": pyo.units.mol,
                "temperature": pyo.units.K,
            }
        )


# End IDAES required model components
############################################


############################################
# Begin Experiment Class for membrane model
class MSContactorExperiment(Experiment):
    def __init__(self, data=None, nfe=10):
        """
        Arguments
        ---------
        data: dict, object containing vital experimental information
        nfe: int, number of finite elements, default 10
        """
        if data is None:
            self.data = {}
            self.data["Q_feed"] = 100.0
            self.data["C_Li_feed"] = 1.7
            self.data["C_Co_feed"] = 17
            self.data["Q_diafiltrate"] = 30.0
            self.data["C_Li_diafiltrate"] = 0.1
            self.data["C_Co_diafiltrate"] = 0.2
        else:
            self.data = data
        self.nfe = nfe
        self.model = None

        #############################
        # End constructor definition

    def get_labeled_model(self):
        if self.model is None:
            self.create_model()
            self.finalize_model()
            self.label_experiment()
        return self.model

    def create_model(self):
        """
        Builds a 3-stage MS Contactor model.
        TODO: Make this n-stage?
        TODO: Should we only do 1 stage?
        """
        # Building flowsheet
        self.model = m = pyo.ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)

        # Adding properties from IDAES components
        # from code above
        m.fs.properties = LiCoParameters()

        # Adding stage 1
        ########################
        m.fs.stage1 = MSContactor(
            number_of_finite_elements=10,
            streams={
                "retentate": {
                    "property_package": m.fs.properties,
                    "has_energy_balance": False,
                    "has_pressure_balance": False,
                },
                "permeate": {
                    "property_package": m.fs.properties,
                    "has_feed": False,
                    "has_energy_balance": False,
                    "has_pressure_balance": False,
                },
            },
        )

        # Defining global variables
        m.J = 0.1 * pyo.units.m / pyo.units.hour  # Flux across membrane
        m.w = 1.5 * pyo.units.m  # Width of the membranes in the stage

        # Add participating species
        m.fs.solutes = pyo.Set(initialize=["Li", "Co"])

        # Add sieving coefficient values
        m.fs.sieving_coefficient = pyo.Var(m.fs.solutes, units=pyo.units.dimensionless)
        m.fs.sieving_coefficient["Li"].fix(1.3)  # TODO: Add as input at constructor?
        m.fs.sieving_coefficient["Co"].fix(0.5)  # TODO: Add as input at constructor?

        # Define stage 1 length
        m.fs.stage1.length = pyo.Var(units=pyo.units.m)
        m.fs.stage1.length.fix(10)  # TODO: Make stage length(s) an input?

        # Defining solvent flux
        def solvent_rule(b, s):
            """
            Defines membrane solvent flux

            Argument:
                s: stage elements

            Returns:
                Solvent flux equation
            """
            return (
                b.material_transfer_term[0, s, "permeate", "retentate", "solvent"]
                == m.J * b.length * m.w * m.fs.properties.dens_H2O / 10
            )

        # Adding solvent flux constraint
        m.fs.stage1.solvent_flux = pyo.Constraint(
            m.fs.stage1.elements,
            rule=solvent_rule,
        )

        # Defining solute flux
        def solute_rule(b, s, j):
            """
            Defines membrane solute flux

            Arguments:
                s: stage elements
                j: solutes

            Returns:
                Solute flux equation"""
            if s == 1:
                in_state = b.retentate_inlet_state[0]
            else:
                sp = b.elements.prev(s)
                in_state = b.retentate[0, sp]

            return pyo.log(b.retentate[0, s].conc_mass_solute[j]) + (
                m.fs.sieving_coefficient[j] - 1
            ) * pyo.log(in_state.flow_vol) == pyo.log(in_state.conc_mass_solute[j]) + (
                m.fs.sieving_coefficient[j] - 1
            ) * pyo.log(
                b.retentate[0, s].flow_vol
            )

        # Adding solute flux constraint
        m.fs.stage1.solute_sieving = pyo.Constraint(
            m.fs.stage1.elements,
            m.fs.solutes,
            rule=solute_rule,
        )

        # initial guess for stage 1 recycle stream (retentate of stage 3)
        m.fs.stage1.retentate_inlet.flow_vol[0].fix(self.data["Q_feed"])
        m.fs.stage1.retentate_inlet.conc_mass_solute[0, "Li"].fix(self.data["C_Li_feed"])
        m.fs.stage1.retentate_inlet.conc_mass_solute[0, "Co"].fix(self.data["C_Co_feed"])

        ########################
        # Finish stage 1 addition

        # Initializing stage 1
        initializer = MSContactorInitializer()
        initializer.initialize(m.fs.stage1)

        # Adding stage 2
        ########################
        m.fs.stage2 = MSContactor(
            number_of_finite_elements=10,
            streams={
                "retentate": {
                    "property_package": m.fs.properties,
                    "has_energy_balance": False,
                    "has_pressure_balance": False,
                },
                "permeate": {
                    "property_package": m.fs.properties,
                    "has_feed": False,
                    "has_energy_balance": False,
                    "has_pressure_balance": False,
                },
            },
        )

        # Define stage 2 length
        m.fs.stage2.length = pyo.Var(units=pyo.units.m)
        m.fs.stage2.length.fix(10)  # TODO: Make stage length(s) an input?

        # Adding solvent and solute flux constraints
        m.fs.stage2.solvent_flux = pyo.Constraint(m.fs.stage2.elements, rule=solvent_rule)
        m.fs.stage2.solute_sieving = pyo.Constraint(
            m.fs.stage2.elements, m.fs.solutes, rule=solute_rule
        )

        # mixer for permeate of stage 1 and retentate of stage 2
        m.fs.mix1 = Mixer(
            num_inlets=2,
            property_package=m.fs.properties,
            material_balance_type=MaterialBalanceType.componentTotal,
            energy_mixing_type=MixingType.none,
            momentum_mixing_type=MomentumMixingType.none,
        )

        # adding a stream that connects stage 1 permeate and mixer 1 inlet_2
        m.fs.stream1 = Arc(
            source=m.fs.stage1.permeate_outlet,
            destination=m.fs.mix1.inlet_2,
        )

        # adding a stream that connects mixer 1 outlet and stage 2 retentate inlet
        m.fs.stream2 = Arc(
            source=m.fs.mix1.outlet,
            destination=m.fs.stage2.retentate_inlet,
        )

        # adding a stream that connects stage 2 retentate outlet and stage 1 retentate inlet
        m.fs.stream3 = Arc(
            source=m.fs.stage2.retentate_outlet,
            destination=m.fs.stage1.retentate_inlet,
        )

        # Converting connections to constraints
        pyo.TransformationFactory("network.expand_arcs").apply_to(m)

        # unfixing stage 1 retentate inlet
        m.fs.stage1.retentate_inlet.flow_vol[0].unfix()
        m.fs.stage1.retentate_inlet.conc_mass_solute[0, "Li"].unfix()
        m.fs.stage1.retentate_inlet.conc_mass_solute[0, "Co"].unfix()

        # fixing mixer 1 inlet_1
        m.fs.mix1.inlet_1.flow_vol[0].fix(self.data["Q_feed"])
        m.fs.mix1.inlet_1.conc_mass_solute[0, "Li"].fix(self.data["C_Li_feed"])
        m.fs.mix1.inlet_1.conc_mass_solute[0, "Co"].fix(self.data["C_Co_feed"])

        # copying stage 1 permeate outlet conditions to mixer 1 inlet_1
        propagate_state(
            destination=m.fs.mix1.inlet_2,
            source=m.fs.stage1.permeate_outlet,
        )

        # initializing mixer 1
        mix_initializer = MixerInitializer()
        mix_initializer.initialize(m.fs.mix1)

        # copying mixer 1 outlet conditions to stage 2 retentate inlet
        propagate_state(source=m.fs.mix1.outlet, destination=m.fs.stage2.retentate_inlet)

        ########################
        # Finish stage 2 addition

        # Initializing stage 2
        initializer.initialize(m.fs.stage2)

        # Adding stage 3
        ########################
        m.fs.stage3 = MSContactor(
            number_of_finite_elements=10,
            streams={
                "retentate": {
                    "property_package": m.fs.properties,
                    "has_energy_balance": False,
                    "has_pressure_balance": False,
                    "side_streams": [10],
                },
                "permeate": {
                    "property_package": m.fs.properties,
                    "has_feed": False,
                    "has_energy_balance": False,
                    "has_pressure_balance": False,
                },
            },
        )

        # Define stage 3 length
        m.fs.stage3.length = pyo.Var(units=pyo.units.m)
        m.fs.stage3.length.fix(10)  # TODO: Make stage length(s) an input?

        # updating solute flux for stage 3
        def stage3_solute_rule(b, s, j):
            """
            Defines stage 3 solute flux

            Arguments:
                s: stage elements
                j: solutes

            Returns:
                Solute flux equation"""
            if s == 1:
                q_in = b.retentate_inlet_state[0].flow_vol
                c_in = b.retentate_inlet_state[0].conc_mass_solute[j]
            elif s == 10:
                sp = b.elements.prev(s)
                q_in = (
                    b.retentate[0, sp].flow_vol
                    + b.retentate_side_stream_state[0, 10].flow_vol
                )
                c_in = (
                    b.retentate[0, sp].conc_mass_solute[j] * b.retentate[0, sp].flow_vol
                    + b.retentate_side_stream_state[0, 10].conc_mass_solute[j]
                    * b.retentate_side_stream_state[0, 10].flow_vol
                ) / q_in
            else:
                sp = b.elements.prev(s)
                q_in = b.retentate[0, sp].flow_vol
                c_in = b.retentate[0, sp].conc_mass_solute[j]

            return pyo.log(b.retentate[0, s].conc_mass_solute[j]) + (
                m.fs.sieving_coefficient[j] - 1
            ) * pyo.log(q_in) == pyo.log(c_in) + (
                m.fs.sieving_coefficient[j] - 1
            ) * pyo.log(
                b.retentate[0, s].flow_vol
            )

        # adding solvent flux constraint
        m.fs.stage3.solvent_flux = pyo.Constraint(
            m.fs.stage3.elements,
            rule=solvent_rule,
        )

        # adding solute flux constraint
        m.fs.stage3.solute_sieving = pyo.Constraint(
            m.fs.stage3.elements,
            m.fs.solutes,
            rule=stage3_solute_rule,
        )

        # mixer 2 for stage 2 permeate outlet and stage 3 retentate outlet
        m.fs.mix2 = Mixer(
            num_inlets=2,
            property_package=m.fs.properties,
            material_balance_type=MaterialBalanceType.componentTotal,
            energy_mixing_type=MixingType.none,
            momentum_mixing_type=MomentumMixingType.none,
        )

        # adding a stream that connects stage 2 permeate and mixer 2 inlet_2
        m.fs.stream4 = Arc(
            source=m.fs.stage2.permeate_outlet, destination=m.fs.mix2.inlet_2
        )

        # adding a stream that connects mixer 2 outlet and stage 3 retentate inlet
        m.fs.stream5 = Arc(source=m.fs.mix2.outlet, destination=m.fs.stage3.retentate_inlet)

        # adding a stream that connects stage 3 retentate outlet and mixer 1 inlet_1
        m.fs.stream6 = Arc(
            source=m.fs.stage3.retentate_outlet, destination=m.fs.mix1.inlet_1
        )

        # converting connections to contraints
        pyo.TransformationFactory("network.expand_arcs").apply_to(m)

        # unfixing mixer 1 inlet_1
        m.fs.mix1.inlet_1.flow_vol[0].unfix()
        m.fs.mix1.inlet_1.conc_mass_solute[0, "Li"].unfix()
        m.fs.mix1.inlet_1.conc_mass_solute[0, "Co"].unfix()

        # adding and fixing diafiltrate stream to mixer 2 inlet_1
        m.fs.mix2.inlet_1.flow_vol[0].fix(self.data["Q_diafiltrate"])
        m.fs.mix2.inlet_1.flow_vol[0].setub(1e-3)
        m.fs.mix2.inlet_1.flow_vol[0].setub(30)
        m.fs.mix2.inlet_1.conc_mass_solute[0, "Li"].fix(self.data["C_Li_diafiltrate"])
        m.fs.mix2.inlet_1.conc_mass_solute[0, "Co"].fix(self.data["C_Co_diafiltrate"])

        # fixing fresh feed conditions at element 10 of stage 3
        m.fs.stage3.retentate_side_stream_state[0, 10].flow_vol.fix(self.data["Q_feed"])
        m.fs.stage3.retentate_side_stream_state[0, 10].flow_vol.setlb(1e-3)
        m.fs.stage3.retentate_side_stream_state[0, 10].flow_vol.setub(100)
        m.fs.stage3.retentate_side_stream_state[0, 10].conc_mass_solute["Li"].fix(self.data["C_Li_feed"])
        m.fs.stage3.retentate_side_stream_state[0, 10].conc_mass_solute["Co"].fix(self.data["C_Co_feed"])

        # copying stage 2 permeate outlet conditions to mixer 2 inlet_2
        propagate_state(source=m.fs.stage2.permeate_outlet, destination=m.fs.mix2.inlet_2)

        # initializing mixer 2
        mix_initializer.initialize(m.fs.mix2)

        # copying mixer 2 outlet conditions to stage 3 retentate inlet
        propagate_state(source=m.fs.mix2.outlet, destination=m.fs.stage3.retentate_inlet)

        ########################
        # Finish stage 3 addition

        # Initializing stage 3
        initializer.initialize(m.fs.stage3)

        # Some useful expression for later analysis
        ########################
        # calculating lithium and cobalt recovery
        m.Li_recovery = pyo.Expression(
            expr=m.fs.stage3.permeate_outlet.flow_vol[0]
                 * m.fs.stage3.permeate_outlet.conc_mass_solute[0, "Li"]
                 / (
                         m.fs.mix2.inlet_1.flow_vol[0]
                         * m.fs.mix2.inlet_1.conc_mass_solute[0, "Li"]
                         + m.fs.stage3.retentate_side_stream_state[0, 10].flow_vol
                         * m.fs.stage3.retentate_side_stream_state[0, 10].conc_mass_solute["Li"]
                 )
        )

        m.Co_recovery = pyo.Expression(
            expr=m.fs.stage1.retentate_outlet.flow_vol[0]
                 * m.fs.stage1.retentate_outlet.conc_mass_solute[0, "Co"]
                 / (
                         m.fs.mix2.inlet_1.flow_vol[0]
                         * m.fs.mix2.inlet_1.conc_mass_solute[0, "Co"]
                         + m.fs.stage3.retentate_side_stream_state[0, 10].flow_vol
                         * m.fs.stage3.retentate_side_stream_state[0, 10].conc_mass_solute["Co"]
                 )
        )

    def finalize_model(self):
        """
        Finalize the model definition. Much of the initialization
        had to happen in the model build function, so this is
        simply a call on IPOPT to solve the square problem.
        """
        m = self.model

        # solving model
        solver = pyo.SolverFactory("ipopt")
        solver.solve(m.fs)

    def label_experiment(self):
        """
        Annotate the model with all pertinent
        information.
        """
        m = self.model

        # Set measurement labels
        m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        # Add retentate flow rate
        m.experiment_outputs[m.fs.stage1.retentate_outlet.flow_vol[0]] = None
        # Add retentate concentrations (Co and Li)
        m.experiment_outputs[m.fs.stage1.retentate_outlet.conc_mass_solute[0, "Co"]] = None
        m.experiment_outputs[m.fs.stage1.retentate_outlet.conc_mass_solute[0, "Li"]] = None
        # Add permeate concentrations (Co and Li)
        m.experiment_outputs[m.fs.stage3.permeate_outlet.conc_mass_solute[0, "Co"]] = None
        m.experiment_outputs[m.fs.stage3.permeate_outlet.conc_mass_solute[0, "Li"]] = None
        # Currently not adding permeate flow
        # Can be added by uncommenting following line:
        # m.experiment_outputs[m.fs.stage3.permeate_outlet.flow_vol[0]] = None

        # Adding error for measurement values (assuming no covariance and constant error for all measurements)
        m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        flow_error = 2e0  # m^3  (10 Liter error)
        concentration_error = 1e-1  # kg/m^3
        # Add retentate flow rate error
        m.measurement_error[m.fs.stage1.retentate_outlet.flow_vol[0]] = flow_error
        # Add retentate concentration errors (Co and Li)
        m.measurement_error[m.fs.stage1.retentate_outlet.conc_mass_solute[0, "Co"]] = concentration_error
        m.measurement_error[m.fs.stage1.retentate_outlet.conc_mass_solute[0, "Li"]] = concentration_error
        # Add permeate concentration errors (Co and Li)
        m.measurement_error[m.fs.stage3.permeate_outlet.conc_mass_solute[0, "Co"]] = concentration_error
        m.measurement_error[m.fs.stage3.permeate_outlet.conc_mass_solute[0, "Li"]] = concentration_error
        # Currently not adding permeate flow
        # Can be added by uncommenting following line:
        # m.measurement_error[m.fs.stage3.permeate_outlet.flow_vol[0]] = flow_error

        # Identify design variables (experiment inputs) for the model
        m.experiment_inputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        # Add feed flow rate
        m.experiment_inputs[m.fs.stage3.retentate_side_stream_state[0, 10].flow_vol] = None
        # Add feed concentrations (Co and Li)
        # m.experiment_inputs[m.fs.stage3.retentate_side_stream_state[0, 10].conc_mass_solute["Co"]] = None
        # m.experiment_inputs[m.fs.stage3.retentate_side_stream_state[0, 10].conc_mass_solute["Li"]] = None
        # Add diafiltrate flow rate
        m.experiment_inputs[m.fs.mix2.inlet_1.flow_vol[0]] = None
        # Add diafiltrate concentrations (Co and Li)
        # m.experiment_inputs[m.fs.mix2.inlet_1.conc_mass_solute[0, "Co"]] = None
        # m.experiment_inputs[m.fs.mix2.inlet_1.conc_mass_solute[0, "Li"]] = None

        # Add unknown parameter labels
        m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        # Add labels to all unknown parameters with nominal value as the value
        m.unknown_parameters.update((k, pyo.value(k)) for k in [m.fs.sieving_coefficient["Li"], m.fs.sieving_coefficient["Co"]])

        #########################
        # End model labeling