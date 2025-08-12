import pyomo.environ as pyo
from matplotlib import pyplot as plt
from pyomo.network import Arc
import numpy as np
from scipy import stats
import pandas as pd
import pyomo.contrib.parmest.parmest as parmest
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
from idaes.core.util.model_statistics import (degrees_of_freedom, unfixed_variables_set, unfixed_variables_generator,
unfixed_variables_in_activated_equalities_set)
from idaes.core.util.initialization import fix_state_vars
from idaes.models.unit_models import MSContactor, MSContactorInitializer
from idaes.models.unit_models import Mixer, MixingType, MomentumMixingType, MixerInitializer
from idaes.core.util.initialization import propagate_state
from pyomo.dae import ContinuousSet, DerivativeVar, Simulator
from pyomo.contrib.parmest.experiment import Experiment
import pyomo.dae as dae
import logging
from idaes.core.util import DiagnosticsToolbox


# State Block class is used to define utility methods that can be applied to multiple State Block Data instances at one time
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

# State Block Data class contains the actual property variables and constraints and will be used in every unit model
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


# This class serves to define the property package in the flowsheet and a central location for storing global parameters related to properties
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
            units=pyo.units.kg / pyo.units.m ** 3,
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

# creating a function that evaluates and solves diafiltration membrane model
def membrane_model(Q_fd, Li_fd, Co_fd, Q_df, Li_df=0, Co_df=0):
    """
    Solves diafiltration membrane model

    Arguments:
        Q_fd: Fresh feed flowrate in m3/hr
        Li_fd: Lithium concentration in fresh feed in kg/m3
        Co_fd: Cobalt concentration in fresh feed in kg/m3
        Q_df: Diafiltrate flowrate in m3/hr
        Li_df: Lithium concentration in diafiltrate in kg/m3
        Co_df: Cobalt concentration in diafiltrate in kg/m3

    Returns:
        Lithium rich product flowrate in m3/hr and species concentration in kg/m3
        Cobalt rich product flowrate in m3/hr and species concentration in kg/m3
    """

    # building flowsheet
    m = pyo.ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    # adding properties
    m.fs.properties = LiCoParameters()

    # adding stage 1
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

    # defining global parameters
    J = 0.1 * pyo.units.m / pyo.units.hour
    w = 1.5 * pyo.units.m

    # adding flowsheet species
    m.fs.solutes = pyo.Set(initialize=["Li", "Co"])

    # adding species sieving coefficient
    m.fs.sieving_coefficient = pyo.Var(m.fs.solutes, units=pyo.units.dimensionless)
    m.fs.sieving_coefficient["Li"].fix(1.3)
    m.fs.sieving_coefficient["Co"].fix(0.5)

    # defining stage 1 length
    m.fs.stage1.length = pyo.Var(units=pyo.units.m)
    m.fs.stage1.length.fix(10)

    # defining solvent flux
    def solvent_rule(b, s):
        """
        Defines membrane solvent flux

        Argument:
            s: stage elements

        Returns:
            Solvent flux equation"""
        return (
                b.material_transfer_term[0, s, "permeate", "retentate", "solvent"]
                == J * b.length * w * m.fs.properties.dens_H2O / 10
        )

    # adding solvent flux constraint
    m.fs.stage1.solvent_flux = pyo.Constraint(
        m.fs.stage1.elements,
        rule=solvent_rule,
    )

    # defining solute flux
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

        return (
                pyo.log(b.retentate[0, s].conc_mass_solute[j])
                + (m.fs.sieving_coefficient[j] - 1)
                * pyo.log(in_state.flow_vol)
                == pyo.log(in_state.conc_mass_solute[j])
                + (m.fs.sieving_coefficient[j] - 1)
                * pyo.log(b.retentate[0, s].flow_vol)
        )

    # adding solute flux constraint
    m.fs.stage1.solute_sieving = pyo.Constraint(
        m.fs.stage1.elements,
        m.fs.solutes,
        rule=solute_rule,
    )

    # initial guess for stage 1 recycle stream (retentate of stage 3)
    m.fs.stage1.retentate_inlet.flow_vol[0].fix(Q_fd)
    m.fs.stage1.retentate_inlet.conc_mass_solute[0, "Li"].fix(Li_fd)
    m.fs.stage1.retentate_inlet.conc_mass_solute[0, "Co"].fix(Co_fd)

    # initializing stage 1
    initializer = MSContactorInitializer()
    initializer.initialize(m.fs.stage1)

    # adding stage 2
    m.fs.stage2 = MSContactor(
        number_of_finite_elements=10,
        streams={
            "retentate": {
                "property_package": m.fs.properties,
                "has_energy_balance": False,
                "has_pressure_balance": False
            },
            "permeate": {
                "property_package": m.fs.properties,
                "has_feed": False,
                "has_energy_balance": False,
                "has_pressure_balance": False
            }
        }
    )

    # definig stage 2 length
    m.fs.stage2.length = pyo.Var(units=pyo.units.m)
    m.fs.stage2.length.fix(10)

    # adding solvent and solute flux constraints
    m.fs.stage2.solvent_flux = pyo.Constraint(m.fs.stage2.elements, rule=solvent_rule)
    m.fs.stage2.solute_sieving = pyo.Constraint(m.fs.stage2.elements, m.fs.solutes, rule=solute_rule)

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

    # converting connections to contraints
    pyo.TransformationFactory("network.expand_arcs").apply_to(m)

    # unfixing stage 1 retentate inlet
    m.fs.stage1.retentate_inlet.flow_vol[0].unfix()
    m.fs.stage1.retentate_inlet.conc_mass_solute[0, "Li"].unfix()
    m.fs.stage1.retentate_inlet.conc_mass_solute[0, "Co"].unfix()

    # fixing mixer 1 inlet_1
    m.fs.mix1.inlet_1.flow_vol[0].fix(Q_fd)
    m.fs.mix1.inlet_1.conc_mass_solute[0, "Li"].fix(Li_fd)
    m.fs.mix1.inlet_1.conc_mass_solute[0, "Co"].fix(Co_fd)

    # copying stage 1 permeate outlet conditions to mixer 1 inlet_1
    propagate_state(
        destination=m.fs.mix1.inlet_2,
        source=m.fs.stage1.permeate_outlet,
    )

    # initializing mixer 1
    mix_initializer = MixerInitializer()
    mix_initializer.initialize(m.fs.mix1)

    # copying mixer 1 outlet conditions to stage 2 retentate inlet
    propagate_state(
        source=m.fs.mix1.outlet,
        destination=m.fs.stage2.retentate_inlet
    )

    # initializing stage 2
    initializer.initialize(m.fs.stage2)

    # adding stage 3
    m.fs.stage3 = MSContactor(
        number_of_finite_elements=10,
        streams={
            "retentate": {
                "property_package": m.fs.properties,
                "has_energy_balance": False,
                "has_pressure_balance": False,
                "side_streams": [10]
            },
            "permeate": {
                "property_package": m.fs.properties,
                "has_feed": False,
                "has_energy_balance": False,
                "has_pressure_balance": False
            }
        }
    )

    assert isinstance(m.fs.stage3, MSContactor)  # check that stage3 exists and is an MSContactor

    # Retentate side checks
    assert hasattr(m.fs.stage3, "retentate_inlet_state")  # check that there is a retentate feed
    assert hasattr(m.fs.stage3, "retentate_side_stream_state")  # check that a side stream exists
    for k in m.fs.stage3.retentate_side_stream_state:
        assert k == (0, 10)  # check that the side stream only exists at element 10
    assert not hasattr(m.fs.stage3, "retentate_energy_balance")  # check that there are no energy balances
    assert not hasattr(m.fs.stage3, "retentate_pressure_balance")  # check that there are no pressure balances

    # Permeate side checks
    assert not hasattr(m.fs.stage3, "permeate_inlet_state")  # check that there is no permeate feed
    assert not hasattr(m.fs.stage3,
                       "permeate_side_stream_state")  # check that there are no side streams on permeate side
    assert not hasattr(m.fs.stage3, "permeate_energy_balance")  # check that there are no energy balances
    assert not hasattr(m.fs.stage3, "permeate_pressure_balance")  # check that there are no pressure balances

    # defining stage 3 length
    m.fs.stage3.length = pyo.Var(units=pyo.units.m)
    m.fs.stage3.length.fix(10)

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
        ) * pyo.log(q_in) == pyo.log(c_in) + (m.fs.sieving_coefficient[j] - 1) * pyo.log(
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
        momentum_mixing_type=MomentumMixingType.none
    )

    # adding a stream that connects stage 2 permeate and mixer 2 inlet_2
    m.fs.stream4 = Arc(
        source=m.fs.stage2.permeate_outlet,
        destination=m.fs.mix2.inlet_2
    )

    # adding a stream that connects mixer 2 outlet and stage 3 retentate inlet
    m.fs.stream5 = Arc(
        source=m.fs.mix2.outlet,
        destination=m.fs.stage3.retentate_inlet
    )

    # adding a stream that connects stage 3 retentate outlet and mixer 1 inlet_1
    m.fs.stream6 = Arc(
        source=m.fs.stage3.retentate_outlet,
        destination=m.fs.mix1.inlet_1
    )

    # converting connections to contraints
    pyo.TransformationFactory("network.expand_arcs").apply_to(m)

    # unfixing mixer 1 inlet_1
    m.fs.mix1.inlet_1.flow_vol[0].unfix()
    m.fs.mix1.inlet_1.conc_mass_solute[0, "Li"].unfix()
    m.fs.mix1.inlet_1.conc_mass_solute[0, "Co"].unfix()

    # adding and fixing diafiltrate stream to mixer 2 inlet_1
    m.fs.mix2.inlet_1.flow_vol[0].fix(Q_df)
    m.fs.mix2.inlet_1.conc_mass_solute[0, "Li"].fix(Li_df)
    m.fs.mix2.inlet_1.conc_mass_solute[0, "Co"].fix(Co_df)

    # fixing fresh feed conditions at element 10 of stage 3
    m.fs.stage3.retentate_side_stream_state[0, 10].flow_vol.fix(Q_fd)
    m.fs.stage3.retentate_side_stream_state[0, 10].conc_mass_solute["Li"].fix(Li_fd)
    m.fs.stage3.retentate_side_stream_state[0, 10].conc_mass_solute["Co"].fix(Co_fd)

    # copying stage 2 permeate outlet conditions to mixer 2 inlet_2
    propagate_state(
        source=m.fs.stage2.permeate_outlet,
        destination=m.fs.mix2.inlet_2
    )

    # initializing mixer 2
    mix_initializer.initialize(m.fs.mix2)

    # copying mixer 2 outlet conditions to stage 3 retentate inlet
    propagate_state(
        source=m.fs.mix2.outlet,
        destination=m.fs.stage3.retentate_inlet
    )

    # initializing stage 3
    initializer.initialize(m.fs.stage3)

    # solving model
    solver = pyo.SolverFactory("ipopt")
    solver.solve(m.fs)

    # displaying outlets result
    m.fs.stage3.permeate_outlet.display()
    m.fs.stage1.retentate_outlet.display()

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

    # Checking recovery values
    m.Co_recovery.display()
    m.Li_recovery.display()

    # Increase stage length and re-solve model
    m.fs.stage1.length.fix(756.4)
    m.fs.stage2.length.fix(756.4)
    m.fs.stage3.length.fix(756.4)

    solver.solve(m.fs)

    # checking updated recovery values
    m.Co_recovery.display()
    m.Li_recovery.display()

    return [pyo.value(m.fs.stage1.retentate_outlet.flow_vol[0]), pyo.value(m.fs.stage3.permeate_outlet.flow_vol[0]),
            pyo.value(m.fs.stage1.retentate_outlet.conc_mass_solute[0, "Co"]),
            pyo.value(m.fs.stage1.retentate_outlet.conc_mass_solute[0, "Li"]),
            pyo.value(m.fs.stage3.permeate_outlet.conc_mass_solute[0, "Li"]),
            pyo.value(m.fs.stage3.permeate_outlet.conc_mass_solute[0, "Co"])]