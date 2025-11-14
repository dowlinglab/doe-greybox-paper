import pyomo.environ as pyo
from matplotlib import pyplot as plt
from pyomo.network import Arc
import numpy as np
import pandas as pd
import logging
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


# Experiment class for the membrane cascade
class MembraneExperiment(Experiment):
    def __init__(self, data, theta=None):
        """
        Arguments
        ---------
        data: object containing vital experimental information
        temp: temperature in K
        pressure: applied pressure in Pa
        """
        self.data = data
        if theta is None:
            self.theta = {"fs.Lp":1e-7, "fs.constant_sieving_coeff[Li]":1.0, "fs.constant_sieving_coeff[Co]":0.4,
                                  "fs.ionic_strength_coeff[Li]":8e-4, "fs.ionic_strength_coeff[Co]":4e-4}
        else:
            self.theta = theta
        self.model = None

    def get_labeled_model(self):
        if self.model is None:
            self.create_model()
            self.finalize_model()
            self.label_experiment()

        return self.model

    # Create flexible model without data
    def create_model(self):
        """
        Creates pyomo model

        Returns:
            m: pyomo model
        """
        m = self.model = pyo.ConcreteModel()
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
        w = 1.5 * pyo.units.m
        R = 8.314 * pyo.units.J / (pyo.units.mol * pyo.units.K)

        # adding flowsheet species
        m.fs.solutes = pyo.Set(initialize=["Li", "Co"])

        # add the temperature
        m.fs.temperature = pyo.Var(units=pyo.units.K)
        m.fs.temperature.fix(298.15)

        # add the pressure drop
        m.fs.pressure_drop = pyo.Var(units=pyo.units.Pa)
        m.fs.pressure_drop.fix(1e6)

        # add the water permeability constant
        m.fs.Lp = pyo.Var(units=pyo.units.m / (pyo.units.hour * pyo.units.Pa))
        m.fs.Lp.fix(self.theta["fs.Lp"])

        # add the valency of the species
        m.fs.valency = pyo.Var(m.fs.solutes, units=pyo.units.dimensionless)
        m.fs.valency["Li"].fix(1)
        m.fs.valency["Co"].fix(2)

        # add the molar mass of the species
        m.fs.molar_mass = pyo.Var(m.fs.solutes, units=pyo.units.kg / pyo.units.mol)
        m.fs.molar_mass["Li"].fix(0.00694)
        m.fs.molar_mass["Co"].fix(0.05893)

        # add the constant sieving coefficient of the species
        m.fs.constant_sieving_coeff = pyo.Var(m.fs.solutes, units=pyo.units.dimensionless, bounds=(1e-8, None))
        m.fs.constant_sieving_coeff["Li"].fix(self.theta["fs.constant_sieving_coeff[Li]"])
        m.fs.constant_sieving_coeff["Co"].fix(self.theta["fs.constant_sieving_coeff[Co]"])

        # add the ionic strength coefficient of the species
        m.fs.ionic_strength_coeff = pyo.Var(m.fs.solutes, units=pyo.units.m ** 3 / pyo.units.mol, bounds=(1e-8, None))
        m.fs.ionic_strength_coeff["Li"].fix(self.theta["fs.ionic_strength_coeff[Li]"])
        m.fs.ionic_strength_coeff["Co"].fix(self.theta["fs.ionic_strength_coeff[Co]"])

        # defining stage 1 length
        m.fs.stage1.length = pyo.Var(units=pyo.units.m)
        m.fs.stage1.length.fix(200)

        # add the ionic strength of stage 1
        m.fs.stage1.ionic_strength = pyo.Var(m.fs.stage1.elements, units=pyo.units.mol / pyo.units.m ** 3)

        # define the ionic strength
        def ionic_strength_rule(b, s):
            """
            Calculates the ionic strength of the retentate
            entering each stage element

            Argument:
                s: stage elements

            Returns:
                Ionic strength equation"""
            if s == 1:
                in_state = b.retentate_inlet_state[0]
            else:
                sp = b.elements.prev(s)
                in_state = b.retentate[0, sp]

            return (
                    b.ionic_strength[s]
                    == (1 / 2) * sum((in_state.conc_mass_solute[j] / m.fs.molar_mass[j]) * m.fs.valency[j] ** 2
                                     for j in m.fs.solutes)
            )

        # add the ionic strength constraint
        m.fs.stage1.ionic_strength_eqtn = pyo.Constraint(
            m.fs.stage1.elements,
            rule=ionic_strength_rule,
        )

        # adding species sieving coefficient
        m.fs.stage1.sieving_coefficient = pyo.Var(m.fs.stage1.elements, m.fs.solutes,
                                                  units=pyo.units.dimensionless, bounds=(1e-8, None))

        # define the sieving coeffficient
        def sieving_coeff_rule(b, s, j):
            """
            Defines the sieving coefficient
             of solutes across each stage

            Argument:
                s: stage elements
                j: solutes

            Returns:
                Sieving coefficient equation"""
            if j == "Li":
                return (
                        b.sieving_coefficient[s, j]
                        == m.fs.constant_sieving_coeff[j] + m.fs.ionic_strength_coeff[j] * b.ionic_strength[s]
                )
            else:
                return (
                        b.sieving_coefficient[s, j]
                        == m.fs.constant_sieving_coeff[j] - m.fs.ionic_strength_coeff[j] * b.ionic_strength[s]
                )

        # add the sieving coefficient constraint
        m.fs.stage1.sieving_coeff_eqtn = pyo.Constraint(
            m.fs.stage1.elements,
            m.fs.solutes,
            rule=sieving_coeff_rule,
        )

        # add the osmotic pressure
        m.fs.stage1.osmotic_pressure = pyo.Var(m.fs.stage1.elements, units=pyo.units.Pa)

        # define the osmotic pressure
        def osmotic_pressure_rule(b, s):
            """
            Calculates the osmotic pressure drop
            in each stage element

            Argument:
                s: stage elements

            Returns:
                Osmotic pressure drop equation"""
            # calculate the sum of the retentate molar concentration
            c_ret = sum(
                b.retentate[0, s].conc_mass_solute[j] / m.fs.molar_mass[j]
                for j in m.fs.solutes
            )

            # calculate the sum of the permeate molar concentration
            c_perm = sum(
                b.sieving_coefficient[s, j] * b.retentate[0, s].conc_mass_solute[j] / m.fs.molar_mass[j]
                for j in m.fs.solutes
            )

            return (
                    b.osmotic_pressure[s]
                    == R * m.fs.temperature * (c_ret - c_perm)
            )

        # add the osmotic pressure constraint
        m.fs.stage1.osmotic_pressure_eqtn = pyo.Constraint(
            m.fs.stage1.elements,
            rule=osmotic_pressure_rule,
        )

        # add the water flux
        m.fs.stage1.water_flux = pyo.Var(m.fs.stage1.elements, units=pyo.units.m / pyo.units.hour, bounds=(1e-8, None))

        # define the water flux
        def water_flux_rule(b, s):
            """
            Calculates the water flux
            in each stage element

            Argument:
                s: stage elements

            Returns:
                Water flux equation"""
            return (
                    b.water_flux[s]
                    == m.fs.Lp * (m.fs.pressure_drop - b.osmotic_pressure[s])
            )

        # add the water flux constraint
        m.fs.stage1.water_flux_eqtn = pyo.Constraint(
            m.fs.stage1.elements,
            rule=water_flux_rule,
        )

        # defining solvent flux
        def solvent_rule(b, s):
            """
            Defines the membrane solvent flux

            Argument:
                s: stage elements

            Returns:
                Solvent flux equation"""
            return (
                    b.material_transfer_term[0, s, "permeate", "retentate", "solvent"]
                    == b.water_flux[s] * b.length * w * m.fs.properties.dens_H2O / 10
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

            # get the retentate concentration of the current and previous stage
            c_cur = b.retentate[0, s].conc_mass_solute[j]
            c_prev = in_state.conc_mass_solute[j]

            # calculate the length of the stage element
            delta_z = b.length / len(b.elements)

            return (
                    (c_cur - c_prev) / delta_z
                    == - (b.water_flux[s] * w * c_cur * (b.sieving_coefficient[s, j] - 1))
                    / b.retentate[0, s].flow_vol
            )

        # adding solute flux constraint
        m.fs.stage1.solute_sieving = pyo.Constraint(
            m.fs.stage1.elements,
            m.fs.solutes,
            rule=solute_rule,
        )

        # initial guess for stage 1 recycle stream (retentate of stage 3)
        m.fs.stage1.retentate_inlet.flow_vol[0].fix(self.data["Q_feed (m^3/hr)"])
        m.fs.stage1.retentate_inlet.conc_mass_solute[0, "Li"].fix(self.data["C_Li_feed (kg/m^3)"])
        m.fs.stage1.retentate_inlet.conc_mass_solute[0, "Co"].fix(self.data["C_Co_feed (kg/m^3)"])

        # initializing stage 1
        initializer = MSContactorInitializer()
        initializer.initialize(m.fs.stage1, output_level=logging.DEBUG)

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
        m.fs.stage2.length.fix(200)

        # add the ionic strength
        m.fs.stage2.ionic_strength = pyo.Var(m.fs.stage2.elements, units=pyo.units.mol / pyo.units.m ** 3)
        m.fs.stage2.ionic_strength_eqtn = pyo.Constraint(m.fs.stage2.elements, rule=ionic_strength_rule)

        # add the sieving coefficient
        m.fs.stage2.sieving_coefficient = pyo.Var(m.fs.stage2.elements, m.fs.solutes,
                                                  units=pyo.units.dimensionless, bounds=(1e-8, None))
        m.fs.stage2.sieving_coeff_eqtn = pyo.Constraint(m.fs.stage2.elements, m.fs.solutes, rule=sieving_coeff_rule)

        # add the osmotic pressure
        m.fs.stage2.osmotic_pressure = pyo.Var(m.fs.stage2.elements, units=pyo.units.Pa)
        m.fs.stage2.osmotic_pressure_eqtn = pyo.Constraint(m.fs.stage2.elements, rule=osmotic_pressure_rule)

        # add the water flux
        m.fs.stage2.water_flux = pyo.Var(m.fs.stage2.elements, units=pyo.units.m / pyo.units.hour, bounds=(1e-8, None))
        m.fs.stage2.water_flux_eqtn = pyo.Constraint(m.fs.stage2.elements, rule=water_flux_rule)

        # adding solvent flux constraint
        m.fs.stage2.solvent_flux = pyo.Constraint(m.fs.stage2.elements, rule=solvent_rule)

        # adding solvent and solute flux constraints
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
        m.fs.mix1.inlet_1.flow_vol[0].fix(self.data["Q_feed (m^3/hr)"])
        m.fs.mix1.inlet_1.conc_mass_solute[0, "Li"].fix(self.data["C_Li_feed (kg/m^3)"])
        m.fs.mix1.inlet_1.conc_mass_solute[0, "Co"].fix(self.data["C_Co_feed (kg/m^3)"])

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
        m.fs.stage3.length.fix(200)

        # add the ionic strength
        m.fs.stage3.ionic_strength = pyo.Var(m.fs.stage3.elements, units=pyo.units.mol / pyo.units.m ** 3)

        # define the ionic strength of stage 3
        def stage3_ionic_strength_rule(b, s):
            """
            Calculates the ionic strength of the retentate
            entering each stage element

            Argument:
                s: stage elements

            Returns:
                Ionic strength equation"""
            if s == 1:
                in_state = b.retentate_inlet_state[0]

                return (
                        b.ionic_strength[s]
                        == (1 / 2) * sum((in_state.conc_mass_solute[j] / m.fs.molar_mass[j]) * m.fs.valency[j] ** 2
                                         for j in m.fs.solutes)
                )
            elif s == 10:
                sp = b.elements.prev(s)
                in_state = b.retentate[0, sp]
                side_state = b.retentate_side_stream_state[0, 10]
                q_in = (
                        in_state.flow_vol + side_state.flow_vol
                )

                return (
                        b.ionic_strength[s]
                        == (1 / 2) * sum(((in_state.conc_mass_solute[j] * in_state.flow_vol
                                           + side_state.conc_mass_solute[j] * side_state.flow_vol
                                           ) / (q_in * m.fs.molar_mass[j])) * m.fs.valency[j] ** 2 for j in
                                         m.fs.solutes)
                )
            else:
                sp = b.elements.prev(s)
                in_state = b.retentate[0, sp]

                return (
                        b.ionic_strength[s]
                        == (1 / 2) * sum((in_state.conc_mass_solute[j] / m.fs.molar_mass[j]) * m.fs.valency[j] ** 2
                                         for j in m.fs.solutes)
                )

        # add the ionic strength constraint
        m.fs.stage3.ionic_strength_eqtn = pyo.Constraint(m.fs.stage3.elements, rule=stage3_ionic_strength_rule)

        # add the sieving coefficient
        m.fs.stage3.sieving_coefficient = pyo.Var(m.fs.stage3.elements, m.fs.solutes,
                                                  units=pyo.units.dimensionless, bounds=(1e-8, None))
        m.fs.stage3.sieving_coeff_eqtn = pyo.Constraint(m.fs.stage3.elements, m.fs.solutes, rule=sieving_coeff_rule)

        # add the osmotic pressure
        m.fs.stage3.osmotic_pressure = pyo.Var(m.fs.stage3.elements, units=pyo.units.Pa)
        m.fs.stage3.osmotic_pressure_eqtn = pyo.Constraint(m.fs.stage3.elements, rule=osmotic_pressure_rule)

        # add the water flux
        m.fs.stage3.water_flux = pyo.Var(m.fs.stage3.elements, units=pyo.units.m / pyo.units.hour, bounds=(1e-8, None))
        m.fs.stage3.water_flux_eqtn = pyo.Constraint(m.fs.stage3.elements, rule=water_flux_rule)

        # adding solvent flux constraint
        m.fs.stage3.solvent_flux = pyo.Constraint(m.fs.stage3.elements, rule=solvent_rule)

        # update the solute flux for stage 3
        def stage3_solute_rule(b, s, j):
            """
            Defines stage 3 solute flux

            Arguments:
                s: stage elements
                j: solutes

            Returns:
                Solute flux equation"""
            if s == 1:
                in_state = b.retentate_inlet_state[0]
                c_prev = in_state.conc_mass_solute[j]
            elif s == 10:
                sp = b.elements.prev(s)
                in_state = b.retentate[0, sp]
                side_state = b.retentate_side_stream_state[0, 10]
                q_in = (
                        in_state.flow_vol + side_state.flow_vol
                )
                c_prev = (
                                 in_state.conc_mass_solute[j] * in_state.flow_vol
                                 + side_state.conc_mass_solute[j] * side_state.flow_vol
                         ) / q_in
            else:
                sp = b.elements.prev(s)
                in_state = b.retentate[0, sp]
                c_prev = in_state.conc_mass_solute[j]

            # get the retentate concentration of the current and previous stage
            c_cur = b.retentate[0, s].conc_mass_solute[j]

            # calculate the length of the stage element
            delta_z = b.length / len(b.elements)

            return (
                    (c_cur - c_prev) / delta_z
                    == - (b.water_flux[s] * w * c_cur * (b.sieving_coefficient[s, j] - 1))
                    / b.retentate[0, s].flow_vol
            )

        # adding solute flux constraint
        m.fs.stage3.solute_sieving = pyo.Constraint(m.fs.stage3.elements, m.fs.solutes, rule=stage3_solute_rule)

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
        m.fs.mix2.inlet_1.flow_vol[0].fix(self.data["Q_diaf (m^3/hr)"])
        m.fs.mix2.inlet_1.flow_vol[0].setub(27)
        m.fs.mix2.inlet_1.flow_vol[0].setub(33)
        m.fs.mix2.inlet_1.conc_mass_solute[0, "Li"].fix(0)
        m.fs.mix2.inlet_1.conc_mass_solute[0, "Co"].fix(0)

        # fixing fresh feed conditions at element 10 of stage 3
        m.fs.stage3.retentate_side_stream_state[0, 10].flow_vol.fix(self.data["Q_feed (m^3/hr)"])
        m.fs.stage3.retentate_side_stream_state[0, 10].flow_vol.setlb(90)
        m.fs.stage3.retentate_side_stream_state[0, 10].flow_vol.setub(110)
        m.fs.stage3.retentate_side_stream_state[0, 10].conc_mass_solute["Li"].fix(self.data["C_Li_feed (kg/m^3)"])
        m.fs.stage3.retentate_side_stream_state[0, 10].conc_mass_solute["Co"].fix(self.data["C_Co_feed (kg/m^3)"])

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

        return m

    def finalize_model(self):
        """
        Specifies experiment inputs and measured variables

        Returns:
            m: pyomo model
        """
        m = self.model

        # Set experimental design bounds
        m.fs.stage3.retentate_side_stream_state[0, 10].flow_vol.setlb(90)
        m.fs.stage3.retentate_side_stream_state[0, 10].flow_vol.setub(110)
        m.fs.mix2.inlet_1.flow_vol[0].setlb(27)
        m.fs.mix2_inlet_flow_LB_con = pyo.Constraint(expr=m.fs.mix2.inlet_1.flow_vol[0] >= 27)
        m.fs.mix2.inlet_1.flow_vol[0].setub(33)

        # Set other experimental design bounds (concentrations)
        m.fs.stage3.retentate_side_stream_state[0, 10].conc_mass_solute["Li"].setlb(1.5)
        m.fs.stage3.retentate_side_stream_state[0, 10].conc_mass_solute["Li"].setub(2.0)
        m.fs.stage3.retentate_side_stream_state[0, 10].conc_mass_solute["Co"].setlb(15)
        m.fs.stage3.retentate_side_stream_state[0, 10].conc_mass_solute["Co"].setub(20)

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

        return m

    def label_experiment(self):
        """
        Labels the model with a full experiment

        Returns:
            m: pyomo model
        """
        m = self.model

        # Set input labels
        m.experiment_inputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_inputs.update(
            [
                (m.fs.stage3.retentate_side_stream_state[0, 10].flow_vol,
                 self.data["Q_feed (m^3/hr)"]),
                (m.fs.stage3.retentate_side_stream_state[0, 10].conc_mass_solute["Li"],
                 self.data["C_Li_feed (kg/m^3)"]),
                (m.fs.stage3.retentate_side_stream_state[0, 10].conc_mass_solute["Co"],
                 self.data["C_Co_feed (kg/m^3)"]),
                (m.fs.mix2.inlet_1.flow_vol[0], self.data["Q_diaf (m^3/hr)"]),
            ]
        )

        # Set measurement labels (y/experiment data)
        m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_outputs.update(
            [
                (m.fs.stage1.retentate_outlet.flow_vol[0],
                 self.data["Q_Co_prdt (m^3/hr)"]),
                (m.fs.stage3.permeate_outlet.flow_vol[0],
                 self.data["Q_Li_prdt (m^3/hr)"]),
                (m.fs.stage1.retentate_outlet.conc_mass_solute[0, "Co"],
                 self.data["C_Co_Co_prdt (kg/m^3)"]),
                (m.fs.stage1.retentate_outlet.conc_mass_solute[0, "Li"],
                 self.data["C_Li_Co_prdt (kg/m^3)"]),
                (m.fs.stage3.permeate_outlet.conc_mass_solute[0, "Li"],
                 self.data["C_Li_Li_prdt (kg/m^3)"]),
                (m.fs.stage3.permeate_outlet.conc_mass_solute[0, "Co"],
                 self.data["C_Co_Li_prdt (kg/m^3)"]),
            ]
        )

        # Adding error for measurement values (assuming no covariance and constant error for all measurements)
        m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        flowrate_std_dev = 2
        conc_std_dev = 0.1
        # Add measurement error for Co_Co_prdt
        m.measurement_error.update([(m.fs.stage1.retentate_outlet.conc_mass_solute[0, "Co"], conc_std_dev)])
        # Add measurement error for Li_Co_prdt
        m.measurement_error.update([(m.fs.stage1.retentate_outlet.conc_mass_solute[0, "Li"], conc_std_dev)])
        # Add measurement error for Li_Li_prdt
        m.measurement_error.update([(m.fs.stage3.permeate_outlet.conc_mass_solute[0, "Li"], conc_std_dev)])
        # Add measurement error for Co_Li_prdt
        m.measurement_error.update([(m.fs.stage3.permeate_outlet.conc_mass_solute[0, "Co"], conc_std_dev)])
        # Add measurement error for Q_Co_prdt
        m.measurement_error.update([(m.fs.stage1.retentate_outlet.flow_vol[0], flowrate_std_dev)])
        # Add measurement error for Q_Li_prdt
        m.measurement_error.update([(m.fs.stage3.permeate_outlet.flow_vol[0], flowrate_std_dev)])

        # Add unknown parameter labels
        m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        # Add labels to all unknown parameters with nominal value as the value
        m.unknown_parameters.update((k, k.value) for k in [m.fs.Lp,
                                                           m.fs.constant_sieving_coeff["Li"],
                                                           m.fs.constant_sieving_coeff["Co"],
                                                           m.fs.ionic_strength_coeff["Li"],
                                                           m.fs.ionic_strength_coeff["Co"]
        ])

        return m


# function that solves the parameter estimation problem
def membrane_parameter_estimation():
    """
    solves the parameter estimation problem using ParmEst

    Returns:
        theta: dictionary of the estimated parameters
        cov: pd.DataFrame of the covariance matrix
    """

    # get the data
    data = pd.read_csv(".\membrane_cascade_data.csv")

    # Generating a list of experiments
    exp_list = []

    # number of experiments
    n_exp = len(data)

    for i in range(n_exp):  # loop through all experiments
        new_exp = MembraneExperiment(data.loc[i, :])
        exp_list.append(new_exp)

    # create parmest Estimator object
    pest = parmest.Estimator(exp_list, obj_function='SSE_weighted', tee=True)

    # estimate the parameters
    obj, theta = pest.theta_est()

    # calculate the covariance matrix
    cov = pest.cov_est()

    print("The estimated parameters are:", theta)
    print("Covariance matrix:\n", cov)

    return theta, cov
