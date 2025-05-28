import pyomo.environ as pyo

from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxModel, ExternalGreyBoxBlock

import warnings
try:
    import idaes
except:
    warnings.warn("Failed to import idaes. Ensure IPOPT is installed properly.")

class simpleExternalGreyBox(ExternalGreyBoxModel):
    def __init__(self, ):
        self._n_inputs = 1
    
    def input_names(self):
        # Cartesian product gives us matrix indices flattened in row-first format
        # Can use itertools.combinations(self._param_names, 2) with added
        # diagonal elements, or do double for loops if we switch to upper triangular
        # input_names_list = list(itertools.product(self._param_names, self._param_names))
        input_names_list = ['x_in', ]
        return input_names_list

    def output_names(self):
        # ToDo: add output name for the variable. This may have to be
        # an input from the user. Or it could depend on the usage of
        # the ObjectiveLib Enum object, which should have an associated
        # name for the objective function at all times.
        obj_name = 'x_out'
        return [obj_name, ]

    def set_input_values(self, input_values):
        # Set initial values to be flattened initial FIM (aligns with input names)
        np.copyto(self._input_values, input_values)
        # self._input_values = list(self.doe_object.fim_initial.flatten())

    def evaluate_equality_constraints(self):
        # ToDo: are there any objectives that will have constraints?
        return None

    def evaluate_outputs(self):
        obj_value = self._input_values
        return np.asarray([obj_value], dtype=np.float64)

    def finalize_block_construction(self, pyomo_block):
        # Set bounds on the inputs/outputs
        # Set initial values of the inputs/outputs
        # This will depend on the objective used
        pyomo_block.inputs["x_in"] = 1
        pyomo_block.inputs["x_in"].lb = 0
        
        pyomo_block.outputs["x_out"] = 1

    def evaluate_jacobian_equality_constraints(self):
        # ToDo: Do any objectives require constraints?

        # Returns coo_matrix of the correct shape
        return None

    def evaluate_jacobian_outputs(self):
        M_rows = [1,]
        M_cols = [1,]
        
        return coo_matrix(
            (1, (M_rows, M_cols)), shape=(1, 1)
        )

    # Beyond here is for Hessian information
    def set_equality_constraint_multipliers(self, eq_con_multiplier_values):
        # ToDo: Do any objectives require constraints?
        # Assert lengths match
        self._eq_con_mult_values = np.asarray(
            eq_con_multiplier_values, dtype=np.float64
        )

    def set_output_constraint_multipliers(self, output_con_multiplier_values):
        # ToDo: Do any objectives require constraints?
        # Assert length matches
        self._output_con_mult_values = np.asarray(
            output_con_multiplier_values, dtype=np.float64
        )

    def evaluate_hessian_equality_constraints(self):
        # Returns coo_matrix of the correct shape
        # No constraints so this returns `None`
        return None

    def evaluate_hessian_outputs(self, FIM=None):
        hess_rows = [1, ]
        hess_cols = [1, ]
        return coo_matrix(
            (0, (hess_rows, hess_cols)),
            shape=(self._n_inputs, self._n_inputs),
        )    

def create_trivial_problem():
    m = pyo.ConcreteModel()
    
    m.x = pyo.Var(initialize=1, bounds=(0, 10))
    
    trivial_gb_model = simpleExternalGreyBox()
    print("Made the EGB object.")
    m.egb_simple_block = ExternalGreyBoxBlock(external_model=trivial_gb_model)
    
    m.x_con = pyo.Constraint(expr=m.x - m.egb_simple_block.inputs["x_in"]==0)
    
    m.obj = pyo.Objective(expr=m.egb_simple_block.outputs["x_out"], sense=pyo.minimize)
    
    return m
    
# Try to solve a simple problem with MUMPS
try:
    m = create_trivial_problem()
    print("Made the problem.")
    solver = pyo.SolverFactory("cyipopt")
    solver.config.options["linear_solver"] = "mumps"
    solver.solve(m, tee=True)
    print("Test using MUMPS with IPOPT passed")
except:
    warnings.warn("FAILED - IPOPT solve with MUMPS")
    
# Try to solve a simple problem using ma57
try:
    m = create_trivial_problem()
    print("Made the problem.")
    solver = pyo.SolverFactory("cyipopt")
    solver.config.options["linear_solver"] = "ma57"
    solver.solve(m, tee=True)
    print("Test using MA57 with IPOPT passed")
except:
    warnings.warn("FAILED - IPOPT solve with MA57")