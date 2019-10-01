import os
import numpy as np
from opencmiss.iron import iron

class Fitting:
    """
    Fitting class
    """

    def __init__(self, instance=1, fitting_type='guass'):
        """
        Create a new instance of a fitting problem.
        """
        self.results_folder = './'

        # Fitting parameters
        self.num_iterations = 1
        self.tau = 0.01
        self.kappa = 0.005
        self.smoothing_parameters = ([
            self.tau,  # tau_1
            self.kappa,  # kappa_11
            self.tau,  # tau_2
            self.kappa,  # kappa_22
            self.kappa,  # kappa_12
            self.tau,  # tau_3
            self.kappa,  # kappa_33
            self.kappa,  # kappa_13
            self.kappa])  # kappa_23
        
        # Set fitting data default
        self.interpolation_type = iron.FieldInterpolationTypes.GAUSS_POINT_BASED
        
        # OpenCMISS user numbers
        self.data_field_user_num = 100 + instance
        self.dependent_field_user_num = 101 + instance
        self.material_field_user_num = 102 + instance
        self.equations_set_field_user_num = 103 + instance
        self.equations_set_user_num = 100 + instance
        self.problem_user_num = 100 + instance
        
        # Instances for setting up fitting problems
        self.data_field = iron.Field()
        self.dependent_field = iron.Field()
        self.material_field = iron.Field()
        self.equations_set_field = iron.Field()
        self.equations_set = iron.EquationsSet()
        self.problem = iron.Problem()

        if fitting_type == 'guass':
            self.specification = (
                [iron.EquationsSetClasses.FITTING,
                 iron.EquationsSetTypes.GAUSS_FITTING_EQUATION,
                 iron.EquationsSetSubtypes.GAUSS_POINT_FITTING,
                 iron.EquationsSetFittingSmoothingTypes.SOBOLEV_VALUE])
        elif fitting_type == 'data':
            self.specification = (
                [iron.EquationsSetClasses.FITTING,
                 iron.EquationsSetTypes.DATA_FITTING_EQUATION,
                 iron.EquationsSetSubtypes.DATA_POINT_FITTING,
                 iron.EquationsSetFittingSmoothingTypes.SOBOLEV_VALUE])
        else:
            raise ValueError('Specified fitting type not supported')
        self.problem_specification = ([iron.ProblemClasses.FITTING,
                                       iron.ProblemTypes.DATA_FITTING,
                                       iron.ProblemSubtypes.STATIC_FITTING])

    def set_data_position(
            self, xi=None, position=None, dataset_num=1, datapoint_ids=None):
        """
        Sets positions of data as mesh xi or data positions
        """

        if position is not None:
            # Create data points from given positions
            if datapoint_ids is None:
                datapoint_ids = np.arange()
            position = np.array(position)
            num_datapoints = position.shape[0]
            num_components = position.shape[1]
            if num_datapoints != len(datapoint_ids):
                raise ValueError('Mismatch in datapoint ids and positions')

            self.datapoints = iron.DataPoints()
            self.datapoints.CreateStart(
                dataset_num, self.region, num_datapoints)
            for idx, datapoint_id in enumerate(datapoint_ids):
                for component_idx, component in enumerate(
                        np.arange(num_components)+1):
                    self.datapoints.PositionSet(
                        idx, position[idx, component_idx])
                self.datapoints.CreateFinish()


    def set_results_folder(self, results_folder):
        """
        Sets results folder for exporting solutions
        """
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
        self.results_folder = results_folder

    def set_smoothing_parameters(self, smoothing_parameters):
        """
        Sets all smoothing parameters.
        """
        if len(smoothing_parameters) == 9:
            self.smoothing_parameters = smoothing_parameters
            self.update_material_field()
        else:
            raise ValueError('9 smoothing parameters required')

    def set_tau(self, tau):
        """
        Sets tau fitting parameters for all required field components.
        """
        self.tau = tau
        self.smoothing_parameters = ([
            self.tau,  # tau_1
            self.kappa,  # kappa_11
            self.tau,  # tau_2
            self.kappa,  # kappa_22
            self.kappa,  # kappa_12
            self.tau,  # tau_3
            self.kappa,  # kappa_33
            self.kappa,  # kappa_13
            self.kappa])  # kappa_23
        self.update_material_field()

    def set_kappa(self, kappa):
        """
        Sets kappa fitting parameters for all required field components.
        """
        self.kappa = kappa
        self.smoothing_parameters = ([
            self.tau,  # tau_1
            self.kappa,  # kappa_11
            self.tau,  # tau_2
            self.kappa,  # kappa_22
            self.kappa,  # kappa_12
            self.tau,  # tau_3
            self.kappa,  # kappa_33
            self.kappa,  # kappa_13
            self.kappa])  # kappa_23
        self.update_material_field()
        
    def set_region(self, region):
        """
        Sets region for fitting.
        """
        self.region = region
        
    def set_decomposition(self, decomposition):
        """
        Sets decomposition for fitting.
        """
        self.decomposition = decomposition
        
    def set_geometric_field(self, geometric_field):
        """
        Sets geometric field for fitting.
        """
        self.geometric_field = geometric_field

    def set_data(self, data, interpolation_type):
        """
        Sets up the field for a fitting problem.
        """
        self.interpolation_type = iron.FieldInterpolationTypes.GAUSS_POINT_BASED

    def setup_fields(
            self, scaling_type=iron.FieldScalingTypes.NONE, num_components=6):
        """
        Sets up the field for a fitting problem.
        """
        components = range(1, num_components + 1)
        mesh_component = 1

        # Setup data field
        variable_types = [iron.FieldVariableTypes.U, iron.FieldVariableTypes.V]
        variable_labels = ['GaussStress', 'GaussWeight']
        num_variables = len(variable_types)
        self.data_field.CreateStart(self.data_field_user_num, self.region)
        self.data_field.TypeSet(iron.FieldTypes.GENERAL)
        self.data_field.MeshDecompositionSet(self.decomposition)
        self.data_field.GeometricFieldSet(self.geometric_field)
        self.data_field.DependentTypeSet(iron.FieldDependentTypes.DEPENDENT)
        self.data_field.NumberOfVariablesSet(num_variables)
        self.data_field.VariableTypesSet(variable_types)
        for idx, variable_type in enumerate(variable_types):
            self.data_field.VariableLabelSet(variable_type,
                                             variable_labels[idx])
            self.data_field.NumberOfComponentsSet(
                variable_type, num_components)
            for component in components:
                self.data_field.ComponentMeshComponentSet(
                    variable_type, component, mesh_component)
                self.data_field.ComponentInterpolationSet(
                    variable_type, component,
                    iron.FieldInterpolationTypes.GAUSS_POINT_BASED)
        self.data_field.ScalingTypeSet(scaling_type)
        self.data_field.CreateFinish()

        # Initialise Gauss point weight field to 1.0
        variable_type = iron.FieldVariableTypes.V
        for component in components:
            self.data_field.ComponentValuesInitialiseDP(
                variable_type, iron.FieldParameterSetTypes.VALUES, component,
                1.0)

        # Setup fitting solution field
        variable_types = ([iron.FieldVariableTypes.U,
                           iron.FieldVariableTypes.DELUDELN])
        num_variables = len(variable_types)
        
        self.dependent_field.CreateStart(
            self.dependent_field_user_num, self.region)
        self.dependent_field.TypeSet(iron.FieldTypes.GENERAL)
        self.dependent_field.DependentTypeSet(
            iron.FieldDependentTypes.DEPENDENT)
        self.dependent_field.MeshDecompositionSet(self.decomposition)
        self.dependent_field.GeometricFieldSet(self.geometric_field)
        self.dependent_field.VariableLabelSet(
            iron.FieldVariableTypes.U, "FittedStress")
        self.dependent_field.NumberOfVariablesSet(num_variables)
        for variable_type in variable_types:
            self.dependent_field.NumberOfComponentsSet(
                variable_type, num_components)
            for component in components:
                self.dependent_field.ComponentMeshComponentSet(
                    variable_type, component, mesh_component)
        self.dependent_field.ScalingTypeSet(scaling_type)
        self.dependent_field.CreateFinish()

        # Set Sobolev smoothing parameters - kappa and tau
        self.num_smoothing_components = 9
        self.smoothing_components = range(1, self.num_smoothing_components + 1)
        self.material_field.CreateStart(
            self.material_field_user_num, self.region)
        self.material_field.TypeSet(iron.FieldTypes.MATERIAL)
        self.material_field.MeshDecompositionSet(self.decomposition)
        self.material_field.GeometricFieldSet(self.geometric_field)
        self.material_field.NumberOfComponentsSet(
            iron.FieldVariableTypes.U, self.num_smoothing_components)
        self.material_field.VariableLabelSet(
            iron.FieldVariableTypes.U, 'SmoothingParameters')
        self.material_field.ScalingTypeSet(scaling_type)
        self.material_field.CreateFinish()

        # Set smoothing parameters
        self.update_material_field()
        
    def update_material_field(self):
        """
        Updates the material field with a set of smoothing parameters.
        """
        for component_idx, component in enumerate(self.smoothing_components):
            self.material_field.ComponentValuesInitialiseDP(
                iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES,
                component, self.smoothing_parameters[component_idx])

    def setup_equations(self):
        """
        Setup the fitting equations.
        """
        self.equations_set.CreateStart(self.equations_set_user_num,
                                       self.region, self.geometric_field,
                                       self.specification,
                                       self.equations_set_field_user_num,
                                       self.equations_set_field)
        self.equations_set.CreateFinish()

        self.equations_set.DependentCreateStart(self.dependent_field_user_num,
                                                self.dependent_field)
        self.equations_set.DependentCreateFinish()

        self.equations_set.IndependentCreateStart(self.data_field_user_num,
                                                  self.data_field)
        self.equations_set.IndependentCreateFinish()

        self.equations_set.MaterialsCreateStart(self.material_field_user_num,
                                                self.material_field)
        self.equations_set.MaterialsCreateFinish()

        self.equations_set.DerivedCreateStart(
            self.data_field_user_num, self.data_field)
        self.equations_set.DerivedVariableSet(
            iron.EquationsSetDerivedTensorTypes.CAUCHY_STRESS,
            iron.FieldVariableTypes.U)
        self.equations_set.DerivedCreateFinish()

        self.equations = iron.Equations()
        self.equations_set.EquationsCreateStart(self.equations)
        self.equations.SparsityTypeSet(iron.EquationsSparsityTypes.SPARSE)
        self.equations.OutputTypeSet(iron.EquationsOutputTypes.NONE)
        self.equations_set.EquationsCreateFinish()

    def setup_problem(self):
        """
        Setup the fitting problem.
        """
        # Create fitting problem
        self.problem.CreateStart(
            self.problem_user_num, self.problem_specification)
        self.problem.CreateFinish()

        self.problem.ControlLoopCreateStart()
        self.problem.ControlLoopCreateFinish()

        self.solver = iron.Solver()
        self.problem.SolversCreateStart()
        self.problem.SolverGet(
            [iron.ControlLoopIdentifiers.NODE], 1, self.solver)
        # self.solver.OutputTypeSet(iron.SolverOutputTypes.NONE)
        self.solver.OutputTypeSet(iron.SolverOutputTypes.PROGRESS)
        # self.solver.LinearTypeSet(iron.LinearSolverTypes.DIRECT)
        # self.solver.LibraryTypeSet(iron.SolverLibraries.UMFPACK) # UMFPACK/SUPERLU
        self.solver.LinearTypeSet(iron.LinearSolverTypes.ITERATIVE)
        self.solver.LinearIterativeMaximumIterationsSet(5000)
        self.solver.LinearIterativeAbsoluteToleranceSet(1.0E-10)
        self.solver.LinearIterativeRelativeToleranceSet(1.0E-05)
        self.problem.SolversCreateFinish()

        self.solver = iron.Solver()
        self.solver_equations = iron.SolverEquations()
        self.problem.SolverEquationsCreateStart()
        self.problem.SolverGet(
            [iron.ControlLoopIdentifiers.NODE], 1, self.solver)
        self.solver.SolverEquationsGet(self.solver_equations)
        self.solver_equations.SparsityTypeSet(
            iron.SolverEquationsSparsityTypes.SPARSE)
        equationsSetIndex = self.solver_equations.EquationsSetAdd(
            self.equations_set)
        self.problem.SolverEquationsCreateFinish()

        self.boundary_conditions = iron.BoundaryConditions()
        self.solver_equations.BoundaryConditionsCreateStart(
            self.boundary_conditions)
        self.solver_equations.BoundaryConditionsCreateFinish()

    def solve(self):
        """
        Peform fitting.
        """
        # Solve the problem
        for iteration_num in range(1, self.num_iterations + 1):
            self.problem.Solve()
            self.export(iteration_num)

    def finalise(self):
        """
        Finalise the fitting problem.
        """
        self.problem.Finalise()

    def export(self, iteration_num):
        """
        Export the fitting solutions for the selected iteration number.
        """
        self.set_results_folder(self.results_folder)
        export_path = os.path.join(
            self.results_folder,
            "fitting_results_iteration_" + str(iteration_num))
    
        fields = iron.Fields()
        fields.AddField(self.geometric_field)
        fields.AddField(self.dependent_field)
        fields.NodesExport(export_path, "FORTRAN")
        fields.ElementsExport(export_path, "FORTRAN")
        fields.Finalise()
