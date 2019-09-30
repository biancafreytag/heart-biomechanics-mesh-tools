import numpy as np

# Intialise OpenCMISS
from opencmiss.iron import iron

class Fitting:
    """
    Cantilever simluation class
    """

    def __init__(self):
        """
        Create a new cantilever simulation.
        """
        self.dimensions = np.array([30, 10, 10])  # Length, width, height (in mm)
        self.num_elements = [2, 2, 2]
        self.components = [1, 2, 3] # Geometric components
        self.dimension = 3 # 3D coordinates
        self.number_gauss_xi = 4 # Number of Gauss points used for quadrature
        self.num_load_increments = 1
        self.results_folder = './'
        self.numDataPointsPerFace = 6

        # Fitting parameters
        self.tau = 0.01
        self.kappa = 0.005

        # Get the number of computational nodes and this computational node number
        self.numberOfComputationalNodes = iron.ComputationalNumberOfNodesGet()
        self.computationalNodeNumber = iron.ComputationalNodeNumberGet()

        # OpenCMISS user numbers
        self.coordinateSystemUserNumber = 1
        self.regionUserNumber = 1
        self.basisUserNumber = 1
        self.pressureBasisUserNumber = 2
        self.generatedMeshUserNumber = 1
        self.meshUserNumber = 1
        self.decompositionUserNumber = 1
        self.geometricFieldUserNumber = 1
        self.materialFieldUserNumber = 3
        self.dependentFieldUserNumber = 4
        self.equationsSetFieldUserNumber = 6
        self.equationsSetUserNumber = 1
        self.problem_user_num = 100

        # Instances for setting up OpenCMISS problems
        self.coordinate_system = iron.CoordinateSystem()
        self.region = iron.Region()
        self.basis = iron.Basis()
        self.pressureBasis = iron.Basis()
        self.generatedMesh = iron.GeneratedMesh()
        self.mesh = iron.Mesh()
        self.decomposition = iron.Decomposition()
        self.geometricField = iron.Field()
        self.equationsSetField = iron.Field()
        self.equationsSet = iron.EquationsSet()
        self.dependentField = iron.Field()
        self.deformedField = iron.Field()
        self.materialField = iron.Field()
        self.sourceField = iron.Field()
        self.fibreField = iron.Field()
        self.equations = iron.Equations()
        self.equationsSetSpecification = [iron.EquationsSetClasses.ELASTICITY,
            iron.EquationsSetTypes.FINITE_ELASTICITY,
            iron.EquationsSetSubtypes.MOONEY_RIVLIN]

        # Instances for solving OpenCMISS problems
        self.problemSpecification = [iron.ProblemClasses.ELASTICITY,
                iron.ProblemTypes.FINITE_ELASTICITY,
                iron.ProblemSubtypes.NONE]
    
    def setup_mesh(self, ):
        """
        Setup a cantilever simulation
        """
        
        # Create a 3D rectangular cartesian coordinate system
        self.coordinate_system.CreateStart(self.coordinateSystemUserNumber)
        self.coordinate_system.DimensionSet(self.dimension)
        self.coordinate_system.CreateFinish()
    
        # Create a region and assign the coordinate system to the region
        self.region.CreateStart(self.regionUserNumber, iron.WorldRegion)
        self.region.LabelSet("Region 1")
        self.region.CoordinateSystemSet(self.coordinate_system)
        self.region.CreateFinish()
    
        # Define basis
        self.basis.CreateStart(self.basisUserNumber)
        self.basis.TypeSet(iron.BasisTypes.LAGRANGE_HERMITE_TP)
        self.basis.NumberOfXiSet(self.dimension)
        self.basis.interpolationXi = [iron.BasisInterpolationSpecifications.CUBIC_LAGRANGE] * self.dimension
        self.basis.quadratureNumberOfGaussXi = [self.number_gauss_xi] * self.dimension
        self.basis.CreateFinish()
    
        # Start the creation of a generated mesh in the region
        self.generatedMesh.CreateStart(self.generatedMeshUserNumber,
                                       self.region)
        self.generatedMesh.TypeSet(iron.GeneratedMeshTypes.REGULAR)
        self.generatedMesh.BasisSet([self.basis])
        self.generatedMesh.ExtentSet(
            self.dimensions)  # Width, length, height
        self.generatedMesh.NumberOfElementsSet(self.num_elements)
        # Finish the creation of a generated mesh in the region
        self.generatedMesh.CreateFinish(self.meshUserNumber,
                                        self.mesh)

        # Create a decomposition for the mesh
        self.decomposition.CreateStart(self.decompositionUserNumber, self.mesh)
        self.decomposition.TypeSet(iron.DecompositionTypes.CALCULATED)
        self.decomposition.CalculateFacesSet(True)
        self.decomposition.NumberOfDomainsSet(self.numberOfComputationalNodes)
        self.decomposition.CreateFinish()

        # Create a field for the geometry
        self.geometricField.CreateStart(self.geometricFieldUserNumber,
                                        self.region)
        self.geometricField.MeshDecompositionSet(self.decomposition)
        self.geometricField.TypeSet(iron.FieldTypes.GEOMETRIC)
        self.geometricField.VariableLabelSet(iron.FieldVariableTypes.U,
                                             "Geometry")
        for component in self.components:
            self.geometricField.ComponentMeshComponentSet(
                iron.FieldVariableTypes.U, component, 1)
        self.geometricField.CreateFinish()
        
        # Update the geometric field parameters from generated mesh
        self.generatedMesh.GeometricParametersCalculate(
            self.geometricField)
        self.generatedMesh.Destroy()

    def setup_fields(self, num_components=6):
        components = range(1, num_components+1)
        mesh_component = 1

        # Setup data field
        variable_types = [iron.FieldVariableTypes.U, iron.FieldVariableTypes.V]
        variable_labels = ['GaussStress', 'GaussWeight']
        num_variables = len(variable_types)
        self.data_field_user_num = 11
        self.data_field = iron.Field()
        self.data_field.CreateStart(self.data_field_user_num, self.region)
        self.data_field.TypeSet(iron.FieldTypes.GENERAL)
        self.data_field.MeshDecompositionSet(self.decomposition)
        self.data_field.GeometricFieldSet(self.geometricField)
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
        self.dependent_field_user_num = 7
        self.dependent_field = iron.Field()
        self.dependent_field.CreateStart(
            self.dependent_field_user_num, self.region)
        self.dependent_field.TypeSet(iron.FieldTypes.GENERAL)
        self.dependent_field.DependentTypeSet(
            iron.FieldDependentTypes.DEPENDENT)
        self.dependent_field.MeshDecompositionSet(self.decomposition)
        self.dependent_field.GeometricFieldSet(self.geometricField)
        self.dependent_field.VariableLabelSet(
            iron.FieldVariableTypes.U, "FittedStress")
        self.dependent_field.NumberOfVariablesSet(num_variables)
        for variable_type in variable_types:
            self.dependent_field.NumberOfComponentsSet(
                variable_type, num_components)
            for component in components:
                self.dependent_field.ComponentMeshComponentSet(
                    variable_type, component, mesh_component)
        self.dependent_field.CreateFinish()

        # Set Sobolev smoothing parameters - kappa and tau
        num_material_components = 9
        self.material_field_user_num = 15
        self.material_field = iron.Field()
        self.material_field.CreateStart(
            self.material_field_user_num, self.region)
        self.material_field.TypeSet(iron.FieldTypes.MATERIAL)
        self.material_field.MeshDecompositionSet(self.decomposition)
        self.material_field.GeometricFieldSet(self.geometricField)
        self.material_field.NumberOfComponentsSet(
            iron.FieldVariableTypes.U, num_material_components)
        self.material_field.VariableLabelSet(
            iron.FieldVariableTypes.U, 'SmoothingParameters')
        self.material_field.CreateFinish()

        # Initialise smoothing parameters
        smoothing_parameters = ([
            self.tau,    # tau_1
            self.kappa,  # kappa_11
            self.tau,    # tau_2
            self.kappa,  # kappa_22
            self.kappa,  # kappa_12
            self.tau,    # tau_3
            self.kappa,  # kappa_33
            self.kappa,  # kappa_13
            self.kappa]) # kappa_23
        for component_idx, component in enumerate(components):
            self.material_field.ComponentValuesInitialiseDP(
                iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES,
                component, smoothing_parameters[component_idx])

    def setup_equations(self):
        self.equations_set_user_num = 2
        self.equations_set_field_user_num = 13
        self.equations_set_field = iron.Field()
        self.equations_set = iron.EquationsSet()
        self.specification = (
            [iron.EquationsSetClasses.FITTING,
             iron.EquationsSetTypes.GAUSS_FITTING_EQUATION,
             iron.EquationsSetSubtypes.GAUSS_POINT_FITTING,
             iron.EquationsSetFittingSmoothingTypes.SOBOLEV_VALUE])

        self.equations_set.CreateStart(self.equations_set_user_num,
                                             self.region, self.geometricField,
                                             self.specification,
                                             self.equations_set_field_user_num,
                                             self.equations_set_field)
        self.equations_set.CreateFinish()

        self.equations_set.DependentCreateStart(self.dependent_field_user_num, self.dependent_field)
        self.equations_set.DependentCreateFinish()

        self.equations_set.IndependentCreateStart(self.data_field_user_num, self.data_field)
        self.equations_set.IndependentCreateFinish()

        self.equations_set.MaterialsCreateStart(self.material_field_user_num, self.material_field)
        self.equations_set.MaterialsCreateFinish()

        self.equations = iron.Equations()
        self.equations_set.EquationsCreateStart(self.equations)
        self.equations.SparsityTypeSet(iron.EquationsSparsityTypes.SPARSE)
        self.equations.OutputTypeSet(iron.EquationsOutputTypes.NONE)
        self.equations_set.EquationsCreateFinish()

    def setup_problem(self):

        # Create fitting problem
        self.problem = iron.Problem()
        self.problem_specification = ([iron.ProblemClasses.FITTING,
                                       iron.ProblemTypes.DATA_FITTING,
                                       iron.ProblemSubtypes.STATIC_FITTING])
        self.problem.CreateStart(
            self.problem_user_num, self.problem_specification)
        self.problem.CreateFinish()

        self.problem.ControlLoopCreateStart()
        self.problem.ControlLoopCreateFinish()

        self.solver = iron.Solver()
        self.problem.SolversCreateStart()
        self.problem.SolverGet(
            [iron.ControlLoopIdentifiers.NODE], 1, self.solver)
        #self.solver.OutputTypeSet(iron.SolverOutputTypes.NONE)
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
        self.problem.Solve()
        self.problem.Finalise()

if __name__ == '__main__':

    fitting = Fitting()
    fitting.setup_mesh()
    fitting.setup_fields()
    fitting.setup_equations()
    fitting.setup_problem()
    fitting.solve()

