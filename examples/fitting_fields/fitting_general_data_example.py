import numpy as np
from opencmiss.iron import iron
import mesh_tools

def setup_mesh():
    
    # OpenCMISS user numbers
    coor_sys_user_num = 1
    region_user_num = 1
    basis_user_num = 1
    generated_mesh_user_num = 1
    mesh_user_num = 1
    decomposition_user_num = 1
    geometric_field_user_num = 1

    # Instances for setting up OpenCMISS mesh
    coordinate_system = iron.CoordinateSystem()
    region = iron.Region()
    basis = iron.Basis()
    generated_mesh = iron.GeneratedMesh()
    mesh = iron.Mesh()
    decomposition = iron.Decomposition()
    geometric_field = iron.Field()
    
    dimensions = np.array(
        [30, 10, 10])  # Length, width, height (in mm)
    num_elements = [2, 2, 2]
    components = [1, 2, 3]  # Geometric components
    dimension = 3  # 3D coordinates
    number_gauss_xi = 4  # Number of Gauss points used for quadrature
    num_load_increments = 1
    results_folder = './'
    numDataPointsPerFace = 6

    # Get the number of computational nodes and this computational node number
    numberOfComputationalNodes = iron.ComputationalNumberOfNodesGet()
    computationalNodeNumber = iron.ComputationalNodeNumberGet()

    # Create a 3D rectangular cartesian coordinate system
    coordinate_system.CreateStart(coor_sys_user_num)
    coordinate_system.DimensionSet(dimension)
    coordinate_system.CreateFinish()

    # Create a region and assign the coordinate system to the region
    region.CreateStart(region_user_num, iron.WorldRegion)
    region.LabelSet("Region 1")
    region.CoordinateSystemSet(coordinate_system)
    region.CreateFinish()

    # Define basis
    basis.CreateStart(basis_user_num)
    basis.TypeSet(iron.BasisTypes.LAGRANGE_HERMITE_TP)
    basis.NumberOfXiSet(dimension)
    basis.interpolationXi = [
                                     iron.BasisInterpolationSpecifications.CUBIC_LAGRANGE] * dimension
    basis.quadratureNumberOfGaussXi = [
                                               number_gauss_xi] * dimension
    basis.CreateFinish()

    # Start the creation of a generated mesh in the region
    generated_mesh.CreateStart(generated_mesh_user_num,
                                   region)
    generated_mesh.TypeSet(iron.GeneratedMeshTypes.REGULAR)
    generated_mesh.BasisSet([basis])
    generated_mesh.ExtentSet(
        dimensions)  # Width, length, height
    generated_mesh.NumberOfElementsSet(num_elements)
    # Finish the creation of a generated mesh in the region
    generated_mesh.CreateFinish(mesh_user_num,
                                    mesh)

    # Create a decomposition for the mesh
    decomposition.CreateStart(decomposition_user_num, mesh)
    decomposition.TypeSet(iron.DecompositionTypes.CALCULATED)
    decomposition.CalculateFacesSet(True)
    decomposition.NumberOfDomainsSet(numberOfComputationalNodes)
    decomposition.CreateFinish()

    # Create a field for the geometry
    geometric_field.CreateStart(geometric_field_user_num,
                                    region)
    geometric_field.MeshDecompositionSet(decomposition)
    geometric_field.TypeSet(iron.FieldTypes.GEOMETRIC)
    geometric_field.VariableLabelSet(iron.FieldVariableTypes.U,
                                         "Geometry")
    for component in components:
        geometric_field.ComponentMeshComponentSet(
            iron.FieldVariableTypes.U, component, 1)
    geometric_field.CreateFinish()

    # Update the geometric field parameters from generated mesh
    generated_mesh.GeometricParametersCalculate(
        geometric_field)
    generated_mesh.Destroy()
    
    return region, decomposition, geometric_field

if __name__ == '__main__':
    
    region, decomposition, geometric_field = setup_mesh()
    tau = 0.01
    kappa = 0.005
    smoothing_parameters = ([
        tau,  # tau_1
        kappa,  # kappa_11
        tau,  # tau_2
        kappa,  # kappa_22
        kappa,  # kappa_12
        tau,  # tau_3
        kappa,  # kappa_33
        kappa,  # kappa_13
        kappa])  # kappa_23

    fitting = mesh_tools.Fitting()
    fitting.set_results_folder('results/')
    fitting.set_region(region)
    fitting.set_decomposition(decomposition)
    fitting.set_geometric_field(geometric_field)
    fitting.setup_fields()
    fitting.setup_equations()
    fitting.setup_problem()
    fitting.set_smoothing_parameters(smoothing_parameters)
    fitting.solve()
    fitting.finalise()

