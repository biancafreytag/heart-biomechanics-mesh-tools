import h5py
import numpy as np
import os
import mesh_tools
from opencmiss.iron import iron


def load_data():
    filename = '/home/psam012/opt/cvt-data/2018-10-30 pre CT lung/PointCloudAlignedToLaserLinesPreCT.hdf5'
    coords_file = h5py.File(filename, 'r')
    geometric_coords = np.array((coords_file[coords_file.keys()[0]]))
    coords_file.close()
    return geometric_coords


# Parameters
dimension = 2
results_folder='./results/'
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

nodeFilename = '/home/psam012/opt/cvt-data/2018-11-9 Post microCT/MeshFitting/data/left_surface.exnode'
elementFilename = '/home/psam012/opt/cvt-data/2018-11-9 Post microCT/MeshFitting/data/left_surface.exelem'

numberOfComputationalNodes = iron.ComputationalNumberOfNodesGet()
computationalNodeNumber = iron.ComputationalNodeNumberGet()

coordinateSystemUserNumber = 1
regionUserNumber = 3
basisUserNumber = 2
meshUserNumber = 1
decompositionUserNumber = 1
geometricFieldUserNumber = 1

coordinateSystem = iron.CoordinateSystem()
coordinateSystem.CreateStart(coordinateSystemUserNumber)
coordinateSystem.dimension = 3
coordinateSystem.CreateFinish()

basis = iron.Basis()
basis.CreateStart(basisUserNumber)
basis.TypeSet(iron.BasisTypes.LAGRANGE_HERMITE_TP)
basis.NumberOfXiSet(dimension)
basis.InterpolationXiSet([iron.BasisInterpolationSpecifications.LINEAR_LAGRANGE]*dimension)
basis.QuadratureNumberOfGaussXiSet([2]*dimension)
basis.CreateFinish()

region = iron.Region()
region.CreateStart(regionUserNumber, iron.WorldRegion)
region.LabelSet("Region")
region.CoordinateSystemSet(coordinateSystem)
region.CreateFinish()

# Load exfiles
exnode = mesh_tools.Exnode(nodeFilename)

elem_nodes = np.array([[19, 19, 142, 174],
                       [19, 13, 174, 176],
                       [13, 14, 176, 148],
                       [14, 14, 148, 123],
                       [19, 19, 142, 128],
                       [19, 13, 128, 111],
                       [13, 14, 111, 110],
                       [14, 14, 110, 123],
                       [142, 174, 401, 384],
                       [174, 176, 384, 386],
                       [176, 148, 386, 370],
                       [148, 123, 370, 323],
                       [110, 123, 303, 323],
                       [142, 128, 401, 346],
                       [128, 111, 346, 328],
                       [111, 110, 328, 303],
                       [401, 384, 650, 601],
                       [384, 386, 601, 585],
                       [386, 370, 585, 605],
                       [370, 323, 605, 536],
                       [303, 323, 538, 536],
                       [328, 303, 558, 538],
                       [346, 328, 597, 558],
                       [401, 346, 650, 597],
                       [650, 601, 768, 723],
                       [601, 585, 723, 724],
                       [585, 605, 724, 774],
                       [605, 536, 774, 715],
                       [650, 597, 768, 718],
                       [597, 558, 718, 699],
                       [558, 538, 699, 698],
                       [538, 536, 698, 715],
                       [768, 723, 837, 837],
                       [723, 724, 837, 869],
                       [724, 774, 869, 865],
                       [768, 718, 837, 837],
                       [718, 699, 837, 869],
                       [699, 698, 869, 865],
                       [698, 715, 865, 865],
                       [774, 715, 865, 865]]).astype('int32')

totalNumberOfNodes = len(exnode.nodeids)
totalNumberOfElements = len(elem_nodes)

# Start the creation of a manually generated mesh in the region
mesh = iron.Mesh()
mesh.CreateStart(meshUserNumber, region, dimension)
mesh.NumberOfComponentsSet(1)
mesh.NumberOfElementsSet(totalNumberOfElements)

# Define nodes for the mesh
nodes = iron.Nodes()
nodes.CreateStart(region, totalNumberOfNodes)
nodes.UserNumbersAllSet(exnode.nodeids)
nodes.CreateFinish()

MESH_COMPONENT1 = 1

mesh.NumberOfComponentsSet(1)

elements = iron.MeshElements()
elements.CreateStart(mesh, MESH_COMPONENT1, basis)
elemNums = []
for elem_num in range(totalNumberOfElements):
    elemNums.append(elem_num + 1)

elements.UserNumbersAllSet(elemNums)
for elem_idx, elem in enumerate(elem_nodes):
    elem_num = elem_idx + 1
    elements.NodesSet(elem_num, elem_nodes[elem_idx])
elements.CreateFinish()

mesh.CreateFinish()

coordinateField = 'coordinates'
coordinates, node_ids = mesh_tools.extract_exfile_coordinates(nodeFilename, coordinateField)

fields = iron.Fields()
fields.AddField(geometricField)
fields.AddField(dependentField)
fields.NodesExport("geometry", "FORTRAN")
fields.ElementsExport("geometry", "FORTRAN")