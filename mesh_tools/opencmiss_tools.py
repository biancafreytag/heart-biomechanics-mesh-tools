import mesh_tools.fields as fields
import utilities
import numpy as np
from opencmiss.iron import iron

def interpolate_opencmiss_field(field, element_ids=[], xi=None, num_values=4,dimension=3, derivative_number=1, elems=None, face=None, value=0.):
    import mesh_tools.fields as fields

    if xi is None:
        if face == None:
            XiNd = fields.generate_xi_grid_fem(
                num_points=[num_values, num_values, num_values])
        else:
            XiNd = fields.generate_xi_on_face(face, value, num_points=num_values, dim=dimension)

        num_elem_values = XiNd.shape[0]
        num_Xe = len(element_ids)
        total_num_values = num_Xe * num_elem_values
        values = np.zeros((num_Xe, num_elem_values, dimension))
        xi = np.zeros((num_Xe, num_elem_values, dimension))
        elements = np.zeros((num_Xe, num_elem_values, 1))

        for elem_idx, element_id in enumerate(element_ids):
            for point_idx in range(num_elem_values):
                single_xi = XiNd[point_idx,:]
                values[elem_idx, point_idx, :] = field.ParameterSetInterpolateSingleXiDP(iron.FieldVariableTypes.U,
                                                             iron.FieldParameterSetTypes.VALUES, derivative_number, int(element_id), single_xi, dimension)
            xi[elem_idx, :, :] = XiNd
            elements[elem_idx, :] = element_id

        values = np.reshape(values, (total_num_values, dimension))
        xi = np.reshape(xi, (total_num_values, dimension))
        elements = np.reshape(elements, (total_num_values))
        return values, xi, elements
    else:
        num_values = xi.shape[0]
        values = np.zeros((num_values, dimension))
        for point_idx in range(xi.shape[0]):
            element_id = elems[point_idx]
            single_xi = xi[point_idx,:]
            values[point_idx, :] = field.ParameterSetInterpolateSingleXiDP(iron.FieldVariableTypes.U,
                                                         iron.FieldParameterSetTypes.VALUES, derivative_number, int(element_id), single_xi, dimension)
        return values


def interpolate_opencmiss_field_sample(
        field, element_ids=None, num_values=4, dimension=3, derivative_number=1,
        face=None, value=0., unique=False, geometric_field=None):
    """ Interpolates and OpenCMISS field at selected points along xi directions


    Args:
        field (OpenCMISS field object): The general field to interpolate
        element_ids (list): Mesh element ids to interpolate. Defaults to all
                            elements if none are specified.
        num_values (int): The number of values to interpolate at along each
                          element xi direction
        dimension (int): Dimension of the field being interpolated
                         (e.g. 3 for 3D)
        derivative_number (int): The field derivative to interpolate
        face (int): The face number interpolate the field at
        value (int): The xi value of the face to select
        unique(bool): Return interpolated field values at only unique geometric
                      values
        geometric_field(OpenCMISS field object): Required for unique option

        Todo replace face and value with OpenCMISS face variable which combines
        the two variables into one.

    Returns:
        values, xi, elements
    """

    if face == None:
        XiNd = fields.generate_xi_grid_fem(
            num_points=[num_values, num_values, num_values])
    else:
        XiNd = fields.generate_xi_on_face(
            face, value, num_points=num_values, dim=dimension)

    num_elem_values = XiNd.shape[0]
    num_Xe = len(element_ids)
    total_num_values = num_Xe * num_elem_values
    values = np.zeros((num_Xe, num_elem_values, dimension))
    xi = np.zeros((num_Xe, num_elem_values, dimension))
    elements = np.zeros((num_Xe, num_elem_values, 1))

    for elem_idx, element_id in enumerate(element_ids):
        for point_idx in range(num_elem_values):
            single_xi = XiNd[point_idx,:]
            values[elem_idx, point_idx,
            :] = field.ParameterSetInterpolateSingleXiDP(
                iron.FieldVariableTypes.U, iron.FieldParameterSetTypes.VALUES,
                derivative_number, int(element_id), single_xi, dimension)
        xi[elem_idx, :, :] = XiNd
        elements[elem_idx, :] = element_id

    # Reshape interpolated field values into a vector
    values = np.reshape(values, (total_num_values, dimension))
    xi = np.reshape(xi, (total_num_values, dimension))
    elements = np.reshape(elements, (total_num_values))

    if unique:
        # Evaluate geometric field at xi values and select general field values
        # that only have unique coordinates
        if geometric_field is None:
            raise ValueError('Geometric field is required for returning unique field values')
        geometric_values = np.zeros((num_Xe, num_elem_values, dimension))
        for elem_idx, element_id in enumerate(element_ids):
            for point_idx in range(num_elem_values):
                single_xi = XiNd[point_idx, :]
                geometric_values[elem_idx, point_idx,
                :] = geometric_field.ParameterSetInterpolateSingleXiDP(
                    iron.FieldVariableTypes.U,
                    iron.FieldParameterSetTypes.VALUES, derivative_number,
                    int(element_id), single_xi, dimension)

        # Reshape interpolated field values into a vector
        geometric_values = np.reshape(
            geometric_values, (total_num_values, dimension))

        # Identify unique geometric field values
        _, indices = utilities.np_1_13_unique(
            geometric_values, axis=0, return_index=True)
        # Select only unique general field values
        values = values[sorted(indices), :]
        xi = xi[sorted(indices), :]
        elements = elements[sorted(indices)]

    return values, xi, elements

def interpolate_opencmiss_field_xi(
        field, xi, element_ids=[], dimension=3, deriv=1):
    num_values = xi.shape[0]
    values = np.zeros((num_values, dimension))
    for point_idx in range(xi.shape[0]):
        element_id = element_ids[point_idx]
        values[point_idx, :] = field.ParameterSetInterpolateSingleXiDP(
            iron.FieldVariableTypes.U,
            iron.FieldParameterSetTypes.VALUES, deriv,
            int(element_id), xi[point_idx, :], dimension)
    return values


def get_field_values(field, node_nums, derivative=1, dimension=3,
                     variable=iron.FieldVariableTypes.U):
    coordinates = np.zeros((len(node_nums), dimension))
    for node_idx, node in enumerate(node_nums):
        for component_idx, component in enumerate(range(1, dimension + 1)):
            coordinates[node_idx, component_idx] = field.ParameterSetGetNodeDP(
                variable, iron.FieldParameterSetTypes.VALUES, 1, derivative,
                node, component)
    return coordinates


def set_field_values(field, node_nums, coordinates, derivative=1,
                     variable=iron.FieldVariableTypes.U,
                     update_scale_factors=False):
    """
    Update the field parameters
    """
    if update_scale_factors:
        field.ParameterSetUpdateStart(iron.FieldVariableTypes.U,
                                      iron.FieldParameterSetTypes.VALUES)
    for node_idx, node in enumerate(node_nums):
        for component_idx, component in enumerate(
                range(1, coordinates.shape[1] + 1)):
            field.ParameterSetUpdateNodeDP(
                variable, iron.FieldParameterSetTypes.VALUES, 1, derivative,
                node, component, coordinates[node_idx, component_idx])

    if update_scale_factors:
        field.ParameterSetUpdateFinish(iron.FieldVariableTypes.U,
                                       iron.FieldParameterSetTypes.VALUES)

def num_nodes_get(mesh, mesh_component=1):
    nodes = iron.MeshNodes()
    mesh.NodesGet(mesh_component, nodes)
    return nodes.NumberOfNodesGet()

def num_element_get(mesh, mesh_component=1):
    elements = iron.MeshElements()
    mesh.ElementsGet(mesh_component, elements)
    num_elements = mesh.NumberOfElementsGet()
    element_nums = (np.arange(num_elements)+1).tolist()
    return num_elements, element_nums