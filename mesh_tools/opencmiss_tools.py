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
