import numpy as np
import mesh_tools

def generate_points_morphic_face(
        mesh, face, value, num_points=[4, 4], element_ids=[], dim=3):
    """
    Generate a grid of points on faces of selected morphic mesh elements

    Keyword arguments:
    mesh -- mesh to evaluate points in
    face -- face to evaluate points on at the specified xi value
    dim -- the number of xi directions
    """

    xi = mesh_tools.generate_xi_on_face(
        face, value, num_points=num_points, dim=dim)

    if not element_ids:
        # if elem group is empty evaluate all elements
        element_ids = mesh.elements.ids

    num_ne = len(element_ids)
    ne_num_points = np.prod(num_points)
    total_num_points = num_ne * ne_num_points
    points = np.zeros((num_ne, ne_num_points, dim))
    all_xi = np.zeros((num_ne, ne_num_points, dim))
    all_ne = np.zeros((num_ne, ne_num_points))

    for idx, element_id in enumerate(element_ids):
        element = mesh.elements[element_id]
        points[idx, :, :] = element.evaluate(xi)
        all_xi[idx, :, :] = xi
        all_ne[idx, :] = element_id

    points = np.reshape(points, (total_num_points, dim))
    all_xi = np.reshape(all_xi, (total_num_points, dim))
    all_ne = np.reshape(all_ne, (total_num_points))

    return points, all_xi, all_ne

def generate_points_morphic_elements(
        mesh, num_points=[4, 4, 4], element_ids=[], dim=3):
    """
    Generate a grid of points within selected morphic mesh elements

    Keyword arguments:
    mesh -- mesh to evaluate points in
    dim -- the number of xi directions
    """

    xi = mesh_tools.generate_xi_grid_fem(num_points=num_points, dim=3)

    if not element_ids:
        # if elem group is empty evaluate all elements
        element_ids = mesh.elements.ids

    num_ne = len(element_ids)
    ne_num_points = np.prod(num_points)
    total_num_points = num_ne * ne_num_points
    points = np.zeros((num_ne, ne_num_points, dim))
    all_xi = np.zeros((num_ne, ne_num_points, dim))
    all_ne = np.zeros((num_ne, ne_num_points))

    for idx, element_id in enumerate(element_ids):
        element = mesh.elements[element_id]
        points[idx, :, :] = element.evaluate(xi)
        all_xi[idx, :, :] = xi
        all_ne[idx, :] = element_id

    points = np.reshape(points, (total_num_points, dim))
    all_xi = np.reshape(all_xi, (total_num_points, dim))
    all_ne = np.reshape(all_ne, (total_num_points))

    return points, all_xi, all_ne
