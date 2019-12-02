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


def add_fig(viewer, label=''):
    if viewer is not None:
        fig = viewer.Figure(label)
    else:
        fig = None
    return fig

def visualise_mesh(
        mesh, fig, visualise=False, face_colours=(1, 0, 0), pt_size=5,
        label=None,
        text=False, element_ids=False, text_elements=None, opacity=0.5,
        line_opacity=1., elements=None, nodes=None, node_text=False, text_size=3,
        node_size=5, node_colours=(1, 0, 0), elements_to_display_nodes=None):
    if fig is not None:
        if label is None:
            label = mesh.label
        Xnid = mesh.get_node_ids(group='_default')

        if nodes == 'all':
            nodes = Xnid[1]  # [:-5]

        if visualise:
            # View breast surface mesh
            Xs, Ts = mesh.get_surfaces(res=16, elements=elements)
            if Xs.shape[0] == 0:
                Xs, Ts = mesh.get_faces(res=16, elements=elements)
            if elements is None:
                Xl = mesh.get_lines(res=32, internal_lines=False)
            else:
                Xl = mesh.get_lines(res=32, elements=elements,
                                    internal_lines=False)
            # import ipdb; ipdb.set_trace()
            fig.plot_surfaces('{0}_Faces'.format(label), Xs, Ts,
                              color=face_colours,
                              opacity=opacity)
            # fig.plot_points('{0}_Nodes'.format(label), Xn, color=(1,0,1), size=pt_size)
            fig.plot_lines('{0}_Lines'.format(label), Xl, color=(1, 1, 0),
                           size=5, opacity=line_opacity)
            if text_elements is None:
                if text:
                    fig.plot_text('{0}_Text'.format(label), Xnid[0], Xnid[1],
                                  size=text_size)
            if elements_to_display_nodes is not None:
                for element_id in elements_to_display_nodes:
                    element = mesh.elements[element_id]
                    # import ipdb; ipdb.set_trace()
                    eXnid = mesh.get_node_ids(element.node_ids)
                    fig.plot_text(
                        '{0}_text_element{1}'.format(label, element.id),
                        eXnid[0], eXnid[1], size=pt_size, color=(1, 0, 0))
                    fig.plot_points(
                        '{0}_Nodes_element{1}'.format(label, element.id),
                        eXnid[0], color=(1, 0, 1), size=pt_size / 2)
            if element_ids:
                fig.plot_element_ids('{0}_Xecid'.format(label), mesh, size=1,
                                     color=(1, 1, 1))
            if nodes is not None:
                # import ipdb; ipdb.set_trace()
                fig.plot_points(
                    '{0}_Points'.format(label), mesh.get_nodes(nodes),
                    color=node_colours, size=node_size)
                if node_text:
                    fig.plot_text(
                        '{0}_Text'.format(label), mesh.get_nodes(nodes), nodes,
                        size=3)
