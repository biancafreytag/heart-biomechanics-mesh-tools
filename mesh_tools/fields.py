import numpy as np

def generate_xi_on_face(face, value, num_points=[4, 4], dim=3):
    """
    Generate a grid of points within each element

    Keyword arguments:
    face -- face to evaluate points on at the specified xi value
    dim -- the number of xi directions
    """
    num_points = np.atleast_1d(num_points)
    if len(num_points)==1:
        num_points = np.array([num_points[0], num_points[0]])

    if dim == 3:
        if face == "xi1":
            xi1 = [value]
            xi2 = np.linspace(0., 1., num_points[0])
            xi3 = np.linspace(0., 1., num_points[1])
        elif face == "xi2":
            xi1 = np.linspace(0., 1., num_points[0])
            xi2 = [value]
            xi3 = np.linspace(0., 1., num_points[1])
        elif face == "xi3":
            xi1 = np.linspace(0., 1., num_points[0])
            xi2 = np.linspace(0., 1., num_points[1])
            xi3 = [value]
        X, Y, Z = np.meshgrid(xi1, xi2, xi3)
        xi = np.array([
            X.reshape((X.size)),
            Y.reshape((Y.size)),
            Z.reshape((Z.size))]).T
    elif dim == 2:
        xi1 = np.linspace(0., 1., num_points[0])
        xi2 = np.linspace(0., 1., num_points[1])
        X, Y = np.meshgrid(xi1, xi2)
        xi = np.array([
            X.reshape((X.size)),
            Y.reshape((Y.size))]).T

    return xi


def generate_xi_grid_fem(num_points=[4, 4, 4], dim=3):
    # Generate a grid of points within each element
    xi1 = np.linspace(0., 1., num_points[0])
    xi2 = np.linspace(0., 1., num_points[1])
    if dim == 2:
        X, Y = np.meshgrid(xi1, xi2)
        XiNd = np.array([
            X.reshape((X.size)),
            Y.reshape((Y.size))]).T
    else:
        xi3 = np.linspace(0., 1., num_points[2])
        X, Y, Z = np.meshgrid(xi1, xi2, xi3)
        XiNd = np.array([
            Z.reshape((Z.size)),
            X.reshape((X.size)),
            Y.reshape((Y.size))]).T
    return XiNd