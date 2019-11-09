import mesh_tools

if __name__ == '__main__':
    input_mesh = '../meshes/heartventricle1.exf'
    dimension = 3
    xi_locations = [0.5, 0.5, 0.5]
    elements = [1]
    mesh_tools.evaluate(input_mesh, dimension, xi_locations, elements)