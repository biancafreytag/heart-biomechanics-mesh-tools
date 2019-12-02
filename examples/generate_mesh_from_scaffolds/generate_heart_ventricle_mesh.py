import mesh_tools

if __name__ == '__main__':
    input_mesh = './meshes/heartventricle1.exf'
    dimension = 3
    xi_locations = [0.5, 0.5, 0.5]
    elements = [1]

    mesh = mesh_tools.Zinc_mesh(input_mesh, dimension)
    mesh.evaluate(xi_locations, elements)
    mesh.export_vtk('./results/test')
    mesh.find_standard_elements()
