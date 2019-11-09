"""
Tools for evaluating zinc meshes (scaffolds)
"""

def evaluate(input_mesh, dim, xi_loc, elements):
    """
    Evaluate element xi locations frmm an input .exf format mesh, at user defined element xi locations

    input_mesh - Location of the input exf file.
    dim - Dimension of the mesh i.e. 1, 2, or 3.
    xi_loc - The xi coordinates to evaluate.
    elements -
    """

    from opencmiss.zinc.context import Context
    from opencmiss.zinc.status import OK as ZINC_OK

    context = Context("Scaffold")
    region = context.getDefaultRegion()
    region.readFile(input_mesh)
    field_module = region.getFieldmodule()
    field = field_module.findFieldByName("coordinates")
    cache = field_module.createFieldcache()
    xi = xi_loc
    mesh = field_module.findMeshByDimension(dim)
    if len(xi) != dim:
        raise TypeError("Number of xi coordinates is not valid for {} dimension".format(dim))

    el_iter = mesh.createElementiterator()
    elemDict = dict()
    element = el_iter.next()
    while element.isValid():
        elemDict[int(element.getIdentifier())] = element
        element = el_iter.next()
    mesh_elements = elemDict

    for element in elements:
        cache.setMeshLocation(mesh_elements[element], xi)
        result, out_values = field.evaluateReal(cache, 3)
        if result == ZINC_OK:
            print(mesh_elements[element].getIdentifier(), out_values)
        else:
            raise('Error when evaluating element {0}, at xi {1}'.format(element, xi))

def main():
    input_mesh = '../meshes/heartventricle1.exf'
    dimension = 3
    xi_locations = [0.5, 0.5, 0.5]
    elements = [1]
    evaluate(input_mesh, dimension, xi_locations, elements)

if __name__ == '__main__':
    main()
