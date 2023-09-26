def get_nodes(l, dictionary):
    '''
    A function that returns a tuple (n1,n2) containing
    the nodes linked by link l and saved in a dictionary
    with structure {(n1,n2) = l}
    '''
    values = list(dictionary.values())
    index = values.index(l)
    nodes = list(dictionary.keys())[index]
    nodes = nodes.replace("(", "")
    nodes = nodes.replace(")", "")
    return tuple(map(int, nodes.split(', ')))