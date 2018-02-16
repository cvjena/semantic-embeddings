import sys
import argparse
import pydot

from class_hierarchy import ClassHierarchy



def plot_hierarchy(hierarchy, filename, class_names = None):
    
    if isinstance(hierarchy, ClassHierarchy):
        hierarchy = hierarchy.children
    
    graph = pydot.Dot(graph_type = 'digraph', rankdir = 'LR')
    nodes = {}
    for lbl, children in hierarchy.items():
        nodes[lbl] = pydot.Node(lbl, label = lbl if class_names is None else class_names[lbl], style = 'filled', fillcolor = '#ffffff' if len(children) == 0 else '#eaeaea')
        for child in children:
            if child not in hierarchy:
                nodes[child] = pydot.Node(child, label = child if class_names is None else class_names[child], style = 'filled', fillcolor = '#ffffff')
    for node in nodes.values():
        graph.add_node(node)
    
    for parent, children in hierarchy.items():
        for child in children:
            graph.add_edge(pydot.Edge(nodes[parent], nodes[child]))
    
    graph.write_svg(filename, prog = 'dot')



if __name__ == '__main__':
    
    # Parse arguments
    parser = argparse.ArgumentParser(description = 'Creates a graphical visualization of a class taxonomy.', formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--hierarchy', type = str, required = True, help = 'Path to a file containing parent-child or is-a relationships (one per line).')
    parser.add_argument('--is_a', action = 'store_true', default = False, help = 'If given, --hierarchy is assumed to contain is-a instead of parent-child relationships.')
    parser.add_argument('--str_ids', action = 'store_true', default = False, help = 'If given, class IDs are treated as strings instead of integers.')
    parser.add_argument('--class_names', type = str, default = None, help = 'Optionally, a text file mapping class labels to names, given as one comma-separated label-name tuple per line.')
    parser.add_argument('--out', type = str, required = True, help = 'Filename of the resulting SVG plot.')
    args = parser.parse_args()
    id_type = str if args.str_ids else int
    
    # Read hierarchy
    hierarchy = ClassHierarchy.from_file(args.hierarchy, is_a_relations = args.is_a, id_type = id_type)
    
    # Read class names
    if args.class_names is not None:
        with open(args.class_names) as f:
            class_names = { id_type(lbl) : name for l in f if l.strip() != '' for lbl, name in [l.strip().split(maxsplit=1)] }
    else:
        class_names = None
    
    # Plot hierarchy
    plot_hierarchy(hierarchy, args.out, class_names=class_names)