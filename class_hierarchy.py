class ClassHierarchy(object):
    """ Represents a class taxonomy and can be used to find Lowest Common Subsumers or to compute class similarities. """
    
    def __init__(self, parents, children):
        """ Initializes a new ClassHierarchy.
        
        parents - Dictionary mapping class labels to lists of parent class labels in the hierarchy.
        children - Dictionary mapping class labels to lists of children class labels in the hierarchy.
        """
        
        object.__init__(self)
        self.parents = parents
        self.children = children
        self.nodes = set(self.parents.keys()) | set(self.children.keys())
        
        self._depths = { False : {}, True : {} }
        self._hyp_depth_cache = { False : {}, True : {} }
        self._hyp_dist_cache = {}
        self._lcs_cache = {}
        self._wup_cache = {}
        
        self._compute_heights()
        self.max_height = max(self.heights.values())
    
    
    def _compute_heights(self):
        """ Computes the heights of all nodes in the hierarchy. """
        
        def height(id):
            
            if id not in self.heights:
                self.heights[id] = 1 + max((height(child) for child in self.children[id]), default = -1) if id in self.children else 0
            return self.heights[id]
        
        self.heights = {}
        for node in self.nodes:
            height(node)
    
    
    def is_tree(self):
        """ Determines whether the hierarchy is a tree.
        
        Note that some popular hierarchies such as WordNet are not trees, but allow nodes to have multiple parents.
        """
        
        return all(len(parents) <= 1 for parents in self.parents.values())
    
    
    def all_hypernym_depths(self, id, use_min_depth = False):
        """ Determines all hypernyms of a given element (including the element itself) along with their depth in the hierarchy.
        
        id - ID of the element.
        use_min_depth - If set to `True`, use the shortest path from the root to an element to determine its depth, otherwise the longest path.
        
        Returns: dictionary mapping hypernym ids to depths. Root nodes have depth 1.
        """
        
        if id not in self._hyp_depth_cache[use_min_depth]:
            
            depths = {}
            if (id not in self.parents) or (len(self.parents[id]) == 0):
                depths[id] = 1 # root nodes have depth 1
            else:
                for parent in self.parents[id]:
                    for hyp, depth in self.all_hypernym_depths(parent, use_min_depth).items():
                        depths[hyp] = depth
                depths[id] = 1 + min(depths[p] for p in self.parents[id]) if use_min_depth else 1 + max(depths[p] for p in self.parents[id])
            
            self._hyp_depth_cache[use_min_depth][id] = depths
            self._depths[use_min_depth][id] = depths[id]
        
        return self._hyp_depth_cache[use_min_depth][id]
    
    
    def all_hypernym_distances(self, id):
        """ Determines all hypernyms of a given element (including the element itself) along with their distance from the element.
        
        id - ID of the element.
        
        Returns: dictionary mapping hypernym ids to distances, measured in the minimum length of edges between two nodes.
        """
        
        if id not in self._hyp_dist_cache:
        
            distances = { id : 0 }
            if id in self.parents:
                for parent in self.parents[id]:
                    for hyp, dist in self.all_hypernym_distances(parent).items():
                        if (hyp not in distances) or (dist + 1 < distances[hyp]):
                            distances[hyp] = dist + 1

            self._hyp_dist_cache[id] = distances
        
        return self._hyp_dist_cache[id]
    
    
    def root_paths(self, id):
        """ Determines all paths from a given element (excluding the element itself) to a root node in the hierarchy.
        
        id - ID of the element.
        
        Returns: list of lists of node ids, each list beginning with a direct hypernym of the given element and ending with a root node
        """
        
        paths = []
        if id in self.parents:
            for parent in self.parents[id]:
                parent_paths = self.root_paths(parent)
                if len(parent_paths) == 0:
                    paths.append([parent])
                else:
                    for parent_path in parent_paths:
                        paths.append([parent] + parent_path)
        return paths
    
    
    def lcs(self, a, b, use_min_depth = False):
        """ Finds the lowest common subsumer of two elements.
        
        a - The ID of the first term.
        b - The ID of the second term.
        use_min_depth - If set to `True`, use the shortest path from the root to an element to determine its depth, otherwise the longest path.
        
        Returns: the id of the LCS or `None` if the two terms do not share any hypernyms.
        """
        
        if (a,b) not in self._lcs_cache:
        
            hypernym_depths = self.all_hypernym_depths(a, use_min_depth)
            common_hypernyms = set(hypernym_depths.keys()) & set(self.all_hypernym_depths(b, use_min_depth).keys())

            self._lcs_cache[(a,b)] = max(common_hypernyms, key = lambda hyp: hypernym_depths[hyp], default = None)
        
        return self._lcs_cache[(a,b)]
    
    
    def shortest_path_length(self, a, b):
        """ Determines the length of the shortest path between two elements of the hierarchy.
        
        a - The ID of the first term.
        b - The ID of the second term.
        
        Returns: length of the shortest path from `a` to `b`, measured in the number of edges. `None` is returned if there is no path.
        """
        
        dist1 = self.all_hypernym_distances(a)
        dist2 = self.all_hypernym_distances(b)
        common_hypernyms = set(dist1.keys()) & set(dist2.keys())
        
        return min((dist1[hyp] + dist2[hyp] for hyp in common_hypernyms), default = None)
    
    
    def depth(self, id, use_min_depth = False):
        """ Determines the depth of a certain element in the hierarchy.
        
        id - The ID of the element.
        use_min_depth - If set to `True`, use the shortest path from the root to an element to determine its depth, otherwise the longest path.
        
        Returns: the depth of the given element. Root nodes have depth 1.
        """
        
        if id not in self._depths[use_min_depth]:
            
            if (not id in self.parents) or (len(self.parents[id]) == 0):
                self._depths[use_min_depth][id] = 1 # root nodes have depth 1
            else:
                parent_depths = (self.depth(p, use_min_depth) for p in self.parents[id])
                self._depths[use_min_depth][id] = 1 + min(parent_depths) if use_min_depth else 1 + max(parent_depths)
        
        return self._depths[use_min_depth][id]
    
    
    def wup_similarity(self, a, b):
        """ Computes the Wu-Palmer similarity of two elements in the hierarchy.
        
        a - The ID of the first term.
        b - The ID of the second term.
        
        Returns: similarity score in the range (0,1].
        """
        
        if (a,b) not in self._wup_cache:
        
            lcs = self.lcs(a, b)
            ds = self.depth(lcs)
            d1 = ds + self.shortest_path_length(a, lcs)
            d2 = ds + self.shortest_path_length(b, lcs)
            self._wup_cache[(a,b)] = (2.0 * ds) / (d1 + d2)
        
        return self._wup_cache[(a,b)]
    
    
    def lcs_height(self, a, b):
        """ Computes the height of the lowest common subsumer of two elements, divided by the height of the entire hierarchy.
        
        a - The ID of the first term.
        b - The ID of the second term.
        
        Returns: dissimilarity score in the range [0,1].
        """
        
        return self.heights[self.lcs(a, b)] / self.max_height
    
    
    def save(self, filename, is_a_relations = False):
        """ Writes the hierarchy structure to a text file as lines of parent-child or child-parent tuples.
        
        filename - Path to the file to be written.
        is_a_relations - If set to `True`, the hierarchy will be exported is child-parent tuples, otherwise as parent-child tuples.
        """
        
        with open(filename, 'w') as f:
            if is_a_relations:
                for child, parents in self.parents.items():
                    for parent in parents:
                        f.write('{} {}\n'.format(child, parent))
            else:
                for parent, children in self.children.items():
                    for child in children:
                        f.write('{} {}\n'.format(parent, child))
    
    
    @classmethod
    def from_file(cls, rel_file, is_a_relations = False, id_type = str):
        """ Constructs a class hierarchy based on a file with parent-child relations.
        
        rel_file - Path to a file specifying the relations between elements in the hierarchy, given by lines of ID tuples.
        is_a_relations - If set to `True`, `rel_file` is supposed to contain `<child> <parent>` tuples, otherwise `<parent> <child>` tuples.
        id_type - Data type of element IDs.
        
        Returns: a new ClassHierarchy instance
        """
        
        parents, children = {}, {}
        with open(rel_file) as f:
            for l in f:
                if l.strip() != '':
                    
                    parent, child = [id_type(id) for id in l.split(maxsplit = 1)]
                    if is_a_relations:
                        parent, child = child, parent
                    
                    if child in parents:
                        parents[child].append(parent)
                    else:
                        parents[child] = [parent]
                    
                    if parent in children:
                        children[parent].append(child)
                    else:
                        children[parent] = [child]
        
        return cls(parents, children)
