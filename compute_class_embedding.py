import sys
import numpy as np
import scipy.linalg, scipy.spatial.distance

import time
import argparse
import pickle

from class_hierarchy import ClassHierarchy



def hierarchical_class_embedding(class_dist, verbose = 0, return_timing = False):
    """
    Finds an embedding of `n` classes in an `(n-1)`-dimensional space, so that their Euclidean distances correspond
    to pre-defined ones.
    
    class_dist - `n-by-n` matrix specifying the desired distance between each pair of classes.
                 The distances in this matrix *must* define a proper metric that fulfills the triangle inequality.
                 Otherwise, a `RuntimeError` will be raised.
    verbose - Positive values enable debugging output: >=1 will print timing information and >=2 will print embeddings.
    
    Returns: `n-by-(n-1)` matrix with rows being the locations of the corresponding classes in the embedding space.
    """
    
    # Check arguments
    if (class_dist.ndim != 2) or (class_dist.shape[0] != class_dist.shape[1]):
        raise ValueError('Given class_dist has invalid shape. Expected: (n, n). Got: {}'.format(class_dist.shape))
    if (class_dist.shape[0] == 0):
        raise ValueError('Empty class_dist given.')
    
    # Place first class at the origin
    nc = class_dist.shape[0]
    embeddings = np.zeros((nc, nc - 1))
    timing = [[0]]
    if verbose >= 2:
        print('Placed class #1 at {}'.format(embeddings[0]))
    
    # Place second class offset along the first axis by the desired distance 
    if nc > 1:
        embeddings[1,0] = class_dist[0,1]
        timing.append([0])
        if verbose >= 2:
            print('Placed class #2 at {}'.format(embeddings[1]))
    
    # Place third class at the intersection of the two circles around the first two classes
    if nc > 2:
        start_time = time.time()
        intersection = intersect_spheres(embeddings[0,:2], embeddings[1,:2], class_dist[0,2], class_dist[1,2])
        if intersection is None:
            raise RuntimeError('Spheres #1 and #2 do not intersect. Check your target metric!')
        c, r, rot = intersection
        if r < 1e-12:
            if verbose >= 1:
                raise RuntimeError('Spheres #1 and #2 intersect in a single point. Make sure that your '\
                                   'target metric fulfills the _strict_ triangle inequality.')
        embeddings[2,:2] = c + np.dot(rot, [r])
        stop_time = time.time()
        timing.append([stop_time - start_time])
        if verbose >= 2:
            print('Placed class #3 in {:7.3f} s at {}'.format(np.sum(timing[-1]), embeddings[2]))
        elif verbose >= 1:
            print('Placed class #3 in {:7.3f} s'.format(np.sum(timing[-1])))
    
    # Iteratively place all remaining classes.
    # Each new class must be located at an intersection of all hyperspheres centered at the already existing classes
    # with radii corresponding to the target distance to the new class.
    for c in range(3, nc):
        
        centers = embeddings[:c, :c-1]
        radii = class_dist[:c, c]
        
        start_time1 = time.time()
        
        # First, intersect the first (c-1)-sphere with all other (c-1)-spheres to obtain a set of hyperplanes that
        # those cuts lie on.
        planes = []
        for i in range(1, len(centers)):
            
            intersection = intersect_spheres(centers[0], centers[i], radii[0], radii[i], return_base = False)
            
            # If two spheres do not intersect, the target metric does not fulfill the triangle inequality.
            if intersection is None:
                raise RuntimeError('Spheres #1 and #{} do not intersect. Check your target metric!'.format(i+1))
            
            # If the radius of the intersection sphere is 0, it is a single point.
            # This indicates that the target metric does not fulfill the strict triangle inequality, preventing us from
            # finding a valid embedding for subsequent classes.
            if intersection[1] < 1e-12:
                if verbose >= 1:
                    raise RuntimeError('Spheres #1 and #{} intersect in a single point. Make sure that your '\
                                       'target metric fulfills the _strict_ triangle inequality.'.format(i+1))
                
            planes.append((intersection[0], intersection[2]))
        
        stop_time1 = time.time()
        start_time2 = time.time()
        
        # Find point of intersection of all those hyperplanes.
        A = np.array([p[1] for p in planes])
        b = np.sum(A * np.array([p[0] for p in planes]), axis = 1)
        plane_intersect_err = False
        try:
            intersection = np.linalg.solve(A, b)
            if not np.allclose(np.dot(A, intersection), b):
                plane_intersect_err = True
        except np.linalg.LinAlgError:
            plane_intersect_err = True
        
        if plane_intersect_err:
            raise RuntimeError('Failed to place class #{}: Hyperplanes do not intersect in 1-d line.'.format(c + 1))
        
        stop_time2 = time.time()
        start_time3 = time.time()
        
        # When we add an additional dimension to our embedding space, the intersection point computed above
        # corresponds to a line parallel to the newly added axis.
        # Now, we find a point on that 1-d line that corresponds to an intersection with the first sphere
        # and, thus, necessarily with all other spheres as well.
        radius_sq = radii[0] * radii[0]
        d_sq = np.sum(intersection ** 2)
        if d_sq > radius_sq:
            raise RuntimeError('Failed to place class #{}: Intersection of planes does not intersect '\
                               'with first sphere (offset: {}).'.format(c + 1, np.sqrt(d_sq) - np.sqrt(radius_sq)))
        x = np.sqrt(radius_sq - d_sq)

        embeddings[c, :c-1] = intersection
        embeddings[c, c-1] = x
        
        stop_time3 = time.time()
        
        timing.append([stop_time1 - start_time1, stop_time2 - start_time2, stop_time3 - start_time3])
        if verbose >= 2:
            print('Placed class #{} in {:7.3f} s at {}'.format(c + 1, np.sum(timing[-1]), embeddings[c]))
        elif verbose >= 1:
            print('Placed class #{} in {:7.3f} s'.format(c + 1, np.sum(timing[-1])))
    
    return (embeddings, timing) if return_timing else embeddings



def intersect_spheres(c1, c2, r1, r2, compute_radius = True, return_base = True):
    """ Finds the intersection of two n-spheres (hyperspheres in (n+1)-dimensional space). 
    
    c1 - Center of first sphere.
    c2 - Center of second sphere.
    r1 - Radius of first sphere.
    r2 - Radius of second sphere.
    compute_radius - Specifies whether to compute the radius of the intersection sphere.
                     This will require an additional square-root computation.
    return_base - If set to `True`, the plane that the intersection sphere lies on will be defined by an (n+1)-by-n
                  rotation matrix, otherwise by an (n+1)-dimensional normal vector.
    
    Returns: The intersection of both n-spheres, which is an (n-1)-sphere, given as tuple of:
             - center (as (n+1)-dimensional vector)
             - radius (as scalar), will only be present if `compute_radius` was set to `True`
             - rotation of the (n-1)-sphere (as (n+1)-by-n matrix) if `return_base` is `True`,
               otherwise the (n+1)-dimensional normal vector of the plane which the sphere lies on
            
            If the two given spheres don't intersect, `None` is returned.
    """
    
    # Cast centers to arrays
    c1 = np.asarray(c1, dtype = float)
    c2 = np.asarray(c2, dtype = float)
    if c1.size != c2.size:
        raise ValueError('Dimensions of hypersphere centers do not match ({} vs. {})'.format(c1.size, c2.size))
    
    # Compute squared radii
    r1_sq = r1 * r1
    r2_sq = r2 * r2
    
    # Compute distance between centers
    direction = c2 - c1
    d = np.linalg.norm(direction)
    direction /= d
    
    # Check if the spheres are disjoint
    if (r1 + r2 < d) or (np.abs(r1 - r2) > d):
        return None
    
    # Compute center of the intersection sphere
    x = (r1_sq - r2_sq) / (2*d) + d / 2
    
    # Compute radius of intersection sphere as third side of the triangle with hypothenuse r1 and leg x
    if compute_radius:
        r = np.sqrt(max(0, r1_sq - x * x))
    
    # Determine parameters of the hyperplane that the intersection lies on
    if return_base:
        if direction.size == 2:
            rot = np.array([[direction[1]], [-1 * direction[0]]])
        elif direction.size == 3:
            rot = np.ndarray((direction.shape[0], direction.shape[0] - 1))
            rot[:,0] = np.array([direction[1], -1 * direction[0], 0])
            rot[:,0] /= np.linalg.norm(rot[:,0])
            rot[:,1] = np.cross(direction, rot[:,0])
        else:
            rot = scipy.linalg.svd(direction[None,:])[2][1:,:].T
    
    if compute_radius:
        return c1 + x * direction, r, rot if return_base else direction
    else:
        return c1 + x * direction, rot if return_base else direction



def mds(class_dist, num_dim = None):
    """
    Finds an embedding of `n` classes in a `d`-dimensional space, so that their Euclidean distances corresponds
    to pre-defined ones, using classical multidimensional scaling (MDS).
    
    class_dist - `n-by-n` matrix specifying the desired distance between each pair of classes.
                 The distances in this matrix *must* define a proper metric that fulfills the triangle inequality.
                 Otherwise, a `RuntimeError` will be raised.
    num_dim - Optionally, the maximum target dimensionality `d` for the embeddings. If not given, it will be determined
              automatically based on the eigenvalues, but this might not be accurate due to limited machine precision.
    
    Returns: `n-by-d` matrix with rows being the locations of the corresponding classes in the embedding space.
    """

    H = np.eye(class_dist.shape[0], dtype=class_dist.dtype) - np.ones(class_dist.shape, dtype=class_dist.dtype) / class_dist.shape[0]
    B = np.dot(H, np.dot(class_dist ** 2, H)) / -2

    eigval, eigvec = np.linalg.eigh(B)
    nonzero_eigvals = (eigval > np.finfo(class_dist.dtype).eps)
    eigval = eigval[nonzero_eigvals]
    eigvec = eigvec[:,nonzero_eigvals]
    
    if num_dim is not None:
        sort_ind = np.argsort(eigval)[::-1]
        eigval = eigval[sort_ind[:num_dim]]
        eigvec = eigvec[:,sort_ind[:num_dim]]

    embedding = eigvec * np.sqrt(eigval[None,:])
    return embedding



def unitsphere_embedding(class_sim):
    """
    Finds an embedding of `n` classes on a unit sphere in `n`-dimensional space, so that their dot products correspond
    to pre-defined similarities.
    
    class_sim - `n-by-n` matrix specifying the desired similarity between each pair of classes.
    
    Returns: `n-by-n` matrix with rows being the locations of the corresponding classes in the embedding space.
    """
    
    # Check arguments
    if (class_sim.ndim != 2) or (class_sim.shape[0] != class_sim.shape[1]):
        raise ValueError('Given class_sim has invalid shape. Expected: (n, n). Got: {}'.format(class_sim.shape))
    if (class_sim.shape[0] == 0):
        raise ValueError('Empty class_sim given.')
    
    # Place first class
    nc = class_sim.shape[0]
    embeddings = np.zeros((nc, nc))
    embeddings[0,0] = 1.
    
    # Iteratively place all remaining classes
    for c in range(1, nc):
        embeddings[c, :c] = np.linalg.solve(embeddings[:c, :c], class_sim[c, :c])
        embeddings[c, c] = np.sqrt(1. - np.sum(embeddings[c, :c] ** 2))
    
    return embeddings



if __name__ == '__main__':
    
    # Parse arguments
    parser = argparse.ArgumentParser(description = 'Computes embeddings of n classes so that their distance corresponds to 1 minus the height of their LCS in a given hierarchy.', formatter_class = argparse.RawTextHelpFormatter)
    parser.add_argument('--hierarchy', type = str, required = True, help = 'Path to a file containing parent-child or is-a relationships (one per line).')
    parser.add_argument('--is_a', action = 'store_true', default = False, help = 'If given, --hierarchy is assumed to contain is-a instead of parent-child relationships.')
    parser.add_argument('--str_ids', action = 'store_true', default = False, help = 'If given, class IDs are treated as strings instead of integers.')
    parser.add_argument('--class_list', type = str, default = None, help = 'Path to a file containing the IDs of the classes to compute embeddings for (as first words per line). If not given, all leaf nodes in the hierarchy will be considered as target classes.')
    parser.add_argument('--out', type = str, required = True, help = 'Filename of the resulting pickle dump (containing keys "embedding", "ind2label", and "label2ind").')
    parser.add_argument('--method', type = str, default = 'spheres', choices = ['spheres', 'mds', 'unitsphere'],
                        help = '''Which algorithm to use for computing class embeddings. Options are:
    - "spheres": Compute (n-1)-dimensional embeddings so that Euclidean distances of class embeddings correspond to their semantic dissimilarity using successive intersections of hyperspheres.
    - "mds": Compute (n-1)-dimensional embeddings so that Euclidean distances of class embeddings correspond to their semantic dissimilarity using classical multidimensional scaling.
    - "unitsphere": Compute n-dimensional L2-normalized embeddings so that the dot product of class embeddings correspond to their semantic similarity.
Default: "spheres"''')
    args = parser.parse_args()
    id_type = str if args.str_ids else int
    
    # Read hierarchy
    hierarchy = ClassHierarchy.from_file(args.hierarchy, is_a_relations = args.is_a, id_type = id_type)
    
    # Determine target classes
    if args.class_list is not None:
        with open(args.class_list) as class_file:
            unique_labels = list(set(id_type(l.strip().split()[0]) for l in class_file if l.strip() != ''))
    else:
        unique_labels = [lbl for lbl in hierarchy.nodes if (lbl not in hierarchy.children) or (len(hierarchy.children[lbl]) == 0)]
        if not args.str_ids:
            unique_labels.sort()
    linear_labels = { lbl : i for i, lbl in enumerate(unique_labels) }
    
    # Compute target distances between classes
    sem_class_dist = np.zeros((len(unique_labels), len(unique_labels)))
    for i in range(len(unique_labels)):
        for j in range(i + 1, len(unique_labels)):
            sem_class_dist[i,j] = sem_class_dist[j,i] = hierarchy.lcs_height(unique_labels[i], unique_labels[j])
    
    # Compute class embeddings
    start_time = time.time()
    if args.method == 'spheres':
        embedding = hierarchical_class_embedding(sem_class_dist)
    elif args.method == 'mds':
        embedding = mds(sem_class_dist, len(unique_labels) - 1)
    elif args.method == 'unitsphere':
        embedding = unitsphere_embedding(1. - sem_class_dist)
    else:
        raise ValueError('Unknown method: {}'.format(args.method))
    stop_time = time.time()
    print('Computed semantic embeddings for {} classes in {} seconds.'.format(embedding.shape[0], stop_time - start_time))
    if args.method == 'unitsphere':
        sim_error = np.abs(np.dot(embedding, embedding.T) - (1. - sem_class_dist))
        print('Maximum deviation from target similarities: {}'.format(sim_error.max()))
        print('Average deviation from target similarities: {}'.format(sim_error.mean()))
    else:
        dist_error = np.abs(scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(embedding)) - sem_class_dist)
        print('Maximum deviation from target distances: {}'.format(dist_error.max()))
        print('Average deviation from target distances: {}'.format(dist_error.mean()))
    
    # Store results
    with open(args.out, 'wb') as dump_file:
        pickle.dump({
                'ind2label' : unique_labels,
                'label2ind' : linear_labels,
                'embedding' : embedding
        }, dump_file)
