import sys
import numpy as np
import scipy.linalg, scipy.spatial.distance

import time
import argparse
import pickle
from collections import OrderedDict

from class_hierarchy import ClassHierarchy



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



def sim_approx(class_sim, num_dim = None):
    """
    Finds an embedding of `n` classes in an `d`-dimensional space with `d <= n`, so that their
    dot products best approximate pre-defined similarities.
    
    class_sim - `n-by-n` matrix specifying the desired similarity between each pair of classes.
    num_dim - Optionally, the maximum target dimensionality `d` for the embeddings. If not given, it will be equal to `n`.
    
    Returns: `n-by-d` matrix with rows being the locations of the corresponding classes in the embedding space.
    """
    
    # Check arguments
    if (class_sim.ndim != 2) or (class_sim.shape[0] != class_sim.shape[1]):
        raise ValueError('Given class_sim has invalid shape. Expected: (n, n). Got: {}'.format(class_sim.shape))
    if (class_sim.shape[0] == 0):
        raise ValueError('Empty class_sim given.')
    
    # Compute optimal embeddings based on eigendecomposition of similarity matrix
    L, Q = np.linalg.eigh(class_sim)
    if np.any(L < 0):
        raise RuntimeError('Given class_sim is not positive semi-definite.')
    embeddings = Q * np.sqrt(L)[None,:]

    # Approximation using the eigenvectors corresponding to the largest eigenvalues
    if (num_dim is not None) and (num_dim < embeddings.shape[1]):
        embeddings = embeddings[:,-num_dim:]  # pylint: disable=invalid-unary-operand-type
    
    return embeddings



def euclidean_embedding(class_dist, solver = 'general'):
    """
    Finds an embedding of `n` classes in an `(n-1)`-dimensional space, so that their Euclidean distances correspond
    to pre-defined ones.
    
    class_dist - `n-by-n` matrix specifying the desired distance between each pair of classes.
                 The distances in this matrix *must* define a proper metric that fulfills the triangle inequality.
                 Otherwise, a `RuntimeError` will be raised.
    solver - The linear solver to be used. May be either 'general' or 'triangular'. The triangular solver is faster,
             since we are dealing with an equation system in triangular form here, but less accurate than the
             general solver.
    
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
    
    # Place second class offset along the first axis by the desired distance 
    if nc > 1:
        embeddings[1,0] = class_dist[0,1]
    
    # Iteratively place all remaining classes.
    # Each new class must be located at an intersection of all hyperspheres centered at the already existing classes
    # with radii corresponding to the target distance to the new class.
    for c in range(2, nc):

        centers = embeddings[1:c, :c-1]
        radii = class_dist[:c, c] ** 2

        # Compute first c-1 coordinates of the new center
        b = (radii[0] - radii[1:] + np.sum(centers ** 2, axis = 1)) / 2
        solve_err = False
        try:
            if solver == 'general':
                x = np.linalg.solve(centers, b)
            elif solver == 'triangular':
                x = scipy.linalg.solve_triangular(centers, b, lower = True)
            else:
                raise ValueError('Unknown solver: {}'.format(solver))
            if not np.allclose(np.dot(centers, x), b):
                solve_err = True
        except (np.linalg.LinAlgError, scipy.linalg.LinAlgError):
            solve_err = True
        
        if solve_err:
            raise RuntimeError('Failed to place class #{}: Hyperspheres do not intersect.'.format(c + 1))

        # Compute c-th coordindate of the new center
        d_sq = np.sum(x ** 2)
        if d_sq > radii[0]:
            raise RuntimeError('Failed to place class #{}: There is no common intersection of all spheres (offset: {}).'.format(
                               c + 1, np.sqrt(d_sq) - np.sqrt(radii[0])))
        z = np.sqrt(radii[0] - d_sq)

        embeddings[c, :c-1] = x
        embeddings[c, c-1] = z
    
    return embeddings



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



if __name__ == '__main__':
    
    # Parse arguments
    parser = argparse.ArgumentParser(description = 'Computes semantic class embeddings based on a given hierarchy.', formatter_class = argparse.RawTextHelpFormatter)
    parser.add_argument('--hierarchy', type = str, required = True, help = 'Path to a file containing parent-child or is-a relationships (one per line).')
    parser.add_argument('--is_a', action = 'store_true', default = False, help = 'If given, --hierarchy is assumed to contain is-a instead of parent-child relationships.')
    parser.add_argument('--str_ids', action = 'store_true', default = False, help = 'If given, class IDs are treated as strings instead of integers.')
    parser.add_argument('--class_list', type = str, default = None, help = 'Path to a file containing the IDs of the classes to compute embeddings for (as first words per line). If not given, all leaf nodes in the hierarchy will be considered as target classes.')
    parser.add_argument('--out', type = str, required = True, help = 'Filename of the resulting pickle dump (containing keys "embedding", "ind2label", and "label2ind").')
    parser.add_argument('--method', type = str, default = 'unitsphere', choices = ['unitsphere', 'approx_sim', 'spheres', 'mds'],
                        help = '''Which algorithm to use for computing class embeddings. Options are:
    - "unitsphere": Compute n-dimensional L2-normalized embeddings so that the dot products of class embeddings correspond to their semantic similarity.
    - "approx_sim": Compute embeddings of arbitrary dimensionality so that the dot products of class embeddings correspond to their semantic similarity.
    - "spheres": Compute (n-1)-dimensional embeddings so that Euclidean distances of class embeddings correspond to their semantic dissimilarity using successive intersections of hyperspheres.
    - "mds": Compute embeddings of arbitrary dimensionality so that Euclidean distances of class embeddings correspond to their semantic dissimilarity using classical multidimensional scaling.
Default: "unitsphere"''')
    parser.add_argument('--num_dim', type = int, default = None, help = 'Number of embedding dimensions when using the "mds" or "approx_sim" method.')
    args = parser.parse_args()
    id_type = str if args.str_ids else int
    
    # Read hierarchy
    hierarchy = ClassHierarchy.from_file(args.hierarchy, is_a_relations = args.is_a, id_type = id_type)
    
    # Determine target classes
    if args.class_list is not None:
        with open(args.class_list) as class_file:
            unique_labels = list(OrderedDict((id_type(l.strip().split()[0]), None) for l in class_file if l.strip() != '').keys())
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
        embedding = euclidean_embedding(sem_class_dist)
    elif args.method == 'mds':
        embedding = mds(sem_class_dist, args.num_dim if args.num_dim else len(unique_labels) - 1)
    elif args.method == 'unitsphere':
        embedding = unitsphere_embedding(1. - sem_class_dist)
    elif args.method == 'approx_sim':
        embedding = sim_approx(1. - sem_class_dist, args.num_dim)
    else:
        raise ValueError('Unknown method: {}'.format(args.method))
    stop_time = time.time()
    print('Computed {}-dimensional semantic embeddings for {} classes using the "{}" method in {} seconds.'.format(
        embedding.shape[1], embedding.shape[0], args.method, stop_time - start_time)
    )
    if args.method in ('unitsphere', 'approx_sim'):
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
