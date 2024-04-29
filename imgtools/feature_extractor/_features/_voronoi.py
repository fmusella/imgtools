import numpy as np
from scipy.spatial import ConvexHull, Voronoi
from ...cte import ChromatinTracingExperiment

docstring = """Calculates the volume of the Voronoi cell claimed by each spot.
Given a set of spots, the Voronoi Diagram is a partition of the 3D space into regions,
where each region contains all the points that are closer to a single spot in the set.
The volume of the Voronoi cell claimed by a spot is the volume of the region associated with that spot.
There is a boundary issue: spots close to the nuclear envelope would have an infinite volume. We set them as NaN."""

required_keys = {}

def run(cellID: str, cte: ChromatinTracingExperiment, _1, feat_arr: np.ndarray, _2) -> np.ndarray:
    """ Calculates the volume of the Voronoi cell claimed by each spot.
    
    Given a set of spots, the Voronoi Diagram is a partition of the 3D space into regions,
    where each region contains all the points that are closer to a single spot in the set.
    
    The volume of the Voronoi cell claimed by a spot is the volume of the region associated with that spot.
    
    There is a boundary issue: spots close to the nuclear envelope would have an infinite volume. We set them as NaN.
    
    To calculate the Voronoi volume, we can use Convex Hulls on the vertices of the Voronoi cells.
    This should be accurante since the Voronoi cells are convex (https://math.berkeley.edu/~bernd/wednesday.pdf).
    
    If two or more spots are mapped to the same domain, the average of the values is taken.

    Args:
        cellID (str)
        cte (ChromatinTracingExperiment)
        feat_arr (np.ndarray): initialized nan-valued array of shape (n_domains, n_traces) to store the feature value.
        _*: not used, just to match the function signature

    Returns:
        (np.ndarray): updated array of shape (n_domains, n_traces) with the feature values.
    """
    
    # Get the cell data in numpy format and extract the coordinates
    xs, ys, zs, chroms, starts, ends, _, traceIDs, _ = cte.get_data(cellID, format='numpy')
    crds = np.array([xs, ys, zs]).T
    
    # Get the traceID hash table to map traces to their position in the array
    traceID_hash = cte.get_trace_hashmap(cellID)
    
    # Get the index and its hash table
    index = cte.index
    index_hash = index.get_index_hashmap()
    
    # Initialize a dictionary to store the feature values for each domain (we will then take the average)
    feat_per_domain = {}
    
    # Calculate the Voronoi Diagrams
    vor = Voronoi(crds)
    
    # Loop over the spots
    for spotnum, (chrom, start, end, traceID) in enumerate(zip(chroms, starts, ends, traceIDs)):
        
        # Get the position of the trace in the array
        i_trace = traceID_hash[chrom][traceID]
        
        # The Voronoi regions don't have a fixed order, so we need to find the region associated with the spot
        regionID = vor.point_region[spotnum]
        
        # Get the indices of the vertices of the region
        indices = vor.regions[regionID]
        
        # The indices could be empty or contain -1, which means the region is unbounded.
        # In this case, we can skip this spot (the feature value is kept as NaN)
        if -1 in indices or len(indices) == 0:
            continue
        
        # Get the vertices of the region
        vertices = vor.vertices[indices]
        
        # Try to fit a Convex Hull to the vertices
        try:
            hull = ConvexHull(vertices)
        except:
            # If the Convex Hull cannot be calculated, skip this spot (the feature value is kept as NaN)
            continue
        
        # Get the feature value as the volume of the Convex Hull
        feat_val = hull.volume
        
        # If the volume is NaN or infinite, skip this spot (the feature value is kept as NaN)
        if np.isnan(feat_val) or np.isinf(feat_val):
            continue
        
        # Get the position of the spot in the Index array using the hash tables
        i_domain = index_hash[(chrom, start, end)]
        assert len(i_domain) == 1, f"Error: multiple domains found for {chrom}, {start}, {end}"
        i_domain = i_domain[0]
        
        # Initialize the list of values for this domain if necessary
        if (i_domain, i_trace) not in feat_per_domain:
            feat_per_domain[(i_domain, i_trace)] = []
        
        # Add the feature value to the dictionary of values for this domain
        feat_per_domain[(i_domain, i_trace)].append(feat_val)
                
    
    # Compute the average of the values for each domain and add them to the feature array
    for (i_domain, i_trace), vals in feat_per_domain.items():
        feat_arr[i_domain, i_trace] = np.nanmean(vals)
    
    return feat_arr
