import numpy as np
from scipy.spatial import distance
from alabtools.utils import get_index_from_set


# FUNCTIONS TO GET SUMMARY METRICS FROM DATA

def get_index_and_attrs(data: dict, assembly: str = None):
    """ Get the index and attributes from the data.

    Args:
        data (dict): data in dictionary format.
        assembly (str, optional): assembly name. Defaults to None.

    Returns:
        index (alabtools.utils.Index): index object.
        attrs (dict): attributes dictionary.
    """
    
    # Initialize attributes variables
    ncell = len(data)
    nchrom = 0
    nspot = 0
    max_ntrace_per_chrom = 0
    max_nspot_per_trace = 0
    max_nspot_per_domain = 0
    
    # Initialize the spot-per-domain counter
    domain_counter = {}
    
    # Initialize the unique domain set
    domain_set = set()
    
    # Loop over the data
    for cellID in data:
        
        # Get the number of chromosomes in the cell
        nchrom_cell = len(data[cellID])
        # Update the maximum number of chromosomes if necessary
        nchrom = max(nchrom, nchrom_cell)
        
        for chrom in data[cellID]:
            
            # Get the number of traces in the chromosome
            ntrace_chrom = len(data[cellID][chrom])
            # Update the maximum number of traces per chromosome if necessary
            max_ntrace_per_chrom = max(max_ntrace_per_chrom, ntrace_chrom)
            
            for traceID in data[cellID][chrom]:
                
                # Get the number of spots in the trace
                nspot_trace = len(data[cellID][chrom][traceID])
                # Increment the number of spots
                nspot += nspot_trace
                # Update the maximum number of spots per trace if necessary
                max_nspot_per_trace = max(max_nspot_per_trace, nspot_trace)
                
                for spotID in data[cellID][chrom][traceID]:
                    
                    # Isolate the domain
                    start = data[cellID][chrom][traceID][spotID]['start']
                    end = data[cellID][chrom][traceID][spotID]['end']
                    domain = (chrom, start, end)
                    
                    # Update the per-domain counter
                    domain_counter, max_nspot_per_domain = update_domain_counter(domain_counter, cellID, chrom, traceID, domain, max_nspot_per_domain)
                    
                    # Update the unique domain set
                    domain_set.add(domain)
    
    # Create the attributes dictionary
    attrs = {
        'ncell': ncell,
        'nchrom': nchrom,
        'nspot': nspot,
        'max_ntrace_per_chrom': max_ntrace_per_chrom,
        'max_nspot_per_trace': max_nspot_per_trace,
        'max_nspot_per_domain': max_nspot_per_domain
    }
    
    # Create Index object from the domain_set if assembly is provided
    if assembly is not None:
        index = get_index_from_set(domain_set, assembly)
    else:
        index = None
    
    del domain_set, domain_counter

    return index, attrs

def update_domain_counter(domain_counter: dict,
                          cellID: str,
                          chrom: str,
                          traceID: str,
                          domain: tuple,
                          max_nspot_per_domain: int):
    """Update the per-domain spot counter and the maximum number of spots per domain.

    Args:
        domain_counter (dict): per-domain spot counter, format: domain_counter[cellID][chrom][traceID][domain] = nspot
        cellID (str)
        chrom (str)
        traceID (str)
        domain (tuple): chrom (str), start (int), end (int)
        max_nspot_per_domain (int): maximum number of spots per domain

    Returns:
        domain_counter (dict): updated per-domain spot counter.
        max_nspot_per_domain (int): updated maximum number of spots per domain.
    """
    
    # Case 1: cellID not present
    if cellID not in domain_counter:
        # Add cellID - chrom - traceID - domain to counter
        domain_counter[cellID] = {chrom: {traceID: {domain: 1}}}
        # Update max_nspot_per_domain to 1 if necessary
        max_nspot_per_domain = max(max_nspot_per_domain, 1)
    
    # Case 2: chrom not present in cellID
    elif chrom not in domain_counter[cellID]:
        # Add chrom - traceID - domain to cellID counter
        domain_counter[cellID][chrom] = {traceID: {domain: 1}}
        # Update max_nspot_per_domain to 1 if necessary
        max_nspot_per_domain = max(max_nspot_per_domain, 1)
    
    # Case 3: traceID not present in cellID - chrom
    elif traceID not in domain_counter[cellID][chrom]:
        # Add traceID - domain to chrom counter
        domain_counter[cellID][chrom][traceID] = {domain: 1}
        # Update max_nspot_per_domain to 1 if necessary
        max_nspot_per_domain = max(max_nspot_per_domain, 1)
    
    # Case 4: domain not present in cellID - chrom - traceID
    elif domain not in domain_counter[cellID][chrom][traceID]:
        # Add domain to traceID counter
        domain_counter[cellID][chrom][traceID][domain] = 1
        # Update max_nspot_per_domain to 1 if necessary
        max_nspot_per_domain = max(max_nspot_per_domain, 1)
    
    # Case 5: domain already present, increase spot count by 1
    else:
        nspot = domain_counter[cellID][chrom][traceID][domain]
        domain_counter[cellID][chrom][traceID][domain] = nspot + 1
        # Update max_nspot_per_domain to nspot + 1 if necessary
        max_nspot_per_domain = max(max_nspot_per_domain, nspot + 1)
    
    return domain_counter, max_nspot_per_domain


# FUNCTIONS TO CONVERT BETWEEN DICTIONARY AND NUMPY ARRAY FORMAT

def chrom_dict_to_numpy(chrom_data: dict):
    """Convert the data of a single chromosome from dictionary format to numpy array format.

    Args:
        chrom_data (dict): Chromosome data in dictionary format:
                             chrom_data[traceID][spotID] = {'x': x,
                                                            'y': y,
                                                            'z': z,
                                                            'chrom': chrom,
                                                            'start': start,
                                                            'end': end,
                                                            'lum': lum}

    Returns:
        xs (np.array, float): x coordinates of the spots.
        ys (np.array, float): y coordinates of the spots.
        zs (np.array, float): z coordinates of the spots.
        starts (np.array, int): start genomic position of the spots.
        ends (np.array, int): end genominc position of the spots.
        lums (np.array, float): intensities of the spots.
        traceIDs (np.array, str): trace IDs of the spots.
        spotIDs (np.array, str): spot IDs of the spots.
    """
    
    # Initialize lists
    xs, ys, zs, starts, ends, lums, traceIDs, spotIDs = [], [], [], [], [], [], [], []
    
    for traceID in chrom_data:
        
        for spotID in chrom_data[traceID]:
            
            spot_data = chrom_data[traceID][spotID]
                
            xs.append(spot_data['x'])
            ys.append(spot_data['y'])
            zs.append(spot_data['z'])
            starts.append(spot_data['start'])
            ends.append(spot_data['end'])
            lums.append(spot_data['lum'])
            traceIDs.append(traceID)
            spotIDs.append(spotID)
    
    xs = np.array(xs).astype(float)
    ys = np.array(ys).astype(float)
    zs = np.array(zs).astype(float)
    starts = np.array(starts).astype(int)
    ends = np.array(ends).astype(int)
    lums = np.array(lums).astype(float)
    traceIDs = np.array(traceIDs).astype('U20')
    spotIDs = np.array(spotIDs).astype('U20')
    
    return xs, ys, zs, starts, ends, lums, traceIDs, spotIDs

def chrom_numpy_to_dict(chrom: str,
                        xs: np.ndarray,
                        ys: np.ndarray,
                        zs: np.ndarray,
                        starts: np.ndarray,
                        ends: np.ndarray,
                        lums: np.ndarray,
                        traceIDs: np.ndarray,
                        spotIDs: np.ndarray):
    """Convert the data of a single chromosome from numpy array format to dictionary format.

    Args:
        chrom (str): chromosome name.
        xs (np.ndarray, float): x coordinates of the spots.
        ys (np.ndarray, float): y coordinates of the spots.
        zs (np.ndarray, float): z coordinates of the spots.
        starts (np.ndarray, int): start genomic position of the spots.
        ends (np.ndarray): end genominc position of the spots.
        lums (np.ndarray): intensities of the spots.
        traceIDs (np.ndarray): trace IDs of the spots.
        spotIDs (np.ndarray): spot IDs of the spots.
    
    Returns:
        chrom_data (dict): Chromosome data in dictionary format:
                             chrom_data[traceID][spotID] = {'x': x,
                                                            'y': y,
                                                            'z': z,
                                                            'chrom': chrom,
                                                            'start': start,
                                                            'end': end,
                                                            'lum': lum}
    """
    
    chrom_data = {}
    
    for traceID in np.unique(traceIDs):
        
        chrom_data[traceID] = {}
        
        # Loop over the indices where traceID == traceID
        for i in np.where(traceIDs == traceID)[0]:
            
            spot_data_i = {'x': float(xs[i]),
                           'y': float(ys[i]),
                           'z': float(zs[i]),
                           'chrom': str(chrom),
                           'start': int(starts[i]),
                           'end': int(ends[i]),
                           'lum': float(lums[i])
                           }
            
            
            chrom_data[traceID][spotIDs[i]] = spot_data_i

    return chrom_data

def cell_to_numpy(cell_data: dict):
    """Convert the data of a single cell from dictionary format to numpy array format.

    Args:
        cell_data (dict): Cell data in dictionary format:
                          cell_data[chrom][traceID][spotID] = {'x': x,
                                                               'y': y,
                                                               'z': z,
                                                               'chrom': chrom,
                                                               'start': start,
                                                               'end': end,
                                                               'lum': lum}

    Returns:
        xs (np.array, float): x coordinates of the spots.
        ys (np.array, float): y coordinates of the spots.
        zs (np.array, float): z coordinates of the spots.
        chroms (np.array, str): chromosome names of the spots.
        starts (np.array, int): start genomic position of the spots.
        ends (np.array, int): end genominc position of the spots.
        lums (np.array, float): intensities of the spots.
        traceIDs (np.array, str): trace IDs of the spots.
        spotIDs (np.array, str): spot IDs of the spots.
    """
    
    # Initialize lists
    xs, ys, zs, chroms, starts, ends, lums, traceIDs, spotIDs = [], [], [], [], [], [], [], [], []
    
    for chrom in cell_data:
    
        for traceID in cell_data[chrom]:
            
            for spotID in cell_data[chrom][traceID]:
                
                spot_data = cell_data[chrom][traceID][spotID]
                
                xs.append(spot_data['x'])
                ys.append(spot_data['y'])
                zs.append(spot_data['z'])
                chroms.append(spot_data['chrom'])
                starts.append(spot_data['start'])
                ends.append(spot_data['end'])
                lums.append(spot_data['lum'])
                traceIDs.append(traceID)
                spotIDs.append(spotID)
    
    xs = np.array(xs).astype(float)
    ys = np.array(ys).astype(float)
    zs = np.array(zs).astype(float)
    chroms = np.array(chroms).astype('U20')
    starts = np.array(starts).astype(int)
    ends = np.array(ends).astype(int)
    lums = np.array(lums).astype(float)
    traceIDs = np.array(traceIDs).astype('U20')
    spotIDs = np.array(spotIDs).astype('U20')
    
    return xs, ys, zs, chroms, starts, ends, lums, traceIDs, spotIDs

def trace_dict_to_numpy(trace_data: dict):
    """ Convert the data of a single trace from dictionary format to numpy array format.

    Args:
        trace_data (dict)

    Returns:
        xs (np.array, float): x coordinates of the spots.
        ys (np.array, float): y coordinates of the spots.
        zs (np.array, float): z coordinates of the spots.
        chroms (np.array, str): chromosome names of the spots.
        starts (np.array, int): start genomic position of the spots.
        ends (np.array, int): end genominc position of the spots.
        lums (np.array, float): intensities of the spots.
        spotIDs (np.array, str): spot IDs of the spots.
    """
    
    # Initialize lists
    xs, ys, zs, chroms, starts, ends, lums, spotIDs = [], [], [], [], [], [], [], []
    
    for spotID in trace_data:
            
            spot_data = trace_data[spotID]
            
            xs.append(spot_data['x'])
            ys.append(spot_data['y'])
            zs.append(spot_data['z'])
            chroms.append(spot_data['chrom'])
            starts.append(spot_data['start'])
            ends.append(spot_data['end'])
            lums.append(spot_data['lum'])
            spotIDs.append(spotID)
    
    xs = np.array(xs).astype(float)
    ys = np.array(ys).astype(float)
    zs = np.array(zs).astype(float)
    chroms = np.array(chroms).astype('U20')
    starts = np.array(starts).astype(int)
    ends = np.array(ends).astype(int)
    lums = np.array(lums).astype(float)
    spotIDs = np.array(spotIDs).astype('U20')
    
    return xs, ys, zs, chroms, starts, ends, lums, spotIDs


# FUNCTIONS FOR 3D MEDIAN

def spots_3d_median(points: np.ndarray, centroid: np.ndarray):
    """ Given a list of spot points associated with the same domain in a trace,
    selects only one of them with the 3D median criterion.
    
    There are three cases:
        1) If there is only one point, return the index of that point (0)
        2) If there are two points, return the one closer to the centroid
        3) If there are three or more points, return the point that minimizes the sum of distances to all other points
    
    The function returns the index of the selected point.
    
    The computation for point 3, the actual 3D median, is based on the 3D geometric median (https://en.wikipedia.org/wiki/Geometric_median).
    However, in this reference the algorithm finds the point in 3D space that minimizes the sum of distances to all other points,
    so it doesn't return a point of the set. Here we don't want to create 'fake' points, so I adapted the algorithm to only
    consider the points in the set.
    
    Args:
        points (np.ndarray): array of shape (npoints, 3) containing the 3D coordinates of the spots.
        centroid (np.ndarray): array of shape (3,) containing the 3D coordinates of the centroid of the trace.

    Returns:
        median_idx (int): index of the selected point, between 0 and npoints-1.
    """
    
    # Check the points array
    if not isinstance(points, np.ndarray):
        raise TypeError('points must be a numpy array')
    if len(points.shape) != 2:
        raise ValueError('points must be a numpy array of shape (npoints, 3)')
    if points.shape[1] != 3:
        raise ValueError('points must be a numpy array of shape (npoints, 3)')
    if points.shape[0] == 0:
        raise ValueError('points must be a numpy array of shape (npoints, 3)')
    npoints = points.shape[0]  # get the number of points
    
    # Check the centroid array
    if not isinstance(centroid, np.ndarray):
        raise TypeError('centroid must be a numpy array')
    if centroid.shape != (3,):
        raise ValueError('centroid must be a numpy array of shape (3,)')
    
    # If there is only one point, return the index of that point (0)
    if npoints == 1:
        return 0
    
    # If there are two points, return the one closer to the centroid
    elif npoints == 2:
        median_idx = 0 if distance.euclidean(points[0], centroid) <= distance.euclidean(points[1], centroid) else 1
        return median_idx
    
    # Otherwise, find the point that minimizes the sum of distances to all other points
    # Initialize the median index and the minimum distance
    median_idx = None
    dists_min = np.inf
    # Loop over the points
    for i in range(len(points)):
        dists_i = 0  # initialize the sum of distances for point i
        for j in range(len(points)):
            if i == j:
                continue
            dists_i = dists_i + distance.euclidean(points[i], points[j])  # add the distance between point i and point j
        # If the total distance is smaller than the minimum distance, update the median index and the minimum distance
        if dists_i < dists_min:
            median_idx = i
            dists_min = dists_i
        # If the total distances are equal, choose the point closer to the centroid
        if dists_i == dists_min:
            if distance.euclidean(points[i], centroid) < distance.euclidean(points[median_idx], centroid):
                median_idx = i
                dists_min = dists_i
    return median_idx
