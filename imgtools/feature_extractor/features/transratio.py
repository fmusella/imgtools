import numpy as np
from scipy.spatial.distance import cdist
from ...cte import ChromatinTracingExperiment
from ...cte import cte_utils

docstring = """Measures the trans ratio within a sphere centered at each spot.
A spot is defined as 'trans' with respect to the central spot if it belongs to a different
chromosome, or to the same chromosome but a different copy (i.e. a different trace).
A spot is 'cis' only if it belongs to the same chromosome and the same copy (trace) as the central spot.
The trans ratio is the number of trans spots within the sphere divided by the total number of
spots within the sphere (excluding the central spot itself).
If there are no other spots within the sphere, the feature value is kept as NaN."""

required_keys = {
    'radius': {'type': float, 'positive': True},
}

def run(cellID: str, cte: ChromatinTracingExperiment, config: dict, feat_arr: np.ndarray, _) -> np.ndarray:
    """ For each spot, measure the ratio of trans spots within a sphere centered at the spot.

    A spot is 'cis' with respect to the central spot only if it belongs to the same chromosome
    and the same copy (trace); otherwise it is 'trans' (different chromosome, or same chromosome
    but different copy). The trans ratio is the number of trans spots within the sphere divided
    by the total number of spots within the sphere (excluding the central spot itself).

    If two or more spots are mapped to the same domain, the average of the values is taken.

    Args:
        cellID (str)
        cte (ChromatinTracingExperiment)
        config (dict): configuration dictionary with the following keys:
            - radius (float): radius of the sphere centered at the spot.
        feat_arr (np.ndarray): initialized nan-valued array of shape (n_domains, n_traces) to store the feature value.
        _: not used, just to match the function signature

    Returns:
        (np.ndarray): updated array of shape (n_domains, n_traces) with the feature values.
    """

    # Get the parameters from the configuration
    radius = config['radius']

    # Get the cell data in dictionary format
    cell_data = cte.get_data(cellID)

    # Get the traceID hash table to map traces to their position in the array
    traceID_hash = cte.get_trace_hashmap(cellID)

    # Convert the cell data in numpy format and get the coordinates of each spot,
    # as well as the chromosome and trace ID of each spot (needed to define cis/trans)
    d = cte_utils.cell_dict_to_numpy(cell_data)
    xs, ys, zs, chroms, traceIDs = d['xs'], d['ys'], d['zs'], d['chroms'], d['traceIDs']
    crds = np.array([xs, ys, zs]).T

    # Get the index and its hash table
    index = cte.index
    index_hash = index.get_index_hashmap()

    # Initialize a dictionary to store the feature values for each domain (we will then take the average)
    feat_per_domain = {}

    for chrom in cell_data:
        for traceID in cell_data[chrom]:

            # Get the position of the trace in the array
            i_trace = traceID_hash[chrom][traceID]

            for spotID in cell_data[chrom][traceID]:

                # Unpack the spot data
                spot_data = cell_data[chrom][traceID][spotID]
                x, y, z = spot_data['x'], spot_data['y'], spot_data['z']
                start, end = spot_data['start'], spot_data['end']

                # Calculate the distance of this spot to all the other spots
                point = np.array([[x, y, z]])
                dists = cdist(point, crds).flatten()

                # Get the mask to filter the spots within the sphere
                mask_in_sphere = dists < radius

                # Get the total number of spots in the sphere (excluding the central spot itself)
                ncontacts = np.sum(mask_in_sphere) - 1

                # If there are no other spots in the sphere, skip this spot (the feature value is kept as NaN)
                if ncontacts == 0:
                    continue

                # Get the chromosome and trace ID of the spots within the sphere
                chroms_in_sphere = chroms[mask_in_sphere]
                traceIDs_in_sphere = traceIDs[mask_in_sphere]

                # A spot is trans if it is on a different chromosome, or on the same chromosome
                # but a different copy (trace). The central spot itself is cis and is therefore
                # naturally excluded from the trans count.
                trans_mask = (chroms_in_sphere != chrom) | (traceIDs_in_sphere != traceID)
                ncontacts_trans = np.sum(trans_mask)
                feat_val = ncontacts_trans / ncontacts

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
