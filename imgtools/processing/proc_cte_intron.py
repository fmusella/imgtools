import numpy as np
from ..cte import ChromatinTracingExperiment
from ..cte import cte_utils
from .. import parallel


# TRACING INTRON CTE TO DNA CTE

intron_tracing_required_keys = {
    'cte_traced_out_name': {'type': str},
    'cte_dna_name': {'type': str},
    'thresh': {'type': float},
}

def chromosome_tracing(
    cellID: str, chrom: str,
    chrom_rna_data: dict, cte_dna: ChromatinTracingExperiment,
    thresh: float
) -> dict:
    """ Function with the calculations to trace intron spots to DNA traces.

    Args:
        cellID (str)
        chrom (str)
        chrom_rna_data (dict): chromosome data for intron spots, in dictionary format
        cte_dna (ChromatinTracingExperiment): CTE object for the DNA data
        thresh (float): threshold for the distance between the intron spot and the closest trace spot

    Returns:
        (dict): chromosome data for intron spots with assigned traceIDs, in dictionary format
        If no spots are left after filtering, returns None.
    """
    
    # Get the data for both DNA and intron
    d = cte_dna.get_data(cellID, chrom, format='numpy')
    xs_dna, ys_dna, zs_dna, starts_dna, ends_dna, traceIDs_dna = d['xs'], d['ys'], d['zs'], d['starts'], d['ends'], d['traceIDs']
    d = cte_utils.chrom_dict_to_numpy(chrom_rna_data)
    xs_rna, ys_rna, zs_rna, starts_rna, ends_rna, lums_rna, spotIDs_rna, geneIDs_rna = d['xs'], d['ys'], d['zs'], d['starts'], d['ends'], d['lums'], d['spotIDs'], d['geneIDs']
    
    # Hash the CTE DNA data, creating a dictionary with the format:
    #   { (start, end): { 'trace_1': [[x_11, y_11, z_11], [x_12, y_12, z_12], ...], 'trace_2': [[x_21, y_21, z_21], [x_22, y_22, z_22], ...], ... } }
    hash_data_dna = {}
    for x, y, z, start, end, traceID in zip(xs_dna, ys_dna, zs_dna, starts_dna, ends_dna, traceIDs_dna):
        if (start, end) not in hash_data_dna:
            hash_data_dna[(start, end)] = {}
        if traceID not in hash_data_dna[(start, end)]:
            hash_data_dna[(start, end)][traceID] = []
        hash_data_dna[(start, end)][traceID].append(np.array([x, y, z]))
    
    # Loop through the intron data, assign each spot to the trace with the shortest distance.
    # If the shortest distance is larger than a threshold, assign to '-1' (unassigned).
    traceIDs_rna = []
    for x, y, z, start, end in zip(xs_rna, ys_rna, zs_rna, starts_rna, ends_rna):
        crd_rna = np.array([x, y, z])  # coordinates of the intron spot
        
        # If the domain (start, end) is not in the hash_cte, assign '-1'
        if (start, end) not in hash_data_dna:
            traceIDs_rna.append('-1')
            continue
        
        # Initialize the assigned traceID and the minimum distance between
        # the intron spot and the closest trace spot
        traceID_assigned = '-1'
        min_dist = np.inf
        
        # Calculate the closest traceID for the intron spot
        for traceID, crds in hash_data_dna[(start, end)].items():
            for crd in crds:
                dist = np.linalg.norm(crd_rna - crd)
                if dist < min_dist:
                    min_dist = dist
                    traceID_assigned = traceID
        
        # If the minimum distance is larger than the threshold, assign '-1'
        if min_dist > thresh:
            traceID_assigned = '-1'
        
        # Store the assigned traceID for the intron spot
        traceIDs_rna.append(traceID_assigned)
    traceIDs_rna = np.array(traceIDs_rna).astype(str)
    
    # Remove the intron spots that are not assigned to any trace
    mask = traceIDs_rna != '-1'
    xs_rna, ys_rna, zs_rna = xs_rna[mask], ys_rna[mask], zs_rna[mask]
    starts_rna, ends_rna = starts_rna[mask], ends_rna[mask]
    traceIDs_rna, spotIDs_rna = traceIDs_rna[mask], spotIDs_rna[mask]
    geneIDs_rna = geneIDs_rna[mask]
    
    # If there are no spots left after filtering, return None
    if len(xs_rna) == 0:
        return None
    
    # Convert the intron data to a dictionary format to create a new CTE
    chrom_rna_data_traced = cte_utils.chrom_numpy_to_dict(
        chrom,
        xs_rna, ys_rna, zs_rna,
        starts_rna, ends_rna,
        lums_rna,
        traceIDs_rna, spotIDs_rna,
        geneIDs_rna
    )
    
    return chrom_rna_data_traced


def run_intron_tracing_single_chrom(
    cellID: str, chrom: str, cte_rna: ChromatinTracingExperiment, config: dict
) -> ChromatinTracingExperiment:
    """ Run the tracing of intron spots to DNA traces for a single chromosome.
    
    This function calls the 'chromosome_tracing' function, but uses the same format as
    'run_intron_tracing' to allow for simple testing.
    
    The new CTE with the traced intron data is created and returned.

    Args:
        cellID (str)
        chrom (str)
        cte_rna (ChromatinTracingExperiment): CTE for the (untraced) intron RNA data
        config (dict): configuration dictionary with the following keys:
            - 'cte_traced_out_name' (str): name of the CTE to save the traced intron data
            - 'cte_dna_name' (str): name of the CTE with the DNA data
            - 'thresh' (float): threshold for the distance between the intron spot and the closest trace spot

    Returns:
        ChromatinTracingExperiment: a new CTE with the traced intron data for the specified cellID and chromosome.
    """
    
    parallel.check_config(config, intron_tracing_required_keys, parallel=False)
    
    # Get the DNA CTE data to use for reference
    cte_dna = ChromatinTracingExperiment(config['cte_dna_name'], 'r')
    
    # Read the data for the specified cellID
    cell_rna_data = cte_rna.get_data(cellID)
    
    # Perform the tracing on the specified chromosome
    chrom_rna_data_traced = chromosome_tracing(
        cellID, chrom, cell_rna_data[chrom], cte_dna, config['thresh']
    )
    
    # If there are no spots, return None
    if chrom_rna_data_traced is None:
        return None
    
    # Otherwise, create a new CTE with the traced data
    cte_rna_traced = ChromatinTracingExperiment(config['cte_traced_out_name'], 'w')  
    # Add the traced data to the new ChromatinTracingExperiment object
    cte_rna_traced.set_data_attrs_index(data={cellID: {chrom: chrom_rna_data_traced}}, index=cte_rna.index)
    
    return cte_rna_traced


def run_intron_tracing(cte_rna: ChromatinTracingExperiment, config: dict) -> ChromatinTracingExperiment:
    """ Run the tracing of intron spots to DNA traces for all cells in parallel.
    
    The parallelization is performed across cells.
    
    Inside each cell, the function loops across chromosomes, and calls
    the 'chromosome_tracing' function for each.
    
    The parallelization pipeline returns a data dictionary with the traced intron data for all cells.
    Then, a new ChromatinTracingExperiment object is created with the traced data.

    Args:
        cte_rna (ChromatinTracingExperiment): CTE for the (untraced) intron RNA data
        config (dict): configuration dictionary with the following keys:
            - 'cte_traced_out_name' (str): name of the CTE to save the traced intron data
            - 'cte_dna_name' (str): name of the CTE with the DNA data
            - 'thresh' (float): threshold for the distance between the intron spot and the closest trace spot

    Returns:
        ChromatinTracingExperiment: a new CTE with the traced intron data for all cells.
    """
    
    # Calculate the traced intron data for all cells in parallel
    data_rna_traced = parallel.control_func(
        cte_rna, None,
        config, intron_tracing_required_keys,
        func_node, reduce_initialization, reduce_update,
    )
    
    # Create the ChromatinTracingExperiment object for the traced data
    cte_rna_traced = ChromatinTracingExperiment(config['cte_traced_out_name'], 'w')  
    cte_rna_traced.set_data_attrs_index(data=data_rna_traced, index=cte_rna.index)
    
    return cte_rna_traced

def func_node(cellID: str, cte_intron_name: str, _, config: dict) -> dict:
    """ Node-level function to trace intron spots to DNA traces for a single cell.

    Args:
        cellID (str)
        cte_intron_name (str): name of the CTE with the intron RNA data
        _ : not used, just to match the signature of the function
        config (dict): configuration dictionary.

    Returns:
        dict: a dictionary with the traced intron data for the specified cellID.
    """
    
    # Get the DNA CTE data to use for reference
    cte_dna = ChromatinTracingExperiment(config['cte_dna_name'], 'r')
    
    # Open the Chromatin Tracing Experiment for intron
    cte_rna = ChromatinTracingExperiment(cte_intron_name, 'r')
    # Read the data for the specified cellID
    cell_rna_data = cte_rna.get_data(cellID)
    
    # Initialize a dictionary to store the cell traced data
    cell_rna_data_traced = {}
    
    # Loop through the chromosomes
    for chrom in cell_rna_data:
        
        # Get the traced data for the specified chromosome
        chrom_rna_data_traced = chromosome_tracing(
            cellID, chrom, cell_rna_data[chrom], cte_dna, config['thresh']
        )
        # If there are no spots, skip the chromosome
        if chrom_rna_data_traced is None:
            continue
        # Add the traced data to the cell data dictionary
        cell_rna_data_traced[chrom] = chrom_rna_data_traced
    
    return cell_rna_data_traced

def reduce_initialization(_1, _2, _3, _4) -> dict:
    """ Initialization function for the reduction step in parallel processing.
    Simply returns an empty dictionary.

    Args:
        _*: not used, just to match the signature of the function

    Returns:
        dict: an empty dictionary to store the traced data for all cells.
    """
    return {}

def reduce_update(cellID: str, data_traced: dict, cell_data_traced: dict, _1, _2, _3) -> dict:
    """ Update function for the reduction step in parallel processing.
    Adds the traced data for the specified cellID to the data_traced dictionary.

    Args:
        cellID (str)
        data_traced (dict): traced data for all cells, in dictionary format
        cell_data_traced (dict): traced data for the specified cellID, in dictionary format
        _*: not used, just to match the signature of the function

    Returns:
        dict: updated traced data for all cells, in dictionary format
    """
    data_traced[cellID] = cell_data_traced
    return data_traced
