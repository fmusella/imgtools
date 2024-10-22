import os
import sys
import pickle
import tempfile
from functools import partial
import numpy as np
from alabtools.parallel import Controller
from ..cte import ChromatinTracingExperiment
from ..scf import SingleCellFeature
from .cte_impute_utils import impute_cte_trace_data
from .scf_impute_utils import impute_scf_trace_data



# CTE IMPUTATION

def run_CTE_imputation(cte: ChromatinTracingExperiment, config: dict) -> ChromatinTracingExperiment:
    """ Performs the imputation of the CTE data.
    
    Missing data points are interpolated using a linear interpolation between the two
    closest data points to their left and right.
    
    If there are no spots either on the left or on the right, the coordinates
    are simply copied from the closest spot.
    
    Imputed spots are assigned a new spotID, always starting with 'IMPUTED_'.
    
    The luminescence intensity of imputed spots is set to NaN.
    
    Args:
        cte (ChromatinTracingExperiment)

    Returns:
        ChromatinTracingExperiment: a new ChromatinTracingExperiment object with the imputed data.
    """
    
    # Create a temporary directory
    tempdir = tempfile.mkdtemp(dir=os.getcwd())
    sys.stdout.write("Temporary directory for nodes' results: {}\n".format(tempdir))
    
    # create a Controller
    controller = Controller(config)
    
    # Get the triad labels, i.e. the cellID / chrom / traceID combinations
    triad_labels = cte.get_triad_labels()
    
    # Run the parallel and reduce tasks
    parallel_task = partial(
        parallel_cte_imputation,
        cte_name = cte.h5_name,
        tempdir = tempdir
    )
    reduce_task = partial(
        reduce_cte_imputation,
        cte = cte,
        tempdir = tempdir
    )
    cte_imp = controller.map_reduce(
        parallel_task,
        reduce_task,
        args = list(triad_labels)
    )
    
    # Delete the non-empty temporary directory
    os.system('rm -r {}'.format(tempdir))
    
    return cte_imp

def parallel_cte_imputation(triadID: np.ndarray, cte_name: str, tempdir: str) -> tuple:
    """ Parallel function for the imputation of the CTE data.
    
    Acts on the data of a single trace, provided by the triadID (cellID, chrom, traceID).
    
    The result is a dictionary, saved in a pickle file in the temporary directory.

    Args:
        triadID (np.ndarray): cellID, chrom, traceID array of str. shape: (3,)
        cte_name (str)
        tempdir (str)

    Returns:
        cellID (str)
        chrom (str)
        traceID (str)
    """
    
    # Get the cellID, chrom, and traceID from the triadID
    try:
        cellID, chrom, traceID = triadID
    except:
        raise ValueError(f"Parallel function: triadID is wrong: {triadID}")
    
    # Read the CTE file
    cte = ChromatinTracingExperiment(cte_name, 'r')
    
    # Get the trace data
    trace_data = cte.get_data(cellID, chrom, traceID)
    
    # Impute the trace data
    trace_data_imp = impute_cte_trace_data(trace_data, cte.index)
    
    # Save the result in the temporary directory as a pickle file
    out_filename = os.path.join(tempdir, f'{cellID}_{chrom}_{traceID}.pkl')
    with open(out_filename, 'wb') as f:
        pickle.dump(trace_data_imp, f)
    
    cte.close()
    
    return cellID, chrom, traceID

def reduce_cte_imputation(triadIDs: list, cte: ChromatinTracingExperiment, tempdir: str) -> ChromatinTracingExperiment:
    """ Reduce function for the imputation of the CTE data.
    
    Create the imputed CTE object.
    
    It iterates over the triadIDs and collects the imputed data for each chrom / traceID pair.
    The imputed data is then added to the imputed CTE object cell by cell.

    Args:
        triadIDs (list): list of triadIDs (cellID, chrom, traceID) from the parallel functions.
        cte (ChromatinTracingExperiment): the original ChromatinTracingExperiment object.
        tempdir (str)

    Returns:
        ChromatinTracingExperiment: a new ChromatinTracingExperiment object with the imputed data.
    """
    
    # Make sure that the returns of the parallel functions are correct
    assert isinstance(triadIDs, list), "Reduce function: triadIDs should be a list. Got type: {}".format(type(triadIDs))
    assert len(triadIDs) > 0, "Reduce: triadIDs list should not be empty."
    
    # Create a CTE object for the imputed data
    cte_imp_h5name = cte.h5_name.replace('.h5', '_imputed.h5')
    cte_imp = ChromatinTracingExperiment(cte_imp_h5name, 'w')
    
    # Add basic data to the imputed CTE (index, cell_labels, attrs)
    cte_imp.set_index(cte.index)
    cte_imp.set_attrs(cte.attrs)
    cte_imp.set_cell_labels(cte.cell_labels)
    # Add the triad_labels to the imputed CTE
    cte_imp.set_triad_labels(cte.get_triad_labels())
    # Add cell_states and alphashapes if they exist
    if 'cell_states' in cte:
        cte_imp.set_cell_states(cte.cell_states)
    if 'alphashapes' in cte:
        cte_imp.set_alphashapes(cte.get_alphashapes())
    
    # We are going to add the CTE data cell by cell, because the whole data would be too big
    # To do so, we need to iterate over each cell and collect the imputed data for each chrom / traceID pair
    # So here we hash the triadIDs by cellID, so that we can easily get all the data for a cell
    # The structure of the hash is: {cellID: [(chrom, traceID), ...]}
    triad_labels_hash = {}
    for (cellID, chrom, traceID) in triadIDs:
        if cellID not in triad_labels_hash:
            triad_labels_hash[cellID] = []
        triad_labels_hash[cellID].extend([(chrom, traceID)])
    
    # Iterate over the cellIDs
    for cellID in triad_labels_hash:
        
        # Initialize the imputed cell data
        cell_data_imp = {}
        
        # Iterate over the chrom / traceID pairs of the cell
        for (chrom, traceID) in triad_labels_hash[cellID]:
        
            # Get the filename of the imputed trace data
            filename = os.path.join(tempdir, f'{cellID}_{chrom}_{traceID}.pkl')
            assert os.path.isfile(filename), f"Parallel result file for {cellID}, {chrom}, {traceID} not found."
            
            # Load the imputed trace data
            with open(filename, 'rb') as f:
                trace_data_imp = pickle.load(f)
            
            # Add the imputed trace data to the imputed cell data
            if chrom not in cell_data_imp:
                cell_data_imp[chrom] = {}
            cell_data_imp[chrom][traceID] = trace_data_imp
        
        # Add the imputed cell data to the imputed CTE
        cte_imp.set_cell_data(cellID, cell_data_imp)
    
    return cte_imp


def run_CTE_imputation_single_trace(
    cte: ChromatinTracingExperiment, cellID: str, chrom: str, traceID: str
) -> ChromatinTracingExperiment:
    """Performs the imputation on a single chromosomal trace of a cell.

    Args:
        cellID (str)
        chrom (str)
        traceID (str)
    
    Returns:
        (ChromatinTracingExperiment): a new ChromatinTracingExperiment object,
                                    with just the imputated data of the specified trace.
    """
    
    # Get the data from the CTE
    trace_data = cte.get_data(cellID, chrom, traceID)
    index = cte.index
    
    # Perform the imputation of the trace data
    trace_data_imp = impute_cte_trace_data(trace_data, index)
    
    # Create a new CTE object
    cte_imp_h5name = cte.h5_name.replace('.h5', f'_imputed_{cellID}_{chrom}_{traceID}.h5')
    cte_trace_imp = ChromatinTracingExperiment(cte_imp_h5name, 'w')
    
    # Add the imputed data to the new CTE object
    cte_trace_imp.set_data_attrs_index(
        data={cellID: {chrom: {traceID: trace_data_imp}}},
        index=index
    )
    
    return cte_trace_imp




# SCF IMPUTATION

def run_SCF_imputation(scf: SingleCellFeature, config: dict) -> np.ndarray:
    """ Performs the imputation of the SCF data for a single feature.
    
    Missing feature values are interpolated using a linear interpolation between the two
    closest data points to their left and right.
    
    If there are no spots either on the left or on the right, the values
    are simply copied from the closest spot.
    
    Args:
        scf (SingleCellFeature)

    Returns:
        np.ndarray: imputed feature matrix. shape: (ncells, ndomains, ncopies)
    """
    
    # Make sure that config has a valid 'feature' key
    if 'feature' not in config:
        raise KeyError("run_SCF_imputation: 'feature' key not found in config.")
    if config['feature'] not in scf.features:
        raise KeyError(f"run_SCF_imputation: feature '{config['feature']}' not found in SCF.")
    
    # Create a temporary directory
    tempdir = tempfile.mkdtemp(dir=os.getcwd())
    sys.stdout.write("Temporary directory for nodes' results: {}\n".format(tempdir))
    
    # create a Controller
    controller = Controller(config)
    
    # Run the parallel and reduce tasks
    parallel_task = partial(
        parallel_scf_imputation,
        scf_name = scf.h5_name,
        feature = config['feature'],
        tempdir = tempdir
    )
    reduce_task = partial(
        reduce_scf_imputation,
        scf_name = scf.h5_name,
        tempdir = tempdir
    )
    featmat_imp = controller.map_reduce(
        parallel_task,
        reduce_task,
        args = list(scf.cell_labels)
    )
    
    # Delete the non-empty temporary directory
    os.system('rm -r {}'.format(tempdir))
    
    return featmat_imp

def parallel_scf_imputation(cellID: np.ndarray, scf_name: str, feature: str, tempdir: str) -> tuple:
    """ Parallel function for the imputation of the SCF data.
    
    Acts on the data of a single cell, provided by the cellID.
    It iterates over the chromosomes and copies of the cell,
    and for each chromosome/copy feature vector, it calculates
    the imputed feature vector.
    
    The result is a numpy array, saved in a pickle file in the temporary directory.

    Args:
        cellID (str)
        scf_name (str)
        feature (str)
        tempdir (str)

    Returns:
        cellID (str)
    """
    
    # Read the SCF file
    scf = SingleCellFeature(scf_name, 'r')
    index = scf.index
    
    # Get the feature matrix for the cell
    featmat = scf.get_feature(feature, cellID)  # shape: (ndomains, ncopies)
    ndomains, ncopies = featmat.shape
    # Initialize the imputed feature matrix
    featmat_imp = np.copy(featmat)  # shape: (ndomains, ncopies)
    
    # Loop over the chromosomes
    for chrom in index.genome.chroms:
        
        # Get the domain mask for the chromosome
        chrom_mask = index.chromstr == chrom
        # Get the chromosome-specific feature matrix
        featmat_chrom = featmat_imp[chrom_mask]  # shape: (ndomains_chrom, ncopies)
        # Initialize the imputed feature matrix for the chromosome
        featmat_chrom_imp = np.copy(featmat_chrom)
        
        # Loop over the copies
        for copy in range(ncopies):
            
            # Get the copy-specific feature vector
            featarr_chrom_copy = featmat_chrom[:, copy]  # shape: (ndomains_chrom,)
            # If the feature vector is all NaN, skip the imputation
            if np.all(np.isnan(featarr_chrom_copy)):
                continue
            
            # Impute the feature vector
            featarr_chrom_copy_imp = impute_scf_trace_data(featarr_chrom_copy, index)
            # Copy the imputed feature vector to the imputed feature matrix
            featmat_chrom_imp[:, copy] = featarr_chrom_copy_imp
        
        # Copy the imputed chromosome-specific feature matrix to the imputed feature matrix
        featmat_imp[chrom_mask] = featmat_chrom_imp
    
    # Save the result in the temporary directory as a pickle file
    out_filename = os.path.join(tempdir, f'{cellID}.pkl')
    with open(out_filename, 'wb') as f:
        pickle.dump(featmat_imp, f)
    
    scf.close()
    
    return cellID

def reduce_scf_imputation(cellIDs: list, scf_name: str, tempdir: str) -> np.ndarray:
    """ Reduce function for the imputation of the SCF data.
    
    Creates the imputed feature matrix of the whole population.
    Iterates over the single-cell results of the parallel functions
    and updates population-wide data.

    Args:
        cellIDs (list): list of cellIDs from the parallel functions.
        scf_name (str)
        tempdir (str)

    Returns:
        np.ndarray: imputed feature matrix. shape: (ncells, ndomains, ncopies)
    """

    # Make sure that the returns of the parallel functions are correct
    assert isinstance(cellIDs, list), "Reduce function: cellIDs should be a list. Got type: {}".format(type(cellIDs))
    assert len(cellIDs) > 0, "Reduce: cellIDs list should not be empty."
    ncells = len(cellIDs)
    
    # Open the SCF file (needed to convert cellIDs to cell numbers)
    scf = SingleCellFeature(scf_name, 'r')
    
    # Open the first pickle file to get the shape of the feature matrix
    filename = os.path.join(tempdir, f'{cellIDs[0]}.pkl')
    with open(filename, 'rb') as f:
        featmat_cell_imp = pickle.load(f)
    ndomains, ncopies = featmat_cell_imp.shape
    
    # Initialize the imputed feature matrix of the whole population
    featmat_imp = np.full((ncells, ndomains, ncopies), np.nan)
    
    # Iterate over the cellIDs, get the imputed feature matrix, and update the featmat_imp
    for cellID in cellIDs:
        
        # Get the filename of the imputed feature matrix for the cell
        filename = os.path.join(tempdir, f'{cellID}.pkl')
        assert os.path.isfile(filename), f"Parallel result file for {cellID} not found."
        
        # Load the imputed feature matrix of the cell
        with open(filename, 'rb') as f:
            featmat_cell_imp = pickle.load(f)
        
        # Update the result
        cellnum = scf.get_cellnum(cellID)
        featmat_imp[cellnum] = featmat_cell_imp
    
    return featmat_imp
