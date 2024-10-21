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
    
    # Open the CTE file
    cte = ChromatinTracingExperiment(cte, 'r')
    
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
        tempdir = tempdir
    )
    data_imp = controller.map_reduce(
        parallel_task,
        reduce_task,
        args = list(triad_labels)
    )
    
    # Create a CTE object for the imputed data
    cte_imp_h5name = cte.h5_name.replace('.h5', '_imputed.h5')
    cte_imp = ChromatinTracingExperiment(cte_imp_h5name, 'w')
    cte_imp.set_data_attrs_index(data=data_imp, index=cte.index)
    
    # Add the triad_labels to the imputed CTE
    cte_imp.set_triad_labels(triad_labels)
    
    # If the original CTE has a cell_states group, copy it to the imputed CTE
    if 'cell_states' in cte:
        cte_imp.set_cell_states(cte.cell_states)
    
    # If the original CTE has an alphashape group, copy it to the imputed CTE
    if 'alphashapes' in cte:
        cte_imp.set_alphashapes(cte.get_alphashapes())
    
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

def reduce_cte_imputation(triadIDs: list, tempdir: str) -> dict:
    """ Reduce function for the imputation of the CTE data.
    
    Creates a dictionary with the imputed data of the whole population.
    Iterates over the single-trace results of the parallel functions
    and updates population-wide data.

    Args:
        triadIDs (list): list of triadIDs (cellID, chrom, traceID) from the parallel functions.
        tempdir (str)

    Returns:
        dict: imputed data of the whole population.
    """
    
    # Make sure that the returns of the parallel functions are correct
    assert isinstance(triadIDs, list), "Reduce function: triadIDs should be a list. Got type: {}".format(type(triadIDs))
    assert len(triadIDs) > 0, "Reduce: triadIDs list should not be empty."
    
    # Initialize the imputed data of the whole population
    data_imp = {}
    
    # Iterate over the triadIDs, get the imputed trace data, and update the data_imp
    for (cellID, chrom, traceID) in triadIDs:
        
        # Get the filename for the temporary chromosomal volumes of the cell
        filename = os.path.join(tempdir, f'{cellID}_{chrom}_{traceID}.pkl')
        assert os.path.isfile(filename), f"Parallel result file for {cellID}, {chrom}, {traceID} not found."
        
        # Load the imputed trace data
        with open(filename, 'rb') as f:
            trace_data_imp = pickle.load(f)
        
        # Update the result
        if cellID not in data_imp:
            data_imp[cellID] = {}
        if chrom not in data_imp[cellID]:
            data_imp[cellID][chrom] = {}
        data_imp[cellID][chrom][traceID] = trace_data_imp
    
    return data_imp

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
