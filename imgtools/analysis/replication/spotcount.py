import numpy as np
import h5py
from ...cte import ChromatinTracingExperiment
from ...cte import cte_io
from ...cte import cte_parallel
from ...scf import SingleCellFeature


spotcount_required_keys = {'out_name': {'type': str}}

def run_spotcount(cte: ChromatinTracingExperiment, config: dict) -> SingleCellFeature:
    """ Calculate the spot count matrix of a ChromatinTracingExperiment in parallel.

    Args:
        cte (ChromatinTracingExperiment)
        config (dict): configuration dictionary for the spotcount task

    Returns:
        scf (SingleCellFeature): SingleCellFeature object with the spot count matrix
    """
        
    def _nfunc(cellID: str, cte_name: str, _) -> np.ndarray:
        """ Node function for the spotcount task.
        It reads the data of a single cell from the HDF5 file, calculates the spot count and returns it.

        Args:
            cellID (str)
            cte_name (str)
            _: not used, just to match the signature of the function

        Returns:
            count (np.ndarray): count array of the cell, with shape (n_domains, max_ntrace_per_chrom)
        """
        
        # Read attributes, index and data of the cell from the HDF5 file
        with h5py.File(cte_name, 'r') as f:
            attrs = cte_io.load_attrs_from_hdf5(f)
            index = cte_io.load_index_from_hdf5(f)
            cell_data = cte_io.load_cell_data_from_hdf5(cellID, f, format='dict')
        
         # Create a hash table for the index
        index_hash = index.get_index_hashmap()
        
        # Initialize the count array of this cell, with shape (n_domains, max_ntrace_per_chrom)
        count = np.zeros((len(index), attrs['max_ntrace_per_chrom']), dtype = np.int32)
        
        # Fill the count array, looping over all spots of the cell
        for chrom in cell_data:
            
            # Get the traces in the chromosome and hash them
            traceIDs = list(cell_data[chrom].keys())
            traceID_hash = {traceID: i for i, traceID in enumerate(traceIDs)}
            
            for traceID in cell_data[chrom]:
                for spotID in cell_data[chrom][traceID]:
                    
                    spot_data = cell_data[chrom][traceID][spotID]
                    start, end = spot_data['start'], spot_data['end']
                    
                    # Get the position of the spot in the count matrix using the hash tables
                    i_domain = index_hash[(chrom, start, end)]
                    i_trace = traceID_hash[traceID]
                    
                    # Increment the count array
                    count[i_domain, i_trace] += 1
        
        return count
            
    
    def _rfunc_init(_1, cte_name: str, _2) -> np.ndarray:
        """ Initialize the global count matrix for the reduce function.

        Args:
            cte_name (str)
            _*: not used, just to match the signature of the function

        Returns:
            count (np.ndarray): global count matrix, with shape (n_cells, n_domains, max_ntrace_per_chrom)
        """
        
        # Read attributes and index from the HDF5 file
        with h5py.File(cte_name, 'r') as f:
            attrs = cte_io.load_attrs_from_hdf5(f)
            index = cte_io.load_index_from_hdf5(f)
        
        # Initialize the global count matrix of shape (n_cells, n_domains, max_ntrace_per_chrom)
        count = np.zeros((attrs['ncell'], len(index), attrs['max_ntrace_per_chrom']), dtype=np.int32)
        
        return count
    
    def _rfunc_update(cellID: str, count: np.ndarray, cell_count: np.ndarray, cte_name: str, _) -> np.ndarray:
        """ Update the global count matrix with the count of a single cell for the reduce function.

        Args:
            cellID (str)
            count (np.ndarray): global spot count matrix
            cell_count (np.ndarray): single cell spot count matrix
            cte_name (str)
            _: not used, just to match the signature of the function

        Returns:
            (np.ndarray): updated global count matrix
        """
        
        # Read the cell labels from the HDF5 file
        with h5py.File(cte_name, 'r') as f:
            cell_labels = cte_io.load_cell_labels_from_hdf5(f)
        
        # Get the index - along cell_labels - of cellID
        cellnum = np.where(cell_labels == cellID)[0][0]
        
        # Add the count of the cell to the global count matrix
        count[cellnum] = cell_count
        
        return count
        
    
    # Calculate the spot count matrix in parallel
    spot_count = cte_parallel.control_func(
        cte,
        config,
        spotcount_required_keys,
        _nfunc,
        _rfunc_init,
        _rfunc_update
    )
    
    # Create a SingleCellFeature object
    scf = SingleCellFeature(config['out_name'], 'w')
    # Add the index/attributes/cell_labels data
    scf.add_index_attrs_cell_labels(cte.index, cte.attrs, cte.cell_labels)
    # Add the spot count matrix
    scf.add_matrix(spot_count, 'spot_count')
    # Add the volume array if present
    if 'alphashapes' in cte.h5:
        volumes = []
        for cellID in cte.cell_labels:
            volumes.append(cte.get_alphashapes(cellID)['mesh'].volume)
        volumes = np.array(volumes, dtype=np.float32)
        scf.add_volumes(volumes)
    
    return scf
