# Class for extracting structural features from the CTE data

import numpy as np
import h5py
from alabtools.utils import Index
from ...cte import ChromatinTracingExperiment
from ...cte import cte_io
from ...cte import parallelization as cte_parallel
from ...scf import SingleCellFeature
from ._features import _chromdepth


# Available features that can be extracted
AVAILABLE_FEATURES = [
    'chromdepth',
]

def feature_extractor(cte: ChromatinTracingExperiment, scf: SingleCellFeature, config: dict) -> None:
    """ Extract structural features from the CTE data and add them to the SingleCellFeature object.

    Args:
        cte (ChromatinTracingExperiment)
        scf (SingleCellFeature)
        config (dict): configuration for the feature extraction
    """
    
    # Check that the index of CTE and SCF are the same
    if not cte.index == scf.index:
        raise ValueError("The index of the CTE and SCF must be the same.")
    
    # Check if the config file is a dict
    if not isinstance(config, dict):
        raise ValueError("Config must be a dict.")
    
    # Get the list of features to extract
    feature_list = list(config.keys())
    keys_to_remove = ['parallel']
    for key in keys_to_remove:
        if key in feature_list:
            feature_list.remove(key)
    
    # Run each feature
    for feature in feature_list:
        
        if not feature in AVAILABLE_FEATURES:
            raise ValueError("Feature {} is not available.".format(feature))
        
        if feature in scf:
            raise ValueError("Feature {} is already in the SingleCellFeature object.".format(feature))

        # Run the feature and get the single-cell feature matrix
        matrix = run_feature(feature, cte, config[feature])
        
        # Add the matrix to the SingleCellFeature object
        scf.add_matrix(feature, matrix)


def run_feature(feature: str, cte: ChromatinTracingExperiment, config: dict) -> np.ndarray:
    """ Calculate the feature matrix in parallel.

    Args:
        feature (str)
        cte (ChromatinTracingExperiment)
        config (dict): configuration for the feature to extract

    Returns:
        np.ndarray: single-cell feature matrix of shape (n_cells, n_domains, max_ntrace_per_chrom)
    """
        
    def nfunc(cellID: str, cte_name: str, config: dict) -> np.ndarray:
        """ Node function for the parallelization of the feature extraction.

        Args:
            cellID (str)
            cte_name (str)
            config (dict): configuration for the feature
            
        Returns:
            cell_arr (np.ndarray): single-cell feature array of shape (ndomain, max_ntrace_per_chrom)
        """
        
        # Read the data from the HDF5 file of the CTE
        with h5py.File(cte_name, 'r') as f:
            cell_data = cte_io.load_cell_data_from_hdf5(cellID, f, format='dict')
            cell_alphashape = cte_io.load_cell_alphashape_from_hdf5(cellID, f)
            attrs = cte_io.load_attrs_from_hdf5(f)
            index = cte_io.load_index_from_hdf5(f)
        
        # Initialize the single-cell feature array to zeros, with shape (ndomain, max_ntrace_per_chrom)
        cell_arr = np.zeros((len(index), attrs['max_ntrace_per_chrom']), dtype=np.float32)
        
        # Perform the feature calculation
        cell_arr = feature_calculation(feature, cell_arr, cell_data, cell_alphashape, index, config)
        
        del cell_data, cell_alphashape, attrs, index
        
        return cell_arr
        
        
    def rfunc_init(_1, cte_name: str, _2) -> np.ndarray:
        """ Initialize the single-cell feature matrix for the reduction function.

        Args:
            cte_name (str)
            _*: not used, just to match the signature of the function

        Returns:
            mat (np.ndarray): global feature matrix, with shape (n_cells, n_domains, max_ntrace_per_chrom)
        """
        
        # Read attributes and index from the HDF5 file
        with h5py.File(cte_name, 'r') as f:
            attrs = cte_io.load_attrs_from_hdf5(f)
            index = cte_io.load_index_from_hdf5(f)
        
        # Initialize the global count matrix of shape (n_cells, n_domains, max_ntrace_per_chrom)
        mat = np.zeros((attrs['ncell'], len(index), attrs['max_ntrace_per_chrom']), dtype=np.float32)
        
        return mat
    
    def rfunc_update(cellID: str, mat: np.ndarray, cell_arr: np.ndarray, cte_name: str, _) -> np.ndarray:
        """ Update the global featyre matrix with the data of a single cell for the reduce function.

        Args:
            cellID (str)
            mat (np.ndarray): global feature matrix
            cell_arr (np.ndarray): single cell feature array
            cte_name (str)
            _: not used, just to match the signature of the function

        Returns:
            mat (np.ndarray): updated global feature matrix
        """
        
        # Read the cell labels from the HDF5 file
        with h5py.File(cte_name, 'r') as f:
            cell_labels = cte_io.load_cell_labels_from_hdf5(f)
        
        # Get the index - along cell_labels - of cellID
        cellnum = np.where(cell_labels == cellID)[0][0]
        
        # Add the data of the cell to the global feature matrix
        mat[cellnum] = cell_arr
        
        return mat
    
    required_keys = get_required_keys(feature)
    
    # Calculate the feature matrix in parallel
    feature_mat = cte_parallel.control_func(
        cte,
        config,
        required_keys,
        nfunc,
        rfunc_init,
        rfunc_update
    )
    
    return feature_mat


def feature_calculation(
    feature: str,
    cell_arr: np.ndarray,
    cell_data: dict,
    cell_alphashape: dict,
    index: Index,
    config: dict
    ) -> np.ndarray:
    """ Calculate the feature for a single cell.
    Runs a different function for each feature, using the respective module.

    Args:
        feature (str)
        cell_arr (np.ndarray): empty single-cell feature array of shape (ndomain, max_ntrace_per_chrom)
        cell_data (dict): cell data in dictionary format
        index (Index)
        config (dict): configuration for the feature

    Returns:
        (np.ndarray): updated single-cell feature array of shape (ndomain, max_ntrace_per_chrom)
    """
    
    if feature == 'chromdepth':
        return _chromdepth.run(cell_arr, cell_data, index, config)

def get_required_keys(feature: str) -> dict:
    """ Get the required keys for the feature.
    
    Returns:
        (dict): required keys for the feature
    """
    if feature == 'chromdepth':
        return _chromdepth.required_keys
