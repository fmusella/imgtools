# Class for extracting structural features from the CTE data

import sys
import numpy as np
import h5py
from alabtools.utils import Index
from ...cte import ChromatinTracingExperiment
from ...cte import cte_io
from ...cte import cte_parallel
from ...scf import SingleCellFeature
from ._features import _spotcount
from ._features import _lamina
from ._features import _chromsurf
from ._features import _immunof
from ._features import _immunof_tsa


# Available features that can be extracted
AVAILABLE_FEATURES = [
    'spotcount',
    'lamina',
    'chromsurf',
]

def feature_extractor(cte: ChromatinTracingExperiment, scf: SingleCellFeature, config: dict) -> None:
    """ Extract structural features from the CTE data and add them to the SingleCellFeature object.
    
    For each feature in the config, the feature is extracted and added to the SingleCellFeature object.
    
    The configuration dictionary provides the parameters for each feature.

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
    
    sys.stdout.write("\n\nExtracting structural features...\n\n")
    
    # Get the list of features to extract
    feature_list = list(config.keys())
    keys_to_remove = ['parallel']
    for key in keys_to_remove:
        if key in feature_list:
            feature_list.remove(key)
    
    sys.stdout.write(f"Features to extract: {', '.join(feature_list)}\n\n")
    
    # Run each feature
    for feature in feature_list:
        
        sys.stdout.write(f"Extracting feature {feature}...\n")
        
        if not feature in AVAILABLE_FEATURES and not 'ImF_file' in config[feature]:
            raise ValueError("Feature {} is not available.".format(feature))
        
        if feature in scf:
            sys.stdout.write(f"Feature {feature} is already in the SingleCellFeature object. Moving on.\n\n")
            continue
        
        # Add the 'parallel' key to the config of the feature
        config[feature]['parallel'] = config['parallel']

        # Run the feature and get the single-cell feature matrix
        feature_matrix = run_feature(feature, cte, config[feature])
        
        # Add the matrices to the SingleCellFeature object
        if 'ImF_file' in config[feature] and 'tsa_alpha' in config[feature]:
            scf.add_matrix(feature_matrix, feature + '_tsa')
        else:
            scf.add_matrix(feature_matrix, feature)
        
        sys.stdout.write(f"Feature {feature} extracted.\n\n")


def run_feature(feature: str, cte: ChromatinTracingExperiment, config: dict) -> np.ndarray:
    """ Calculate the feature matrix in parallel.
    
    It uses the control_func function of the cte_parallel module to parallelize the feature extraction.

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
            (np.ndarray): single-cell feature array of shape (ndomain, max_ntrace_per_chrom)
        """
        
        # Read the data from the HDF5 file of the CTE
        with h5py.File(cte_name, 'r') as f:
            cell_data = cte_io.load_cell_data_from_hdf5(cellID, f, format='dict')
            cell_alphashape = cte_io.load_cell_alphashape_from_hdf5(cellID, f)
            attrs = cte_io.load_attrs_from_hdf5(f)
            index = cte_io.load_index_from_hdf5(f)
        
        # Initialize the single-cell feature array to zeros, with shape (ndomain, max_ntrace_per_chrom)
        feat_arr = np.zeros((len(index), attrs['max_ntrace_per_chrom']), dtype=np.float32)
        
        # Perform the feature calculation for the feature
        feat_arr = feature_calculation(cellID, feature, feat_arr, cell_data, cell_alphashape, index, config)
        
        del cell_data, cell_alphashape, attrs, index
        
        return feat_arr
        
        
    def rfunc_init(_1, cte_name: str, _2) -> np.ndarray:
        """ Initialize the single-cell feature matrix for the reduction function.

        Args:
            cte_name (str)
            _*: not used, just to match the signature of the function

        Returns:
            (np.ndarray): initialized 0-valued global feature matrix of shape (n_cells, n_domains, max_ntrace_per_chrom)
        """
        
        # Read attributes and index from the HDF5 file
        with h5py.File(cte_name, 'r') as f:
            attrs = cte_io.load_attrs_from_hdf5(f)
            index = cte_io.load_index_from_hdf5(f)
        
        # Initialize the global feature matrix of shape (n_cells, n_domains, max_ntrace_per_chrom)
        feat_mat = np.zeros((attrs['ncell'], len(index), attrs['max_ntrace_per_chrom']), dtype=np.float32)

        return feat_mat
    
    def rfunc_update(cellID: str, feat_mat: np.ndarray, feat_arr: np.ndarray, cte_name: str, _) -> np.ndarray:
        """ Update the global feature matrix with the data of a single cell for the reduce function.

        Args:
            cellID (str)
            feat_mat (np.ndarray): global feature matrix with shape (n_cells, n_domains, max_ntrace_per_chrom)
            feat_arr (np.ndarray): single-cell feature matrix with shape (ndomain, max_ntrace_per_chrom)
            cte_name (str)
            _: not used, just to match the signature of the function

        Returns:
            (np.ndarray): updated global feature matrix of shape (n_cells, n_domains, max_ntrace_per_chrom)
        """
        
        # Read the cell labels from the HDF5 file
        with h5py.File(cte_name, 'r') as f:
            cell_labels = cte_io.load_cell_labels_from_hdf5(f)
        
        # Get the index - along cell_labels - of cellID
        cellnum = np.where(cell_labels == cellID)[0][0]
        
        # Add the data of the cell to the global feature matrix
        feat_mat[cellnum, :, :] = feat_arr
        
        return feat_mat
    
    required_keys = get_required_keys(feature, config)
    
    # Calculate the feature matrix in parallel
    feat_mat = cte_parallel.control_func(
        cte,
        config,
        required_keys,
        nfunc,
        rfunc_init,
        rfunc_update
    )
    
    return feat_mat


def feature_calculation(
    cellID: str,
    feature: str,
    feat_arr: np.ndarray,
    cell_data: dict,
    cell_alphashape: dict,
    index: Index,
    config: dict
    ) -> np.ndarray:
    """ Calculate the feature for a single cell.
    Runs a different function for each feature, using the respective module.

    Args:
        cellID (str)
        feature (str)
        feat_arr (np.ndarray): 0-valued single-cell feature array of shape (ndomain, max_ntrace_per_chrom) to be filled
        cell_data (dict): cell data in dictionary format
        index (Index)
        config (dict): configuration for the feature

    Returns:
        (np.ndarray): updated single-cell feature array of shape (ndomain, max_ntrace_per_chrom)
    """
    
    if 'ImF_file' in config:
        if 'tsa_alpha' in config:
            return _immunof_tsa.run(cellID, feature, feat_arr, cell_data, index, config)
        else:
            return _immunof.run(cellID, feature, feat_arr, cell_data, index, config)
    if feature == 'spotcount':
        return _spotcount.run(feat_arr, cell_data, index)
    if feature == 'lamina':
        return _lamina.run(feat_arr, cell_data, cell_alphashape, index, config)
    if feature == 'chromsurf':
        return _chromsurf.run(feat_arr, cell_data, index, config)

def get_required_keys(feature: str, config: dict) -> dict:
    """ Get the required keys for the feature.
    
    Returns:
        (dict): required keys for the feature
    """
    if 'ImF_file' in config:
        if 'tsa_alpha' in config:
            return _immunof_tsa.required_keys
        else:
            return _immunof.required_keys
    if feature == 'spotcount':
        return {}
    if feature == 'lamina':
        return _lamina.required_keys
    if feature == 'chromsurf':
        return _chromsurf.required_keys
