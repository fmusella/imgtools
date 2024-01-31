# Class for extracting structural features from the CTE data

import numpy as np
from alabtools.utils import Index
from ..cte.parallelization import control_func
from ..cte import ChromatinTracingExperiment
from ..scmatrix import SingleCellMatrix
from . import _chromdepth


# Available features that can be extracted
AVAILABLE_FEATURES = [
    'chromdepth',
]

def feature_extractor(cte: ChromatinTracingExperiment, scm: SingleCellMatrix, config: dict) -> None:
    """
    Extract structural features from the CTE data.
    """
    # Read features from config file
    features = config['features']
    assert isinstance(features, dict), "Features must be a dict."
    
    # Run each feature
    for feature in features:
        
        if not feature in AVAILABLE_FEATURES:
            raise ValueError("Feature {} is not available.".format(feature))

        matrix = run_feature(feature, cte, config)
        
        scm.add_matrix(feature, matrix)


def run_feature(feature: str, cte: ChromatinTracingExperiment, config: dict) -> np.ndarray:
    
    pfunc = feature_parallel(feature, cte.data, cte.attrs, cte.index, config)
        
    def rfunc_init(cellIDs: list, data_attrs: dict, index: Index, config: dict) -> np.ndarray:
        """ Initialization for the reduction function."""
        # Get the three dimensions of the matrix
        ncell = len(cellIDs)
        ndomain = len(index)
        max_ntrace_per_chrom = data_attrs['max_ntrace_per_chrom']
        # Initialize the matrix
        mat = np.zeros((ncell, ndomain, max_ntrace_per_chrom), dtype=np.float32)
        return mat
    
    def rfunc_update(cellID: str, result: np.ndarray, cell_result: np.ndarray, cellIDs, data_attrs, index, config) -> dict:
        """ Update for the reduction function."""
        # Get the cell index
        cell_idx = np.where(np.array(cellIDs) == cellID)[0][0]
        # Update the matrix
        result[cell_idx, :, :] = cell_result
    
    feature_mat = control_func(
        cte.data,
        cte.attrs,
        cte.index,
        config,
        required_keys,
        pfunc,
        rfunc_init,
        rfunc_update
    )
    
    return feature_mat

def feature_parallel(feature: str, cell_data: dict, data_attrs, index, config: dict):
    
    if feature == 'chromdepth':
        _chromdepth.run(cell_data, data_attrs, index, config)

def required_keys(feature: str) -> list:
    """
    Get the required keys for the feature.
    """
    if feature == 'chromdepth':
        return _chromdepth.required_keys
