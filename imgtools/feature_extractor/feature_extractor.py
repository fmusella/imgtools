import sys
import numpy as np
import h5py
from functools import partial
from .. import utils
from ..cte import ChromatinTracingExperiment
from ..cte import cte_io
from ..cte import cte_parallel
from ..scf import SingleCellFeature
from . import _features


class FeatureExtractor:
    """ Class to extract structural features from a ChromatinTracingExperiment.
    
    The data is stored in a SingleCellFeature object.
    
    The only method of interest is run, which extracts the features from the CTE.
    
    Usage:
        fe = FeatureExtractor(cte, scf, config)
        fe.run()
    
    The config file must be a dict with the following structure:
        {
            'features': {
                'envsurf': {
                    'module': 'envsurf',
                    ...  
                },
                'chromsurf': {
                    'module': 'chromsurf',
                    ...  
                },
                'rg_500nm': {
                    'module': 'rg',
                    ...
                },
                ...etc...
            },
            'parallel: {'controller': 'ipyparallel'}
        }
    
    --- Attributes ---
    cte (ChromatinTracingExperiment): ChromatinTracingExperiment object
    scf (SingleCellFeature): SingleCellFeature object
    config (dict): configuration for the feature extraction
    feature_list (list): list of features to extract
    
    """
    
    def __init__(self, cte: ChromatinTracingExperiment, scf: SingleCellFeature, config: dict) -> None:
        """ Initialize the FeatureExtractor object.
        
        Args:
            cte (ChromatinTracingExperiment)
            scf (SingleCellFeature)
            config (dict): configuration for the feature extraction
        """
        
        self.cte = cte
        self.scf = scf
        self.config = config
        # Get the list of features to extract
        self.feature_list = self.get_feature_list()
        # Convert the relative paths in the config to absolute paths
        self.config_to_abspath()
        # Expand the config to include the 'parallel' key to each feature
        self.expand_config()
        # Check the requirements for the feature extraction
        self.check_requirements()
    
    
    # PREPARATION METHODS
    
    def get_feature_list(self) -> list:
        """ Get the list of features to extract from the config.
        
        Returns:
            (list): list of features to extract
        """
        # Initialize the list of features
        feature_list = []
        for key in self.config['features']:
            feature_list.append(key)
        return feature_list
    
    def config_to_abspath(self) -> None:
        """ Convert the relative paths in the config to absolute paths.
        """
        utils.convert_to_abs_path(self.config)
    
    def expand_config(self) -> None:
        """ Expand the config dictionary, adding the 'parallel' key to the config of each feature.
        """       
        # Add the 'parallel' key to the config of each feature
        for feature in self.feature_list:
            self.config['features'][feature]['parallel'] = self.config['parallel']
    
    def check_requirements(self) -> None:
        """ Check the requirements for the feature extraction.
        
        The requirements are:
            - The index of the CTE and SCF must be the same.
            - The config file must be a dict.
            - The config contains a key "features", whose value is a dict,
                and a key "parallel", whose value is a dict too.
            - Each feature in config['features'] must have a 'module' key, whose value must be in MODULES.
        """
        if not self.cte.index == self.scf.index:
            raise ValueError("The index of the CTE and SCF must be the same.")
        if not isinstance(self.config, dict):
            raise ValueError("Config must be a dict.")
        if not 'features' in self.config:
            raise ValueError("Config must contain a 'features' key.")
        if not isinstance(self.config['features'], dict):
            raise ValueError("The value of the 'features' key must be a dict.")
        if not 'parallel' in self.config:
            raise ValueError("Config must contain a 'parallel' key.")
        if not isinstance(self.config['parallel'], dict):
            raise ValueError("The value of the 'parallel' key must be a dict.")
        for feature in self.feature_list:
            if not 'module' in self.config['features'][feature]:
                raise ValueError(f"Feature {feature} must have a 'module' key in the config.")
            if not self.config['features'][feature]['module'] in _features.MODULES:
                raise ValueError(f"Module {self.config['features'][feature]['module']} is not available.")
    
    
    # RUN METHOD (MAIN METHOD) AND HELPER METHODS
    
    def run(self) -> None:
        """ Run the feature extraction for each feature in the config.
        """
        
        sys.stdout.write("\n\nExtracting structural features...\n\n")
        
        sys.stdout.write("Feature list:\n")
        for feature in self.feature_list:
            sys.stdout.write(f"- {feature}\n")
        sys.stdout.write("\n")
        
        for feature in self.feature_list:
            
            sys.stdout.write(f"Extracting feature {feature}...\n")
            
            # If the feature is already in the SCF object, move on
            if feature in self.scf:
                sys.stdout.write(f"Feature {feature} is already in the SCF object. Moving on.\n\n")
                continue
            
            # Get the module to use for the feature extraction
            module = self.config['features'][feature]['module']
            
            # Run the feature and get the single-cell feature matrix
            feature_matrix = self.run_feature(feature, module)
            
            # Add the matrix to the SCF object
            self.scf.add_feature(feature_matrix, feature, doc=_features.MODULES[module].docstring)
            
            sys.stdout.write(f"Feature {feature} extracted.\n\n")
            
            del feature_matrix
        
        sys.stdout.write("All features extracted.\n\n")
    
    def run_feature(self, feature: str, module: str) -> np.ndarray:
        """ Run the feature extraction for a single feature.
        
        Uses the general parallelization scheme of the CTE object (see cte_parallel).

        Args:
            feature (str): feature to extract.
            module (str): module to use for the feature extraction.

        Returns:
            np.ndarray: single-cell feature matrix of shape (n_cells, n_domains, max_ntrace_per_chrom)
        """
        
        required_keys = _features.MODULES[module].required_keys
    
        # Calculate the feature matrix in parallel
        feat_mat = cte_parallel.control_func(
            self.cte,
            self.config['features'][feature],
            required_keys,
            partial(self.nfunc, feature=feature, module=module),
            self.rfunc_init,
            self.rfunc_update
        )

        return feat_mat
    
    @staticmethod
    def nfunc(cellID: str, cte_name: str, config: dict, feature: str, module: str) -> np.ndarray:
        """ Node function for the parallelization of the feature extraction.

        Args:
            cellID (str)
            cte_name (str)
            config (dict): configuration for the feature
            feature (str): feature to extract
            module (str): module to use for the feature extraction

        Returns:
            (np.ndarray): single-cell feature array of shape (ndomain, max_ntrace_per_chrom)
        """
        
        # Read the CTE file
        cte = ChromatinTracingExperiment(cte_name, 'r')
        
        # Initialize the single-cell feature array to NaN values
        feat_arr = np.full((len(cte.index), cte.attrs['max_ntrace_per_chrom']), np.nan, dtype=np.float32)
        
        # Perform the feature calculation for the feature
        feat_arr = _features.MODULES[module].run(cellID, cte, config, feat_arr, feature)

        cte.close()
        
        return feat_arr
    
    @staticmethod
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
            cell_labels = cte_io.load_cell_labels_from_hdf5(f)
        
        # Initialize the global feature matrix of shape (n_cells, n_domains, max_ntrace_per_chrom)
        feat_mat = np.zeros((len(cell_labels), len(index), attrs['max_ntrace_per_chrom']), dtype=np.float32)

        return feat_mat
    
    @staticmethod
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
