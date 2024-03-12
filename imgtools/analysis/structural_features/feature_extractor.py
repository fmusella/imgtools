import sys
import numpy as np
import h5py
from functools import partial
from alabtools.utils import Index
from ... import utils
from ...cte import ChromatinTracingExperiment
from ...cte import cte_io
from ...cte import cte_parallel
from ...scf import SingleCellFeature
from ._features import _spotcount
from ._features import _envsurf
from ._features import _chromsurf
from ._features import _immunof
from ._features import _immunof_tsa


# Available features that can be extracted
AVAILABLE_FEATURES = [
    'spotcount',
    'envsurf',
    'chromsurf',
    'immunof',
]


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
                'envsurf': {...},
                'chromsurf': {...},
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
        # Expand the config to include the 'tsa' and 'contact' features and the 'parallel' key
        self.expand_config()
        # Check the requirements for the feature extraction
        self.check_requirements()
    
    
    # PREPARATION METHODS
    
    def get_feature_list(self) -> list:
        """ Get the list of features to extract from the config.
        
        If the feature has a 'tsa_alpha' key, it adds a new feature to the list
        (right after the original feature) with the suffix '_tsa'.
        
        If the feature has a 'contact_threshold' key, it adds a new feature to the list
        (right after the original feature and the '_tsa' feature, if it exists) with the suffix '_contact'.
        
        Returns:
            (list): list of features to extract
        """
        # Initialize the list of features
        feature_list = []
        for key in self.config['features']:
            feature_list.append(key)
            if 'tsa_alpha' in self.config['features'][key]:
                feature_list.append(key + '_tsa')
            if 'contact_threshold' in self.config['features'][key]:
                feature_list.append(key + '_contact')
        return feature_list
    
    def config_to_abspath(self) -> None:
        """ Convert the relative paths in the config to absolute paths.
        """
        utils.convert_to_abs_path(self.config)
    
    def expand_config(self) -> None:
        """ Expand the config dictionary:
            - Add the 'tsa' and 'contact' features, copying the same configuration as the original feature.
            - Add the 'parallel' key to the config of each feature.
        """
        
        # Expand the config to include the 'tsa' and 'contact' features
        # Create a new key for these feature with the same configuration as the original feature
        for feature in self.feature_list:
            if feature[-4:] == '_tsa':
                self.config['features'][feature] = self.config['features'][feature[:-4]]
            if feature[-8:] == '_contact':
                self.config['features'][feature] = self.config['features'][feature[:-8]]
        
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
            
            # If the feature is not available, raise an error
            # To be available, the feature must either be in AVAILABLE_FEATURES,
            # or have an 'ImF_file' key in the config (in which case an ImmunoFluorescence data is required)
            if not feature in AVAILABLE_FEATURES and not 'ImF_file' in self.config['features'][feature]:
                raise ValueError(f"Feature {feature} is not available.")
            
            # Run the feature and get the single-cell feature matrix
            feature_matrix = self.run_feature(feature)
            
            # Add the matrix to the SCF object
            self.scf.add_matrix(feature_matrix, feature)
            
            sys.stdout.write(f"Feature {feature} extracted.\n\n")
            
            del feature_matrix
        
        sys.stdout.write("All features extracted.\n\n")
    
    def run_feature(self, feature: str) -> np.ndarray:
        """ Run the feature extraction for a single feature.
        
        Uses the general parallelization scheme of the CTE object (see cte_parallel).

        Args:
            feature (str): feature to extract.

        Returns:
            np.ndarray: single-cell feature matrix of shape (n_cells, n_domains, max_ntrace_per_chrom)
        """
        
        required_keys = get_required_keys(feature, self.config['features'][feature])
    
        # Calculate the feature matrix in parallel
        feat_mat = cte_parallel.control_func(
            self.cte,
            self.config['features'][feature],
            required_keys,
            partial(self.nfunc, feature=feature),
            self.rfunc_init,
            self.rfunc_update
        )

        return feat_mat
    
    @staticmethod
    def nfunc(cellID: str, cte_name: str, config: dict, feature: str) -> np.ndarray:
        """ Node function for the parallelization of the feature extraction.

        Args:
            cellID (str)
            cte_name (str)
            config (dict): configuration for the feature
            feature (str): feature to extract

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
        if feature[-4:] == '_tsa':
            return _immunof_tsa.run(cellID, feature, feat_arr, cell_data, index, config)
        elif feature[-8:] == '_contact':
            raise NotImplementedError("Contact feature is not available for immunofluorescence.")
        else:
            return _immunof.run(cellID, feature, feat_arr, cell_data, index, config)
    if feature == 'spotcount':
        return _spotcount.run(feat_arr, cell_data, index)
    if feature == 'envsurf':
        return _envsurf.run(feat_arr, cell_data, cell_alphashape, index)
    if feature == 'chromsurf':
        return _chromsurf.run(feat_arr, cell_data, index, config)

def get_required_keys(feature: str, config: dict) -> dict:
    """ Get the required keys for the feature.
    
    Returns:
        (dict): required keys for the feature
    """
    if 'ImF_file' in config:
        if feature[-4:] == '_tsa':
            return _immunof_tsa.required_keys
        elif feature[-8:] == '_contact':
            return {}
        else:
            return _immunof.required_keys
    if feature == 'spotcount':
        return {}
    if feature == 'envsurf':
        return _envsurf.required_keys
    if feature == 'chromsurf':
        return _chromsurf.required_keys
