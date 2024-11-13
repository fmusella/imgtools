import numpy as np
from sklearn.mixture import GaussianMixture
from ...scf import SingleCellFeature
from .synchronizer import CellCycleSynchronizer

class CellCycleGaussianMixture(CellCycleSynchronizer):
    """ Class to perform the Gaussian Mixture model to classify for the cell cycle synchronization.
        
    It first identifies the top and bottom percentiles of the RT distribution (percentile from the config).
    The regions with the top RT are the very early replicating domains, while the bottom RT are the late replicating domains.
    
    We assume that the early replicating regions are instantly replicated in all S cells,
    while the late replicating regions are replicated almost exclusively in G2 cells.
    
    Thus, we fit two Gaussian Mixture models using the rowsum of the feature matrix for the top and bottom RT regions.
    - The first model is fitted to the top RT regions to separate G1 from S/G2 cells.
    - The second model is fitted to the bottom RT regions to separate G2 from G1/S cells.
    
    Inherits from CellCycleSynchronizer.
    
    --- Attributes (inherit from CellCycleSynchronizer) ---
    scf (SingleCellFeature): SingleCellFeature object.
    index (Index): Index of the SingleCellFeature.
    config (dict): Configuration dictionary for the synchronization method.
    rt_file (str): Path to the replication timing file.
    usechroms (list): List of chromosome strings to be used in the synchronization.
    smooth_k (int or None): Smoothing parameter k.
    feature (str): Name of the feature to be used in the synchronization.
    smooth_chromstr (np.array or None): Chromosome strings for the smoothing function.
    matrix (np.array): Matrix of the feature for the chromosomes specified in usechroms.
    rowmean (np.array): Row-wise mean of the matrix.
    rt_index (Index): Index of the RT data.
    rt (np.array): RT signal for the chromosomes specified in usechroms.
    states_ (np.array): Array of strings with the states of the cells, e.g. ['G', 'S', 'G', ...], to be updated in the run method.
    
    --- Attributes (specific to CellCycleAnnealer) ---
    percentile (int): Percentile to be used to identify the top and bottom RT regions.
    
    --- Methods (for users) ---
    run: Run the Gaussian Mixture model method.

    Args:
        CellCycleSynchronizer (_type_): _description_
    """
    
    def __init__(self, scf: SingleCellFeature, config: dict, initial_states: np.array = None) -> None:
        """ Initialize the CellCycleGaussianMixture class.
        Inherits from CellCycleSynchronizer.

        Args:
            scf (SingleCellFeature)
            config (dict): configuration dictionary for the greedy algorithm.
            initial_states (np.array, optional): Initial states of the cells, e.g. ['G', 'S', 'G', ...].
                            If None, the states are initialized randomly.
        """
        
        super().__init__(scf, config, initial_states)
        
        # Check that the configuration dictionary is correct
        self.check_config()
        
        # Extract the parameters from the configuration dictionary
        self.percentile = self.config['percentile']
    
    
    def check_config(self):
        """ Check the configuration dictionary for the Gaussian Mixture model.
        """
        
        required_keys = {
            'percentile': int,
        }
        for key, dtype in required_keys.items():
            if not key in self.config:
                raise ValueError(f'Key "{key}" not found in config.')
            if not isinstance(self.config[key], dtype):
                raise ValueError(f'Key "{key}" should be of type {dtype}. Got {type(self.config[key])} instead.')
    
    
    def run(self) -> None:
        """ Run the Gaussian Mixture model to classify the cells into G1, S, and G2 phases.
        
        It first identifies the top and bottom percentiles of the RT distribution (percentile from the config).
        The regions with the top RT are the very early replicating domains, while the bottom RT are the late replicating domains.
        
        We assume that the early replicating regions are instantly replicated in all S cells,
        while the late replicating regions are replicated almost exclusively in G2 cells.
        
        Thus, we fit two Gaussian Mixture models using the rowsum of the feature matrix for the top and bottom RT regions.
        - The first model is fitted to the top RT regions to separate G1 from S/G2 cells.
        - The second model is fitted to the bottom RT regions to separate G2 from G1/S cells.
        """
        
        # Select top and bottom percentiles from the RT
        top_idx = np.where(self.rt > np.nanpercentile(self.rt, 100 - self.percentile))[0]
        bottom_idx = np.where(self.rt < np.nanpercentile(self.rt, self.percentile))[0]
        
        # Calculate the rowsum of the feature matrix for the top and bottom percentiles
        rowsum_top = np.nansum(self.matrix[:, top_idx], axis=1)  # shape: (ncell,)
        rowsum_bottom = np.nansum(self.matrix[:, bottom_idx], axis=1)  # shape: (ncell,)

        # Fit a Gaussian Mixture model to the top signal to separate G1 from S/G2
        X = np.array([rowsum_top]).T
        gm_top = GaussianMixture(n_components=2).fit(X)
        y = gm_top.predict(X)
        # y is a binary array of 0s and 1s. We assign 'G1' to the cluster with the lowest mean
        states_top = np.full(len(y), 'S/G2', dtype='U20')
        mean0, mean1 = gm_top.means_
        if mean0 < mean1:
            states_top[y == 0] = 'G1'
        else:
            states_top[y == 1] = 'G1'

        # Do the same for the bottom signal to separate G1/S from G2
        X = np.array([rowsum_bottom]).T
        gm_bottom = GaussianMixture(n_components=2).fit(X)
        y = gm_bottom.predict(X)
        # Now we assign 'G2' to the cluster with the largest mean
        states_bottom = np.full(len(y), 'G1/S', dtype='U20')
        mean0, mean1 = gm_bottom.means_
        if mean0 > mean1:
            states_bottom[y == 0] = 'G2'
        else:
            states_bottom[y == 1] = 'G2'
        
        # Make sure that the lengths of the states are consistent
        assert len(states_top) == len(states_bottom) == self.matrix.shape[0], 'Shape mismatch between states and matrix.'
        
        # Raise an error if the states are not consistent
        if np.any(np.logical_and(states_top == 'G1', states_bottom == 'G2')):
            n_inconsistent = np.sum(np.logical_and(states_top == 'G1', states_bottom == 'G2'))
            print(f"Warning: {n_inconsistent} cells are classified as both G1 and G2.")
        
        # Combine the states
        states = np.full(len(states_top), 'S', dtype='U20')
        states[states_top == 'G1'] = 'G1'
        states[states_bottom == 'G2'] = 'G2'
        states[np.logical_and(states_top == 'G1', states_bottom == 'G2')] = 'NA'
        
        # Save the results to the object
        self.states_ = states
        self.gm_top_ = gm_top
        self.gm_bottom_ = gm_bottom
