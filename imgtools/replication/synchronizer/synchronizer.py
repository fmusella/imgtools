import os
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from alabtools.utils import Index, get_index_from_bed, get_index_from_bigwig
from ...scf import SingleCellFeature
from ... import utils


class CellCycleSynchronizer:
    """ Parent class for cell cycle synchronization.
    
    The synchronization consists of assigning cell to either G1, G2 or S phase.
    To achieve this, we use the following general approach:
        1. We get the feature matrix from the SingleCellFeature,
        2. We get the replication timing (RT) signal from a bed or bigwig file,
        3. We employ a search algorithm to look for many possible states of the cells,
           i.e. many possible assignments of cells to G1/G2 or S phase,
        4. We simulate the RT signal using the feature matrix and the states of the cells,
        5. We compare the simulated RT signal with the real RT signal,
        6. We select the states that maximize the correlation.
    
    The search algorithm is specific to each synchronization method, and it is implemented in the run method.
    
    This class is not meant to be used directly, but to be inherited by specific synchronization methods.
    
    --- Input Arguments ---
    scf (SingleCellFeature): SingleCellFeature object.
    config (dict): Configuration dictionary for the synchronization method.
    initial_states (np.array or None): Array of strings with the states of the cells, e.g. ['G', 'S', 'G', ...].
        If None, the states are initialized with 50% of cells in G phase (either G1 or G2) and 50% in S phase.
    
    --- Attributes ---
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
    
    --- Methods (for users) ---
    run: Run the synchronization algorithm. To be overridden by the specific synchronization method.
    """
    
    def __init__(self, scf: SingleCellFeature, config: dict, initial_states: np.array = None) -> None:
        """ Initializes the CellCycleSynchronizer object.
        
        It checks the type of the input scf and config, and then adds the scf, its index,
        and the configuration dictionary to the object.
        
        Here, the method checks that the configuration dictionary has the essential parameters
        required for any synchronization method,
        
        Args:
            scf (SingleCellFeature)
            config (dict): Configuration dictionary for the synchronization method.
            initial_states (np.array or None): Array of strings with the states of the cells, e.g. ['G', 'S', 'G', ...].
                If None, the states are initialized with 50% of cells in G phase (either G1 or G2) and 50% in S phase.
        """
        
        # Check the type of the input scf and config
        if not isinstance(scf, SingleCellFeature):
            raise TypeError("The input scf must be a SingleCellFeature.")
        if not isinstance(config, dict):
            raise TypeError("The input config must be a dictionary.")
        
        # Add the scf, its index, and the configuration dictionary to the object
        self.scf = scf
        self.index = scf.index
        self.config = config
        
        # Check that the configuration dictionary has the essential parameters for any synchronization method
        self.check_basic_config()
        
        # Read the essential parameters from the configuration dictionary
        self.rt_file = self.config['rt_file']
        self.usechroms = self.config['usechroms']
        self.feature = self.config['feature']
        
        # Prepare the matrix for the synchronization algorithm
        self.matrix, self.rowmean = self.prepare_matrix()
        
        # Read the RT file from the configuration
        self.rt_index = self.read_RT()
        # Prepare the RT signal for the synchronization algorithm
        self.rt = self.prepare_RT()
        
        # Try to get the smooth_k paramter from the config, otherwise set it to None
        try:
            smooth_k = self.config['smooth_k']
            # Check that the smoothing parameter k is an integer and positive
            if not isinstance(smooth_k, int):
                raise TypeError("The smoothing parameter k must be an integer.")
            if smooth_k <= 0:
                raise ValueError("The smoothing parameter k must be a positive integer.")
        except KeyError:
            smooth_k = None
        self.smooth_k = smooth_k
        # If smooth_k is not None, get the chromosome strings for the smoothing function
        if self.smooth_k is not None:
            self.smooth_chromstr = self.index.chromstr[np.isin(self.index.chromstr, self.usechroms)]
        else:
            self.smooth_chromstr = None
        
        # Initialize the cell cycle states (G1/G2 treated as G), which will be updated in the run method
        # If initial_states is None, we start with 50% of cells in G phase (either G1 or G2) and 50% in S phase
        if initial_states is None:
            self.states_ = self.initialize_states()
        else:
            self.states_ = initial_states
        
        # Check the states
        self.check_states()
    
    
    # INITIALIZATION METHODS
    
    def check_basic_config(self) -> None:
        """ Checks that the configuration dictionary contains the essential parameters for any synchronization method.
        
        It checks that:
        - scf is a SingleCellFeature
        - config is a dictionary
        - config has the following keys: rt_file, feature, usechroms, smooth_k
        - rt_file exists
        - rt_file is a bed or bigwig file
        - feature is present in the SingleCellFeature
        - usechroms is a subset of the chromosomes present in the SingleCellFeature, or '#' (meaning all autosomes)
        """
        # Check that scf is a SingleCellFeature
        if not isinstance(self.scf, SingleCellFeature):
            raise TypeError("The input scf must be a SingleCellFeature.")
        # Check that config is a dictionary
        if not isinstance(self.config, dict):
            raise TypeError("The input config must be a dictionary.")
        # Check that config has the following keys
        required_keys = ['rt_file', 'feature', 'usechroms']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"The key {key} is missing from the configuration dictionary.")
        # Check that the rt_file exists
        if not os.path.exists(self.config['rt_file']):
            raise FileNotFoundError(f"The file {self.config['rt_file']} does not exist.")
        # Check that the rt_file is a bed or bigwig file
        accepted_endings = ['.bed', '.bedgraph', '.BedGraph', 'bg', '.bw', '.bigwig', '.BigWig']
        if not any(self.config['rt_file'].endswith(ending) for ending in accepted_endings):
            raise ValueError(f"The file {self.config['rt_file']} is not a bed or bigwig file.")
        # Check that the feature is present in the SingleCellFeature
        if self.config['feature'] not in self.scf:
            raise ValueError(f"The feature {self.config['feature']} is not present in the SingleCellFeature.")
        # If usechroms is '#', it means all autosomes: create it
        if self.config['usechroms'] == ['#']:
            # Get the list of autosomes from 'chr1' to 'chr22'
            autosomes = [f'chr{i}' for i in range(1, 23)]
            # Subsample the autosomes on the chromosomes present in the SingleCellFeature
            usechroms = set(autosomes).intersection(self.index.genome.chroms)
            # Convert usechroms to a list and assign it to the configuration dictionary
            self.config['usechroms'] = list(usechroms)
        # Check that usechroms is a subset of the chromosomes present in the Index of the SingleCellFeature
        if not set(self.config['usechroms']).issubset(self.index.genome.chroms):
            raise ValueError(f"The chromosomes {self.config['usechroms']} are not present in the SingleCellFeature.")
    
    def prepare_matrix(self) -> tuple:
        """ Prepare the matrix for the simulated annealing algorithm.
        
        The matrix in SCF has shape (ncell, ndomain, max_ncopy_per_chrom).
        
        We sum off the last axis to get a matrix of shape (ncell, ndomain).
        
        Then, we isolate only the chromosomes of interest specified in usechroms.
        
        Finally, we get the row-wise mean, i.e. the average signal for each cell,
        which is used to normalize the cells in the bias computation, so that the bias
        is not dominated by cells with high signal.

        Returns:
            np.array, shape=(ncell, ndomain_usechr): matrix (3-rd axis summed off)
                            of the feature for the chromosomes specified in usechroms.
            np.array, shape=(ncell,): row-wise mean of the matrix.
        """
        
        # Get the matrix from the SingleCellFeature
        matrix = self.scf.get_feature(self.feature)  # (ncell, ndomain, max_ncopy_per_chrom)
        
        # The matrix in SCF has shape (ncell, ndomain, max_ncopy_per_chrom).
        # Sum off the last axis to get a matrix of shape (ncell, ndomain).
        matrix = np.nansum(matrix, axis=2)  # (ncell, ndomain)
        
        # Isolate only the chromosomes of interest specified in usechroms
        mask = np.isin(self.index.chromstr, self.usechroms)
        matrix = matrix[:, mask]  # (ncell, ndomain_usechr)
        
        # Get the row-wise mean, i.e. the average signal for each cell
        # This is used to normalize the cells in the bias computation,
        # so that the bias is not dominated by cells with high signal.
        rowmean = np.nanmean(matrix, axis=1)  # (ncell,)
        
        return matrix, rowmean
    
    def read_RT(self) -> Index:
        """Read the replication timing data from the RT file.
        
        The name of the file is taken from the configuration dictionary, and can be either a bed or bigwig file.
        
        The function then makes sure that - on the subset of chromosomes to be used in the synchronization -
        the index of the RT data matches the index of the SingleCellFeature.

        Returns:
            Index: Index of the RT data.
        """
        
        # We can read the RT data from either a bed or a bigwig file
        bed_endings = ['.bed', '.bedgraph', '.BedGraph', 'bg']
        bw_endings = ['.bw', '.bigwig', '.BigWig']
        # Read the RT data as bed
        if any(self.rt_file.endswith(ending) for ending in bed_endings):
            rt_index = get_index_from_bed(self.rt_file, genome=self.index.genome)
        # Read the RT data as bigwig
        elif any(self.rt_file.endswith(ending) for ending in bw_endings):
            rt_index = get_index_from_bigwig(self.rt_file, genome=self.index.genome, res=self.index)
        
        # Make sure that rt_index has a 'track0' attribute (the RT data) and it is a numpy array of floats
        if not hasattr(rt_index, 'track0'):
            raise AttributeError("The index of the RT data must have a 'track0' attribute.")
        if not isinstance(rt_index.track0, np.ndarray):
            raise TypeError("The 'track0' attribute of the index of the RT data must be a numpy array.")
        if not np.issubdtype(rt_index.track0.dtype, np.floating):
            raise TypeError("The 'track0' attribute of the index of the RT data must be a numpy array of floats.")
        
        # Make sure the index of the RT data matches the index of the SingleCellFeature
        # for the subset of chromosomes used in the Simulated Annealing
        if not utils.compare_index(self.index, rt_index, self.usechroms):
            raise ValueError(f"The index of the RT data does not match the index of the SingleCellFeature on the chromosomes {self.usechroms}.")
        
        return rt_index
    
    def prepare_RT(self) -> np.array:
        """ Prepare the RT signal for the synchronization algorithm.
        
        We isolate the RT signal for chromosomes specified in usechroms.

        Returns:
            np.array, shape=(ndomain_usechroms): RT signal for the chromosomes specified in usechroms.
        """
        
        # Get the RT signal
        rt = self.rt_index.track0  # (ndomain,)
        
        # Isolate the RT signal for chromosomes specified in usechroms
        rt = rt[np.isin(self.rt_index.chromstr, self.usechroms)]
        
        # Make sure that the shape of the RT signal matches the shape of the matrix
        if len(rt) != self.matrix.shape[1]:
            raise ValueError("The shape of the RT signal does not match the shape of the matrix.")
        
        return rt
    
    def initialize_states(self) -> np.array:
        """ Initialize the states of the cells.
        We start with 50% of cells in G phase (either G1 or G2) and 50% in S phase.

        Returns:
            np.array: Array of strings with the states of the cells, e.g. ['G', 'S', 'G', ...]
        """
        
        # Initialize the states as an empty array of strings
        states = np.full(len(self.scf.cell_labels), '', dtype='U20')
        
        # We start with 50% of cells in G phase (either G1 or G2) and 50% in S phase
        nG = int(len(states) / 2)
        states[:nG] = 'G'
        states[nG:] = 'S'
        
        # Randomly shuffle the states
        np.random.shuffle(states)
        
        return states
    
    def check_states(self) -> None:
        """ Check the states of the cells.
        
        It checks that:
        - states_ is a numpy array
        - states_ is a numpy array of strings U20
        - states_ has the same length as the cell labels in the SingleCellFeature
        - states_ contains only the strings 'G' and 'S'.
        """
        if not isinstance(self.states_, np.ndarray):
            raise TypeError("The states must be a numpy array.")
        if self.states_.dtype != 'U20':
            raise TypeError("The states must be a numpy array of strings.")
        if len(self.states_) != len(self.scf.cell_labels):
            raise ValueError("The states must have the same length as the cell labels.")
        if not np.all(np.isin(self.states_, ['G', 'S'])):
            raise ValueError("The states must contain only the strings 'G' and 'S'.")    
    
    
    # RUN METHOD (MAIN)
    
    def run(self) -> None:
        """ Run the synchronization algorithm.
        
        This method is meant to be overridden by the specific synchronization method.
        """
        raise NotImplementedError("The run method must be overridden by the specific synchronization method.")
    
    
    # METHOD TO SEPARATE 'G' INTO 'G1' AND 'G2'
    def separate_G1G2(self) -> None:
        """ Separate G1 and G2 cells based on the number of spots and the rowmean of the feature matrix.
    
        The function clusters the G cells into 2 groups based on these two variables.
        
        The clusters are identified with the AgglomerativeClustering algorithm.
        Then, the cluster with the smaller volume and rowmean is called 'G1' and the other 'G2'.
        If the clusters are not well separated, the function raises an error.
        
        This method is needed when the Synchronizer doesn't separate G1 and G2 cells.
        In particular, it's needed for the SimulatedAnnealing and Greed methods.
        """
        
        # Make sure that the states exist, contain only 'G' and 'S', and have the same length as the cell labels
        assert hasattr(self, 'states_'), "The states have not been extracted."
        assert np.all(np.isin(self.states_, ['G', 'S'])), "The states must contain only 'G' and 'S'."
        assert len(self.states_) == len(self.scf.cell_labels), "The states must have the same length as the cell labels."
        states = self.states_

        # Get the cell nuclei volumes
        volumes = self.scf.volumes
        
        # Cluster the G cells into 2 groups based on volumes and rowmean
        X = np.column_stack((volumes[states == 'G'], self.rowmean[states == 'G']))
        agg = AgglomerativeClustering(n_clusters=2).fit(X)
        labels = agg.labels_
        
        # Get the centers of the clusters
        center_0 = np.mean(X[labels == 0], axis=0)  # (2,)
        center_1 = np.mean(X[labels == 1], axis=0)  # (2,)
        
        # Call 'G1' the cluster with the smaller volume/rowmean
        states_G = np.full(len(labels), '').astype(states.dtype)
        if center_0[0] < center_1[0] and center_0[1] < center_1[1]:
            states_G[labels == 0] = 'G1'
            states_G[labels == 1] = 'G2'
        elif center_0[0] > center_1[0] and center_0[1] > center_1[1]:
            states_G[labels == 0] = 'G2'
            states_G[labels == 1] = 'G1'
        else:
            raise ValueError('Cannot determine G1/G2: the clusters are not well separated')

        # Create a new states array that is 'S' for S cells and 'G1'/'G2' for G cells
        states_new = np.full(len(states), '').astype(states.dtype)
        idx_G = np.where(states == 'G')[0]
        assert len(idx_G) == len(states_G)
        idx_G1 = idx_G[states_G == 'G1']
        idx_G2 = idx_G[states_G == 'G2']
        states_new[idx_G1] = 'G1'
        states_new[idx_G2] = 'G2'
        states_new[states == 'S'] = 'S'
        
        # Replace the states with the new states
        self.states_ = states_new
        


# Function to simulate the RT signal
# It is defined outside the class because we want to use it in parallel nodes,
# and the class methods cannot be pickled.

def simulate_rt(
    matrix: np.ndarray, rowmean: np.ndarray, states: np.array,
    smooth_k: int = None, smooth_chromstr: np.array = None
)-> np.array:
    """ Simulate the RT signal using the feature matrix and the states of the cells.
    
    The bias in G1/G2 cells is first estimated, and then the matrix in S phase is normalized by it.
    
    Args:
        matrix (np.array): matrix of the feature for the chromosomes specified in usechroms.
        rowmean (np.array): row-wise mean of the matrix.
        states (np.array): array of strings with the states of the cells, e.g. ['G', 'S', 'G', ...]
        smooth_k (int or None): Smoothing parameter k.
        smooth_chromstr (np.array): Chromosome strings for the smoothing function.
    
    Returns:
        np.array: simulated RT signal.
    """
    
    # Isolate the G and S submatrices and the rowmean for G cells
    matrix_s = matrix[states == 'S', :]
    matrix_g = matrix[states == 'G', :]
    rowmean_g = rowmean[states == 'G']
    
    # Get the bias array for G cells
    matrix_g = matrix_g / rowmean_g[:, np.newaxis]
    bias = np.nanmean(matrix_g, axis=0)  # shape (ndomain,)
    
    # Simulate the RT signal
    rt_sim = np.nansum(matrix_s, axis=0) / bias  # shape (ndomain,)
    
    # Smooth the RT signal if required
    if smooth_k is not None:
        rt_sim = utils.smooth(rt_sim, smooth_chromstr, smooth_k)
    
    del matrix_s, matrix_g, rowmean_g, bias
    
    return rt_sim
