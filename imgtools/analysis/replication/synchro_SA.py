import os
import sys
import numpy as np
from alabtools.utils import Index, get_index_from_bed, get_index_from_bigwig
from ...scf import SingleCellFeature
from ... import utils
from . import repliseq


class CellCycleAnnealer:
    """Class to perform the simulated annealing algorithm for the cell cycle.
    """
    
    def __init__(self, scf: SingleCellFeature, config: dict) -> None:
        
        self.scf = scf
        self.index = scf.index
        self.config = config
        self.check_requirements()
        
        # Read the parameters from the configuration dictionary
        self.rt_file = self.config['rt_file']
        self.usechroms = self.config['usechroms']
        self.smooth = self.config['smooth']
        self.feature = self.config['feature']
        self.sa_temp0 = self.config['sa_temp0']
        self.sa_alpha = self.config['sa_alpha']
        self.sa_nstep = self.config['sa_nstep']
        
        # Read the feature matrix from the SingleCellFeature
        self.matrix = self.scf.get_matrix(self.feature)
        
        # Read the RT file from the configuration
        self.rt_index = self.read_RT()
        
        # Initialize the cell cycle states (not yet separating G1 and G2)
        self.states_ = self.initialize_states()
    
    
    # AUXILIARY METHODS FOR INITIALIZATION
    
    def check_requirements(self) -> None:
        """ Checks that the input data in __init__ is correct.
        It checks that:
        - scf is a SingleCellFeature
        - config is a dictionary
        - config has the following keys: rt_file, feature, usechroms, smooth, sa_temp0, sa_alpha, sa_nstep
        - rt_file exists
        - rt_file is a bed or bigwig file
        - feature is present in the SingleCellFeature
        - usechroms is a subset of the chromosomes present in the SingleCellFeature
        - smooth is a boolean
        """
        
        # Check that scf is a SingleCellFeature
        if not isinstance(self.scf, SingleCellFeature):
            raise TypeError("The input scf must be a SingleCellFeature.")
        # Check that config is a dictionary
        if not isinstance(self.config, dict):
            raise TypeError("The input config must be a dictionary.")
        # Check that config has the following keys
        required_keys = ['rt_file', 'feature', 'usechroms', 'smooth', 'sa_temp0', 'sa_alpha', 'sa_nstep']
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
        # Check that usechroms is a subset of the chromosomes present in the Index of the SingleCellFeature
        if not set(self.config['usechroms']).issubset(self.index.genome.chroms):
            raise ValueError(f"The chromosomes {self.config['usechroms']} are not present in the SingleCellFeature.")
        # Check that smooth is a boolean
        if not isinstance(self.config['smooth'], bool):
            raise TypeError("The smooth parameter must be a boolean.")
    
    def read_RT(self) -> Index:
        """Read the replication timing data from the RT file.
        
        The name of the file is taken from the configuration dictionary, and can be either a bed or bigwig file.
        
        The function than makes sure that - on the subset of chromosomes to be used in the Simulated Annealing -
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
        
        # Make sure that rt_index has a 'track0' attribute (the RT data) and it is a numpy array
        if not hasattr(rt_index, 'track0'):
            raise AttributeError("The index of the RT data must have a 'track0' attribute.")
        if not isinstance(rt_index.track0, np.ndarray):
            raise TypeError("The 'track0' attribute of the index of the RT data must be a numpy array.")
        
        # Make sure the index of the RT data matches the index of the SingleCellFeature
        # for the subset of chromosomes used in the Simulated Annealing
        if not utils.compare_index(self.index, rt_index, self.usechroms):
            raise ValueError(f"The index of the RT data does not match the index of the SingleCellFeature
                             on the chromosomes {self.usechroms}.")
        
        return rt_index
    
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
    
    
    # MAIN METHOD (RUN) OF SIMULATED ANNEALING AND AUXILIARY METHODS
    
    def run(self) -> None:
        
        # Create lists to store the cost and acceptance probability
        costs = list()
        probs = list()
        
        # Define the temperature schedule
        temps = self.sa_temp0 * self.sa_alpha ** np.arange(self.sa_nstep)
        
        # Initialize the cost function to be +∞
        cost = np.inf
        
        # Loop over the temperature schedule
        for temp in temps:
            
            # Update the states of the cells
            states_new = self.update_states()

            # Compute the new cost
            cost_new = self.cost_function(states_new)
            
            # Calculate the acceptance probability
            prob = self.accept_probability(cost, cost_new, temp)
            
            # Rejection condition: don't update and move to the next iteration
            if prob < np.random.uniform():
                continue
            
            # Acceptance condition: update the states, the cost, append to lists
            self.states_ = states_new
            cost = cost_new
            costs.append(cost)
            probs.append(prob)
    
    
    # CALCULATION METHODS NECESSARY FOR THE SIMULATED ANNEALING
    
    def simulate_rt(self, states: np.array) -> np.array:
        """ Simulate the RT signal using the feature matrix and the states of the cells.
        
        The bias in G1/G2 cells is first estimated, and then the matrix in S phase is normalized by it.
        
        Args:
            states (np.array): states of the cells, e.g. ['G', 'S', 'G', ...]
        
        Returns:
            np.array: simulated RT signal.
        """
        
        # Get the bias array for G cells
        bias = repliseq.get_bias(self.matrix, states)  # shape (ndomain,)
        # Reshape the bias array to broadcast with the matrix
        bias = np.reshape(bias, (1, len(bias), 1))  # shape (1, ndomain, 1)
        
        # Isolate the S phase submatrix
        matrix_s = self.matrix[states == 'S', :, :]
        
        # Normalize the matrix
        matrix_s_norm = matrix_s / bias
        
        # Compute the simulated RT signal
        rt_sim = np.nansum(matrix_s_norm, axis=(0, 2))
        
        del bias, matrix_s, matrix_s_norm
        
        return rt_sim
    
    def correlate_rt(self, rt_sim: np.array) -> float:
        """ Compute the Pearson correlation coefficient between the simulated and the observed RT signals.

        Args:
            rt_sim (np.array): RT signal simulated from the imaging data.

        Returns:
            float: Pearson correlation coefficient.
        """
        
        rt_exp = self.rt_index.track0
        
        # Isolate the RT signals for chromosomes specified in usechroms
        rt_sim_usechr = rt_sim[np.isin(self.index.chromstr, self.usechroms)]
        rt_exp_usechr = rt_exp[np.isin(self.rt_index.chromstr, self.usechroms)]
        
        # Compute the Pearson correlation coefficient
        r = utils.clean_pearsonr(rt_sim_usechr, rt_exp_usechr)
        
        return r
    
    
    # SIMULATED ANNEALING METHODS
    
    def cost_function(self, states: np.ndarray) -> float:
        """ Compute the cost function for the simulated annealing algorithm.
        
        First, the RT signal is simulated from the imaging data.
        Then, the Pearson r is computed between the simulated and the observed RT signals.
        Finally, the cost function is computed as the (opposite of) inverse hyperbolic tangent of r.
        
        atanh(x) is a bijection between (-1, 1) and (-∞, ∞), and it is smooth, so it is a good cost function:
              atanh(x) = 0.5 * log((1+x) / (1-x))
        
        To make sure that the cost is minimized when r = 1, we put a - sign in front of the atanh function.
        
        Also, to make the cost function more sensitive to small changes in r, we divide by log(1.01), so that
        we are using the base 1.01 logarithm.
        
        Args:
            states (np.array): states of the cells, e.g. ['G', 'S', 'G', ...]

        Returns:
            float: cost given the current states of the cells.
        """
        
        # Simulate the RT signal
        rt_sim = self.simulate_rt(states)
        
        # Compute the Pearson correlation coefficient
        r = self.correlate_rt(rt_sim)
        
        # Compute the cost function using the atanh function
        np.arctanh
        cost = - 0.5 * np.log((1+r) / (1-r)) / np.log(1.01)
        
        return cost
    
    def update_states(self) -> np.array:
        """ Update the states of the cells by switching a random cell from G to S or vice versa.

        Returns:
            np.array: updated states of the cells.
        """
        
        # Select a random cell to switch
        idx = np.random.choice(len(self.states_))
        
        # Switch the state of the cell
        states_updated = self.states_.copy()
        if self.states_[idx] == 'G':
            states_updated[idx] = 'S'
        else:
            states_updated[idx] = 'G'
        
        return states_updated

    @staticmethod
    def accept_probability(cost, cost_new, tmp):
        """Compute the acceptance probability.

        Args:
            cost (float): Cost value.
            cost_new (float): Updated cost value.
            tmp (float): Temperature.

        Returns:
            (float): Acceptance probability.
        """
        if cost_new <= cost:
            return 1.
        if cost_new > cost:
            return np.exp(-(cost_new - cost) / tmp)
