import os
import tqdm
import numpy as np
from alabtools.utils import Index, get_index_from_bed, get_index_from_bigwig
from ...scf import SingleCellFeature
from ... import utils


AVAILABLE_SCHEDULES = ['linear', 'geometric', 'logarithmic']

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
        self.smooth_k = self.config['smooth_k']
        self.feature = self.config['feature']
        self.temp_0 = self.config['temp_0']
        self.temp_f = self.config['temp_f']
        self.nstep = self.config['nstep']
        self.schedule = self.config['schedule']
        
        # Prepare the matrix for the simulated annealing algorithm
        self.matrix, self.rowmean = self.prepare_matrix()
        
        # Read the RT file from the configuration
        self.rt_index = self.read_RT()
        # Prepare the RT signal for the simulated annealing algorithm
        self.rt = self.prepare_RT()
        
        # If the smoothing parameter is not None,
        # we need the chromstr array subsampled on usechroms for the smoothing function
        if self.smooth_k is not None:
            self.smooth_chromstr = self.index.chromstr[np.isin(self.index.chromstr, self.usechroms)]
        
        # Initialize the cell cycle states (not yet separating G1 and G2), which will be updated in the run method
        self.states_ = self.initialize_states()
        
        # Initialize the SA cost, cost diffs, and acceptance probability lists, which will be filled in the run method
        self.costs_ = list()
        self.costs_diff_ = list()
        self.probs_ = list()
    
    
    # AUXILIARY METHODS FOR INITIALIZATION
    
    def check_requirements(self) -> None:
        """ Checks that the input data in __init__ is correct.
        It checks that:
        - scf is a SingleCellFeature
        - config is a dictionary
        - config has the following keys: rt_file, feature, usechroms, smooth_k, temp_0, temp_f, nstep, schedule
        - the annealing schedule is one of the available ones (in AVAILABLE_SCHEDULES)
        - rt_file exists
        - rt_file is a bed or bigwig file
        - feature is present in the SingleCellFeature
        - usechroms is a subset of the chromosomes present in the SingleCellFeature
        - smooth_k is either None or a positive integer
        """
        
        # Check that scf is a SingleCellFeature
        if not isinstance(self.scf, SingleCellFeature):
            raise TypeError("The input scf must be a SingleCellFeature.")
        # Check that config is a dictionary
        if not isinstance(self.config, dict):
            raise TypeError("The input config must be a dictionary.")
        # Check that config has the following keys
        required_keys = ['rt_file', 'feature', 'usechroms', 'smooth_k', 'temp_0', 'temp_f', 'nstep', 'schedule']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"The key {key} is missing from the configuration dictionary.")
        # Check that the annealing schedule is one of the available ones
        if self.config['schedule'] not in AVAILABLE_SCHEDULES:
            raise ValueError(f"The annealing schedule {self.config['schedule']} is not available. Please choose one of {AVAILABLE_SCHEDULES}.")
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
        # Check that smoothing k parameter is either None or a positive integer
        if self.config['smooth_k'] is not None and not isinstance(self.config['smooth_k'], int):
            raise TypeError("The smoothing parameter k must be either None or an integer.")
        if isinstance(self.config['smooth_k'], int) and self.config['smooth_k'] <= 0:
            raise ValueError("The smoothing parameter k must be a positive integer.")
    
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
        matrix = self.scf.get_matrix(self.feature)  # (ncell, ndomain, max_ncopy_per_chrom)
        
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
        """ Prepare the RT signal for the Simulated Annealing algorithm.
        
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
    
    
    # MAIN METHOD (RUN) OF SIMULATED ANNEALING AND AUXILIARY METHODS
    
    def run(self) -> None:
        
        # Define the temperature schedule
        temps = self.annealing_schedule()
        
        # Initialize the cost function to be +∞
        cost = np.inf
        
        # Loop over the temperature schedule
        for temp in tqdm.tqdm(temps, desc='Simulated Annealing'):
             
            # Update the states of the cells
            states_new = self.update_states()

            # Compute the new cost
            cost_new = self.cost_function(states_new)
            
            # Calculate the acceptance probability
            prob = self.accept_probability(cost, cost_new, temp)
            
            # Append the cost, cost diff, and acceptance probability to the lists
            self.costs_.append(cost_new)
            self.costs_diff_.append(cost_new - cost)
            self.probs_.append(prob)
            
            # Rejection condition: don't update and move to the next iteration
            if prob < np.random.uniform():
                continue
            
            # Acceptance condition: update the states and the cost
            self.states_ = states_new
            cost = cost_new

    # CALCULATION METHODS NECESSARY FOR THE SIMULATED ANNEALING
    
    def annealing_schedule(self) -> np.array:
        """ Compute the annealing schedule.

        Returns:
            np.array: Annealing schedule, i.e. the temperature for each step.
        """
    
        if self.schedule == 'linear':
            # T(n) = T0 - α * n
            alpha = (self.temp_0 - self.temp_f) / self.nstep
            return self.temp_0 - alpha * np.arange(self.nstep)
        
        elif self.schedule == 'geometric':
            # T(n) = T0 * α^n
            alpha = (self.temp_f / self.temp_0) ** (1 / self.nstep)
            return self.temp_0 * alpha ** np.arange(self.nstep)
        
        elif self.schedule == 'logarithmic':
            # T(n) = β + α / log(e + n)
            alpha = (self.temp_0 - self.temp_f) / (1 - 1 / np.log(np.e + self.nstep))
            beta = self.temp_0 - alpha
            return beta + alpha / np.log(np.e + np.arange(self.nstep))
    
    def simulate_rt(self, states: np.array) -> np.array:
        """ Simulate the RT signal using the feature matrix and the states of the cells.
        
        The bias in G1/G2 cells is first estimated, and then the matrix in S phase is normalized by it.
        
        Args:
            states (np.array): states of the cells, e.g. ['G', 'S', 'G', ...]
        
        Returns:
            np.array: simulated RT signal.
        """
        
        # Isolate the G and S submatrices and the rowmean for G cells
        matrix_s = self.matrix[states == 'S', :]
        matrix_g = self.matrix[states == 'G', :]
        rowmean_g = self.rowmean[states == 'G']
        
        # Get the bias array for G cells
        matrix_g = matrix_g / rowmean_g[:, np.newaxis]
        bias = np.nanmean(matrix_g, axis=0)  # shape (ndomain,)
        
        # Simulate the RT signal
        rt_sim = np.nansum(matrix_s, axis=0) / bias  # shape (ndomain,)
        
        # Smooth the RT signal if required
        if self.smooth_k is not None:
            rt_sim = utils.smooth(rt_sim, self.smooth_chromstr, self.smooth_k)
        
        del matrix_s, matrix_g, rowmean_g, bias
        
        return rt_sim
    
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
        r = utils.clean_pearsonr(rt_sim, self.rt)
        
        # Compute the cost function using the (opposite of) the atanh function:
        #       atanh(x) = 0.5 * log((1+x) / (1-x))
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
