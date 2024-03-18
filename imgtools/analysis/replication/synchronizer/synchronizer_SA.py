import numpy as np
import tqdm
from ....scf import SingleCellFeature
from .... import utils
from .synchronizer import CellCycleSynchronizer, simulate_rt


class CellCycleAnnealer(CellCycleSynchronizer):
    """Class to perform the simulated annealing algorithm for the cell cycle synchronization.
    
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
    AVAILABLE_SCHEDULES (list): List of available annealing schedules.
    temp_0 (float): Initial temperature.
    temp_f (float): Final temperature.
    nstep (int): Number of steps.
    schedule (str): Annealing schedule, one of ['linear', 'geometric', 'logarithmic'].
    costs_ (list): List of costs at each step.
    costs_diff_ (list): List of cost differences at each step.
    probs_ (list): List of acceptance probabilities at each step.
    
    --- Methods (for users) ---
    run: Run the simulated annealing algorithm.
    """
    
    def __init__(self, scf: SingleCellFeature, config: dict, initial_states: np.array = None) -> None:
        """ Initialize the CellCycleAnnealer object.
        Inherits from CellCycleSynchronizer.
        
        Args:
            scf (SingleCellFeature)
            config (dict): configuration dictionary for the simulated annealing algorithm.
            initial_states (np.array, optional): Initial states of the cells, e.g. ['G', 'S', 'G', ...].
                            If None, the states are initialized randomly.
        """
        
        super(CellCycleAnnealer, self).__init__(scf, config, initial_states)
        
        # Define the available annealing schedules
        self.AVAILABLE_SCHEDULES = ['linear', 'geometric', 'logarithmic']
        
        # Check that the configuration dictionary has the required keys
        self.check_config()
        
        # Extract the configuration parameters for the SA algorithm
        self.temp_0 = self.config['temp_0']
        self.temp_f = self.config['temp_f']
        self.nstep = self.config['nstep']
        self.schedule = self.config['schedule']
        
        # Initialize the SA cost, cost diffs, and acceptance probability lists, which will be filled in the run method
        self.costs_ = list()
        self.costs_diff_ = list()
        self.probs_ = list()
    
    
    def check_config(self) -> None:
        """ Check that the configuration dictionary has the required SA keys
        and that the annealing schedule is one of the available ones.
        """
        
        # Check that config has the following keys
        required_keys = ['temp_0', 'temp_f', 'nstep', 'schedule']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"The key {key} is missing from the configuration dictionary.")
            
        # Check that the annealing schedule is one of the available ones
        if self.config['schedule'] not in self.AVAILABLE_SCHEDULES:
            raise ValueError(f"The annealing schedule {self.config['schedule']} is not available. Please choose one of {self.AVAILABLE_SCHEDULES}.")
    
    
    # MAIN METHOD (RUN) OF SIMULATED ANNEALING AND AUXILIARY METHODS
    
    def run(self) -> None:
        """ Run the simulated annealing algorithm.
        
        The algorithm is implemented as follows:
            1. Initialize the cost function to be +∞.
            2. Define the temperature schedule, i.e. the temperature for each step with T(n+1) < T(n).
            3. Loop over the temperatures:
                a. Randomly update the states of the cells.
                b. Compute the new cost.
                c. Calculate the acceptance probability.
                d. Append the cost, cost diff, and acceptance probability to the lists.
                e. If the acceptance probability is less than a random number sampled from U(0,1), don't update and move to the next iteration.
                f. If the acceptance probability is greater than a random number sampled from U(0,1), update the states and the cost.
        """
        
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
        rt_sim = simulate_rt(self.matrix, self.rowmean, states, self.smooth_k, self.smooth_chromstr)
        
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
        """ Compute the acceptance probability:
                p = 1 if cost_new <= cost
                p = exp(-(cost_new - cost) / tmp) if cost_new > cost

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
