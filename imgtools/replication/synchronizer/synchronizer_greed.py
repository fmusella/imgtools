import os
import sys
import tempfile
import pickle
from functools import partial
import numpy as np
from alabtools.parallel import Controller
from ...scf import SingleCellFeature
from ... import utils
from .synchronizer import CellCycleSynchronizer, simulate_rt


class CellCycleGreeder(CellCycleSynchronizer):
    """ Greedy algorithm to synchronize the cell cycle.
    
    The synchronization consists of assigning cell to either G1, G2 or S phase (see CellCycleSynchronizer).
    
    Here we use a greedy algorithm to synchronize the cell cycle. The algorithm is implemented as follows:
    Repeat:
        1. Simulate the RT with the current states.
        2. In parallel, change the state of each cell and compute a new correlation for each change.
        3. If the correlation has not improved, break the loop.
        4. If the correlation has improved, update the states with the best change.
    
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
    
    --- Attributes (specific to CellCycleGreeder) ---
    correlations_ (list): List of correlations between the simulated RT and the experimental one, to be updated in the run method.
    updated_indices_ (list): List of indices of the cells that have been updated, to be updated in the run method.
    
    --- Methods (for users) ---
    run: Run the greedy algorithm to synchronize the cell cycle.
    """
    
    def __init__(self, scf: SingleCellFeature, config: dict, initial_states: np.array = None) -> None:
        """ Initialize the CellCycleGreeder object.
        Inherits from CellCycleSynchronizer.

        Args:
            scf (SingleCellFeature)
            config (dict): configuration dictionary for the greedy algorithm.
            initial_states (np.array, optional): Initial states of the cells, e.g. ['G', 'S', 'G', ...].
                            If None, the states are initialized randomly.
        """
        
        super().__init__(scf, config, initial_states)
        
        # Initialize the correlations list and the indices list
        self.correlations_ = []
        self.updated_indices_ = []
        
        # If the initial states specify 'G1' or 'G2', change them to 'G'
        self.states_[self.states_ == 'G1'] = 'G'
        self.states_[self.states_ == 'G2'] = 'G'
    
    def check_config(self) -> None:
        """ Checks that the configuration dictionary contains the parameters needed for the Volume Synchronizer.

        In particular, checks that the parallel configuration is present and that the controller is either 'serial' or 'ipyparallel'.
        """
        
        # Check the parallel configuration
        if 'parallel' not in self.config:
            raise ValueError('parallel must be present in the configuration dictionary')
        if 'controller' not in self.config['parallel']:
            raise ValueError('controller must be present in the parallel configuration dictionary')
        if self.config['parallel']['controller'] not in ['serial', 'ipyparallel']:
            raise ValueError('controller must be either serial or ipyparallel')
    
    
    def run(self) -> None:
        """ Run the greedy algorithm to synchronize the cell cycle.
        
        The algorithm is implemented as follows:
        Repeat:
            1. Simulate the RT with the current states.
            2. In parallel, change the state of each cell and compute a new correlation for each change.
            3. If the correlation has not improved, break the loop.
            4. If the correlation has improved, update the states with the best change.
        """
        
        # Create a temporary directory
        tempdir = tempfile.mkdtemp(dir=os.getcwd())
        
        # Save the matrix, rowmean,rt, smooth_k, smooth_chromstr to the temporary directory
        with open(os.path.join(tempdir, 'data.pickle'), 'wb') as f:
            pickle.dump({
                'matrix': self.matrix,
                'rowmean': self.rowmean,
                'rt': self.rt,
                'smooth_k': self.smooth_k,
                'smooth_chromstr': self.smooth_chromstr
            }, f)
        
        # Create the controller
        controller = Controller(self.config)
        
        # Compute the initial correlation between the simulated RT and the experimental one
        rt_sim = simulate_rt(self.matrix, self.rowmean, self.states_, self.smooth_k, self.smooth_chromstr)
        r = utils.clean_correlation(self.rt, rt_sim)
        
        # Print the initial correlation
        sys.stdout.write(f"Starting correlation: {r}\n")
        
        # Loop: change the states until the correlation does not improve
        while True:
            
            # Update the the correlations list
            self.correlations_.append(r)
            
            # Save the states to the temporary directory
            with open(os.path.join(tempdir, 'states.pickle'), 'wb') as f:
                pickle.dump(self.states_, f)
            
            # Find the index of the state that improves the correlation the most (and the new correlation)
            i, r_new = controller.map_reduce(
                partial(self.parallel_task, tempdir=tempdir),
                self.reduce_task,
                np.arange(len(self.states_))
            )
            
            # Print the iteration results
            sys.stdout.write(f"\rIteration {len(self.correlations_)}\n")
            sys.stdout.write(f"Index: {i}\n")
            sys.stdout.write(f"New correlation: {r_new}\n")
            sys.stdout.write(f"Number of cells in S: {np.sum(self.states_ == 'S')}\n")
            sys.stdout.write(f"\n")
            
            # If the correlation has not improved, break the loop
            if r_new <= r:
                break
            
            # Update the correlation, the states and the indices
            r = r_new
            self.states_[i] = 'S' if self.states_[i] == 'G' else 'G'
            self.updated_indices_.append(i)
        
        # Remove the non-empty temporary directory
        os.system(f"rm -r {tempdir}")
        
        # Separate the 'G' states to G1 and G2
        self.separate_G1G2()
    
    
    @staticmethod
    def parallel_task(i: int, tempdir: str,) -> tuple:
        """ Parallel task to simulate the RT and compute the correlation
        between the simulated RT and the experimental one.

        Args:
            i (int): Index of the state to change.
            tempdir (str): Temporary directory where the data is stored.

        Returns:
            int: Index of the state investigated.
            float: Correlation between the simulated RT and the experimental one after the change of the state of i.
        """
        
        # Read the data from the temporary directory
        with open(os.path.join(tempdir, 'data.pickle'), 'rb') as f:
            data = pickle.load(f)
        matrix = data['matrix']
        rowmean = data['rowmean']
        rt = data['rt']
        smooth_k = data['smooth_k']
        smooth_chromstr = data['smooth_chromstr']
        del data
        
        # Read the states from the temporary directory
        with open(os.path.join(tempdir, 'states.pickle'), 'rb') as f:
            states = pickle.load(f)
        
        # Change the state of i
        states[i] = 'S' if states[i] == 'G' else 'G'
        
        # Simulate the RT
        rt_sim = simulate_rt(matrix, rowmean, states, smooth_k, smooth_chromstr)
        
        # Compute the correlation between the simulated RT and the experimental one
        r = utils.clean_correlation(rt, rt_sim)
        
        del states, matrix, rowmean, rt, smooth_k, smooth_chromstr, rt_sim
        
        return i, r
    
    @staticmethod
    def reduce_task(parallel_results: list) -> tuple:
        """ Reduce the results of the parallel tasks.
        
        Finds the index of the state that improves the correlation the most,
        returns it along with the new correlation.

        Args:
            parallel_results (list): List of tuples (i, r) from the parallel tasks.

        Returns:
            int: Index of the state that improves the correlation the most.
            float: Best correlation.
        """
        
        # Find the index of the state that improves the correlation the most
        i_best, r_best = -1, -1
        for i, r in parallel_results:
            if r > r_best:
                i_best, r_best = i, r
        
        return i_best, r_best
