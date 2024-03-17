import os
import tempfile
import pickle
import numpy as np
from alabtools.parallel import Controller
from ....scf import SingleCellFeature
from .... import utils
from .synchronizer import CellCycleSynchronizer


class CellCycleGreeder(CellCycleSynchronizer):
    """Greedy algorithm to synchronize the cell cycle.
    
    Inherits from CellCycleSynchronizer.
    
    --- Attributes ---
    
    --- Methods (for users) ---
    run: Run the greedy algorithm to synchronize the cell cycle.
    """
    
    def __init__(self, scf: SingleCellFeature, config: dict, initial_states: np.array = None) -> None:
        
        super(CellCycleGreeder, self).__init__(scf, config, initial_states)
        
        # Initialize the number of iterations and the correlations list (to be updated during the run)
        self.niter_ = 0
        self.correlations_ = []
    
    
    def run(self) -> None:
        
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
        r = self.compute_correlation()
        
        # Loop: change the states until the correlation does not improve
        while True:
            
            # Update the number of iterations and the correlations list
            self.niter_ += 1
            self.correlations_.append(r)
            
            # Save the states to the temporary directory
            with open(os.path.join(tempdir, 'states.pickle'), 'wb') as f:
                pickle.dump(self.states_, f)
            
            # Find the index of the state that improves the correlation the most (and the new correlation)
            i, r_new = controller.map_reduce(
                self.parallel_task,
                self.reduce_task,
                np.arange(len(self.states_))
            )
            
            # If the correlation has not improved, break the loop
            if r_new <= r:
                break
            
            # Update the correlation and the states
            r = r_new
            self.states_[i] = 'S' if self.states_[i] == 'G' else 'G'
        
        # Remove the temporary directory
        os.rmdir(tempdir)
    
    
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
        
        # Read the matrix, rowmean,rt, smooth_k, smooth_chromstr from the temporary directory
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
        
        # Simulate the RT
        rt_sim = simulate_rt(matrix, rowmean, states, smooth_k, smooth_chromstr)
        
        # Compute the correlation between the simulated RT and the experimental one
        r = utils.clean_pearsonr(rt, rt_sim)
        
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
