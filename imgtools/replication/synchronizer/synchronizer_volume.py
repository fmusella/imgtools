import os
import tempfile
import pickle
from functools import partial
import numpy as np
from alabtools.parallel import Controller
from ...scf import SingleCellFeature
from ... import utils
from .synchronizer import CellCycleSynchronizer, simulate_rt


class CellCycleVolumer(CellCycleSynchronizer):
    """ Algorithm to synchronize the cell cycle using the volume of the cells.
    
    It assumes that the bottom X% of the cells are in G1, the top Y% are in G2 and the rest are in S,
    and it estimates X and Y by maximizing the correlation of the simulated RT with the experimental one.
    
    Since the search space is large, the algorithm allows to input the min/max percentiles for G1 and G2 cells
    to look for the best combination of X and Y.
    
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
    
    --- Attributes (specific to CellCycleVolumer) ---
    volumes (np.array): Array of volumes.
    G1_min_percentile (float): Minimum percentile for G1 cells.
    G1_max_percentile (float): Maximum percentile for G1 cells.
    G2_min_percentile (float): Minimum percentile for G2 cells.
    G2_max_percentile (float): Maximum percentile for G2 cells.
    nsegment_ (int): Number of possible G1/G2 segmentations.
    r_ (float): Correlation between the simulated and the experimental RT for the best segmentation.
    ncell_g1_ (int): Number of cells in G1 for the best segmentation.
    ncell_g2_ (int): Number of cells in G2 for the best segmentation.
    
    --- Methods (for users) ---
    run: Run the greedy algorithm to synchronize the cell cycle.
    """
    
    def __init__(self, scf: SingleCellFeature, config: dict, initial_states: np.array = None) -> None:
        """ Initialize the CellCycleVolumer object.
        Inherits from CellCycleSynchronizer.

        Args:
            scf (SingleCellFeature)
            config (dict): configuration dictionary for the volume synchronization method.
            initial_states (np.array, optional): Initial states of the cells, e.g. ['G', 'S', 'G', ...].
                            If None, the states are initialized randomly.
        """
        
        super().__init__(scf, config, initial_states)
        
        # Add the volumes from the SingleCellFeature
        if 'volumes' not in scf:
            raise ValueError('volumes must be present in the SingleCellFeature')
        self.volumes = scf.volumes
        
        # Check the configuration
        self.check_config()
        
        # Add the G1 and G2 percentiles to the object
        self.G1_min_percentile = self.config['G1_min_percentile']
        self.G1_max_percentile = self.config['G1_max_percentile']
        self.G2_min_percentile = self.config['G2_min_percentile']
        self.G2_max_percentile = self.config['G2_max_percentile']
        
        # Initialize the parameters for the simulation, to be updated in the run method
        self.nsegment_ = None
        self.r_ = None
        self.ncell_g1_ = None
        self.ncell_g2_ = None
        
    
    def check_config(self) -> None:
        """ Checks that the configuration dictionary contains the parameters needed for the Volume Synchronizer.
        
        If they are not present, they are added with default values.
        
        It checks that config has the following `keys`:
        - G1_min_percentile (float): Minimum percentile for G1 cells.
        - G1_max_percentile (float): Maximum percentile for G1 cells.
        - G2_min_percentile (float): Minimum percentile for G2 cells.
        - G2_max_percentile (float): Maximum percentile for G2 cells.
        - 'parallel', whose value is a dictionary whose key is 'controller', with either 'serial' or 'ipyparallel' as value.
        """
        
        # Default percentiles for G1 and G2
        default_percentiles = {
            'G1_min_percentile': 0.,
            'G1_max_percentile': 0.4,
            'G2_min_percentile': 0.,
            'G2_max_percentile': 0.4,
        }
        
        for key in default_percentiles:
            # If the keys are not present, add them with the default values
            if key not in self.config:
                self.config[key] = default_percentiles[key]
            # Check that the values are floats between 0 and 1
            if not isinstance(self.config[key], float):
                raise ValueError(f'{key} must be a float')
            if self.config[key] < 0 or self.config[key] > 1:
                raise ValueError(f'{key} must be between 0 and 1')
        
        # Check that the min percentiles are smaller than the max percentiles
        if self.config['G1_min_percentile'] >= self.config['G1_max_percentile']:
            raise ValueError('G1_min_percentile must be smaller than G1_max_percentile')
        if self.config['G2_min_percentile'] >= self.config['G2_max_percentile']:
            raise ValueError('G2_min_percentile must be smaller than G2_max_percentile')

        # Check the parallel configuration
        if 'parallel' not in self.config:
            raise ValueError('parallel must be present in the configuration dictionary')
        if 'controller' not in self.config['parallel']:
            raise ValueError('controller must be present in the parallel configuration dictionary')
        if self.config['parallel']['controller'] not in ['serial', 'ipyparallel']:
            raise ValueError('controller must be either serial or ipyparallel')
    
    
    def get_segmentations(self) -> np.array:
        """ Get all the possible G1/G2 segmentations.
        
        Returns:
            segmentation (np.array(nsegment, 2), dtype=int): segmentation array.
                    Each row is a possible G1/G2 segmentation, i.e. [ncell_g1, ncell_g2].
        """
        
        ncell = len(self.volumes)
        
        # Get the min/max absolute number of cells in G1 and G2
        min_g1 = int(self.G1_min_percentile * ncell) - 1
        min_g2 = int(self.G2_min_percentile * ncell) - 1
        max_g1 = int(self.G1_max_percentile * ncell) + 1
        max_g2 = int(self.G2_max_percentile * ncell) + 1
        
        # Adjust the min/max values to be within the range of the number of cells
        min_g1 = max(1, min_g1)
        min_g2 = max(1, min_g2)
        max_g1 = min(ncell - 1, max_g1)
        max_g2 = min(ncell - 1, max_g2)
        
        # Initialize the segmentations array
        segmentations = []
        
        # Loop over all the possible G1/G2 segmentations and append them to the segmentation array
        for ncell_g1 in range(min_g1, max_g1):
            for ncell_g2 in range(min_g2, max_g2):
                segmentations.append([ncell_g1, ncell_g2])
        segmentations = np.array(segmentations)
        
        return segmentations
    
    
    def run(self) -> None:
        """ Run the greedy algorithm to synchronize the cell cycle.
        
        The algorithm is implemented as follows:
        1. Get all the possible G1/G2 segmentations.
        2. In parallel, for each segmentation, simulate the RT signal and calculate the correlation with the experimental RT.
        3. Reduce the results to get the segmentation with the highest correlation.        
        """
        
        # Create a temporary directory
        tempdir = tempfile.mkdtemp(dir=os.getcwd())
        
        # Get the segmentations
        segmentations = self.get_segmentations()
        self.nsegment_ = len(segmentations)
        
        # Save the matrix, rowmean,rt, smooth_k, smooth_chromstr to the temporary directory
        with open(os.path.join(tempdir, 'data.pickle'), 'wb') as f:
            pickle.dump({
                'matrix': self.matrix,
                'rowmean': self.rowmean,
                'volumes': self.volumes,
                'rt': self.rt,
                'smooth_k': self.smooth_k,
                'smooth_chromstr': self.smooth_chromstr,
                'segmentations': segmentations,
            }, f)
        
        # Create the controller
        controller = Controller(self.config)

        # run the parallel and reduce tasks
        segmentID_best, r_best = controller.map_reduce(
            partial(self.parallel_task, tempdir=tempdir),
            self.reduce_task,
            args=np.arange(self.nsegment_),
        )
        
        # Delete the non-empty temporary directory
        os.system(f"rm -r {tempdir}")
        
        # Get the states array from the best segmentation
        ncell_g1, ncell_g2 = segmentations[segmentID_best]
        ncell_g1, ncell_g2 = int(ncell_g1), int(ncell_g2)
        self.states_ = get_states_by_volume_sorting(self.volumes, ncell_g1, ncell_g2)
        
        # Save the best segmentation and the correlation to the object
        self.r_ = r_best
        self.ncell_g1_ = ncell_g1
        self.ncell_g2_ = ncell_g2
    
    @staticmethod
    def parallel_task(segmentID: int, tempdir: os.PathLike) -> tuple:
        """ Parallel task to simulate the RT signal and calculate the correlation
        with the experimental RT for a given segmentation.

        Args:
            segmentID (int): Index of the segmentation.
            tempdir (os.PathLike): Path to the temporary directory.

        Returns:
            int: Index of the segmentation.
            float: Correlation between the simulated and the experimental RT.
        """
        
        # Read the data from the temporary directory
        with open(os.path.join(tempdir, 'data.pickle'), 'rb') as f:
            data = pickle.load(f)
        matrix = data['matrix']
        rowmean = data['rowmean']
        volumes = data['volumes']
        rt = data['rt']
        smooth_k = data['smooth_k']
        smooth_chromstr = data['smooth_chromstr']
        segmentations = data['segmentations']
        del data
        
        # Get the number of cells and the G1/G2 segmentation
        ncell_g1, ncell_g2 = segmentations[segmentID]
        ncell_g1, ncell_g2 = int(ncell_g1), int(ncell_g2)
        
        # Get the states array from the G1/G2 segmentation
        states = get_states_by_volume_sorting(volumes, ncell_g1, ncell_g2)
        
        # Simulate the RT signal
        rt_sim = simulate_rt(matrix, rowmean, states, smooth_k, smooth_chromstr)
        
        # Calculate the correlation between the simulated and the experimental RT
        r = utils.clean_pearsonr(rt, rt_sim)
        
        del matrix, rowmean, volumes, rt, smooth_chromstr, segmentations, states, rt_sim
        
        return segmentID, r
        
    
    @staticmethod
    def reduce_task(parallel_results: list) -> tuple:
        """ Reduce task to find the segmentation with the highest correlation.

        Args:
            parallel_results (list): List of tuples (segmentID, r) with the results from the parallel task.

        Returns:
            int: Index of the segmentation with the highest correlation.
            float: Correlation between the simulated and the experimental RT for the best segmentation.
        """
        
        # Find the segmentation with the highest correlation
        segmentID_best, r_best = -1, -1
        for segmentID, r in parallel_results:
            if r > r_best:
                segmentID_best, r_best = segmentID, r
        
        return segmentID_best, r_best
    


def get_states_by_volume_sorting(volumes: np.array, ncell_g1: int, ncell_g2: int) -> np.array:
    """ Get the states array from the G1/G2 segmentation.
    
    The first ncell_g1 cells are set to 'G', the last ncell_g2 cells are set to 'G' and the rest are set to 'S'.
    
    Args:
        volumes (np.array(ncell,)): Array of volumes.
        ncell_g1 (int): Number of cells in G1.
        ncell_g2 (int): Number of cells in G2.
    
    Returns:
        states (np.array(ncell,)): Array of strings with the states of the cells, e.g. ['G', 'S', 'G', ...].
    """
    
    ncell = len(volumes)
    
    # Initialize the states as an empty array of strings
    states = np.full(len(volumes), 'S', dtype='U20')

    # Set the first ncell_g1 cells and the last ncell_g2 cells to 'G'
    states[:ncell_g1] = 'G'
    states[(ncell - ncell_g2):] = 'G'
    
    # The states array is sorted by volume (low to high)
    # Sort it back to the original order
    states = states[np.argsort(np.argsort(volumes))]
    
    return states
        