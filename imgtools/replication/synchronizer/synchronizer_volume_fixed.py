import numpy as np
from ...scf import SingleCellFeature
from .synchronizer import CellCycleSynchronizer


class CellCycleVolumerFixed(CellCycleSynchronizer):
    """ Algorithm to synchronize the cell cycle using the volume of the cells.
    
    It uses the same principle of the CellCycleVolumer: bottom X% of the cells are in G1,
    top Y% of the cells are in G2, and the rest are in S phase.
    
    However, the percentiles X and Y are fixed parameters, set in the configuration dictionary.
    
    Inherits from CellCycleSynchronizer.
    
    --- Attributes (inherit from CellCycleSynchronizer) ---
    scf (SingleCellFeature): SingleCellFeature object.
    index (Index): Index of the SingleCellFeature.
    config (dict): Configuration dictionary for the synchronization method.
    states_ (np.array): Array of strings with the states of the cells, e.g. ['G', 'S', 'G', ...], to be updated in the run method.
    
    --- Attributes (specific to CellCycleVolumer) ---
    volumes (np.array): Array of volumes.
    G1_percentile (float): Percentile for G1 cells.
    G2_percentile (float): Percentile for G2 cells.
    
    --- Methods (for users) ---
    run: Run the algorithm to synchronize the cell cycle.
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
        self.G1_percentile = self.config['G1_percentile']
        self.G2_percentile = self.config['G2_percentile']
        
    
    def check_config(self) -> None:
        """ Checks that the configuration dictionary contains the parameters needed for the Volume Synchronizer.
        
        If they are not present, they are added with default values.
        
        It checks that config has the following `keys`:
        - G1_percentile (float): Percentile for G1 cells.
        - G2_percentile (float): Percentile for G2 cells.
        """
        
        for key in ['G1_percentile', 'G2_percentile']:
            if key not in self.config:
                raise ValueError(f'{key} must be present in the configuration dictionary')
    
    
    def run(self) -> None:
        """ Run the algorithm to synchronize the cell cycle.
        
        The algorithm is implemented as follows:
        1. Sorts cells by volume.
        2. Sets the bottom X% of the cells to G1, using the G1_percentile parameter from the configuration dictionary.
        3. Sets the top Y% of the cells to G2, using the G2_percentile parameter from the configuration dictionary.
        4. The rest of the cells are set to S phase.
        """
        
        # Get the number of cells
        ncell = len(self.volumes)
        # Get the number of G1 and G2 cells from the percentiles
        ncell_g1 = int(np.floor(self.G1_percentile * ncell))
        ncell_g2 = int(np.floor(self.G2_percentile * ncell))
    
        # Initialize the states to 'S'
        states = np.full(len(self.volumes), 'S', dtype='U20')

        # Set the first ncell_g1 cells and the last ncell_g2 cells to 'G'
        states[:ncell_g1] = 'G1'
        states[(ncell - ncell_g2):] = 'G2'
        
        # The states array is sorted by volume (low to high)
        # Sort it back to the original order
        states = states[np.argsort(np.argsort(self.volumes))]

        # Update the states_ attribute
        self.states_ = states
