import os
import numpy as np
import h5py
from alabtools.utils import Index
from ..scf import SingleCellFeature
from ..scf import scf_utils


class SimulatedRepliSeqExperiment:
    """ A class to perform a simulated Repli-Seq experiment on a SingleCellFeature object.
    
    The aim of the simulated Repli-Seq experiment is to estimate the replication probability for each locus/cell.
    
    The experiment is divided into three main steps:
    1. Locus-dependent analysis:
        Assumes that different cells are independent realizations of the same locus-dependent process.
    2. Cell-dependent analysis:
        Assumes that different loci are independent realizations of the same cell-dependent process.
    3. Sliding window analysis:
        Relaxes the above assumptions, now every locus and cell can have different distributions.
        Assumes that the replication state is consistent within a sliding window.
    
    The object can be saved and loaded with an HDF5 file.
    
    ----------
    Attributes:
        ncells (int): number of cells in the SCF data.
        nloci (int): number of loci in the SCF data.
        ncopies (int): number of copies in the SCF data.
    
    ----------
    Datasets (from the SCF data):
        index (alabtools.utils.Index): index of the SCF data.
        states (np.ndarray): cell states of the SCF data, can be 'G1', 'S' or 'G2'. shape: (ncells).
        volumes (np.ndarray): cell nuclear volumes of the SCF data. shape: (ncells).
        n_ic (np.ndarray): spotcount of the SCF, i.e. number of spots per cell and per locus. shape: (ncells, nloci, ncopies).
    
    ----------
    Datasets created by the analysis:
        Locus-dependent analysis:
            pS_i (np.ndarray): S-phase replication probability for each locus, shape: (nloci).
            eps_i (np.ndarray): detection efficiency for each locus, shape: (nloci).
            b_i (np.ndarray): average multiplicative bias for each locus, shape: (nloci).
            pS_i_ (np.ndarray): S-phase replication probability for each locus, shape: (nloci).
            csi_i_ (np.ndarray): additive bias for each locus, shape: (nloci).
        Cell-dependent analysis:
            p_c (np.ndarray): replication probability for each cell, shape: (ncells).
            eps_c (np.ndarray): detection efficiency for each cell, shape: (ncells).
            b_c (np.ndarray): average multiplicative bias for each cell, shape: (ncells).
            eps_c_ (np.ndarray): approximate efficiency using only early replicating loci, shape: (ncells).
            b_c_ (np.ndarray): approximate b using only early replicating loci, shape: (ncells).
        Sliding window analysis:
            p_ic (np.ndarray): replication probability for each sliding window of locus/cell, shape: (ncells, nloci, ncopies).
            q_ic (np.ndarray): quality of the sliding window, True if enough statistical confidence, shape: (ncells, nloci, ncopies).
            eps_ic (np.ndarray): detection efficiency for each sliding window, shape: (ncells, nloci, ncopies).
            b_ic (np.ndarray): average multiplicative bias for each sliding window, shape: (ncells, nloci, ncopies).
        After sliding window analysis:
            r_ic (np.ndarray): replication state for each sliding window of locus/cell, shape: (ncells, nloci, ncopies).
    """
    
    
    # INITIALIZATION METHODS
    
    def __init__(self) -> None:
        """ Initialize the SimulatedRepliSeqExperiment object.
        
        This method just initializes the attributes as None.
        Then, either 'from_hdf5' or 'from_scf' should be called to initialize the object,
        either from a HDF5 file or from a SingleCellFeature object.
        """
        
        # Initialize as None the attributes that will be set later
        self.ncells = None
        self.nloci = None
        self.ncopies = None
        self.index = None
        self.states = None
        self.volumes = None
        self.n_ic = None
    
    @classmethod
    def from_hdf5(cls, filename: str) -> 'SimulatedRepliSeqExperiment':
        """ Initializes the SimulatedRepliSeqExperiment object by loading the data from an HDF5 file.
        
        Args:
            filename (str): name, with path, of the HDF5 file to load the data.
        
        Returns:
            SimulatedRepliSeqExperiment
        """
        
        # Create a new object
        obj = cls()
        # Load the data from the HDF5 file
        obj.load_from_hdf5(filename)
        return obj
    
    
    @classmethod
    def from_scf(cls, scf: SingleCellFeature) -> 'SimulatedRepliSeqExperiment':
        """ Initializes the SimulatedRepliSeqExperiment object from a SingleCellFeature object.
        
        Args:
            scf (SingleCellFeature)
        
        Returns:
            SimulatedRepliSeqExperiment
        """
        
        obj = cls()
        
        # Check the input SingleCellFeature object
        obj._check_scf(scf)
        
        obj.index = scf.index
        obj.states = scf.cell_states
        obj.volumes = scf.volumes
        obj.n_ic = scf.get_feature('spotcount')
        obj.ncells, obj.nloci, obj.ncopies = obj.n_ic.shape
        
        return obj
    
    @staticmethod
    def _check_scf(scf: SingleCellFeature) -> None:
        """ Check the input SingleCellFeature object.
        
        It checks that:
         - the input is a SingleCellFeature object,
         - the SCF contains the 'spotcount' feature,
         - the SCF contains the 'cell_states' feature,
         - the 'cell_states' feature only contains 'G1', 'S' and 'G2',
         - the index of the SCF has a valid resolution with consecutive loci.

        Args:
            scf (SingleCellFeature)
        """
        
        if not isinstance(scf, SingleCellFeature):
            raise TypeError("The input scf must be a SingleCellFeature.")
        
        if 'spotcount' not in scf.feature_list:
            raise ValueError("The input scf must contain the 'spotcount' feature.")
        if 'cell_states' not in scf:
            raise ValueError("The input scf must contain the 'cell_states' dataset.")
        if not all([state in ['G1', 'S', 'G2'] for state in scf.cell_states]):
            raise ValueError("The 'cell_states' feature must only contain 'G1', 'S' and 'G2'.")
        if 'volumes' not in scf:
            raise ValueError("The input scf must contain the 'volumes' dataset.")
        
        if scf.index.resolution() is None:
            raise ValueError("The index of the input SCF must have a valid resolution.")
        if not scf.index.consecutive():
            raise ValueError("The index of the input SCF must have consecutive loci.")
    
    
    # INPUT/OUTPUT METHODS
    
    def save_to_hdf5(self, filename: str) -> None:
        """ Save the data of the object to an HDF5 file.
        
        To identify the data to store, it uses the keys of the object's __dict__ attribute.
        It doesn't store a few keys that are not relevant to the analysis.
        
        Args:
            filename (str): name, with path, of the HDF5 file to save the data.
        """
        
        # Check that the filename has a valid path
        if not os.path.exists(os.path.dirname(filename)):
            raise ValueError(f"Invalid path: {filename}")
        # Check that the filename doesn't already exist
        if os.path.exists(filename):
            print(f"Warning: {filename} already exists. Can't overwrite it.")
            return
        
        # Create the HDF5 file and save the data
        with h5py.File(filename, 'w') as f:
            
            # Save the index
            self.index.save(f)
            
            # If the object has a config dictionary, save it as a group
            if hasattr(self, 'config'):
                config_group = f.create_group('config')
                for key, value in self.config.items():
                    config_group.attrs[key] = value
            
            # Loop over the items of the object to save arrays as datasets
            for key, value in self.__dict__.items():

                # Ignore the keys that are saved in a different way
                keys_to_ignore = ['config', 'genome', 'index', 'ncells', 'nloci', 'ncopies']
                if key in keys_to_ignore:
                    continue
                # Ignore the keys that are not numpy arrays
                if not isinstance(value, np.ndarray):
                    continue
                # If the array is a string, save as S type
                if value.dtype.kind in ['U', 'S']:
                    f.create_dataset(key, data=value.astype('S'))
                # Otherwise, save with the default type
                else:
                    f.create_dataset(key, data=value)
    
    def load_from_hdf5(self, filename: str) -> None:
        """ Load the data of the object from an HDF5 file.
        
        It loads the data from the HDF5 file to the object's attributes.
        It doesn't load a few keys that are not relevant to the analysis.
        
        Args:
            filename (str): name, with path, of the HDF5 file to load the data.
        """
        
        # Check that the filename exists
        if not os.path.exists(filename):
            raise ValueError(f"File not found: {filename}")
        
        # Load the data from the HDF5 file
        with h5py.File(filename, 'r') as f:
            
            # Loop over the items of the object and load the data
            for key in f.keys():
                
                # If the key is 'config', load as a dictionary
                if key == 'config':
                    self.config = {k: v for k, v in f[key].attrs.items()}
                    continue
                
                # If the key is 'genome', skip it (it is loaded in the Index object)
                if key == 'genome':
                    continue
                
                # If the key is 'index', load as an Index object
                if key == 'index':
                    self.index = Index(f)
                    continue
                
                # Otherwise, load as a numpy array
                arr = f[key][:]
                # If the array is a string, convert to unicode string
                if arr.dtype.kind in ['U', 'S']:
                    arr = arr.astype(str)
                # Store the array in the object
                self.__dict__[key] = arr
        
        # Set the number of cells, loci and copies as attributes
        self.ncells, self.nloci, self.ncopies = self.n_ic.shape


    # RUN METHODS
    
    def run(self, config) -> None:
        """ Run the simulated Repli-Seq analysis on the SCF data.
        
        It performs three main steps:
        1. Locus-dependent analysis:
            Assumes that different cells are independent realizations of the same locus-dependent process.
            It thus estimates average values for each locus, e.g. the S-phase replication probability pS_i.
        2. Cell-dependent analysis:
            Assumes that different loci are independent realizations of the same cell-dependent process.
            It thus estimates average values for each cell, e.g. the replication probability p_c.
        3. Sliding window analysis:
            Relaxes the above assumptions, now every locus and cell can have different distributions.
            It assumes that the replication state is consistent within a sliding window,
            and estimates the replication probability p_ic for each locus/cell.
        
        Args:
            config (dict): configuration dictionary. Must contain the following keys:
                            - sex,
                            - sliding_window_size,
                            - sliding_window_f0_threshold,
                            - sliding_window_efficiency_threshold.
        """
        self._check_config(config)
        self.config = config
        self.locus_dependent_run()
        self.cell_dependent_run()
        self.sliding_window_run()
    
    @staticmethod
    def _check_config(config: dict) -> None:
        """ Check the input config dictionary.
        
        It checks that the input is a dictionary and that it contains the required keys:
         - sex,
         - sliding_window_size,
         - sliding_window_f0_threshold,
         - sliding_window_efficiency_threshold,
         
        It also checks that the 'sex' key is a string and that it is either 'male' or 'female'.

        Args:
            config (dict)
        """
            
        if not isinstance(config, dict):
            raise TypeError("The input config must be a dictionary.")
        
        required_keys = [
            'sex',
            'sliding_window_size',
            'sliding_window_f0_threshold',
            'sliding_window_efficiency_threshold',
        ]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing key '{key}' in the input config.")
        
        if not isinstance(config['sex'], str):
            raise TypeError(f"Input sex in config must be str. Got type {type(config['sex'])} instead.")
        if not config['sex'] in ['male', 'female']:
            raise ValueError(f"Input sex in config must be either 'male' or 'female'")
    
    def locus_dependent_run(self) -> None:
        """ Run the locus-dependent analysis.
        
        It assumes that different cells are independent realizations of the same locus-dependent process,
        and estimates average values for each locus.
        
        It estimates the following values:
        - pS_i: S-phase replication probability for each locus.
        - eps_i: detection efficiency for each locus.
        - b_i: average multiplicative bias for each locus.
        - pS_i_: S-phase replication probability for each locus,
                 without the assumption that the additive bias csi is 0.
        - csi_i_: additive bias for each locus.
        """
        
        # Calculate the average number of spots and the fraction of zeros per locus
        # for 4 cases: 1) all cells, 2) G1 cells, 3) S cells, 4) G2 cells.
        n_i = {}
        f0_i = {}
        for s in ['all', 'G1', 'S', 'G2']:
            # Create a mask for the state
            if s == 'all':
                mask_state = np.ones(self.ncells, dtype=bool)
            else:
                mask_state = self.states == s
            # Calculate the average number of spots for each locus
            n_i[s] = np.mean(self.n_ic[mask_state, :, :], axis=(0, 2))  # shape: (nloci)
            f0_i[s] = np.mean(self.n_ic[mask_state, :, :] == 0, axis=(0, 2))  # shape: (nloci)
            # Fix the values for the X and Y chromosomes if sex is male, since there is only one copy
            # In the SCF file, this means that the second copy is all 0s, and thus we have to adjust averages
            if self.config['sex'] == 'male':
                mask_XY = np.logical_or(self.index.chromstr == 'chrX', self.index.chromstr == 'chrY')
                # Double the average number of spots, since one copy is all 0s
                n_i[s][mask_XY] = n_i[s][mask_XY] * 2
                # Fix the fraction of zeros
                f0_i[s][mask_XY] = 2 * f0_i[s][mask_XY] - 1
        
        # Calculate z and the S-phase replication probability for each locus, pS
        z_i = (n_i['G1'] + n_i['G2'] / 2) / 2
        pS_i = n_i['S'] / z_i - 1
        
        # Calculate the replication probability, averaged across G1, S, G2, per locus
        nG1, nS, nG2 = np.sum(self.states == 'G1'), np.sum(self.states == 'S'), np.sum(self.states == 'G2')
        p_i = pS_i * nS / (nG1 + nS + nG2) + nG2 / (nG1 + nS + nG2)

        # Calculate the detection efficiency per locus
        eps_i = (1 + p_i - np.sqrt((1 + p_i) ** 2 - 4 * p_i * (1 - f0_i['all']))) / (2 * p_i)
        # Calculate the average B values per locus
        b_i = z_i / eps_i
        
        # Re-calculate z, pS and also csi using the formula without the assumption that csi = 0
        # (in this case the efficiency and B cannot be calculated)
        z_i_ = n_i['G2'] - n_i['G1']
        csi_i_ = 2 * n_i['G1'] - n_i['G2']
        pS_i_ = (n_i['S'] - csi_i_) / z_i_ - 1
        
        # Store the results
        self.pS_i = pS_i
        self.eps_i = eps_i
        self.b_i = b_i
        self.pS_i_ = pS_i_
        self.csi_i_ = csi_i_

    def cell_dependent_run(self) -> None:
        """ Run the cell-dependent analysis.
        
        It assumes that different loci are independent realizations of the same cell-dependent process,
        and estimates average values for each cell.
        
        It estimates the following values:
        - p_c: replication probability for each cell (= 0 for G1, = 1 for G2).
        - eps_c: detection efficiency for each cell.
        - b_c: average multiplicative bias for each cell.
        - eps_c_: approximate efficiency using only early replicating loci.
        - b_c_: approximate b using only early replicating loci.
        """
        
        # Calculate the average number of spots and the fraction of zeros per cell
        # using either all autosomic loci or the early replicating autosomic loci.
        n_c = {}
        f0_c = {}
        for loci in ['all', 'early']:
            # Create the loci mask
            if loci == 'all':
                mask_loci = np.logical_and(self.index.chromstr != 'chrX', self.index.chromstr != 'chrY')
            else:
                mask_loci = np.logical_and(
                    np.logical_and(self.index.chromstr != 'chrX', self.index.chromstr != 'chrY'),
                    self.pS_i > 0.9
                )
            # Calculate the average number of spots and the fraction of zeros for each cell
            n_c[loci] = np.mean(self.n_ic[:, mask_loci, :], axis=(1, 2))  # shape: (ncells)
            f0_c[loci] = np.mean(self.n_ic[:, mask_loci, :] == 0, axis=(1, 2))
        
        # Calculate the approximate efficiency and b for G1, S, G2 using the early replicating loci
        eps_c_ = np.full(self.ncells, np.nan)
        eps_c_[self.states == 'G1'] = 1 - f0_c['early'][self.states == 'G1']
        eps_c_[self.states == 'S'] = 1 - f0_c['early'][self.states == 'S'] ** 0.5
        eps_c_[self.states == 'G2'] = 1 - f0_c['early'][self.states == 'G2'] ** 0.5
        b_c_ = np.full(self.ncells, np.nan)
        b_c_[self.states == 'G1'] = n_c['early'][self.states == 'G1'] / eps_c_[self.states == 'G1']
        b_c_[self.states == 'S'] = n_c['early'][self.states == 'S'] / (2 * eps_c_[self.states == 'S'])
        b_c_[self.states == 'G2'] = n_c['early'][self.states == 'G2'] / (2 * eps_c_[self.states == 'G2'])
        
        # Calculate the efficiency for G1 and G2
        eps_c = np.full(self.ncells, np.nan)
        eps_c[self.states == 'G1'] = 1 - f0_c['all'][self.states == 'G1']
        eps_c[self.states == 'G2'] = 1 - f0_c['all'][self.states == 'G2'] ** 0.5
        
        # Calculate b_c for G1 and G2
        b_c = np.full(self.ncells, np.nan)
        b_c[self.states == 'G1'] = n_c['all'][self.states == 'G1'] / eps_c[self.states == 'G1']
        b_c[self.states == 'G2'] = n_c['all'][self.states == 'G2'] / (2 * eps_c[self.states == 'G2'])
        
        # Use the approximate b for S
        b_c[self.states == 'S'] = b_c_[self.states == 'S']
        
        # Calculate the efficiency for S
        dS_c = n_c['all'][self.states == 'S'] / b_c[self.states == 'S']
        eps_S_c = (dS_c / 2) * (1 + np.sqrt(1 - 4 * (f0_c['all'][self.states == 'S'] + dS_c - 1) / dS_c ** 2))
        # Use the approximate efficiency for S if the formula gives NaN
        eps_S_c[np.isnan(eps_S_c)] = eps_c_[self.states == 'S'][np.isnan(eps_S_c)]
        eps_c[self.states == 'S'] = eps_S_c
        
        # Calculate the replication probability
        p_c = np.full(self.ncells, np.nan)
        p_c[self.states == 'G1'] = 0
        p_c[self.states == 'G2'] = 1
        p_c[self.states == 'S'] = n_c['all'][self.states == 'S'] / (eps_c[self.states == 'S'] * b_c[self.states == 'S']) - 1
        
        # Store the results
        self.p_c = p_c
        self.eps_c = eps_c
        self.eps_c_ = eps_c_
        self.b_c = b_c
        self.b_c_ = b_c_
    
    def sliding_window_run(self) -> None:
        """ Run the sliding window analysis.
        
        It relaxes the assumptions of the locus-dependent and cell-dependent analyses,
        and estimates the replication probability for each locus/cell.
        
        It estimates the following values:
        - p_ic: replication probability for each sliding window of locus/cell.
        - q_ic: quality of the sliding window, True if enough statistical confidence.
        - eps_ic: detection efficiency for each sliding window.
        - b_ic: average multiplicative bias for each sliding window.
        
        Note that we drop the 'sw' notation in the variable names when storing the results.
        """
        
        # Copy the n_ic array
        n_ic = np.copy(self.n_ic)
        
        # Set counts in n_ic larger than a threshold to NaN
        n_ic[n_ic >= 4] = np.nan
        
        # Take eps_i, eps_c, b_i, b_c
        eps_i = self.eps_i.copy()
        eps_c = self.eps_c.copy()
        b_i = self.b_i.copy()
        b_c = self.b_c.copy()
        # Set edge cases to NaN
        eps_i[eps_i < 0] = np.nan
        eps_c[eps_c < 0] = np.nan
        eps_i[eps_i > 1] = np.nan
        eps_c[eps_c > 1] = np.nan
        b_i[b_i < 1] = np.nan
        b_c[b_c < 1] = np.nan
        
        # Calculate the B and eps matrices, using the formula
        #   b_ic = b_i + b_c - (<b_i> + <b_c>) / 2
        #   eps_ic = eps_i + eps_c - (<eps_i> + <eps_c>) / 2
        b_ic = b_i[np.newaxis, :] + b_c[:, np.newaxis] - (np.nanmean(b_i) + np.nanmean(b_c)) / 2
        b_ic = np.repeat(b_ic[:, :, np.newaxis], self.ncopies, axis=2)
        eps_ic = eps_i[np.newaxis, :] + eps_c[:, np.newaxis] - (np.nanmean(eps_i) + np.nanmean(eps_c)) / 2
        eps_ic = np.repeat(eps_ic[:, :, np.newaxis], self.ncopies, axis=2)
        assert b_ic.shape == n_ic.shape, f"b_ic shape: {b_ic.shape} != n_sw_ic shape: {n_sw_ic.shape}"
        assert eps_ic.shape == n_ic.shape, f"eps_ic shape: {eps_ic.shape} != n_sw_ic shape: {n_sw_ic.shape}"
        
        # Get the window size in units of loci
        window = int(np.ceil(self.config['sliding_window_size'] / self.index.resolution()))
        
        # Calculate the sliding window averages
        n_sw_ic = scf_utils.sliding_matrix(self.n_ic, self.index, window=window, method='mean')
        b_sw_ic = scf_utils.sliding_matrix(b_ic, self.index, window=window, method='mean')
        eps_sw_ic = scf_utils.sliding_matrix(eps_ic, self.index, window=window, method='mean')
        
        # Calculate the fraction of zeros in the sliding windows
        n0_ic = np.zeros(n_ic.shape, dtype=float)
        n0_ic[n_ic == 0] = 1
        n0_ic[np.isnan(n_ic)] = np.nan  # ignore NaN values, i.e. values larger than 4
        f0_sw_ic = scf_utils.sliding_matrix(n0_ic, self.index, window=window, method='mean')
        
        # Create a quality array: it's True for regions with enough statistical confidence:
        # we want that the fraction of zeros is smaller than a threshold and the efficiency is larger than another threshold
        f0_sw_ok_ic = f0_sw_ic < self.config['sliding_window_f0_threshold']
        eps_sw_ok_ic = eps_sw_ic > self.config['sliding_window_efficiency_threshold']
        q_sw_ic = np.logical_and(f0_sw_ok_ic, eps_sw_ok_ic)
        
        # Calculate the replication probability
        p_sw_ic = n_sw_ic / (b_sw_ic * eps_sw_ic) - 1

        # Store the results
        self.p_ic = p_sw_ic
        self.q_ic = q_sw_ic
        self.eps_ic = eps_sw_ic
        self.b_ic = b_sw_ic
    
    
    # MISCELLANEOUS METHODS
    
    def sort_by_cellcycle(self) -> np.ndarray:
        """ Sort the cells by cell cycle pseudo-time.
        
        Returns a sorter array that sorts the cells by cell cycle pseudo-time.
        
        The cells are first sorted by state: G1, S and then G2.
        
        Then, the cells are sorted by nuclear volume within G1 and G2,
        and by cell-specific replication probability within S.
        
        Usage:
            sorter = self.sort_by_cellcycle()
            x: np.array  # shape: (ncells,)
            x_sorted = x[sorter]
        
        Returns:
            np.ndarray: the sorted cell indices.
        """
        
        # Check that the states, volumes and p_c attributes exist
        for attr in ['states', 'volumes', 'p_c']:
            if not hasattr(self, attr):
                raise ValueError(f"The attribute {attr} has not been set yet.")
        
        # To implement the sorting, we create a sorter array,
        # where its values are monotonically increasing with the desired sorting order.
        # In G1 and G2, the sorter value is the nuclear volume.
        # In S, it's the replication probability.
        # To make sure that the sorter puts G1 before S and S before G2,
        # we add a quantity (delta) to S and double that (2 * delta) to G2,
        # such that the sorter values in G1 < sorter values in S < sorter values in G2.
        delta = 10 * (np.max(self.volumes) + np.max(self.p_c))
        sorter = np.full(self.ncells, np.nan)
        sorter[self.states == 'G1'] = self.volumes[self.states == 'G1']
        sorter[self.states == 'S'] = self.p_c[self.states == 'S'] + delta
        sorter[self.states == 'G2'] = self.volumes[self.states == 'G2'] + 2 * delta
        
        # We then sort the sorter array and return the indices
        return np.argsort(sorter)
    
    def group_by_cellcycle(self, ncells_per_group: int) -> np.ndarray:
        """ Group the cells by cell cycle progression.
        
        Given the input number of cells per group, it:
            - Determines the number of groups for each state,
            - Sorts the cells by cell cycle pseudo-time,
            - Assigns the sorted cells to the groups.
        
        Note: the last group may have fewer cells than the input number.
        
        E.g. {
            'G1_1': [12, 34, ..., 3],
            'G1_2': [45, 67, ..., 89],
            ...
            'S_1': [4, 12, ..., 56],
            ...,
            'G2_3': [74, 23, ..., 48]
        }

        Args:
            ncells_per_group (int): number of cells per group.

        Returns:
            dict: the groups dictionary, with the group names as keys and the list of cell indices as values.
        """
        
        # Get the sorter array
        sorter = self.sort_by_cellcycle()
        
        # Subset the sorter in G1, S and G2
        nG1 = np.sum(self.states == 'G1')
        nS = np.sum(self.states == 'S')
        sorter_bystate = {
            'G1': sorter[:nG1],
            'S': sorter[nG1: nG1 + nS],
            'G2': sorter[nG1 + nS:]
        }
        
        # Initialize the groups dictionary
        groups = {}
        
        # Loop over the states and create the groups
        for state in ['G1', 'S', 'G2']:
            
            # Get the sorter array for the state
            sorter_state = sorter_bystate[state]
            
            # Determine the number of groups
            ngroups = int(np.ceil(len(sorter_state) / ncells_per_group))
            
            # Loop over the groups and assign the cells
            for i in range(ngroups):
                group_indices = sorter_state[i * ncells_per_group: (i+1) * ncells_per_group]
                groups[f"{state}_{i+1}"] = group_indices
        
        return groups


def simple_simulate_rt(
    scf: SingleCellFeature,
    states: np.array = None,
) -> np.array:
    """ Simulate the replication timing (RT) from the spotcount matrix in the SCF object.
    
    Uses a simplified method that doesn't require to distinguish between G1 and G2 cells.
    
    However, the output RT is not a probability value between 0 and 1, and it has no clear units.
    
    Requires a cell cycle state array with values 'G1', 'S' and 'G2', either provided or from the SCF object.
    
    The function estimates the 'Z_i' profile (product of efficiency and bias) in a simplified way:
    - Isolates the spotcount matrix for the G1/G2 cells,
    - Normalizes the matrix so that each cell has the same mean = 1,
    - Calculates the average of the normalized matrix to a profile of shape (nloci,).
    The normalization to 1 is performed in order not to give more weight to the G2 cells, since they have more spots.
    
    The RT is calculated as in the SimulatedReplication class, by dividing the S-phase profile by the 'Z_i' profile.

    Args:
        scf (SingleCellFeature): 
        states (np.array, optional): cell cycle states, with values 'G1', 'S' and 'G2'.
            If not provided, it uses the states from the SCF object. Error if not found.

    Returns:
        rt (np.array): the replication timing profile, shape: (nloci,)
    """
    
    # Get the states from the SCF if not provided
    if states is None:
        try:
            states = scf.cell_states
        except AttributeError:
            raise ValueError("The input SCF does not have the cell states. Provide it.")
    
    # Get the spotcount matrix
    matrix = scf.get_feature('spotcount')  # shape: (ncells, nloci, ncopies)
    
    # Isolate S and G1/G2 matrices
    matrix_s = matrix[states == 'S', :, :]  # shape: (nS, nloci, ncopies)
    matrix_g1g2 = matrix[states != 'S', :, :]  # shape: (nG1+nG2, nloci, ncopies)
    
    # Get the bias from the G1/G2 matrix
    # Sum off the third axis, to get an array of shape (nG1+nG2, nloci)
    matrix_g1g2 = np.nansum(matrix_g1g2, axis=2)
    # Normalize the rows so that each cell has the same mean = 1
    row_mean = np.nanmean(matrix_g1g2, axis=1)  # shape: (nG1+nG2)
    matrix_g1g2_norm = matrix_g1g2 / row_mean[:, np.newaxis]
    # Get the average of the normalized matrix
    bias = np.nanmean(matrix_g1g2_norm, axis=0)  # shape: (nloci)
    
    # Get the RT from the S matrix by dividing by the bias
    rt = np.nanmean(matrix_s, axis=(0, 2)) / bias  # shape: (nloci)
    
    return rt
