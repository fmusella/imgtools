import os
import numpy as np
import h5py
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
    
    def __init__(self, scf: SingleCellFeature) -> None:
        """ Initialize the SimulatedRepliSeqExperiment object.

        Args:
            scf (SingleCellFeature)
        """
        
        # Check the input
        self._check_scf(scf)
        
        # Get the data from the input scf
        self.index = scf.index
        self.states = scf.cell_states
        self.volumes = scf.volumes
        # We write the spotcount feature as n_ic, where i is the cell index and c is the locus index
        # It implicitly assumed that the actual quantity is a 3D tensor, where the third dimension is the copy,
        # but we write it like this to match the notation of the mathematical formulas.
        self.n_ic = scf.get_feature('spotcount')  # shape: (ncells, nloci, ncopies)
        self.ncells, self.nloci, self.ncopies = self.n_ic.shape
    
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
            
            # Loop over the items of the object and save them
            for key, value in self.__dict__.items():

                # Ignore the keys that are not relevant to the analysis
                keys_to_ignore = ['config', 'genome', 'index', 'states', 'volumes', 'n_ic', 'ncells', 'nloci', 'ncopies']
                if key in keys_to_ignore:
                    continue
                
                # Save the data
                if isinstance(value, np.ndarray):
                    f.create_dataset(key, data=value)
        
            # Save the config dictionary in a separate group
            if not hasattr(self, 'config'):
                return None  # no config to save
            config_group = f.create_group('config')
            for key, value in self.config.items():
                config_group.attrs[key] = value
    
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
                
                # Otherwise, load as a numpy array
                self.__dict__[key] = f[key][:]


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
        
        # First set counts in n_ic larger than a threshold to NaN
        self.n_ic[self.n_ic >= 4] = np.nan
        
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
        assert b_ic.shape == self.n_ic.shape, f"b_ic shape: {b_ic.shape} != n_sw_ic shape: {n_sw_ic.shape}"
        assert eps_ic.shape == self.n_ic.shape, f"eps_ic shape: {eps_ic.shape} != n_sw_ic shape: {n_sw_ic.shape}"
        
        # Get the window size in units of loci
        window = int(np.ceil(self.config['sliding_window_size'] / self.index.resolution()))
        
        # Calculate the sliding window averages
        n_sw_ic = scf_utils.sliding_matrix(self.n_ic, self.index, window=window, method='mean')
        b_sw_ic = scf_utils.sliding_matrix(b_ic, self.index, window=window, method='mean')
        eps_sw_ic = scf_utils.sliding_matrix(eps_ic, self.index, window=window, method='mean')
        
        # Calculate the fraction of zeros in the sliding windows
        n0_ic = np.zeros(self.n_ic.shape, dtype=float)
        n0_ic[self.n_ic == 0] = 1
        n0_ic[np.isnan(self.n_ic)] = np.nan  # ignore NaN values, i.e. values larger than 4
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
    
    def set_replication_states(self, low: float, high: float) -> None:
        """ Set the replication states based on the replication probability.
        
        It sets the replication state to 1 (non-replicating) if the replication probability is below a low threshold,
        and to 2 (replicating) if the replication probability is above a high threshold.
        The in-between values are set to 0 (not enough confidence).
        
        Stores the replication state as r_ic in the object.

        Args:
            low (float): low threshold for non-replicating regions.
            high (float): high threshold for replicating regions.
        """
        r_ic = np.zeros(self.n_ic.shape, dtype=float)
        r_ic[self.p_ic < low] = 1
        r_ic[self.p_ic > high] = 2
        self.r_ic = r_ic
    
    
    # PERFORMANCE EVALUATION METHODS
    
    def yielding(self) -> dict:
        """ Calculate the yield of the replication classification,
        i.e. the fraction of significantly-predicted replicating/non-replicating/determined loci.
        
        The yield is calculated per locus, per cell and total, and is given as absolute and relative values:
        - absolute yield: the fraction of rep/nonrep/det loci over the total number of loci
        - relative yield: the fraction of rep/nonrep/det loci over the total number of imaged loci (where n_ic > 0)
        
        The yield should be measured for the S-phase cells, but G1 and G2 yields are also calculated for validation.

        Returns:
            dict: the yield dictionaries, with the following keys:
                - abs_rep_yield_i: absolute yield of replicating loci per locus.
                - abs_nonrep_yield_i: absolute yield of non-replicating loci per locus.
                - abs_det_yield_i: absolute yield of determined loci per locus.
                - abs_rep_yield_c: absolute yield of replicating loci per cell.
                - abs_nonrep_yield_c: absolute yield of non-replicating loci per cell.
                - abs_det_yield_c: absolute yield of determined loci per cell.
                - abs_rep_yield: absolute yield of replicating loci.
                - abs_nonrep_yield: absolute yield of non-replicating loci.
                - abs_det_yield: absolute yield of determined loci.
                - rel_rep_yield_i: relative yield of replicating loci per locus.
                - rel_nonrep_yield_i: relative yield of non-replicating loci per locus.
                - rel_det_yield_i: relative yield of determined loci per locus.
                - rel_rep_yield_c: relative yield of replicating loci per cell.
                - rel_nonrep_yield_c: relative yield of non-replicating loci per cell.
                - rel_det_yield_c: relative yield of determined loci per cell.
                - rel_rep_yield: relative yield of replicating loci.
                - rel_nonrep_yield: relative yield of non-replicating loci.
                - rel_det_yield: relative yield of determined loci.
                - abs_rep_yield_G1: absolute yield of replicating loci in G1.
                - abs_nonrep_yield_G1: absolute yield of non-replicating loci in G1.
                - abs_det_yield_G1: absolute yield of determined loci in G1.
                - abs_rep_yield_G2: absolute yield of replicating loci in G2.
                - abs_nonrep_yield_G2: absolute yield of non-replicating loci in G2.
                - abs_det_yield_G2: absolute yield of determined loci in G2.
                - rel_rep_yield_G1: relative yield of replicating loci in G1.
                - rel_nonrep_yield_G1: relative yield of non-replicating loci in G1.
                - rel_det_yield_G1: relative yield of determined loci in G1.
                - rel_rep_yield_G2: relative yield of replicating loci in G2.
                - rel_nonrep_yield_G2: relative yield of non-replicating loci in G2.
                - rel_det_yield_G2: relative yield of determined loci in G2.
        """
        
        # Get a copy of the replication states
        r_ic = np.copy(self.r_ic)
        
        # Set low quality regions to NaN
        r_ic[~self.q_ic] = np.nan
        # Set 0s to NaN (non-determined regions)
        r_ic[self.r_ic == 0] = np.nan
        
        # Isolate the S-phase cells
        rS_ic = r_ic[self.states == 'S', :, :]
        nS_ic = self.n_ic[self.states == 'S', :, :]
        nScells = np.sum(self.states == 'S')
        
        # Get the total number of non-replicated, replicated and determined loci:
        # per locus,
        rep_i = np.sum(rS_ic == 2, axis=(0, 2))
        nonrep_i = np.sum(rS_ic == 1, axis=(0, 2))
        det_i = np.sum(~np.isnan(rS_ic), axis=(0, 2))
        # per cell,
        rep_c = np.sum(rS_ic == 2, axis=(1, 2))
        nonrep_c = np.sum(rS_ic == 1, axis=(1, 2))
        det_c = np.sum(~np.isnan(rS_ic), axis=(1, 2))
        # and total
        rep = np.sum(rS_ic == 2)
        nonrep = np.sum(rS_ic == 1)
        det = np.sum(~np.isnan(rS_ic))
        
        # To calculate the yield, we consider the following:
        # - absolute yield: the fraction of rep/nonrep/det loci over the total number of loci
        # - relative yield: the fraction of rep/nonrep/det loci over the total number of imaged loci (where n_ic > 0)
        abs_rep_yield_i = rep_i / (nScells * self.ncopies)
        abs_nonrep_yield_i = nonrep_i / (nScells * self.ncopies)
        abs_det_yield_i = det_i / (nScells * self.ncopies)
        abs_rep_yield_c = rep_c / (self.nloci * self.ncopies)
        abs_nonrep_yield_c = nonrep_c / (self.nloci * self.ncopies)
        abs_det_yield_c = det_c / (self.nloci * self.ncopies)
        abs_rep_yield = rep / (nScells * self.nloci * self.ncopies)
        abs_nonrep_yield = nonrep / (nScells * self.nloci * self.ncopies)
        abs_det_yield = det / (nScells * self.nloci * self.ncopies)
        rel_rep_yield_i = rep_i / np.sum(nS_ic > 0, axis=(0, 2))
        rel_nonrep_yield_i = nonrep_i / np.sum(nS_ic > 0, axis=(0, 2))
        rel_det_yield_i = det_i / np.sum(nS_ic > 0, axis=(0, 2))
        rel_rep_yield_c = rep_c / np.sum(nS_ic > 0, axis=(1, 2))
        rel_nonrep_yield_c = nonrep_c / np.sum(nS_ic > 0, axis=(1, 2))
        rel_det_yield_c = det_c / np.sum(nS_ic > 0, axis=(1, 2))
        rel_rep_yield = rep / np.sum(nS_ic > 0)
        rel_nonrep_yield = nonrep / np.sum(nS_ic > 0)
        rel_det_yield = det / np.sum(nS_ic > 0)
        
        
        # Get the absolute and relative yield in G1/G2 for validation
        rG1_ic = r_ic[self.states == 'G1', :, :]
        rG2_ic = r_ic[self.states == 'G2', :, :]
        nG1_ic = self.n_ic[self.states == 'G1', :, :]
        nG2_ic = self.n_ic[self.states == 'G2', :, :]
        nG1cells = np.sum(self.states == 'G1')
        nG2cells = np.sum(self.states == 'G2')
        repG1 = np.sum(rG1_ic == 2)
        nonrepG1 = np.sum(rG1_ic == 1)
        detG1 = np.sum(~np.isnan(rG1_ic))
        repG2 = np.sum(rG2_ic == 2)
        nonrepG2 = np.sum(rG2_ic == 1)
        detG2 = np.sum(~np.isnan(rG2_ic))
        abs_rep_yield_G1 = repG1 / (nG1cells * self.nloci * self.ncopies)
        abs_nonrep_yield_G1 = nonrepG1 / (nG1cells * self.nloci * self.ncopies)
        abs_det_yield_G1 = detG1 / (nG1cells * self.nloci * self.ncopies)
        abs_rep_yield_G2 = repG2 / (nG2cells * self.nloci * self.ncopies)
        abs_nonrep_yield_G2 = nonrepG2 / (nG2cells * self.nloci * self.ncopies)
        abs_det_yield_G2 = detG2 / (nG2cells * self.nloci * self.ncopies)
        rel_rep_yield_G1 = repG1 / np.sum(nG1_ic > 0)
        rel_nonrep_yield_G1 = nonrepG1 / np.sum(nG1_ic > 0)
        rel_det_yield_G1 = detG1 / np.sum(nG1_ic > 0)
        rel_rep_yield_G2 = repG2 / np.sum(nG2_ic > 0)
        rel_nonrep_yield_G2 = nonrepG2 / np.sum(nG2_ic > 0)
        rel_det_yield_G2 = detG2 / np.sum(nG2_ic > 0)
        
        return {
            'abs_rep_yield_i': abs_rep_yield_i,
            'abs_nonrep_yield_i': abs_nonrep_yield_i,
            'abs_det_yield_i': abs_det_yield_i,
            'abs_rep_yield_c': abs_rep_yield_c,
            'abs_nonrep_yield_c': abs_nonrep_yield_c,
            'abs_det_yield_c': abs_det_yield_c,
            'abs_rep_yield': abs_rep_yield,
            'abs_nonrep_yield': abs_nonrep_yield,
            'abs_det_yield': abs_det_yield,
            'rel_rep_yield_i': rel_rep_yield_i,
            'rel_nonrep_yield_i': rel_nonrep_yield_i,
            'rel_det_yield_i': rel_det_yield_i,
            'rel_rep_yield_c': rel_rep_yield_c,
            'rel_nonrep_yield_c': rel_nonrep_yield_c,
            'rel_det_yield_c': rel_det_yield_c,
            'rel_rep_yield': rel_rep_yield,
            'rel_nonrep_yield': rel_nonrep_yield,
            'rel_det_yield': rel_det_yield,
            'abs_rep_yield_G1': abs_rep_yield_G1,
            'abs_nonrep_yield_G1': abs_nonrep_yield_G1,
            'abs_det_yield_G1': abs_det_yield_G1,
            'abs_rep_yield_G2': abs_rep_yield_G2,
            'abs_nonrep_yield_G2': abs_nonrep_yield_G2,
            'abs_det_yield_G2': abs_det_yield_G2,
            'rel_rep_yield_G1': rel_rep_yield_G1,
            'rel_nonrep_yield_G1': rel_nonrep_yield_G1,
            'rel_det_yield_G1': rel_det_yield_G1,
            'rel_rep_yield_G2': rel_rep_yield_G2,
            'rel_nonrep_yield_G2': rel_nonrep_yield_G2,
            'rel_det_yield_G2': rel_det_yield_G2,
        }

    
    def performance(self) -> dict:
        """ Evaluate the performance of the replication classification.
        
        This method evaluates the performance of the replication classification using two methods:
        1. Using G1/G2 as ground truth.
        2. Using early/late replicating loci in S as ground truth.

        Returns:
            dict: the performance dictionaries for the two methods.
        """
        
        if not hasattr(self, 'r_ic'):
            raise ValueError("The replication states have not been set yet.")
        
        perf_1 = self.performance_method_1(self.r_ic)
        perf_2 = self.performance_method_2(self.r_ic)
        
        return {'Method 1': perf_1, 'Method 2': perf_2}
    
    def evaluate_thresholds(self, ranges: list) -> None:
        """ Evaluate the performance of the replication classification for different thresholds.
        
        For each pair of low and high thresholds, it calculates the replication state based on the replication probability,
        and evaluates the performance of the replication classification using two methods:
        1. Using G1/G2 as ground truth.
        2. Using early/late replicating loci in S as ground truth.

        Args:
            ranges (list): list of two ranges, one for the low threshold and one for the high threshold.
        """
        
        # To find the best thresholds, initialize the best one to 0
        best_low, best_high, best_acc = None, None, 0
        
        # Loop over the ranges of low threshold (non-replicating) and high threshold (replicating)
        for low in ranges[0]:
            for high in ranges[1]:
                
                # Calculate the replication state: 1 non-replicating, 2 replicating, 0 not enough confidence
                r_ic = np.zeros(self.n_ic.shape, dtype=float)
                r_ic[self.p_ic < low] = 1
                r_ic[self.p_ic > high] = 2
                
                # Calculate the performance of the replication classification with the two methods
                perf_1 = self.performance_method_1(r_ic)
                perf_2 = self.performance_method_2(r_ic)
                
                # Print the performance dictionaries
                print(f"Low: {low}, High: {high}")
                print("Method 1:")
                print(perf_1)
                print("Method 2:")
                print(perf_2)
                print("\n")
                
                # Calculate the average accuracy of the two methods
                acc = (perf_1['accuracy'] + perf_2['accuracy']) / 2
                
                # Update the best thresholds if the accuracy is better
                if acc > best_acc:
                    best_low, best_high, best_acc = low, high, acc
        
        # Print the best thresholds
        print(f"Best thresholds: low = {best_low}, high = {best_high}, accuracy = {best_acc}")
        print("\n\n")
    
    def performance_method_1(
        self,
        r_ic: np.ndarray
    ) -> dict:
        """ Evaluate the performance of the replication classification using G1/G2 as ground truth.
        
        It evaluates true positives and true negatives, respectively, as the number of regions
        classified as replicating in G2 and non-replicating in G1.

        Args:
            r_ic (np.ndarray): replication state array, shape: (ncells, nloci, ncopies).

        Returns:
            dict: the confusion matrix and the performance metrics, with the following keys:
                - TP: true positives.
                - TN: true negatives.
                - FP: false positives.
                - FN: false negatives.
                - accuracy: accuracy.
                - ppv: positive predictive value.
                - npv: negative predictive value.
                - tpr: true positive rate.
                - tnr: true negative rate.
        """
        
        # Set low quality regions to NaN
        r_ic[~self.q_ic] = np.nan
        
        # Get the replication states and volumes for G1 and G2 cells
        r_ic_G1 = r_ic[self.states == 'G1', :, :]
        r_ic_G2 = r_ic[self.states == 'G2', :, :]
        vol_G1 = self.volumes[self.states == 'G1']
        vol_G2 = self.volumes[self.states == 'G2']
        
        # Remove the top 20% of the volumes for G1 and the bottom 20% of the volumes for G2,
        # to make sure that there is no S-phase contamination
        G1_max_vol = np.percentile(vol_G1, 80)
        G2_min_vol = np.percentile(vol_G2, 20)
        r_ic_G1 = r_ic_G1[vol_G1 < G1_max_vol, :, :]
        r_ic_G2 = r_ic_G2[vol_G2 > G2_min_vol, :, :]
        
        # Remove the NaN values and flatten the arrays
        r_ic_G1 = r_ic_G1[~np.isnan(r_ic_G1)].flatten()
        r_ic_G2 = r_ic_G2[~np.isnan(r_ic_G2)].flatten()
        
        # Subsample the two arrays to have the same length,
        # so that the performance metrics are not biased by different sample sizes
        np.random.seed(0)  # for reproducibility
        min_len = min(len(r_ic_G1), len(r_ic_G2))
        r_ic_G1 = np.random.choice(r_ic_G1, min_len, replace=False)
        r_ic_G2 = np.random.choice(r_ic_G2, min_len, replace=False)
        
        # Return the confusion matrix
        return self.confusion_matrix(r_ic_G1, r_ic_G2)
    
    def performance_method_2(
        self,
        r_ic: np.ndarray
    ) -> dict:
        """ Evaluate the performance of the replication classification using early/late replicating loci in S as ground truth.
        
        It evaluates true positives and true negatives, respectively, as the number of regions
        classified as replicating for the early replicating loci and non-replicating for the late replicating loci.

        Args:
            r_ic (np.ndarray): replication state array, shape: (ncells, nloci, ncopies).

        Returns:
            dict: the confusion matrix and the performance metrics, with the following keys:
                - TP: true positives.
                - TN: true negatives.
                - FP: false positives.
                - FN: false negatives.
                - accuracy: accuracy.
                - ppv: positive predictive value.
                - npv: negative predictive value.
                - tpr: true positive rate.
                - tnr: true negative rate.
        """
        
        # Set low quality regions to NaN
        r_ic[~self.q_ic] = np.nan
        
        # Get the replication states for S cells
        rS_ic = r_ic[self.states == 'S', :, :]

        # Identify the early/late replicating loci
        early = self.pS_i > np.nanpercentile(self.pS_i, 90)
        late = self.pS_i < np.nanpercentile(self.pS_i, 10)
        assert np.sum(early) == np.sum(late), "The number of early and late replicating loci should be the same"
        
        # Get the replication states in S for the early/late replicating loci
        rS_ic_early = rS_ic[:, early, :]
        rS_ic_late = rS_ic[:, late, :]
        
        # Consider only cells at the end of S phase for the early replicating loci (ensuring ground truth of r=2),
        # and at the beginning for the late replicating loci (ensuring ground truth of r=1)
        pS_c = self.p_c[self.states == 'S']
        rS_ic_early = rS_ic_early[pS_c > np.nanpercentile(pS_c, 90), :, :]
        rS_ic_late = rS_ic_late[pS_c < np.nanpercentile(pS_c, 10), :, :]
        
        # Remove the NaN values and flatten the arrays
        rS_ic_early = rS_ic_early[~np.isnan(rS_ic_early)].flatten()
        rS_ic_late = rS_ic_late[~np.isnan(rS_ic_late)].flatten()
        
        # Subsample the two arrays to have the same length,
        # so that the performance metrics are not biased by different sample sizes
        np.random.seed(0)
        min_len = min(len(rS_ic_early), len(rS_ic_late))
        rS_ic_early = np.random.choice(rS_ic_early, min_len, replace=False)
        rS_ic_late = np.random.choice(rS_ic_late, min_len, replace=False)
        
        # Return the confusion matrix
        return self.confusion_matrix(rS_ic_late, rS_ic_early)
        
    @staticmethod
    def confusion_matrix(y1: np.ndarray, y2: np.ndarray) -> dict:
        """ Calculate the confusion matrix of the replication classification.
        
        It calculates the classification counts as:
        - True positive: replicated regions correctly classified as replicated.
        - True negative: non-replicated regions correctly classified as non-replicated.
        - False positive: non-replicated regions incorrectly classified as replicated.
        - False negative: replicated regions incorrectly classified as non-replicated.
        
        And the performance metrics:
        - Accuracy: (TP + TN) / (TP + TN + FP + FN).
        - Positive predictive value (PPV): TP / (TP + FP).
        - Negative predictive value (NPV): TN / (TN + FN).
        - True positive rate (TPR): TP / (TP + FN).
        - True negative rate (TNR): TN / (TN + FP).

        Args:
            y1 (np.ndarray): predicted replication states, where the ground truth is 1 (non-replicated)
            y2 (np.ndarray): predicted replication states, where the ground truth is 2 (replicated)

        Returns:
            dict: the confusion matrix and the performance metrics, with the following keys:
                - TP: true positives.
                - TN: true negatives.
                - FP: false positives.
                - FN: false negatives.
                - accuracy: accuracy.
                - ppv: positive predictive value.
                - npv: negative predictive value.
                - tpr: true positive rate.
                - tnr: true negative rate.
        """
        
        # Calculate the true positives, true negatives, false positives and false negatives
        tp = np.sum(y2 == 2)  # True positives
        tn = np.sum(y1 == 1)  # True negatives
        fp = np.sum(y1 == 2)  # False positives
        fn = np.sum(y2 == 1)  # False negatives
        
        # Calculate the performance metrics
        acc = 100 * (tp + tn) / (tp + tn + fp + fn)  # Accuracy
        ppv = 100 * tp / (tp + fp)  # Positive predictive value (PPV)
        npv = 100 * tn / (tn + fn)  # Negative predictive value (NPV)
        tpr = 100 * tp / (tp + fn)  # True positive rate (TPR)
        tnr = 100 * tn / (tn + fp)  # True negative rate (TNR)

        # Return the results as a dictionary
        return {
            'TP': tp,
            'TN': tn,
            'FP': fp,
            'FN': fn,
            'accuracy': acc,
            'ppv': ppv,
            'npv': npv,
            'tpr': tpr,
            'tnr': tnr
        }
    
    
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
