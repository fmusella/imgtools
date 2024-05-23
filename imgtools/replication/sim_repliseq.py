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
    
    The data can be saved to an HDF5 file.
    
    ----------
    Attributes:
        ncells (int): number of cells in the SCF data.
        nloci (int): number of loci in the SCF data.
        ncopies (int): number of copies in the SCF data.
        sex (str): sex of the organism
    
    ----------
    Properties (from the SCF data):
        index (alabtools.utils.Index): index of the SCF data.
        states (np.ndarray): cell states of the SCF data, can be 'G1', 'S' or 'G2'.
        n_ic (np.ndarray): number of spots per cell and per locus, shape: (ncells, nloci, ncopies).
    
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
            b_c_ (np.ndarray): approximate b using only early replicating loci, shape: (ncells).
        Sliding window analysis:
            p_sw_ic (np.ndarray): replication probability for each sliding window of locus/cell, shape: (ncells, nloci, ncopies).
            q_sw_ic (np.ndarray): quality of the sliding window, True if enough statistical confidence, shape: (ncells, nloci, ncopies).
            r_sw_ic (np.ndarray): replication state for each sliding window,
                            1 for non-replicating, 2 for replicating, NaN for not enough confidence, shape: (ncells, nloci, ncopies).
            eps_sw_ic (np.ndarray): detection efficiency for each sliding window, shape: (ncells, nloci, ncopies).
            b_sw_ic (np.ndarray): average multiplicative bias for each sliding window, shape: (ncells, nloci, ncopies).
    """
    
    def __init__(self, scf: SingleCellFeature, sex: str = 'male') -> None:
        """ Initialize the SimulatedRepliSeqExperiment object.

        Args:
            scf (SingleCellFeature)
            sex (str, optional): sex of the organism, can be either 'male' or 'female.
                                 Defaults to 'male'.
        """
        
        # Check the input scf:
        # 1. The input scf must be a SingleCellFeature.
        # 2. The input scf must contain the 'spotcount' feature.
        # 3. The input scf must contain the 'cell_states' feature.
        # 4. The 'cell_states' feature must only contain 'G1', 'S' and 'G2'.
        if not isinstance(scf, SingleCellFeature):
            raise TypeError("The input scf must be a SingleCellFeature.")
        if 'spotcount' not in scf.feature_list:
            raise ValueError("The input scf must contain the 'spotcount' feature.")
        if 'cell_states' not in scf:
            raise ValueError("The input scf must contain the 'cell_states' feature.")
        if not all([state in ['G1', 'S', 'G2'] for state in scf.cell_states]):
            raise ValueError("The 'cell_states' feature must only contain 'G1', 'S' and 'G2'.")
        if not isinstance(sex, str):
            raise TypeError(f"Input sex must be str. Got type {type(sex)} instead.")
        if not sex in ['male', 'female']:
            raise ValueError(f"Input sex must be either 'male' or 'female'")
        
        # Get the data from the input scf
        self.index = scf.index
        self.states = scf.cell_states
        # We write the spotcount feature as n_ic, where i is the cell index and c is the locus index
        # It implicitly assumed that the actual quantity is a 3D tensor, where the third dimension is the copy,
        # but we write it like this to match the notation of the mathematical formulas.
        self.n_ic = scf.get_feature('spotcount')  # shape: (ncells, nloci, ncopies)
        self.ncells, self.nloci, self.ncopies = self.n_ic.shape
        self.sex = sex
    
    
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
                keys_to_ignore = ['genome', 'index', 'states', 'n_ic', 'ncells', 'nloci', 'ncopies', 'sex']
                if key in keys_to_ignore:
                    continue
                
                # Save the data
                if isinstance(value, np.ndarray):
                    f.create_dataset(key, data=value)
        

    def run(self) -> None:
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
        """
        self.locus_dependent_run()
        self.cell_dependent_run()
        self.sliding_window_run()
    
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
            if self.sex == 'male':
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
        
        # Calculate the approximate b for G1, S, G2 using the early replicating loci
        b_c_ = np.full(self.ncells, np.nan)
        b_c_[self.states == 'G1'] = n_c['early'][self.states == 'G1'] / (1 - f0_c['early'][self.states == 'G1'])
        b_c_[self.states == 'S'] = n_c['early'][self.states == 'S'] / (2 * (1 - f0_c['early'][self.states == 'S'] ** 0.5))
        b_c_[self.states == 'G2'] = n_c['early'][self.states == 'G2'] / (2 * (1 - f0_c['early'][self.states == 'G2'] ** 0.5))
        
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
        eps_c[self.states == 'S'] = (dS_c / 2) * (1 + np.sqrt(1 - 4 * (f0_c['all'][self.states == 'S'] + dS_c - 1) / dS_c ** 2))
        
        # Calculate the replication probability
        p_c = np.full(self.ncells, np.nan)
        p_c[self.states == 'G1'] = 0
        p_c[self.states == 'G2'] = 1
        p_c[self.states == 'S'] = n_c['all'][self.states == 'S'] / (eps_c[self.states == 'S'] * b_c[self.states == 'S']) - 1
        
        # Store the results
        self.p_c = p_c
        self.eps_c = eps_c
        self.b_c = b_c
        self.b_c_ = b_c_
    
    def sliding_window_run(self) -> None:
        """ Run the sliding window analysis.
        
        It relaxes the assumptions of the locus-dependent and cell-dependent analyses,
        and estimates the replication probability for each locus/cell.
        
        It estimates the following values:
        - p_sw_ic: replication probability for each sliding window of locus/cell.
        - q_sw_ic: quality of the sliding window, True if enough statistical confidence.
        - r_sw_ic: replication state for each sliding window, 1 for non-replicating, 2 for replicating, NaN for not enough confidence.
        - eps_sw_ic: detection efficiency for each sliding window.
        - b_sw_ic: average multiplicative bias for each sliding window.
        """
        
        # First set counts in n_ic larger than a threshold to NaN
        self.n_ic[self.n_ic >= 4] = np.nan
        
        # Take eps_i, eps_c, b_i, b_c and remove some edge cases
        eps_i = self.eps_i
        eps_c = self.eps_c
        eps_i[eps_i < 0] = np.nan
        eps_c[eps_c < 0] = np.nan
        b_i = self.b_i
        b_c = self.b_c
        b_i[b_i < 1] = np.nan
        b_c[b_c < 1] = np.nan
        b_i[b_i > 1.1] = np.nan
        b_c[b_c > 1.1] = np.nan
        
        # Calculate the B and eps matrices, using the formula
        #   b_ic = b_i + b_c - (<b_i> + <b_c>) / 2
        #   eps_ic = eps_i + eps_c - (<eps_i> + <eps_c>) / 2
        b_ic = b_i[np.newaxis, :] + b_c[:, np.newaxis] - (np.nanmean(b_i) + np.nanmean(b_c)) / 2
        b_ic = np.repeat(b_ic[:, :, np.newaxis], self.ncopies, axis=2)
        eps_ic = eps_i[np.newaxis, :] + eps_c[:, np.newaxis] - (np.nanmean(eps_i) + np.nanmean(eps_c)) / 2
        eps_ic = np.repeat(eps_ic[:, :, np.newaxis], self.ncopies, axis=2)
        assert b_ic.shape == self.n_ic.shape, f"b_ic shape: {b_ic.shape} != n_sw_ic shape: {n_sw_ic.shape}"
        assert eps_ic.shape == self.n_ic.shape, f"eps_ic shape: {eps_ic.shape} != n_sw_ic shape: {n_sw_ic.shape}"
        
        # Calculate the sliding window averages
        window = 40
        n_sw_ic = scf_utils.sliding_matrix(self.n_ic, self.index, window=window, method='mean')
        b_sw_ic = scf_utils.sliding_matrix(b_ic, self.index, window=window, method='mean')
        eps_sw_ic = scf_utils.sliding_matrix(eps_ic, self.index, window=window, method='mean')
        
        # Calculate the fraction of zeros in the sliding windows
        n0_ic = np.zeros(self.n_ic.shape, dtype=float)
        n0_ic[self.n_ic == 0] = 1
        n0_ic[np.isnan(self.n_ic)] = np.nan
        f0_sw_ic = scf_utils.sliding_matrix(n0_ic, self.index, window=window, method='mean')
        
        # Create a quality array: it's True for regions with enough statistical confidence:
        # we want that the fraction of zeros is smaller than 0.9 and that the detection efficiency is larger than 0.18
        f0_sw_ok_ic = f0_sw_ic < 0.9
        eps_sw_ok_ic = eps_sw_ic > 0.18
        q_sw_ic = np.logical_and(f0_sw_ok_ic, eps_sw_ok_ic)
        
        # Calculate the replication probability
        p_sw_ic = n_sw_ic / (b_sw_ic * eps_sw_ic) - 1
        
        # Calculate the replication tensor: 1 for non-replicating, 2 for replicating, NaN for not enough confidence
        # We use a thresholding method, where we set the replication state to 1 if p < -0.3 and to 2 if p > 1.8
        r_sw_ic = np.full(self.n_ic.shape, np.nan, dtype=float)
        r_sw_ic[p_sw_ic < -0.3] = 1
        r_sw_ic[p_sw_ic > 1.8] = 2

        # Store the results
        self.p_sw_ic = p_sw_ic
        self.q_sw_ic = q_sw_ic
        self.r_sw_ic = r_sw_ic
        self.eps_sw_ic = eps_sw_ic
        self.b_sw_ic = b_sw_ic
        