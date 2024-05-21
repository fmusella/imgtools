import os
import numpy as np
import h5py
from alabtools.utils import Index
from ..scf import SingleCellFeature
from ..scf import scf_utils


class SimulatedRepliSeqExperiment:
    
    def __init__(self, scf: SingleCellFeature, sex: str = 'male') -> None:
        
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
        self.n_ic = scf.get_feature('spotcount')  # shape: (ncells, nloci, ncopies)
        self.ncells, self.nloci, self.ncopies = self.n_ic.shape
        self.sex = sex
        

    def run(self) -> None:
        
        self._locus_dependent_run()
        
        self._cell_dependent_run()
        
        self._sliding_window_run()

    
    def _locus_dependent_run(self) -> None:
        
        # Calculate the average number of spots and the fraction of zeros per locus
        # for 4 cases: 1) all cells, 2) G1 cells, 3) S cells, 4) G2 cells.
        n_i = {}
        f0_i = {}
        for s in ['all', 'G1', 'S', 'G2']:
            # Create a cell mask for the state
            if s == 'all':
                mask_state = np.ones(self.ncells, dtype=bool)
            else:
                mask_state = self.states == s
            # Calculate the average number of spots for each locus
            n_i[s] = np.mean(self.n_ic[mask_state, :, :], axis=(0, 2))  # shape: (nloci)
            f0_i[s] = np.mean(self.n_ic[mask_state, :, :] == 0, axis=(0, 2))  # shape: (nloci)
            # Fix the values for the X and Y chromosomes if sex is male
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
        self.z_i = z_i
        self.pS_i = pS_i
        self.eps_i = eps_i
        self.b_i = b_i
        self.z_i_ = z_i_
        self.pS_i_ = pS_i_
        self.csi_i_ = csi_i_


    def _cell_dependent_run(self) -> None:
        
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
        delta_c = n_c['all'][self.states == 'S'] / b_c[self.states == 'S']
        eps_c[self.states == 'S'] = (delta_c / 2) * (1 + np.sqrt(1 - 4 * (f0_c['all'][self.states == 'S'] + delta_c - 1) / delta_c ** 2))
        
        # Calculate the replication probability
        p_c = np.full(self.ncells, np.nan)
        p_c[self.states == 'G1'] = 0
        p_c[self.states == 'G2'] = 1
        p_c[self.states == 'S'] = n_c['all'][self.states == 'S'] / (eps_c[self.states == 'S'] * b_c[self.states == 'S']) - 1
        
        # Store the results
        self.b_c = b_c
        self.eps_c = eps_c
        self.p_c = p_c
        self.b_c_ = b_c_
    
    def _sliding_window_run(self) -> None:
        
        # First set counts in n_ic larger than a threshold to NaN
        self.n_ic[self.n_ic >= 4] = np.nan
        
        # Take eps_i, eps_c, b_i, b_c and remove some edge cases
        eps_i = self.eps_i
        eps_c = self.eps_c
        eps_i[eps_i < 0.15] = np.nan
        eps_c[eps_c < 0.15] = np.nan
        b_i = self.b_i
        b_c = self.b_c
        b_i[b_i < 1] = np.nan
        b_c[b_c < 1] = np.nan
        b_i[b_i > 1.07] = np.nan
        b_c[b_c > 1.07] = np.nan
        
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
        
        # Remove regions where the average number of spots is less than 5/window (at least 5 spots per window)
        n_sw_ic[n_sw_ic < 4/window] = np.nan
        
        # Calculate the locus/cell replication probability
        p_ic = n_sw_ic / (b_sw_ic * eps_sw_ic) - 1

        # Store the results
        self.b_ic = b_sw_ic
        self.eps_ic = eps_sw_ic
        self.p_ic = p_ic
        