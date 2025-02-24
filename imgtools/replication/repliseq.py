import os
import numpy as np
import h5py
from alabtools.utils import Index
from ..scf import SingleCellFeature
from ..scf import scf_utils
from ..utils import resample_array
from .GMM_solver import GMM_solve


class SimulatedRepliSeqExperiment:
    """ A class to perform a simulated Repli-Seq experiment.
    
    The simulated Repli-Seq experiment is based on the following model for the spotcounts:
        N = R * E + B,
    where N, R, E and B are random variables:
        - N models the spotcount,
        - R models the replication state (0 or 1),
        - E models the detection state (0, 0.5 or 1),
        - B models the overcounting state (any integer >= 0).
    
    The model is hierarchical, with the following conditional distributions:
        - R is independent.
          It's a Bernoulli variable with probability p.
        - E depends on R.
          It has a Binomial conditional distribution, with a parameter eps as detection efficiency:
            P(E = 0 | R = 1) = 1 - eps,
            P(E = 0.5 | R = 1) = 0,
            P(E = 1 | R = 1) = eps,
            P(E = 0 | R = 2) = (1 - eps) ** 2,
            P(E = 0.5 | R = 2) = 2 * eps * (1 - eps),
            P(E = 1 | R = 2) = eps ** 2.
          Can be written more rigorously as:
            P(E = e | R = r) = Binomial(r  * e ; r, eps).
        - B depends on both R and E.
          It has a Poisson conditional distribution with a parameter beta as bias rate.
          If E = 0 the rate is 0, so B = 0 with probability 1.
          If R = 2 and E = 1, the rate is doubled.
            P(B = b | R = 1, E = 0) = 1 if b = 0, 0 otherwise,
            P(B = b | R = 2, E = 0) = 1 if b = 0, 0 otherwise,
            P(B = b | R = 1, E = 1) = Poisson(b ; beta),
            P(B = b | R = 2, E = 0.5) = Poisson(b ; beta),
            P(B = b | R = 2, E = 1) = Poisson(b ; 2 * beta).
          Can be written more rigorously as:
            P(B = b | R = r, E = e) = Poisson(b ; r * e * beta).
    
    The equations are solved using the Generalized Method of Moments (G-MoM), obtaining the following equations:
        <N> = (1 + p) * eps * beta,
        P(N = 0) = 1 - (1 + p) * eps + p * eps^2.
    Notice that we have replaced (1 + beta) with beta for simplicity.
    
    This class aims to solve the equations, estimating the parameters p, eps and beta in different ways,
    each with a different biological interpretation.
    The solution is done in several steps:
        1. Population-wide analysis.
        2. Feature-dependent analysis.
        3. Locus-dependent analysis.
        4. Locus and feature-dependent analysis.
        5. Cell-dependent analysis.
        6. Cell and feature-dependent analysis.
    By feature run, we mean that we calculate the average p, eps, beta for each quantized interval of a feature,
    for example Speckle distance.
    
    The object can be saved and loaded with an HDF5 file.
    
    ----------
    Attributes:
        h5_name (str): name of the HDF5 file.
        h5 (h5py.File): HDF5 file object.
        
    ----------
    Run-time Attributes:
        The function _load_to_memory() loads the data from the HDF5 file to memory before running the analysis.
        For simplicity, the data is stored in the following attributes:
            - index (Index): index of the HDF5 file.
            - states (np.ndarray): cell states. shape: (ncells,).
            - volumes (np.ndarray): cell volumes. shape: (ncells,).
            - N (np.ndarray): spotcount data. shape: (ncells, nloci, ncopies).
            - featdata (dict): data for each feature:
                - F (np.ndarray): feature data. shape: (ncells, nloci, ncopies).
                - Fq (np.ndarray): quantized feature data. shape: (ncells, nloci, ncopies).
                - quants (np.ndarray): quantiles of the feature data. shape: (nquants).
            - nquants (int): number of quantiles.
            - ncells (int): number of cells.
            - nloci (int): number of loci.
            - ncopies (int): number of copies.
            - G1s (np.ndarray): mask for G1 cells. shape: (ncells,).
            - G2s (np.ndarray): mask for G2 cells. shape: (ncells,).
            - Ss (np.ndarray): mask for S cells. shape: (ncells,).
            - loaded (bool): whether the data has been loaded.
                This boolean is used to avoid loading the data again.
    """
    
    
    # INITIALIZATION METHODS
    
    def __init__(
        self, h5_name: str, mode: str = 'r',
        scf: SingleCellFeature = None, feature_list: list = [], nquants: int = 10
    ) -> None:
        """ Initialize the SimulatedRepliSeqExperiment object.
        
        There are two ways to initialize the object:
            - from a previously saved HDF5 file,
            - from a SingleCellFeature object. A new HDF5 file is created.

        Args:
            h5_name (str)
            mode (str, optional): Mode to open the HDF5 file. Defaults to 'r'.
            scf (SingleCellFeature, optional)
            feature_list (list, optional): list of features to include. Defaults to [].
            nquants (int, optional): number of quantiles to divide the feature data. Defaults to 10.
        """
        
        # Extend the name with its absolute path
        h5_name = os.path.abspath(h5_name)
        
        # Check that file has a valid path
        if not os.path.exists(os.path.dirname(h5_name)):
            raise FileNotFoundError("The path of the HDF5 file does not exist.")
        
        # Store the name of the HDF5 file
        self.h5_name = h5_name
        
        # If the SCF file provided, create the HDF5 file from it
        if scf is not None:
            # Check that the mode is 'w'
            if mode != 'w':
                raise ValueError("The mode must be 'w' when creating the HDF5 file from a SCF.")
            self.h5 = self.from_scf(scf, feature_list, nquants)
            return
        
        # Otherwise, read the HDF5 file
        self.h5 = h5py.File(h5_name, mode=mode)
    
    def from_scf(
        self, scf: SingleCellFeature, feature_list: list = [], nquants: int = 10
    ) -> h5py.File:
        """ Creates a new HDF5 file from a SingleCellFeature object.
        
        The data stored are:
            - the index of the SCF,
            - the cell states,
            - the cell volumes,
            - the spotcount data (N),
            - the feature data (F, Fq, quants).
        
        For the spotcount and feature data, we:
            - curate missing chromosomes, setting whole missing chromosomes to NaN,
            - quantize the feature data. Saved as Fq.

        Args:
            scf (SingleCellFeature)
            feature_list (list, optional): list of features to include. Defaults to [].
            nquants (int, optional): number of quantiles to divide the feature data. Defaults to 10.

        Returns:
            h5py.File: initialized HDF5 file for the SimulatedRepliSeqExperiment.
        """
        
        # Check that the SCF file is consistent
        self._check_scf(scf, feature_list)
        
        # Create the HDF5 file
        h5 = h5py.File(self.h5_name, 'w')
        
        # Save the index
        scf.index.save(h5)
        
        # Save the datasets for states and volumes
        h5.create_dataset('states', data=scf.cell_states.astype('S'))
        h5.create_dataset('volumes', data=scf.volumes)
        
        # Read the spotcount data
        N = scf.get_feature('spotcount')
        # Curate missing chromosomes, setting whole missing chromosomes to NaN
        self._curate_missing_chromosomes(N, scf.index)
        # Save the spotcount data
        h5.create_dataset('N', data=scf.get_feature('spotcount'))
        
        # If the feature list is empty, just exit
        if len(feature_list) == 0:
            return h5
        
        # Otherwise, create the group to store the feature data
        group = h5.create_group('featdata')
        # Loop over the features and add a subgroup for each
        for feat in feature_list:
            subgroup = group.create_group(feat)
            # Read the feature data
            F = scf.get_feature(feat)
            # Curate missing chromosomes
            self._curate_missing_chromosomes(F, scf.index)
            # Quantize the feature data
            Fq, quants = self._quantize_feat(F, nquants)
            # Save the feature data
            subgroup.create_dataset('F', data=F)
            subgroup.create_dataset('Fq', data=Fq)
            subgroup.create_dataset('quants', data=quants)
        
        return h5
    
    @staticmethod
    def _check_scf(scf: SingleCellFeature, feats: list = []) -> None:
        """ Check the input SingleCellFeature object.
        
        It checks that:
         - the input is a SingleCellFeature object,
         - the SCF contains the 'spotcount' feature,
         - the SCF contains the features in the feats list,
         - the SCF contains the 'cell_states' feature,
         - the 'cell_states' feature only contains 'G1', 'S' and 'G2',
         - the SCF contains the 'volumes' feature,

        Args:
            scf (SingleCellFeature)
            feats (list): list of features to include
        """
        
        if not isinstance(scf, SingleCellFeature):
            raise TypeError("The input scf must be a SingleCellFeature.")
        
        if 'spotcount' not in scf.feature_list:
            raise ValueError("The input scf must contain the 'spotcount' feature.")
        for feat in feats:
            if feat not in scf.feature_list:
                raise ValueError(f"The input scf must contain the '{feat}' feature.")
        if 'cell_states' not in scf:
            raise ValueError("The input scf must contain the 'cell_states' dataset.")
        if not all([state in ['G1', 'S', 'G2'] for state in scf.cell_states]):
            raise ValueError("The 'cell_states' feature must only contain 'G1', 'S' and 'G2'.")
        if 'volumes' not in scf:
            raise ValueError("The input scf must contain the 'volumes' dataset.")

    @staticmethod
    def _curate_missing_chromosomes(m: np.ndarray, index: Index) -> None:
        """ Set the entries of a matrix of shape (ncells, nloci, ncopies) to NaN
        for missing chromosomal traces.
        
        Changes the input matrix in place.

        Args:
            m (np.ndarray): matrix of shape (ncells, nloci, ncopies).
        """
        
        # Check the shape of the input matrix, it must be (ncells, nloci, ncopies)
        try:
            ncells, _, ncopies = m.shape
        except ValueError:
            raise ValueError("The input matrix must have shape (ncells, nloci, ncopies).")
        
        # Loop over cells
        for cellnum in range(ncells):
        
            # Loop over the chromosomes and mask them
            for chrom in index.genome.chroms:
                mask_chrom = index.chromstr == chrom  # shape: (nloci)
                
                # Loop over the copies
                for copynum in range(ncopies):
                    
                    # If the matrix of the cell/chrom/copy is made of only 0s, set it as NaN in the object
                    if np.all(m[cellnum, mask_chrom, copynum] == 0):
                        m[cellnum, mask_chrom, copynum] = np.nan

    @staticmethod
    def _quantize_feat(F: np.ndarray, nquants: int) -> np.ndarray:
        """ Quantize the feature values separately for each cell.
        
        Creates a quantized version of the feature data: Fq: (ncells, nloci, ncopies).
        This is an int array, where each value Fq[c, i, h] is the quantized value of F[c, i, h]
        with respect to the other values in the same cell, F[c, :, :].

        Args:
            F (np.ndarray): feature data. shape: (ncells, nloci, ncopies).
            nquants (int): number of quantiles to divide the feature data.

        Returns:
            Fq (np.ndarray): quantized feature data. shape: (ncells, nloci, ncopies).
            quants (np.ndarray): quantiles of the feature data. shape: (nquants).
        """
        
        # Check the shape of the input matrix, it must be (ncells, nloci, ncopies)
        try:
            ncells, _, _ = F.shape
        except ValueError:
            raise ValueError("The input matrix must have shape (ncells, nloci, ncopies).")
        
        # Initialize the quantized feature
        # We initialize with -1: the NaN values in the feature will remain as -1
        Fq = np.full(F.shape, -1, dtype=int)  # shape: (ncells, nloci, ncopies)
        
        # Loop over the cells
        for c in range(ncells):
            
            # Get the feature data for the cell
            F_c = F[c, :, :]  # shape: (nloci, ncopies)
            
            # Initialize the quantized data for the cell
            Fq_c = np.full(F_c.shape, -1, dtype=int)  # shape: (nloci, ncopies)
            
            # Get the quantiles of the cell
            quants_c = np.nanquantile(F_c, np.linspace(0, 1, nquants + 1))  # shape: (nquants + 1)
            
            # Loop over the quantiles
            for q in range(nquants):
                # Get the mask for the quantile
                if q == nquants - 1:
                    mask_q = F_c >= quants_c[q]  # include the last value if it's the last quantile
                else:
                    mask_q = np.logical_and(F_c >= quants_c[q], F_c < quants_c[q + 1])
                # Assign the quantile to the quantized data
                Fq_c[mask_q] = q
            
            # Store the quantized data for the cell
            Fq[c, :, :] = Fq_c
        
        # Get the quantiles as an array
        quants = np.arange(nquants)
        
        return Fq, quants
    

    # RUN METHODS
    
    def run(self, overwrite: bool = False, schedule: list = ['#']) -> None:
        """ Run the simulated Repli-Seq experiment.
        
        Perform the analysis in the following steps:
            1. Population-wide analysis.
            2. Feature-dependent analyses.
            3. Locus-dependent analysis.
            4. Locus and feature-dependent analyses.
            5. Cell-dependent analysis.
            6. Cell and feature-dependent analyses.
        
        If the key 'overwrite' is True, the previous results are deleted,
        otherwise previously-done runs are skipped.
        
        The key 'schedule' is list specifying which runs to perform
        and which to skip ('#' performs all the runs).
        
        The results are stored in the object's HDF5 file.
        
        Args:
            overwrite (bool, optional): whether to overwrite previous results. Defaults to False.
            schedule (list, optional): list of runs to perform. Defaults to ['#'].
        """
        
        # Check the schedule. The accepted runs are:
        accepted_schedule = [
            'population_run',
            'feat_run',
            'locus_run',
            'locus_feat_run',
            'cell_run',
            'cell_feat_run',
        ]
        # If the schedule only contains '#', get all the runs
        if schedule == ['#']:
            schedule = accepted_schedule
        # Check that all runs in the schedule are accepted
        for run in schedule:
            if run not in accepted_schedule:
                raise ValueError(f"The run '{run}' is not accepted.")
        
        # Load the data to memory
        # It's done to avoid reading the file every time we access the data
        # All data are stored in the object's attributes
        self._load_to_memory()
        
        # Population-wide analysis
        if 'population_run' in schedule:
            if 'population_run' not in self.h5 or overwrite:
                self.population_run()
        
        # Feature-dependent analysis
        if 'feat_run' in schedule:
            for feat in self.featdata:
                if 'feat_run' not in self.h5 or feat not in self.h5['feat_run'] or overwrite:
                    self.feat_run(feat)
            
        # Locus-dependent analysis
        if 'locus_run' in schedule:
            if 'locus_run' not in self.h5 or overwrite:
                self.locus_run()
        
        # Locus and feature-dependent analysis
        if 'locus_feat_run' in schedule:
            for feat in self.featdata:
                if 'locus_feat_run' not in self.h5 or feat not in self.h5['locus_feat_run'] or overwrite:
                    self.locus_feat_run(feat)
        
        # Cell-dependent analysis
        if 'cell_run' in schedule:
            if 'cell_run' not in self.h5 or overwrite:
                self.cell_run()
        
        # Cell and feature-dependent analysis
        if 'cell_feat_run' in schedule:
            for feat in self.featdata:
                if 'cell_feat_run' not in self.h5 or feat not in self.h5['cell_feat_run'] or overwrite:
                    self.cell_feat_run(feat)
        
        """self.complete_eps_beta()
        self.sliding_window_run()"""
    
    def _load_to_memory(self):
        """ Load the data from the HDF5 file to memory.
        It's done to avoid reading the file every time we access the data.
        Adds also an attribute 'loaded' so that, if the data is already loaded,
        we don't do it again.
        """
        
        # Check if the data has already been loaded
        if hasattr(self, 'loaded') and self.loaded:
            return
        
        # Add the loaded attribute
        self.loaded = True
        
        # Load the data from the HDF5 file
        self.index = Index(self.h5)
        
        self.states = self.h5['states'][:].astype(str)
        self.G1s = self.states == 'G1'
        self.G2s = self.states == 'G2'
        self.Ss = self.states == 'S'
        
        self.volumes = self.h5['volumes'][:]
        
        self.N = self.h5['N'][:]
        self.ncells, self.nloci, self.ncopies = self.N.shape
        
        # If the h5 file doesn't contain featdata, finish the loading
        if 'featdata' not in self.h5:
            return
        
        # Otherwise, load the feature data
        self.featdata = {}
        for feat in self.h5['featdata']:
            self.featdata[feat] = {
                'F': self.h5['featdata'][feat]['F'][:],
                'Fq': self.h5['featdata'][feat]['Fq'][:],
                'quants': self.h5['featdata'][feat]['quants'][:]
            }
            # The number of quantiles is the same for all features,
            # so we can just re-write it
            self.nquants = len(self.featdata[feat]['quants'])
    
    def population_run(self) -> None:
        """ Run the population-wide analysis.
        Separately for G1, S, G2, it combines the data from all cells and loci to estimate average values.
        In S phase, since there are two equations and three unknowns, we assume that the efficiency is the average
        of G1 and G2.
        Estimates:
            - eps_G1, detection efficiency in G1. float,
            - eps_G1_err, error in eps_G1. float,
            - beta_G1, bias rate in G1. float,
            - beta_G1_err, error in beta_G1. float,
            - eps_G2, detection efficiency in G2. float,
            - eps_G2_err, error in eps_G2. float,
            - beta_G2, bias rate in G2. float,
            - beta_G2_err, error in beta_G2. float,
            - eps_S, detection efficiency in S. float,
            - eps_S_err, error in eps_S. float,
            - beta_S, bias rate in S. float,
            - beta_S_err, error in beta_S. float,
            - p_S, replication probability in S. float,
            - p_S_err, error in p_S. float.
        """
        
        print('POPULATION RUN')
        print('---------------')
        
        # Delete the previous results if they exist
        if 'population_run' in self.h5:
            del self.h5['population_run']
        
        # Ignore the X and Y chromosomes
        mask_XY = np.logical_or(self.index.chromstr == 'chrX', self.index.chromstr == 'chrY') 
        N = self.N[:, ~mask_XY, :]
        
        # Initialize the summary statistics dictionary
        stat = {}
        
        # We calculate average/std for each state
        for s in ['G1', 'S', 'G2']:
            
            # Mask for the state
            mask_state = self.states == s
            N_s = N[mask_state, :, :]
            
            # Create a zero-indicator version of N_s: 1 if N_s = 0, 0 otherwise
            B_s = (N_s == 0).astype(float)
            B_s[np.isnan(N_s)] = np.nan
            
            # Calculate the average number of spots and the fraction of zeros, along with their std and cov
            nsamples = np.sum(~np.isnan(N_s))  # int
            n = np.nanmean(N_s)  # float
            f = np.nanmean(B_s)
            stat[s] = {
                'nsamples': nsamples,  # int
                'n': n,  # float
                'n_var': np.nanvar(N_s, ddof=1) / nsamples,
                'f': f,
                'f_var': np.nanvar(B_s, ddof=1) / nsamples,
                'nf_cov': - n * f / nsamples
            }
        
        # Calculate efficiency and bias in G1 and G2
        eps_G1, beta_G1, eps_G1_err, beta_G1_err = GMM_solve(stat['G1'], p='G1')
        eps_G2, beta_G2, eps_G2_err, beta_G2_err = GMM_solve(stat['G2'], p='G2')
        
        # We assume that the efficiency in S is the average of G1 and G2
        eps_S = (eps_G1 + eps_G2) / 2
        eps_S_err = np.sqrt(eps_G1_err ** 2 + eps_G2_err ** 2) / 2
        
        # Calculate replication probability and bias in S
        p_S, beta_S, p_S_err, beta_S_err = GMM_solve(stat['S'], eps=eps_S, eps_err=eps_S_err)
        
        # Store the results in the h5 file as a group
        # The group is created if it doesn't exist
        group = self.h5.create_group('population_run')        
        group.create_dataset('eps_G1', data=eps_G1)
        group.create_dataset('eps_G1_err', data=eps_G1_err)
        group.create_dataset('beta_G1', data=beta_G1)
        group.create_dataset('beta_G1_err', data=beta_G1_err)
        group.create_dataset('eps_G2', data=eps_G2)
        group.create_dataset('eps_G2_err', data=eps_G2_err)
        group.create_dataset('beta_G2', data=beta_G2)
        group.create_dataset('beta_G2_err', data=beta_G2_err)
        group.create_dataset('eps_S', data=eps_S)
        group.create_dataset('eps_S_err', data=eps_S_err)
        group.create_dataset('beta_S', data=beta_S)
        group.create_dataset('beta_S_err', data=beta_S_err)
        group.create_dataset('p_S', data=p_S)
        group.create_dataset('p_S_err', data=p_S_err)
        
        print('OVER.')
        print('\n\n')
    
    def feat_run(self, feat: str) -> None:
        """ Run the feature-dependent analysis.
        Treats each feature quantile independently, combining the data from all cells and loci.
        
        Estimates:
            - eps_q_G1, detection efficiency in G1. shape: (nquants),
            - eps_q_G1_err, error in eps_q_G1. shape: (nquants),
            - beta_q_G1, bias rate in G1. shape: (nquants),
            - beta_q_G1_err, error in beta_q_G1. shape: (nquants),
            - eps_q_G2, detection efficiency in G2. shape: (nquants),
            - eps_q_G2_err, error in eps_q_G2. shape: (nquants),
            - beta_q_G2, bias rate in G2. shape: (nquants),
            - beta_q_G2_err, error in beta_q_G2. shape: (nquants),
            - eps_q_S, detection efficiency in S. shape: (nquants),
            - eps_q_S_err, error in eps_q_S. shape: (nquants),
            - beta_q_S, bias rate in S. shape: (nquants),
            - beta_q_S_err, error in beta_q_S. shape: (nquants),
            - p_q_S, replication probability in S. shape: (nquants),
            - p_q_S_err, error in p_q_S. shape: (nquants).

        Args:
            feat (str)
        """
        
        print(f'FEAT-DEPENDENT RUN ({feat})')
        print('---------------')
        
        # Delete the previous results if they exist
        if 'feat_run' in self.h5:
            if feat in self.h5['feat_run']:
                del self.h5['feat_run'][feat]
        
        # Ignore the X and Y chromosomes
        mask_XY = np.logical_or(self.index.chromstr == 'chrX', self.index.chromstr == 'chrY') 
        N = self.N[:, ~mask_XY, :]
        Fq = self.featdata[feat]['Fq'][:, ~mask_XY, :]
        
        # Initialize the summary statistics dictionary
        stat = {}
        
        # Loop over the states
        for s in ['G1', 'S', 'G2']:
            
            # Mask for the state
            mask_state = self.states == s
            N_s = N[mask_state, :, :]
            Fq_s = Fq[mask_state, :, :]
            
            # Initialize the arrays to store quantile-dependent averages
            stat[s] = {
                'nsamples': np.zeros(self.nquants),  # shape: (nquants)
                'n': np.zeros(self.nquants),
                'n_var': np.zeros(self.nquants),
                'f': np.zeros(self.nquants),
                'f_var': np.zeros(self.nquants),
                'nf_cov': np.zeros(self.nquants)
            }
            
            # Loop over the quantiles
            for q in self.featdata[feat]['quants']:
                
                # Mask for the quantile
                mask_q = Fq_s == q
                N_s_q = N_s[mask_q]
                
                # Create a zero-indicator version of N_s_q: 1 if N_s = 0, 0 otherwise
                B_s_q = (N_s_q == 0).astype(float)
                B_s_q[np.isnan(N_s_q)] = np.nan
                
                # Calculate average/std for the quantile
                nsamples = np.sum(~np.isnan(N_s_q))  # int
                stat[s]['nsamples'][q] = nsamples
                stat[s]['n'][q] = np.nanmean(N_s_q)
                stat[s]['n_var'][q] = np.nanvar(N_s_q, ddof=1) / nsamples
                stat[s]['f'][q] = np.nanmean(B_s_q)
                stat[s]['f_var'][q] = np.nanvar(B_s_q, ddof=1) / nsamples
                stat[s]['nf_cov'][q] = - stat[s]['n'][q] * stat[s]['f'][q] / nsamples
        
        # Calculate efficiency and bias in G1 and G2
        eps_q_G1, beta_q_G1, eps_q_G1_err, beta_q_G1_err = GMM_solve(stat['G1'], p='G1')
        eps_q_G2, beta_q_G2, eps_q_G2_err, beta_q_G2_err = GMM_solve(stat['G2'], p='G2')
        eps_q_G1 = self.print_n_clip('eps_q_G1', eps_q_G1, 0, 1)
        eps_q_G2 = self.print_n_clip('eps_q_G2', eps_q_G2, 0, 1)
        beta_q_G1 = self.print_n_clip('beta_q_G1', beta_q_G1, 0, None)
        beta_q_G2 = self.print_n_clip('beta_q_G2', beta_q_G2, 0, None)
        
        # We assume that the efficiency in S is the average of G1 and G2
        eps_q_S = (eps_q_G1 + eps_q_G2) / 2
        eps_q_S_err = np.sqrt(eps_q_G1_err ** 2 + eps_q_G2_err ** 2) / 2
        
        # Calculate replication probability and bias in S
        p_q_S, beta_q_S, p_q_S_err, beta_q_S_err = GMM_solve(stat['S'], eps=eps_q_S, eps_err=eps_q_S_err)
        p_q_S = self.print_n_clip('p_q_S', p_q_S, 0, 1)
        beta_q_S = self.print_n_clip('beta_q_S', beta_q_S, 0, None)
        
        # Store the results
        group = self.h5.require_group('feat_run')
        subgroup = group.create_group(feat)
        subgroup.create_dataset('eps_q_G1', data=eps_q_G1)
        subgroup.create_dataset('eps_q_G1_err', data=eps_q_G1_err)
        subgroup.create_dataset('beta_q_G1', data=beta_q_G1)
        subgroup.create_dataset('beta_q_G1_err', data=beta_q_G1_err)
        subgroup.create_dataset('eps_q_G2', data=eps_q_G2)
        subgroup.create_dataset('eps_q_G2_err', data=eps_q_G2_err)
        subgroup.create_dataset('beta_q_G2', data=beta_q_G2)
        subgroup.create_dataset('beta_q_G2_err', data=beta_q_G2_err)
        subgroup.create_dataset('eps_q_S', data=eps_q_S)
        subgroup.create_dataset('eps_q_S_err', data=eps_q_S_err)
        subgroup.create_dataset('beta_q_S', data=beta_q_S)
        subgroup.create_dataset('beta_q_S_err', data=beta_q_S_err)
        subgroup.create_dataset('p_q_S', data=p_q_S)
        subgroup.create_dataset('p_q_S_err', data=p_q_S_err)
        
        print('OVER.')
        print('\n\n')
    
    def locus_run(self) -> None:
        """ Run the locus-dependent analysis.
        Treats each locus independently, assuming that different cells are independent realizations
        of the same locus-dependent proces (separately for G1, S and G2).
        In S phase, since there are two equations and three unknowns, we assume that the efficiency
        signal is the locus-dependent average of G1 and G2.
        Estimates:
            - eps_i_G1, detection efficiency in G1. shape: (nloci),
            - eps_i_G1_err, error in eps_i_G1. shape: (nloci),
            - beta_i_G1, bias rate in G1. shape: (nloci),
            - beta_i_G1_err, error in beta_i_G1. shape: (nloci),
            - eps_i_G2, detection efficiency in G2. shape: (nloci),
            - eps_i_G2_err, error in eps_i_G2. shape: (nloci),
            - beta_i_G2, bias rate in G2. shape: (nloci),
            - beta_i_G2_err, error in beta_i_G2. shape: (nloci),
            - eps_i_S, detection efficiency in S. shape: (nloci),
            - eps_i_S_err, error in eps_i_S. shape: (nloci),
            - beta_i_S, bias rate in S. shape: (nloci),
            - beta_i_S_err, error in beta_i_S. shape: (nloci),
            - p_i_S, replication probability in S. shape: (nloci),
            - p_i_S_err, error in p_i_S. shape: (nloci).
        """
        
        print('LOCUS-DEPENDENT RUN')
        print('-------------------')
        
        # Delete the previous results if they exist
        if 'locus_run' in self.h5:
            del self.h5['locus_run']
        
        # Initialize the summary statistics dictionary
        stat = {}
        
        # Loop over the states
        for s in ['G1', 'S', 'G2']:
            
            # Mask for the state
            mask_state = self.states == s
            N_s = self.N[mask_state, :, :]
            
            # Create a zero-indicator version of N_s: 1 if N_s = 0, 0 otherwise
            B_s = (N_s == 0).astype(float)
            B_s[np.isnan(N_s)] = np.nan
            
            # Calculate average/std for each locus
            nsamples = np.sum(~np.isnan(N_s), axis=(0, 2))  # shape: (nloci)
            n = np.nanmean(N_s, axis=(0, 2))
            f = np.nanmean(B_s, axis=(0, 2))
            stat[s] = {
                'nsamples': nsamples,
                'n': n,
                'n_var': np.nanvar(N_s, ddof=1, axis=(0, 2)) / nsamples,
                'f': f,
                'f_var': np.nanvar(B_s, ddof=1, axis=(0, 2)) / nsamples,
                'nf_cov': - n * f / nsamples
            }
        
        # Calculate efficiency and bias in G1 and G2
        eps_i_G1, beta_i_G1, eps_i_G1_err, beta_i_G1_err = GMM_solve(stat['G1'], p='G1')
        eps_i_G2, beta_i_G2, eps_i_G2_err, beta_i_G2_err = GMM_solve(stat['G2'], p='G2')
        eps_i_G1 = self.print_n_clip('eps_i_G1', eps_i_G1, 0, 1)
        eps_i_G2 = self.print_n_clip('eps_i_G2', eps_i_G2, 0, 1)
        beta_i_G1 = self.print_n_clip('beta_i_G1', beta_i_G1, 0, None)
        beta_i_G2 = self.print_n_clip('beta_i_G2', beta_i_G2, 0, None)

        # Assume that the efficiency in S is the average of G1 and G2
        eps_i_S = (eps_i_G1 + eps_i_G2) / 2
        eps_i_S_err = np.sqrt(eps_i_G1_err ** 2 + eps_i_G2_err ** 2) / 2
        
        # Calculate replication probability and bias in S
        p_i_S, beta_i_S, p_i_S_err, beta_i_S_err = GMM_solve(stat['S'], eps=eps_i_S, eps_err=eps_i_S_err)
        p_i_S = self.print_n_clip('p_i_S', p_i_S, 0, 1)
        beta_i_S = self.print_n_clip('beta_i_S', beta_i_S, 0, None)
        
        # Store the results
        group = self.h5.create_group('locus_run')
        group.create_dataset('eps_i_G1', data=eps_i_G1)
        group.create_dataset('eps_i_G1_err', data=eps_i_G1_err)
        group.create_dataset('beta_i_G1', data=beta_i_G1)
        group.create_dataset('beta_i_G1_err', data=beta_i_G1_err)
        group.create_dataset('eps_i_G2', data=eps_i_G2)
        group.create_dataset('eps_i_G2_err', data=eps_i_G2_err)
        group.create_dataset('beta_i_G2', data=beta_i_G2)
        group.create_dataset('beta_i_G2_err', data=beta_i_G2_err)
        group.create_dataset('eps_i_S', data=eps_i_S)
        group.create_dataset('eps_i_S_err', data=eps_i_S_err)
        group.create_dataset('beta_i_S', data=beta_i_S)
        group.create_dataset('beta_i_S_err', data=beta_i_S_err)
        group.create_dataset('p_i_S', data=p_i_S)
        group.create_dataset('p_i_S_err', data=p_i_S_err)
        
        print('OVER.')
        print('\n\n')
    
    def locus_feat_run(self, feat: str) -> None:
        """ Run the locus and feature-dependent analysis.
        Treats each locus and feature quantile independently, combining the data from all cells.
        
        Estimates:
            - eps_iq_G1, detection efficiency in G1. shape: (nloci, nquants),
            - eps_iq_G1_err, error in eps_iq_G1. shape: (nloci, nquants),
            - beta_iq_G1, bias rate in G1. shape: (nloci, nquants),
            - beta_iq_G1_err, error in beta_iq_G1. shape: (nloci, nquants),
            - eps_iq_G2, detection efficiency in G2. shape: (nloci, nquants),
            - eps_iq_G2_err, error in eps_iq_G2. shape: (nloci, nquants),
            - beta_iq_G2, bias rate in G2. shape: (nloci, nquants),
            - beta_iq_G2_err, error in beta_iq_G2. shape: (nloci, nquants),
            - eps_iq_S, detection efficiency in S. shape: (nloci, nquants),
            - eps_iq_S_err, error in eps_iq_S. shape: (nloci, nquants),
            - beta_iq_S, bias rate in S. shape: (nloci, nquants),
            - beta_iq_S_err, error in beta_iq_S. shape: (nloci, nquants),
            - p_iq_S, replication probability in S. shape: (nloci, nquants),
            - p_iq_S_err, error in p_iq_S. shape: (nloci, nquants).

        Args:
            feat (str)
        """
        
        print(f'LOCUS AND FEAT-DEPENDENT RUN ({feat})')
        print('---------------')
        
        # Delete the previous results if they exist
        if 'locus_feat_run' in self.h5:
            if feat in self.h5['locus_feat_run']:
                del self.h5['locus_feat_run'][feat]
        
        # Initialize the summary statistics dictionary
        stat = {}
        
        # Loop over the states
        for s in ['G1', 'S', 'G2']:
            
            # Create the state mask
            mask_state = self.states == s   
            # Subsample the N and Fq matrices
            N_s = self.N[mask_state, :, :]
            Fq_s = self.featdata[feat]['Fq'][mask_state, :, :]
            
            # Initialize the arrays to store locus-and-quantile-dependent averages
            stat[s] = {
                'nsamples': np.zeros((self.nloci, self.nquants)),  # shape: (nloci, nquants)
                'n': np.zeros((self.nloci, self.nquants)),
                'n_var': np.zeros((self.nloci, self.nquants)),
                'f': np.zeros((self.nloci, self.nquants)),
                'f_var': np.zeros((self.nloci, self.nquants)),
                'nf_cov': np.zeros((self.nloci, self.nquants)),
            }
            
            # Loop over the quantiles
            for q in self.featdata[feat]['quants']:
                
                # Create the quantile mask
                mask_q = Fq_s == q
                
                # To exclude data from other quantiles, we create an array N_s_q
                # that is NaN where the mask_q is False
                N_s_q = np.where(mask_q, N_s, np.nan)
                
                # Create a zero-indicator version of N_s_q: 1 if N_s = 0, 0 otherwise
                B_s_q = (N_s_q == 0).astype(float)
                B_s_q[np.isnan(N_s_q)] = np.nan
                
                # Calculate average/std for each locus and quantile
                nsamples = np.sum(~np.isnan(N_s_q), axis=(0, 2))
                n = np.nanmean(N_s_q, axis=(0, 2))
                f = np.nanmean(B_s_q, axis=(0, 2))
                stat[s]['nsamples'][:, q] = nsamples
                stat[s]['n'][:, q] = n
                stat[s]['n_var'][:, q] = np.nanvar(N_s_q, ddof=1, axis=(0, 2)) / nsamples
                stat[s]['f'][:, q] = f
                stat[s]['f_var'][:, q] = np.nanvar(B_s_q, ddof=1, axis=(0, 2)) / nsamples
                stat[s]['nf_cov'][:, q] = - n * f / nsamples
        
        # Calculate the efficiency in G1 and G2
        eps_iq_G1, beta_iq_G1, eps_iq_G1_err, beta_iq_G1_err = GMM_solve(stat['G1'], p='G1')
        eps_iq_G2, beta_iq_G2, eps_iq_G2_err, beta_iq_G2_err = GMM_solve(stat['G2'], p='G2')
        eps_iq_G1 = self.print_n_clip('eps_iq_G1', eps_iq_G1, 0, 1)
        eps_iq_G2 = self.print_n_clip('eps_iq_G2', eps_iq_G2, 0, 1)
        beta_iq_G1 = self.print_n_clip('beta_iq_G1', beta_iq_G1, 0, None)
        beta_iq_G2 = self.print_n_clip('beta_iq_G2', beta_iq_G2, 0, None)
        
        # Assume that the efficiency in S is the average of G1 and G2
        eps_iq_S = (eps_iq_G1 + eps_iq_G2) / 2
        eps_iq_S_err = np.sqrt(eps_iq_G1_err ** 2 + eps_iq_G2_err ** 2) / 2
        
        # Calculate the probability of replication in S
        p_iq_S, beta_iq_S, p_iq_S_err, beta_iq_S_err = GMM_solve(stat['S'], eps=eps_iq_S, eps_err=eps_iq_S_err)
        p_iq_S = self.print_n_clip('p_iq_S', p_iq_S, 0, 1)
        beta_iq_S = self.print_n_clip('beta_iq_S', beta_iq_S, 0, None)
        
        # Store the results
        group = self.h5.require_group('locus_feat_run')
        subgroup = group.create_group(feat)
        subgroup.create_dataset('eps_iq_G1', data=eps_iq_G1)
        subgroup.create_dataset('eps_iq_G1_err', data=eps_iq_G1_err)
        subgroup.create_dataset('beta_iq_G1', data=beta_iq_G1)
        subgroup.create_dataset('beta_iq_G1_err', data=beta_iq_G1_err)
        subgroup.create_dataset('eps_iq_G2', data=eps_iq_G2)
        subgroup.create_dataset('eps_iq_G2_err', data=eps_iq_G2_err)
        subgroup.create_dataset('beta_iq_G2', data=beta_iq_G2)
        subgroup.create_dataset('beta_iq_G2_err', data=beta_iq_G2_err)
        subgroup.create_dataset('eps_iq_S', data=eps_iq_S)
        subgroup.create_dataset('eps_iq_S_err', data=eps_iq_S_err)
        subgroup.create_dataset('beta_iq_S', data=beta_iq_S)
        subgroup.create_dataset('beta_iq_S_err', data=beta_iq_S_err)
        subgroup.create_dataset('p_iq_S', data=p_iq_S)
        subgroup.create_dataset('p_iq_S_err', data=p_iq_S_err)
        
        print('OVER.')
        print('\n\n')

    def cell_run(self) -> None:
        """ Run the cell-dependent analysis.
        Treats each cell independently, assuming that different loci are independent realizations
        of the same cell-dependent process.
        Assumes a constant efficiency for each cell, taken from the population-wide analysis.
        As for the error, it is estimated from G1/G2 as the RMSD between the cell-dependent
        and the population-wide efficiencies.
        Estimates:
            - eps_c, detection efficiency. shape: (ncells),
            - eps_c_err, error in eps_c. shape: (ncells),
            - beta_c, bias rate. shape: (ncells),
            - beta_c_err, error in beta_c. shape: (ncells),
            - p_c, replication probability. shape: (ncells),
            - p_c_err, error in p_c. shape: (ncells).
        """
        
        print('CELL-DEPENDENT RUN')
        print('------------------')
        
        # Delete the previous results if they exist
        if 'cell_run' in self.h5:
            del self.h5['cell_run']
        
        # Initialize the summary statistics dictionary
        stat = {}
            
        # Ignore the X and Y chromosomes
        mask_XY = np.logical_or(self.index.chromstr == 'chrX', self.index.chromstr == 'chrY') 
        N = self.N[:, ~mask_XY, :]
        
        # Create a zero-indicator version of N: 1 if N = 0, 0 otherwise
        B = (N == 0).astype(float)
        B[np.isnan(N)] = np.nan
            
        # Calculate average/std for each cell
        nsamples = np.sum(~np.isnan(N), axis=(1, 2))  # shape: (ncells,)
        n = np.nanmean(N, axis=(1, 2))
        n_var = np.nanvar(N, ddof=1, axis=(1, 2)) / nsamples
        f = np.nanmean(B, axis=(1, 2))
        f_var = np.nanvar(B, ddof=1, axis=(1, 2)) / nsamples
        nf_cov = - n * f / nsamples
            
        # Save the statistics separately for G1, S, G2 and combined (G1SG2)
        for s in ['G1', 'S', 'G2', 'G1SG2']:
            
            if s == 'G1SG2':
                mask_state = np.ones(self.ncells, dtype=bool)
            else:
                mask_state = self.states == s
            
            stat[s] = {
                'nsamples': nsamples[mask_state],  # shape: (ncells_state,)
                'n': n[mask_state],
                'n_var': n_var[mask_state],
                'f': f[mask_state],
                'f_var': f_var[mask_state],
                'nf_cov': nf_cov[mask_state]
            }
        
        # Calculate the single-cell efficiency in G1 and G2
        eps_c = np.full(self.ncells, np.nan)
        for state in ['G1', 'G2']:
            mask = self.states == state
            eps_c_s, _, _, _ = GMM_solve(stat[state], p=state)
            eps_c[mask] = eps_c_s
        
        # Calculate the RMSD between the G1/G2 efficiency to the population-wide average efficiency
        eps_G1 = self.h5['population_run']['eps_G1'][()]
        eps_G2 = self.h5['population_run']['eps_G2'][()]
        rmsd_G1 = np.sqrt(np.mean((eps_G1 - eps_c[self.G1s]) ** 2))
        rmsd_G2 = np.sqrt(np.mean((eps_G2 - eps_c[self.G2s]) ** 2))
        
        # Use the RMSD as error in the efficiency
        eps_c_err = np.full(self.ncells, np.nan)
        eps_c_err[self.G1s] = rmsd_G1
        eps_c_err[self.G2s] = rmsd_G2
        eps_c_err[self.Ss] = rmsd_G2  # Use the same error for S phase
        print(f'Efficiency: RMSD in G1: {rmsd_G1}, RMSD in G2: {rmsd_G2}')
        
        # Now use the population-wide efficiencies for G1, S, G2
        eps_S = self.h5['population_run']['eps_S'][()]
        eps_c[self.Ss] = eps_S
        eps_c[self.G1s] = eps_G1
        eps_c[self.G2s] = eps_G2
        
        # Calculate the replication probability and bias per cell
        p_c, beta_c, p_c_err, beta_c_err = GMM_solve(stat['G1SG2'], eps=eps_c, eps_err=eps_c_err)
        
        # In S phase, min-max normalize the replication probability such that
        # the minimum is pmin and the maximum is pmax
        print(f'Before normalization: pmin = {np.min(p_c[self.Ss])}, pmax = {np.max(p_c[self.Ss])}')
        pmin, pmax = 0.18, 0.90
        p_c_S = p_c[self.Ss]
        p_c_S = (p_c_S - np.min(p_c_S)) / (np.max(p_c_S) - np.min(p_c_S)) * (pmax - pmin) + pmin
        p_c[self.Ss] = p_c_S
        print(f'After normalization: pmin = {np.min(p_c_S)}, pmax = {np.max(p_c_S)}')
        
        eps_c = self.print_n_clip('eps_c', eps_c, 0, 1)
        beta_c = self.print_n_clip('beta_c', beta_c, 0, None)
        
        # Store the results
        group = self.h5.create_group('cell_run')
        group.create_dataset('eps_c', data=eps_c)
        group.create_dataset('eps_c_err', data=eps_c_err)
        group.create_dataset('beta_c', data=beta_c)
        group.create_dataset('beta_c_err', data=beta_c_err)
        group.create_dataset('p_c', data=p_c)
        group.create_dataset('p_c_err', data=p_c_err)
        
        print('OVER.')
        print('\n\n')
    
    def cell_feat_run(self, feat: str) -> None:
        """ Run the cell and feature-dependent analysis.
        Treats each cell and feature quantile independently, combining the data from all loci.
        For S phase, approximates the replication probability using the results of the cell and feature run:
            p_cq_S = p_c_S * p_q_S / mean(p_q_S).
            
        Estimates:
            - eps_cq, detection efficiency. shape: (ncells, nquants),
            - eps_cq_err, error in eps_cq. shape: (ncells, nquants),
            - beta_cq, bias rate. shape: (ncells, nquants),
            - beta_cq_err, error in beta_cq. shape: (ncells, nquants),
            - p_cq_S, replication probability in S. shape: (ncells, nquants),
            - p_cq_S_err, error in p_cq_S. shape: (ncells, nquants).

        Args:
            feat (str)
        """
        
        print(f'CELL AND FEAT-DEPENDENT RUN ({feat})')
        print('------------------------')
        
        # Delete the previous results if they exist
        if 'cell_feat_run' in self.h5:
            if feat in self.h5['cell_feat_run']:
                del self.h5['cell_feat_run'][feat]
        
        # Initialize the summary statistics dictionary
        stat = {}
        for s in ['G1', 'S', 'G2']:
            ncells_s = np.sum(self.states == s)
            stat[s] = {
                'nsamples': np.zeros((ncells_s, self.nquants)),  # shape: (ncells_s, nquants)
                'n': np.zeros((ncells_s, self.nquants)),
                'n_var': np.zeros((ncells_s, self.nquants)),
                'f': np.zeros((ncells_s, self.nquants)),
                'f_var': np.zeros((ncells_s, self.nquants)),
                'nf_cov': np.zeros((ncells_s, self.nquants)),
            }
        
        # Remove the X and Y chromosomes
        mask_XY = np.logical_or(self.index.chromstr == 'chrX', self.index.chromstr == 'chrY')
        N = self.N[:, ~mask_XY, :]
        Fq = self.featdata[feat]['Fq'][:, ~mask_XY, :]
        
        # Loop over the feature quantiles
        for q in self.featdata[feat]['quants']:
            
            # Create the quantile mask
            mask_q = Fq == q
            
            # To exclude data from other quantiles, we create an array N_q
            # that is NaN where the mask_q is False
            N_q = np.where(mask_q, N, np.nan)
            
            # Create a zero-indicator version of N_q: 1 if N_q = 0, 0 otherwise
            B_q = (N_q == 0).astype(float)
            B_q[np.isnan(N_q)] = np.nan
            
            # Calculate average/std for each cell
            nsamples = np.sum(~np.isnan(N_q), axis=(1, 2))  # shape: (ncells,)
            n = np.nanmean(N_q, axis=(1, 2))
            n_var = np.nanvar(N_q, ddof=1, axis=(1, 2)) / nsamples
            f = np.nanmean(B_q, axis=(1, 2))
            f_var = np.nanvar(B_q, ddof=1, axis=(1, 2)) / nsamples
            nf_cov = - n * f / nsamples
            
            # Save the statistics separately for G1, S, G2
            for s in ['G1', 'S', 'G2']:
                mask_state = self.states == s
                stat[s]['nsamples'][:, q] = nsamples[mask_state]
                stat[s]['n'][:, q] = n[mask_state]
                stat[s]['n_var'][:, q] = n_var[mask_state]
                stat[s]['f'][:, q] = f[mask_state]
                stat[s]['f_var'][:, q] = f_var[mask_state]
                stat[s]['nf_cov'][:, q] = nf_cov[mask_state]

        # Calculate efficiency and bias for G1 and G2
        eps_cq_G1, beta_cq_G1, eps_cq_G1_err, beta_cq_G1_err = GMM_solve(stat['G1'], p='G1')
        eps_cq_G2, beta_cq_G2, eps_cq_G2_err, beta_cq_G2_err = GMM_solve(stat['G2'], p='G2')
        # Create arrays for all cells and fill them
        eps_cq = np.full((self.ncells, self.nquants), np.nan)  # shape: (ncells, nquants)
        eps_cq[self.G1s, :] = eps_cq_G1
        eps_cq[self.G2s, :] = eps_cq_G2
        eps_cq_err = np.full((self.ncells, self.nquants), np.nan)  # shape: (ncells, nquants)
        eps_cq_err[self.G1s, :] = eps_cq_G1_err
        eps_cq_err[self.G2s, :] = eps_cq_G2_err
        beta_cq = np.full((self.ncells, self.nquants), np.nan)  # shape: (ncells, nquants)
        beta_cq[self.G1s, :] = beta_cq_G1
        beta_cq[self.G2s, :] = beta_cq_G2
        beta_cq_err = np.full((self.ncells, self.nquants), np.nan)  # shape: (ncells, nquants)
        beta_cq_err[self.G1s, :] = beta_cq_G1_err
        beta_cq_err[self.G2s, :] = beta_cq_G2_err
        
        # We approximate the replication probability using our previous results,
        # in particular the cell run and the feature run.
        # We start from the p_c values, and we tile them
        p_c_S = self.h5['cell_run']['p_c'][self.Ss]
        p_c_S_err = self.h5['cell_run']['p_c_err'][self.Ss]
        p_c_S = np.tile(p_c_S[:, np.newaxis], (1, self.nquants))  # shape: (ncells_S, nquants)
        p_c_S_err = np.tile(p_c_S_err[:, np.newaxis], (1, self.nquants))
        # Then we calculate the rescaling factors for each quantile from p_q_S,
        # i.e. the ratio between each p_q value and their average
        # (we ignore the error from the feat run, since it's much smaller than the one from the cell run)
        p_q_S = self.h5['feat_run'][feat]['p_q_S'][:]
        x_q_S = p_q_S / np.nanmean(p_q_S)
        x_q_S = np.tile(x_q_S[np.newaxis, :], (np.sum(self.Ss), 1))  # shape: (ncells_S, nquants)
        # Finally, we define the cell-and-quantile dependent replication probability as the product of the two
        p_cq_S = p_c_S * x_q_S
        p_cq_S_err = p_c_S_err * x_q_S
        p_cq_S = self.print_n_clip('p_cq_S', p_cq_S, 0, 1)
        # Create a full p_cq matrix to store the results
        p_cq = np.full((self.ncells, self.nquants), np.nan)  # shape: (ncells, nquants)
        p_cq[self.G1s, :] = 0
        p_cq[self.G2s, :] = 1
        p_cq[self.Ss, :] = p_cq_S
        p_cq_err = np.full((self.ncells, self.nquants), np.nan)  # shape: (ncells, nquants)
        p_cq_err[self.Ss, :] = p_cq_S_err
        
        # We then calculate the efficiency and bias for S
        eps_cq_S, beta_cq_S, eps_cq_S_err, beta_cq_S_err = GMM_solve(stat['S'], p=p_cq_S, p_err=p_cq_S_err)
        eps_cq[self.Ss, :] = eps_cq_S
        beta_cq[self.Ss, :] = beta_cq_S
        eps_cq_err[self.Ss, :] = eps_cq_S_err
        beta_cq_err[self.Ss, :] = beta_cq_S_err
        eps_cq = self.print_n_clip('eps_cq', eps_cq, 0, 1)
        beta_cq = self.print_n_clip('beta_cq', beta_cq, 0, None)
        
        # Store the results
        group = self.h5.require_group('cell_feat_run')
        subgroup = group.create_group(feat)
        subgroup.create_dataset('eps_cq', data=eps_cq)
        subgroup.create_dataset('eps_cq_err', data=eps_cq_err)
        subgroup.create_dataset('beta_cq', data=beta_cq)
        subgroup.create_dataset('beta_cq_err', data=beta_cq_err)
        subgroup.create_dataset('p_cq', data=p_cq)
        subgroup.create_dataset('p_cq_err', data=p_cq_err)
        
        print('OVER.')
        print('\n\n')
    
    def complete_eps_beta(self) -> None:
        """ TODO: fix with new data structure. """
        
        print('COMPLETE EPS AND BETA')
        print('------------------')
        
        # Create two eps tensors of shapes (ncells, nloci, zquants, ncopies) and (ncells, nloci, radquants, ncopies)
        eps_icz = np.zeros((self.ncells, self.nloci, len(self.zquants), self.ncopies), dtype=float)
        eps_icd = np.zeros((self.ncells, self.nloci, len(self.radquants), self.ncopies), dtype=float)
        for cellnum, state in enumerate(self.states):
            for copynum in range(self.ncopies):
                if state == 'G1':
                    eps_icz[cellnum, :, :, copynum] = self.eps_iz_G1
                    eps_icd[cellnum, :, :, copynum] = self.eps_id_G1
                elif state == 'S':
                    eps_icz[cellnum, :, :, copynum] = self.eps_iz_S
                    eps_icd[cellnum, :, :, copynum] = self.eps_id_S
                elif state == 'G2':
                    eps_icz[cellnum, :, :, copynum] = self.eps_iz_G2
                    eps_icd[cellnum, :, :, copynum] = self.eps_id_G2

        # Create two beta tensors of shapes (ncells, nloci, zquants, ncopies) and (ncells, nloci, radquants, ncopies)
        beta_icz = np.tile(self.beta_cz[:, np.newaxis, :, np.newaxis], (1, self.nloci, 1, self.ncopies))
        beta_icd = np.tile(self.beta_cd[:, np.newaxis, :, np.newaxis], (1, self.nloci, 1, self.ncopies))

        # Create a locus, cell, copy dependent tensors by selecting the actual z and rad quantiles
        eps_ic = np.full((self.ncells, self.nloci, self.ncopies), np.nan)
        beta_ic = np.full((self.ncells, self.nloci, self.ncopies), np.nan)
        for z in self.zquants:
            for d in self.radquants:
                mask_zd = np.logical_and(self.zq_ic == z, self.radq_ic == d)
                eps_ic[mask_zd] = (eps_icz[:, :, z, :][mask_zd] + eps_icd[:, :, d, :][mask_zd]) / 2
                beta_ic[mask_zd] = (beta_icz[:, :, z, :][mask_zd] + beta_icd[:, :, d, :][mask_zd]) / 2
        # Loop over cells and z to correct eps_ic and beta_ic
        for cellnum in range(self.ncells):
            for copynum in range(self.ncopies):
                for z in self.zquants:
                    for d in self.radquants:
                        # Get the efficiency and bias for the current cell, locus, copy and rad
                        mask_d = self.radq_ic[cellnum, :, copynum] == d
                        eps_ic_ = eps_ic[cellnum, :, copynum][mask_d]
                        beta_ic_ = beta_ic[cellnum, :, copynum][mask_d]
                        # Rescale the efficiency and bias by cell_n_d estimates
                        eps_ic_ = eps_ic_ * self.eps_cd[cellnum, d] / np.nanmean(eps_ic_)
                        beta_ic_ = beta_ic_ * self.beta_cd[cellnum, d] / np.nanmean(beta_ic_)
                        # Assign the corrected values
                        eps_ic[cellnum, :, copynum][mask_d] = eps_ic_
                        beta_ic[cellnum, :, copynum][mask_d] = beta_ic_
                    # Get the efficiency and bias for the current cell, locus, copy and z
                    mask_z = self.zq_ic[cellnum, :, copynum] == z
                    eps_ic_ = eps_ic[cellnum, :, copynum][mask_z]
                    beta_ic_ = beta_ic[cellnum, :, copynum][mask_z]
                    # Rescale the efficiency and bias by cell_n_z estimates
                    eps_ic_ = eps_ic_ * self.eps_cz[cellnum, z] / np.nanmean(eps_ic_)
                    beta_ic_ = beta_ic_ * self.beta_cz[cellnum, z] / np.nanmean(beta_ic_)
                    # Assign the corrected values
                    eps_ic[cellnum, :, copynum][mask_z] = eps_ic_
                    beta_ic[cellnum, :, copynum][mask_z] = beta_ic_
            # Correct the efficiency and bias by cell estimates
            eps_ic[cellnum, :, :] = eps_ic[cellnum, :, :] * self.eps_c[cellnum] / np.nanmean(eps_ic[cellnum, :, :])
            beta_ic[cellnum, :, :] = beta_ic[cellnum, :, :] * self.beta_c[cellnum] / np.nanmean(beta_ic[cellnum, :, :])
        eps_ic = self.print_n_clip('eps_ic', eps_ic, 0, 1)
        beta_ic = self.print_n_clip('beta_ic', beta_ic, 0, None)
        
        # Store the results
        self.eps_ic = eps_ic
        self.beta_ic = beta_ic
        
        print('OVER.')
        print('\n\n')
    
    def sliding_window_run(self) -> None:
        """ TODO: fix with new data structure."""
        
        print('SLIDING WINDOW RUN')
        print('------------------')
        
        # Get the window size in units of loci
        window = int(np.ceil(self.config['sliding_window_size'] / self.index.resolution()))
        
        # Calculate the sliding window averages
        n_ic_SW = scf_utils.sliding_matrix(self.n_ic, self.index, window=window, method='mean')
        eps_ic_SW = scf_utils.sliding_matrix(self.eps_ic, self.index, window=window, method='mean')
        beta_ic_SW = scf_utils.sliding_matrix(self.beta_ic, self.index, window=window, method='mean')
        
        # Calculate the replication probability
        p_ic_SW = n_ic_SW / (eps_ic_SW * beta_ic_SW) - 1

        # Store the results
        self.eps_ic_SW = eps_ic_SW
        self.beta_ic_SW = beta_ic_SW
        self.p_ic_SW = p_ic_SW
        
        print('OVER.')
        print('\n\n')        


    def feat_loci_repliprob(
        self, loci: np.ndarray, feat: str = 'z', S_stage: float = 0.5, quantiles: dict = None
    ) -> dict:
        
        # Load the data from the HDF5 file into memory
        self._load_to_memory()
        
        # If the quantiles are not provided, we use the default ones
        if quantiles is None:
            quantiles = {q: np.array([q]) for q in self.featdata[feat]['quants']}
        nquants = len(quantiles)
        
        # Initialize the summary statistics dictionary
        stat = {}
        for s in ['G1', 'S', 'G2']:
            stat[s] = {
                    'nsamples': np.zeros(nquants),  # shape: (nquants)
                    'n': np.zeros(nquants),
                    'n_var': np.zeros(nquants),
                    'f': np.zeros(nquants),
                    'f_var': np.zeros(nquants),
                    'nf_cov': np.zeros(nquants)
                }
        
        # Loop over the states
        for s in ['G1', 'S', 'G2']:
            
            # Mask for the state
            mask_state = self.states == s
            
            # If the state is S, we need to subsample only those cells that are in the required stage
            if s == 'S':
                
                # Get the cell progression probabilities from the cell run
                p_c = self.h5['cell_run']['p_c'][:]
                
                # Get the cells with replication probability less than S_stage
                mask_p_c = np.logical_and(p_c > 0, p_c <= S_stage)
                mask_state = np.logical_and(mask_state, mask_p_c)
            
            # Subsample the N and Fq matrices
            N_s = self.N[mask_state, :, :][:, loci, :]
            Fq_s = self.featdata[feat]['Fq'][mask_state, :, :][:, loci, :]
            
            # Remove the bottom and top 2 z quantiles from the analysis
            zq_s = self.featdata['z']['Fq'][mask_state, :, :][:, loci, :]
            mask_z = np.logical_and(zq_s > 2, zq_s < self.nquants - 2)
            N_s[~mask_z] = np.nan
            Fq_s[~mask_z] = -1
            
            # Remove the first envsurf quantile from the analysis
            envq_s = self.featdata['envsurf_imputed']['Fq'][mask_state, :, :][:, loci, :]
            mask_env = envq_s > 0
            N_s[~mask_env] = np.nan
            Fq_s[~mask_env] = -1
            
            # Loop over the quantiles
            for q in quantiles:
                
                # Create the quantile mask
                mask_q = np.full(Fq_s.shape, False)
                for qq in quantiles[q]:
                    mask_q = np.logical_or(mask_q, Fq_s == qq)
                
                # Subsample the N matrix
                N_s_q = N_s[mask_q]
                
                # Create a zero-indicator version of N_s_q: 1 if N_s = 0, 0 otherwise
                B_s_q = (N_s_q == 0).astype(float)
                B_s_q[np.isnan(N_s_q)] = np.nan
                
                # Calculate average/std for the quantile
                nsamples = np.sum(~np.isnan(N_s_q))  # int
                n = np.nanmean(N_s_q)
                f = np.nanmean(B_s_q)
                stat[s]['nsamples'][q] = nsamples
                stat[s]['n'][q] = n
                stat[s]['n_var'][q] = np.nanvar(N_s_q, ddof=1) / nsamples
                stat[s]['f'][q] = f
                stat[s]['f_var'][q] = np.nanvar(B_s_q, ddof=1) / nsamples
                stat[s]['nf_cov'][q] = - n * f / nsamples

        # Calculate efficiency and bias in G1 and G2
        eps_q_G1, beta_q_G1, eps_q_G1_err, beta_q_G1_err = GMM_solve(stat['G1'], p='G1')
        eps_q_G2, beta_q_G2, eps_q_G2_err, beta_q_G2_err = GMM_solve(stat['G2'], p='G2')
        eps_q_G1 = self.print_n_clip('eps_q_G1', eps_q_G1, 0, 1)
        eps_q_G2 = self.print_n_clip('eps_q_G2', eps_q_G2, 0, 1)
        beta_q_G1 = self.print_n_clip('beta_q_G1', beta_q_G1, 0, None)
        beta_q_G2 = self.print_n_clip('beta_q_G2', beta_q_G2, 0, None)
        
        # We assume that the efficiency and bias in S are the average of G1 and G2
        eps_q_S = (eps_q_G1 + eps_q_G2) / 2
        beta_q_S = (beta_q_G1 + beta_q_G2) / 2
        eps_q_S_err = np.sqrt(eps_q_G1_err**2 + eps_q_G2_err**2) / 2
        beta_q_S_err = np.sqrt(beta_q_G1_err**2 + beta_q_G2_err**2) / 2
        
        # Calculate replication probability in S
        # p_q_S, beta_q_S, p_q_S_err, beta_q_S_err = GMM_solve(stat['S'], eps=eps_q_S, eps_err=eps_q_S_err)
        p_q_S, p_q_S_err = GMM_solve(stat['S'], eps=eps_q_S, beta=beta_q_S, eps_err=eps_q_S_err, beta_err=beta_q_S_err)
        p_q_S = self.print_n_clip('p_q_S', p_q_S, 0, 1)
        beta_q_S = self.print_n_clip('beta_q_S', beta_q_S, 0, None)
        
        # Now we re-calculate the efficiency using the replication probability
        # to weigh between the G1 and G2 estimates
        p_q_S_avg = np.nanmean(p_q_S)
        eps_q_S = (eps_q_G1 * (1 - p_q_S_avg) + eps_q_G2 * p_q_S_avg)
        beta_q_S = (beta_q_G1 * (1 - p_q_S_avg) + beta_q_G2 * p_q_S_avg)
        eps_q_S_err = np.sqrt(eps_q_G1_err**2 * (1 - p_q_S_avg)**2 + eps_q_G2_err**2 * p_q_S_avg**2)
        beta_q_S_err = np.sqrt(beta_q_G1_err**2 * (1 - p_q_S_avg)**2 + beta_q_G2_err**2 * p_q_S_avg**2)
        
        # And we re-calculate the replication probability using the new efficiency
        # p_q_S, beta_q_S, p_q_S_err, beta_q_S_err = GMM_solve(stat['S'], eps=eps_q_S, eps_err=eps_q_S_err)
        p_q_S, p_q_S_err = GMM_solve(stat['S'], eps=eps_q_S, beta=beta_q_S, eps_err=eps_q_S_err, beta_err=beta_q_S_err)
        p_q_S = self.print_n_clip('p_q_S', p_q_S, 0, 1)
        beta_q_S = self.print_n_clip('beta_q_S', beta_q_S, 0, None)
        
        # Return the results
        results = {
            'eps_q_G1': eps_q_G1, 'beta_q_G1': beta_q_G1, 'eps_q_G1_err': eps_q_G1_err, 'beta_q_G1_err': beta_q_G1_err,
            'eps_q_G2': eps_q_G2, 'beta_q_G2': beta_q_G2, 'eps_q_G2_err': eps_q_G2_err, 'beta_q_G2_err': beta_q_G2_err,
            'eps_q_S': eps_q_S, 'beta_q_S': beta_q_S, 'eps_q_S_err': eps_q_S_err, 'beta_q_S_err': beta_q_S_err,
            'p_q_S': p_q_S, 'p_q_S_err': p_q_S_err
            
        }
        return results
        

    def simple_repliprob(self, mask: np.ndarray, feat: str = 'z') -> float:
        
        # Load the data from the HDF5 file into memory
        self._load_to_memory()
        
        # Get the target cells, i.e. those with at least one locus present in the mask
        tcells = np.where(np.sum(mask, axis=(1, 2)) > 0)[0]  # shape: (ntcells), dtype: int
        
        # Make sure that the target cells are all S
        if not np.all(self.states[tcells] == 'S'):
            raise ValueError('The target cells must be all in S phase.')
        
        # Check that the feature for the correction is valid
        if feat not in self.featdata.keys():
            raise ValueError('The feature for the correction is not valid.')
        
        # First we need to calculate the average number of spots, zq, radq,
        # eps_c_S, beta_c_S and p_c_S for the target S cells
        N = self.N[mask]
        n = np.nanmean(N)
        f = np.sum(N == 0) / np.sum(~np.isnan(N))
        
        # Get the quantile for the chosen feature
        q = np.nanmean(self.featdata[feat]['Fq'][mask])
        q = int(np.round(q))
        
        # Get the beta from the cell_feat_run for the chosen feature
        beta_cq = self.h5['cell_feat_run'][feat]['beta_cq'][:]
        
        # Get the beta
        beta = np.nanmean(beta_cq[tcells, q])
        
        # Get eps and repliprob
        p, eps = GMM_solve(n, f, beta=beta)
        
        return p
        

    def calculate_repliprob(self, mask: np.ndarray, nrepeat: int = 1, feat: str = 'z') -> list:
        """
        Calculates the replication probability for a given mask.
        
        mask is a boolean numpy array of shape (ncells, ndomains, ncopies),
        indicating for which loci in which cells we have to calculate the
        replication probability.
        
        The function does the following:
            - makes sure that the mask only contains True values for S cells,
            - estimates eps and beta from G1 and G2 by bootstrapping,
            - corrects eps and beta with cell and feature-dependent estimates,
            - calculates the replication probability for the original mask,
            - the process can be repeated multiple times to get a more robust estimate.
        
        The input feature specifies which feature to use for the correction.
            
        Returns a list of length nrepeat containing the replication probabilities
        for each repetition.

        Args:
            mask (np.ndarray): A boolean numpy array of shape (ncells, ndomains, ncopies).
            nrepeat (int): The number of times the process is repeated.

        Returns:
            list: A list of length nrepeat containing the replication probabilities.
        """
        
        # NOTE: for now I am calculating the quantiles of each feature,
        # even though we only need one feature. I am doing it because we might need it later.
        # However, if at the end I see that we don't need it, I can change the code
        # to calculate the quantiles only for the chosen feature.
        
        # Load the data from the HDF5 file into memory
        self._load_to_memory()
        
        # Get the target cells, i.e. those with at least one locus present in the mask
        tcells = np.where(np.sum(mask, axis=(1, 2)) > 0)[0]  # shape: (ntcells), dtype: int
        
        # Make sure that the target cells are all S
        if not np.all(self.states[tcells] == 'S'):
            raise ValueError('The target cells must be all in S phase.')
        
        # Check that the feature for the correction is valid
        if feat not in self.featdata.keys():
            raise ValueError('The feature for the correction is not valid.')

        # Now we estimate eps and beta in G1 and G2 by bootstrapping
        # We repeat this process nrepeat times to get a more robust estimate
        G1G2_results = {
            'tcells_G1': [],
            'tcells_G2': [],
            'eps_G1': [],
            'eps_G2': [],
            'beta_G1': [],
            'beta_G2': [],
            'fq_G1': {feat: [] for feat in self.featdata.keys()},
            'fq_G2': {feat: [] for feat in self.featdata.keys()}
        }
        for r in range(nrepeat):
            G1G2_results = self.bootstrap_G1G2(tcells, mask, G1G2_results)
        
        # Calculate the replication probability for the target S cells
        # using the estimates from G1 and G2
        
        # First we need to calculate the average number of spots, zq, radq,
        # eps_c_S, beta_c_S and p_c_S for the target S cells
        N_S = self.N[mask]
        n_S = np.nanmean(N_S)
        f_S = np.sum(N_S == 0) / np.sum(~np.isnan(N_S))
        
        # Calculate the average feature quantiles
        fq_S = {}
        for feat in self.featdata.keys():
            fq_S[feat] = np.nanmean(self.featdata[feat]['Fq'][mask])
            # Round to the nearest integer
            fq_S[feat] = int(np.round(fq_S[feat]))
        
        # Initialize the list of inferred replication probabilities
        p_Ss = []
        
        # Loop over the repetitions
        for r in range(nrepeat):
            
            # Get the G1G2 randomization results for the current repetition
            tcells_G1 = G1G2_results['tcells_G1'][r]
            tcells_G2 = G1G2_results['tcells_G2'][r]
            eps_G1 = G1G2_results['eps_G1'][r]
            eps_G2 = G1G2_results['eps_G2'][r]
            beta_G1 = G1G2_results['beta_G1'][r]
            beta_G2 = G1G2_results['beta_G2'][r]
            
            # Get the feature quantiles in S, G1 and G2 (for the current repetition)
            q_S = fq_S[feat]
            q_G1 = G1G2_results['fq_G1'][feat][r]
            q_G2 = G1G2_results['fq_G2'][feat][r]
            # Get eps and beta of the cell_feat_run for the chosen feature
            eps_cq = self.h5['cell_feat_run'][feat]['eps_cq'][:]
            beta_cq = self.h5['cell_feat_run'][feat]['beta_cq'][:]
            # Perform the correction of eps_G1, eps_G2, beta_G1 and beta_G2
            eps_G1 = eps_G1 + np.nanmean(eps_cq[tcells, q_S]) - np.nanmean(eps_cq[tcells_G1, q_G1])
            eps_G2 = eps_G2 + np.nanmean(eps_cq[tcells, q_S]) - np.nanmean(eps_cq[tcells_G2, q_G2])
            beta_G1 = beta_G1 + np.nanmean(beta_cq[tcells, q_S]) - np.nanmean(beta_cq[tcells_G1, q_G1])
            beta_G2 = beta_G2 + np.nanmean(beta_cq[tcells, q_S]) - np.nanmean(beta_cq[tcells_G2, q_G2])
            
            # Assign eps_S and beta_S as the average of G1 and G2
            eps_S = (eps_G1 + eps_G2) / 2
            beta_S = (beta_G1 + beta_G2) / 2
            
            # Calculate the replication probability in S
            p_S = GMM_solve(n_S, f_S, eps=eps_S, beta=beta_S)
            p_Ss.append(p_S)
        
        return p_Ss
    
    def bootstrap_G1G2(self, tcells: np.ndarray, mask: np.ndarray, G1G2_results: dict) -> tuple:
        """ Estimate efficiency and bias in G1/G2 given the current S-phase mask by
        randomly bootstrapping G1/G2 cells with the same loci distribution as the S cells.
        
        This is achieved by randomly mapping S cells to G1/G2: c -> cG1, c -> cG2.
        Then we take the mask from the S cell, mask[c, :, :], and apply it to cG1 and cG2.
        This is done for each S cell.

        Args:
            tcells (np.ndarray): An array containing the indices of the target cells.
            mask (np.ndarray): A boolean numpy array of shape (ncells, ndomains, ncopies).
            G1G2_results (dict): A dictionary containing the results of previous randomizations. Contains:
                - tcells_G1 (list): A list of lists containing the indices of the target G1 cells.
                - tcells_G2 (list): A list of lists containing the indices of the target G2 cells.
                - eps_G1 (list): A list of the estimated efficiencies for G1.
                - eps_G2 (list): A list of the estimated efficiencies for G2.
                - beta_G1 (list): A list of the estimated biases for G1.
                - beta_G2 (list): A list of the estimated biases for G2.
                - fq_G1 (dict): A dictionary containing average feature quantiles in G1.
                - fq_G2 (dict): A dictionary containing average feature quantiles in G2.

        Returns:
            G1G2_results (dict): The updated dictionary containing the results of the current randomization.
        """
        
        # Randomly select target cells from G1 and G2,
        # resampling them to have the same number of cells as S
        tcells_G1 = resample_array(len(tcells), np.where(self.G1s)[0])
        tcells_G2 = resample_array(len(tcells), np.where(self.G2s)[0])
        
        # Randomly map these cells to the target cells
        map_G1 = {cS: cG1 for cS, cG1 in zip(tcells, tcells_G1)}
        map_G2 = {cS: cG2 for cS, cG2 in zip(tcells, tcells_G2)}
        
        # Initialize the lists to store the G1, G2 randomized data
        N_G1, N_G2 = np.array([]), np.array([])
        Fq_G1 = {feat: np.array([]) for feat in self.featdata.keys()}
        Fq_G2 = {feat: np.array([]) for feat in self.featdata.keys()}
        
        # Loop over the target cells
        for c in tcells:
            cG1, cG2 = map_G1[c], map_G2[c]
            
            # Get the mask to apply from the S cell
            mask_c = mask[c, :, :]  # shape: (ndomains, ncopies)
            
            # Apply the mask to the N and Fq matrices for cG1 and cG2
            N_G1 = np.concatenate((N_G1, self.N[cG1, mask_c]))
            N_G2 = np.concatenate((N_G2, self.N[cG2, mask_c]))
            for feat in self.featdata.keys():
                Fq_G1[feat] = np.concatenate(
                    (Fq_G1[feat], self.featdata[feat]['Fq'][cG1, mask_c])
                )
                Fq_G2[feat] = np.concatenate(
                    (Fq_G2[feat], self.featdata[feat]['Fq'][cG2, mask_c])
                )
        
        # Calculate the average number of spots and the fraction of zeros
        n_G1, n_G2 = np.nanmean(N_G1), np.nanmean(N_G2)
        f_G1 = np.sum(N_G1 == 0) / np.sum(~np.isnan(N_G1))
        f_G2 = np.sum(N_G2 == 0) / np.sum(~np.isnan(N_G2))
        
        # Calculate the average feature quantiles
        fq_G1 = {feat: np.nanmean(Fq_G1[feat]) for feat in Fq_G1.keys()}
        fq_G2 = {feat: np.nanmean(Fq_G2[feat]) for feat in Fq_G2.keys()}
        # Round to the nearest integer
        for feat in self.featdata.keys():
            fq_G1[feat] = int(np.round(fq_G1[feat]))
            fq_G2[feat] = int(np.round(fq_G2[feat]))
        
        # Calculate the efficiency and bias
        eps_G1, beta_G1 = GMM_solve(n_G1, f_G1, p='G1')
        eps_G2, beta_G2 = GMM_solve(n_G2, f_G2, p='G2')
        eps_G1 = np.clip(eps_G1, 0, 1)
        eps_G2 = np.clip(eps_G2, 0, 1)
        beta_G1 = np.clip(beta_G1, 0, None)
        beta_G2 = np.clip(beta_G2, 0, None)
        
        # Append the results to the dictionary
        G1G2_results['tcells_G1'].append(tcells_G1)
        G1G2_results['tcells_G2'].append(tcells_G2)
        G1G2_results['eps_G1'].append(eps_G1)
        G1G2_results['eps_G2'].append(eps_G2)
        G1G2_results['beta_G1'].append(beta_G1)
        G1G2_results['beta_G2'].append(beta_G2)
        for feat in self.featdata.keys():
            G1G2_results['fq_G1'][feat].append(fq_G1[feat])
            G1G2_results['fq_G2'][feat].append(fq_G2[feat])
        
        return G1G2_results
    
    
    def sliding_window_run_old(self) -> None:
        """ Run the sliding window analysis.
        Treats each cell, locus independently.
        The estimations are performed by taking sliding windows of a given size centered at each locus.
        The efficiency and bias for each cell, locus and z quantile are calculated from the previous steps.
        Estimates:
            - eps_ic, detection efficiency. shape: (ncells, nloci, ncopies),
            - beta_ic, bias rate. shape: (ncells, nloci, ncopies),
            - p_ic, replication probability. shape: (ncells, nloci, ncopies).
        """
        
        print('SLIDING WINDOW RUN')
        print('------------------')
        
        # Create two eps tensors of shapes (ncells, nloci, zquants, ncopies) and (ncells, nloci, radquants, ncopies)
        eps_icz = np.zeros((self.ncells, self.nloci, len(self.zquants), self.ncopies), dtype=float)
        eps_icd = np.zeros((self.ncells, self.nloci, len(self.radquants), self.ncopies), dtype=float)
        for cellnum, state in enumerate(self.states):
            for copynum in range(self.ncopies):
                if state == 'G1':
                    eps_icz[cellnum, :, :, copynum] = self.eps_iz_G1
                    eps_icd[cellnum, :, :, copynum] = self.eps_id_G1
                elif state == 'S':
                    eps_icz[cellnum, :, :, copynum] = self.eps_iz_S
                    eps_icd[cellnum, :, :, copynum] = self.eps_id_S
                elif state == 'G2':
                    eps_icz[cellnum, :, :, copynum] = self.eps_iz_G2
                    eps_icd[cellnum, :, :, copynum] = self.eps_id_G2

        # Create two beta tensors of shapes (ncells, nloci, zquants, ncopies) and (ncells, nloci, radquants, ncopies)
        beta_icz = np.tile(self.beta_cz[:, np.newaxis, :, np.newaxis], (1, self.nloci, 1, self.ncopies))
        beta_icd = np.tile(self.beta_cd[:, np.newaxis, :, np.newaxis], (1, self.nloci, 1, self.ncopies))

        # Create a locus, cell, copy dependent tensors by selecting the actual z and rad quantiles
        eps_ic = np.full((self.ncells, self.nloci, self.ncopies), np.nan)
        beta_ic = np.full((self.ncells, self.nloci, self.ncopies), np.nan)
        for z in self.zquants:
            for d in self.radquants:
                mask_zd = np.logical_and(self.zq_ic == z, self.radq_ic == d)
                eps_ic[mask_zd] = (eps_icz[:, :, z, :][mask_zd] + eps_icd[:, :, d, :][mask_zd]) / 2
                beta_ic[mask_zd] = (beta_icz[:, :, z, :][mask_zd] + beta_icd[:, :, d, :][mask_zd]) / 2
        # Loop over cells and z to correct eps_ic and beta_ic
        for cellnum in range(self.ncells):
            for copynum in range(self.ncopies):
                for z in self.zquants:
                    for d in self.radquants:
                        # Get the efficiency and bias for the current cell, locus, copy and rad
                        mask_d = self.radq_ic[cellnum, :, copynum] == d
                        eps_ic_ = eps_ic[cellnum, :, copynum][mask_d]
                        beta_ic_ = beta_ic[cellnum, :, copynum][mask_d]
                        # Rescale the efficiency and bias by cell_n_d estimates
                        eps_ic_ = eps_ic_ * self.eps_cd[cellnum, d] / np.nanmean(eps_ic_)
                        beta_ic_ = beta_ic_ * self.beta_cd[cellnum, d] / np.nanmean(beta_ic_)
                        # Assign the corrected values
                        eps_ic[cellnum, :, copynum][mask_d] = eps_ic_
                        beta_ic[cellnum, :, copynum][mask_d] = beta_ic_
                    # Get the efficiency and bias for the current cell, locus, copy and z
                    mask_z = self.zq_ic[cellnum, :, copynum] == z
                    eps_ic_ = eps_ic[cellnum, :, copynum][mask_z]
                    beta_ic_ = beta_ic[cellnum, :, copynum][mask_z]
                    # Rescale the efficiency and bias by cell_n_z estimates
                    eps_ic_ = eps_ic_ * self.eps_cz[cellnum, z] / np.nanmean(eps_ic_)
                    beta_ic_ = beta_ic_ * self.beta_cz[cellnum, z] / np.nanmean(beta_ic_)
                    # Assign the corrected values
                    eps_ic[cellnum, :, copynum][mask_z] = eps_ic_
                    beta_ic[cellnum, :, copynum][mask_z] = beta_ic_
            # Correct the efficiency and bias by cell estimates
            eps_ic[cellnum, :, :] = eps_ic[cellnum, :, :] * self.eps_c[cellnum] / np.nanmean(eps_ic[cellnum, :, :])
            beta_ic[cellnum, :, :] = beta_ic[cellnum, :, :] * self.beta_c[cellnum] / np.nanmean(beta_ic[cellnum, :, :])
        eps_ic = self.print_n_clip('eps_ic', eps_ic, 0, 1)
        beta_ic = self.print_n_clip('beta_ic', beta_ic, 0, None)
        
        # Clip n_ic up to 4 to avoid large overestimations of the replication probability
        # n_ic = self.print_n_clip('n_ic', self.n_ic, 0, 4)
        
        # Get the window size in units of loci
        window = int(np.ceil(self.config['sliding_window_size'] / self.index.resolution()))
        
        # Calculate the sliding window averages
        n_ic_SW = scf_utils.sliding_matrix(self.n_ic, self.index, window=window, method='mean')
        eps_ic_SW = scf_utils.sliding_matrix(eps_ic, self.index, window=window, method='mean')
        beta_ic_SW = scf_utils.sliding_matrix(beta_ic, self.index, window=window, method='mean')
        
        # Calculate the replication probability
        p_ic_SW = n_ic_SW / (eps_ic_SW * beta_ic_SW) - 1

        # Store the results
        self.eps_ic = eps_ic_SW
        self.beta_ic = beta_ic_SW
        self.p_ic = p_ic_SW
        
        print('OVER.')
        print('\n\n')
    
    
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
        
        # Get the cell states, volumes and p_c
        try:
            states = self.h5['states'][:].astype(str)
        except KeyError:
            raise KeyError('The states array is not available.')
        try:
            volumes = self.h5['volumes'][:]
        except KeyError:
            raise KeyError('The volumes array is not available.')
        try:
            p_c = self.h5['cell_run']['p_c'][:]
        except KeyError:
            raise KeyError('The p_c array is not available.')
        
        # To implement the sorting, we create a sorter array,
        # where its values are monotonically increasing with the desired sorting order.
        # In G1 and G2, the sorter value is the nuclear volume.
        # In S, it's the replication probability.
        # To make sure that the sorter puts G1 before S and S before G2,
        # we add a quantity (delta) to S and double that (2 * delta) to G2,
        # such that the sorter values in G1 < sorter values in S < sorter values in G2.
        delta = 10 * (np.nanmax(volumes) + np.nanmax(p_c))
        sorter = np.full(states.shape, np.nan)
        sorter[states == 'G1'] = volumes[states == 'G1']
        sorter[states == 'S'] = p_c[states == 'S'] + delta
        sorter[states == 'G2'] = volumes[states == 'G2'] + 2 * delta
        
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
        
        # Get the cell states
        states = self.h5['states'][:].astype(str)
        
        # Subset the sorter in G1, S and G2
        nG1 = np.nansum(states == 'G1')
        nS = np.nansum(states == 'S')
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
    
    @staticmethod
    def print_n_clip(x_name: str, x: np.ndarray, v1: float = None, v2: float = None) -> np.ndarray:
        """ Given an array and its name, print the fraction of infs,
        and the fraction of non-NaN values below and above two thresholds v1 and v2,
        converts infs to NaNs and then clip the values.

        Args:
            x_name (str): name of the array.
            x (np.ndarray): array to clip.
            v1 (float, optional): lower threshold. Defaults to None.
            v2 (float, optional): upper threshold. Defaults to None.

        Returns:
            np.ndarray: the clipped array.
        """
        # Print fraction of infs and then convert them to NaN
        infs = np.mean(np.isinf(x))
        print(f"Fraction of {x_name} infs: {infs}")
        x = np.where(np.isinf(x), np.nan, x)
        # Print fraction values below and above the thresholds
        if v1 is not None:
            below_v1 = np.nanmean(x < v1)
            print(f"Fraction of {x_name} below {v1}: {below_v1}")
        if v2 is not None:
            above_v2 = np.nanmean(x > v2)
            print(f"Fraction of {x_name} above {v2}: {above_v2}")
        # Clip the values
        x_clipped = np.clip(x, v1, v2)
        return x_clipped


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
