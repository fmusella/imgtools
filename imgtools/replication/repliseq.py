import os
import numpy as np
import h5py
from functools import partial
from scipy.optimize import minimize
from alabtools.utils import Index
from ..scf import SingleCellFeature
from ..scf import scf_utils
from ..utils import smooth, clean_pearsonr


class SimulatedRepliSeqExperiment:
    """ A class to perform a simulated Repli-Seq experiment on a SingleCellFeature object.
    
    The simulated Repli-Seq experiment is based on the following model for the spotcounts:
        N = R * E + B,
    where N, R, E and B are random variables:
        - N models the spotcount,
        - R models the replication state (0 or 1),
        - E models the detection state (0, 0.5 or 1),
        - B models the bias rate (any integer >= 0).
    
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
            P(E = e | R = r) = Binomial(e ; r, eps) / r.
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
            P(B = b | R = r, E = e) = Poisson(b; r * e * beta).
    
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
        7. Sliding window analysis.
    By feature run, we mean that we calculate the average p, eps, beta for each quantized interval of the feature,
    for example Speckle distance.
    
    The object can be saved and loaded with an HDF5 file.
    
    ----------
    Attributes:
        h5_name (str): name of the HDF5 file.
        h5 (h5py.File): HDF5 file object.
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
        
        # Create the datasets for states and volumes
        h5.create_dataset('states', data=scf.cell_states.astype('S'))
        h5.create_dataset('volumes', data=scf.volumes)
        
        # Read the spotcount data
        N = scf.get_feature('spotcount')
        # Curate missing chromosomes, setting whole missing chromosomes to NaN
        self._curate_missing_chromosomes(N, scf.index)
        # Save the spotcount data
        h5.create_dataset('N', data=scf.get_feature('spotcount'))
        
        # Create the group to store the feature data
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
            subgroup.create_dataset('F', data=scf.get_feature(feat))
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
         - the index of the SCF has a valid resolution with consecutive loci.

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
        
        if scf.index.resolution() is None:
            raise ValueError("The index of the input SCF must have a valid resolution.")
        if not scf.index.consecutive():
            raise ValueError("The index of the input SCF must have consecutive loci.")

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
            7. Sliding window analysis.
        
        The results are stored in the object's HDF5 file.
        """
        
        # Check the schedule. The accepted runs are:
        accepted_schedule = [
            'population_run', 'feat_run',
            'locus_run', 'locus_feat_run',
            'cell_run', 'cell_feat_run',
        ]
        # If the schedule only contains '#', get all the runs
        if schedule == ['#']:
            schedule = accepted_schedule
        # Check that all the runs in the schedule are accepted
        for run in schedule:
            if run not in accepted_schedule:
                raise ValueError(f"The run '{run}' is not accepted.")
        
        # Load the data to memory
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
        """
        
        # Load the data from the HDF5 file
        self.index = Index(self.h5)
        self.states = self.h5['states'][:].astype(str)
        self.volumes = self.h5['volumes'][:]
        self.N = self.h5['N'][:]
        self.featdata = {}
        for feat in self.h5['featdata']:
            self.featdata[feat] = {
                'F': self.h5['featdata'][feat]['F'][:],
                'Fq': self.h5['featdata'][feat]['Fq'][:],
                'quants': self.h5['featdata'][feat]['quants'][:]
            }
        
        # Get the number of quantiles from the first feature
        feat = list(self.featdata.keys())[0]
        self.nquants = len(self.featdata[feat]['quants'])
        
        # Get the number of cells, loci and copies from the N matrix
        self.ncells, self.nloci, self.ncopies = self.N.shape
        
        # Create masks for the cell states
        self.G1s = self.states == 'G1'
        self.G2s = self.states == 'G2'
        self.Ss = self.states == 'S'
    
    def population_run(self) -> None:
        """ Run the population-wide analysis.
        Separately for G1, S, G2, it combines the data from all cells and loci to estimate average values.
        In S phase, since there are two equations and three unknowns, we assume that the efficiency is the average
        of G1 and G2.
        Estimates:
            - eps_G1, detection efficiency in G1. float,
            - beta_G1, bias rate in G1. float,
            - eps_G2, detection efficiency in G2. float,
            - beta_G2, bias rate in G2. float,
            - eps_S, detection efficiency in S. float,
            - beta_S, bias rate in S. float,
            - p_S, replication probability in S. float.
        """
        
        print('POPULATION RUN')
        print('---------------')
        
        # Delete the previous results if they exist
        if 'population_run' in self.h5:
            del self.h5['population_run']
        
        # Calculate the average number of spots for G1, S and G2, and their fractions of zeros
        n = {}
        f0 = {}
        for s in ['G1', 'S', 'G2']:
            
            # Create the state mask
            mask_state = self.states == s
            
            # Create a mask for the X and Y chromosomes (to be ignored)
            mask_XY = np.logical_or(self.index.chromstr == 'chrX', self.index.chromstr == 'chrY')
            
            # Subsample the N matrix
            N_s = self.N[mask_state, :, :][:, ~mask_XY, :]
            
            # Calculate the average number of spots and the fraction of zeros
            n[s] = np.nanmean(N_s)  # float
            f0[s] = np.sum(N_s == 0) / np.sum(~np.isnan(N_s))
        
        # Calculate efficiency and bias in G1 and G2
        eps_G1, beta_G1 = GMM_solve(n['G1'], f0['G1'], p='G1')
        eps_G2, beta_G2 = GMM_solve(n['G2'], f0['G2'], p='G2')
        
        # We assume that the efficiency in S is the average of G1 and G2
        eps_S = (eps_G1 + eps_G2) / 2
        
        # Calculate replication probability and bias in S
        p_S, beta_S = GMM_solve(n['S'], f0['S'], eps=eps_S)
        
        # Store the results in the h5 file as a group
        # The group is created if it doesn't exist
        group = self.h5.create_group('population_run')
        group.create_dataset('eps_G1', data=eps_G1)
        group.create_dataset('beta_G1', data=beta_G1)
        group.create_dataset('eps_G2', data=eps_G2)
        group.create_dataset('beta_G2', data=beta_G2)
        group.create_dataset('eps_S', data=eps_S)
        group.create_dataset('beta_S', data=beta_S)
        group.create_dataset('p_S', data=p_S)
        
        print('OVER.')
        print('\n\n')
    
    def feat_run(self, feat: str) -> None:
        """ Run the feature-dependent analysis.
        Treats each feature quantile independently, combining the data from all cells and loci.
        
        Estimates:
            - eps_q_G1, detection efficiency in G1. shape: (nquants),
            - beta_q_G1, bias rate in G1. shape: (nquants),
            - eps_q_G2, detection efficiency in G2. shape: (nquants),
            - beta_q_G2, bias rate in G2. shape: (nquants),
            - eps_q_S, detection efficiency in S. shape: (nquants),
            - beta_q_S, bias rate in S. shape: (nquants),
            - p_q_S, replication probability in S. shape: (nquants).

        Args:
            feat (str)
        """
        
        print(f'FEAT-DEPENDENT RUN ({feat})')
        print('---------------')
        
        # Delete the previous results if they exist
        if 'feat_run' in self.h5:
            if feat in self.h5['feat_run']:
                del self.h5['feat_run'][feat]
        
        # Calculate the average number of spots and the fraction of zeros per feature quantile
        n = {}
        f0 = {}
        for s in ['G1', 'S', 'G2']:
            
            # Create the state mask
            mask_state = self.states == s
            # Create a mask for the X and Y chromosomes (to be ignored)
            mask_XY = np.logical_or(self.index.chromstr == 'chrX', self.index.chromstr == 'chrY') 
            # Subsample the N and Fq matrices
            N_s = self.N[mask_state, :, :][:, ~mask_XY, :]
            Fq_s = self.featdata[feat]['Fq'][mask_state, :, :][:, ~mask_XY, :]
            
            # Initialize the dictionaries to store quantile-dependent averages
            n[s] = np.zeros(self.nquants)  # shape: (nquants)
            f0[s] = np.zeros(self.nquants)  # shape: (nquants)
            # Loop over the quantiles
            for q in self.featdata[feat]['quants']:
                
                # Create the quantile mask
                mask_q = Fq_s == q
                # Subsample the N matrix
                N_s_q = N_s[mask_q]
                
                # Calculate the average number of spots and the fraction of zeros
                n[s][q] = np.nanmean(N_s_q)
                f0[s][q] = np.sum(N_s_q == 0) / np.sum(~np.isnan(N_s_q))

        # Calculate efficiency and bias in G1 and G2
        eps_q_G1, beta_q_G1 = GMM_solve(n['G1'], f0['G1'], p='G1')
        eps_q_G2, beta_q_G2 = GMM_solve(n['G2'], f0['G2'], p='G2')
        eps_q_G1 = self.print_n_clip('eps_q_G1', eps_q_G1, 0, 1)
        eps_q_G2 = self.print_n_clip('eps_q_G2', eps_q_G2, 0, 1)
        beta_q_G1 = self.print_n_clip('beta_q_G1', beta_q_G1, 0, None)
        beta_q_G2 = self.print_n_clip('beta_q_G2', beta_q_G2, 0, None)
        
        # We assume that the efficiency in S is the average of G1 and G2
        eps_q_S = (eps_q_G1 + eps_q_G2) / 2
        
        # Calculate replication probability and bias in S
        p_q_S, beta_q_S = GMM_solve(n['S'], f0['S'], eps=eps_q_S)
        p_q_S = self.print_n_clip('p_q_S', p_q_S, 0, 1)
        beta_q_S = self.print_n_clip('beta_q_S', beta_q_S, 0, None)
        
        # Store the results
        group = self.h5.require_group('feat_run')
        subgroup = group.create_group(feat)
        subgroup.create_dataset('eps_q_G1', data=eps_q_G1)
        subgroup.create_dataset('beta_q_G1', data=beta_q_G1)
        subgroup.create_dataset('eps_q_G2', data=eps_q_G2)
        subgroup.create_dataset('beta_q_G2', data=beta_q_G2)
        subgroup.create_dataset('eps_q_S', data=eps_q_S)
        subgroup.create_dataset('beta_q_S', data=beta_q_S)
        subgroup.create_dataset('p_q_S', data=p_q_S)
        
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
            - beta_i_G1, bias rate in G1. shape: (nloci),
            - eps_i_G2, detection efficiency in G2. shape: (nloci),
            - beta_i_G2, bias rate in G2. shape: (nloci),
            - eps_i_S, detection efficiency in S. shape: (nloci),
            - beta_i_S, bias rate in S. shape: (nloci),
            - p_i_S, replication probability in S. shape: (nloci).
        """
        
        print('LOCUS-DEPENDENT RUN')
        print('-------------------')
        
        # Delete the previous results if they exist
        if 'locus_run' in self.h5:
            del self.h5['locus_run']
        
        # Calculate the average number of spots for G1, S and G2, and their fractions of zeros
        n_i = {}
        f0_i = {}
        for s in ['G1', 'S', 'G2']:
            
            # Create the state mask
            mask_state = self.states == s
            N_s = self.N[mask_state, :, :]
            
            # Calculate the average number of spots and the fraction of zeros for each locus
            n_i[s] = np.nanmean(N_s, axis=(0, 2))  # shape: (nloci)
            f0_i[s] = np.sum(N_s == 0, axis=(0, 2)) / np.sum(~np.isnan(N_s), axis=(0, 2))
        
        # Calculate efficiency and bias in G1 and G2
        eps_i_G1, beta_i_G1 = GMM_solve(n_i['G1'], f0_i['G1'], p='G1')
        eps_i_G2, beta_i_G2 = GMM_solve(n_i['G2'], f0_i['G2'], p='G2')
        eps_i_G1 = self.print_n_clip('eps_i_G1', eps_i_G1, 0, 1)
        eps_i_G2 = self.print_n_clip('eps_i_G2', eps_i_G2, 0, 1)
        beta_i_G1 = self.print_n_clip('beta_i_G1', beta_i_G1, 0, None)
        beta_i_G2 = self.print_n_clip('beta_i_G2', beta_i_G2, 0, None)

        # Assume that the efficiency in S is the average of G1 and G2
        eps_i_S = (eps_i_G1 + eps_i_G2) / 2
        
        # Note: since we assume that beta doesn't depend on i,
        # we could use beta_S to estimate both eps_i_S and p_i_S
        # However, I think that the statistical power is not good enough to
        # estimate two parameters. Indeed, the results looked bad.
        # Note that here, for the locus-dependent analysis, we don't have
        # a lot of data for the estimation: there are ~250 cells in G1,
        # ~250 cells in G2, and ~500 cells in S. Since there are two copies
        # we multiply these number by 2, but it's still very little.
        
        # Calculate replication probability and bias in S
        p_i_S, beta_i_S = GMM_solve(n_i['S'], f0_i['S'], eps=eps_i_S)
        p_i_S = self.print_n_clip('p_i_S', p_i_S, 0, 1)
        beta_i_S = self.print_n_clip('beta_i_S', beta_i_S, 0, None)
        
        # Store the results
        group = self.h5.create_group('locus_run')
        group.create_dataset('eps_i_G1', data=eps_i_G1)
        group.create_dataset('beta_i_G1', data=beta_i_G1)
        group.create_dataset('eps_i_G2', data=eps_i_G2)
        group.create_dataset('beta_i_G2', data=beta_i_G2)
        group.create_dataset('eps_i_S', data=eps_i_S)
        group.create_dataset('beta_i_S', data=beta_i_S)
        group.create_dataset('p_i_S', data=p_i_S)
        
        print('OVER.')
        print('\n\n')
    
    def locus_feat_run(self, feat: str) -> None:
        """ Run the locus and feature-dependent analysis.
        Treats each locus and feature quantile independently, combining the data from all cells.
        
        Estimates:
            - eps_iq_G1, detection efficiency in G1. shape: (nloci, nquants),
            - eps_iq_G2, detection efficiency in G2. shape: (nloci, nquants),
            - eps_iq_S, detection efficiency in S. shape: (nloci, nquants),
            - beta_iq_S, bias rate in S. shape: (nloci, nquants),
            - p_iq_S, replication probability in S. shape: (nloci, nquants).

        Args:
            feat (str)
        """
        
        print(f'LOCUS AND FEAT-DEPENDENT RUN ({feat})')
        print('---------------')
        
        # Delete the previous results if they exist
        if 'locus_feat_run' in self.h5:
            if feat in self.h5['locus_feat_run']:
                del self.h5['locus_feat_run'][feat]
        
        # Calculate the average number of spots and the fraction of zeros
        # per locus and feature quantile, separately for G1, S and G2
        n_iq = {}
        f0_iq = {}
        for s in ['G1', 'S', 'G2']:
            
            # Create the state mask
            mask_state = self.states == s   
            # Subsample the N and Fq matrices
            N_s = self.N[mask_state, :, :]
            Fq_s = self.featdata[feat]['Fq'][mask_state, :, :]
            
            # Initialize the dictionaries to store average values
            n_iq[s] = np.zeros((self.nloci, self.nquants))  # shape: (nloci, nquants)
            f0_iq[s] = np.zeros((self.nloci, self.nquants))  # shape: (nloci, nquants)
            # Loop over the quantiles
            for q in self.featdata[feat]['quants']:
                
                # Create the quantile mask
                mask_q = Fq_s == q
                
                # To exclude data from other quantiles, we create an array N_s_q
                # that is NaN where the mask_q is False
                N_s_q = np.where(mask_q, N_s, np.nan)
                
                # Calculate the average number of spots and the fraction of zeros
                n_iq[s][:, q] = np.nanmean(N_s_q, axis=(0, 2))  # shape: (nloci)
                f0_iq[s][:, q] = np.sum(N_s_q == 0, axis=(0, 2)) / np.sum(~np.isnan(N_s_q), axis=(0, 2))
        
        # Calculate the efficiency in G1 and G2
        # We assume that the bias rate is uniform across loci, so we ignore the beta value
        eps_iq_G1, _ = GMM_solve(n_iq['G1'], f0_iq['G1'], p='G1')
        eps_iq_G2, _ = GMM_solve(n_iq['G2'], f0_iq['G2'], p='G2')
        eps_iq_G1 = self.print_n_clip('eps_iq_G1', eps_iq_G1, 0, 1)
        eps_iq_G2 = self.print_n_clip('eps_iq_G2', eps_iq_G2, 0, 1)
        
        # Assume that the efficiency in S is the average of G1 and G2
        eps_iq_S = (eps_iq_G1 + eps_iq_G2) / 2
        
        # For S, since we assume that the bias rate is uniform across loci,
        # we can just use the beta value from the feat-dependent analysis and tile it
        beta_q_S = self.h5['feat_run'][feat]['beta_q_S'][:]
        beta_iq_S = np.tile(beta_q_S[np.newaxis, :], (self.nloci, 1))  # shape: (nloci, nquants)
        
        # Calculate the probability of replication in S
        p_iq_S = GMM_solve(n_iq['S'], f0_iq['S'], eps=eps_iq_S, beta=beta_iq_S)
        p_iq_S = self.print_n_clip('p_iq_S', p_iq_S, 0, 1)
        
        # Store the results
        group = self.h5.require_group('locus_feat_run')
        subgroup = group.create_group(feat)
        subgroup.create_dataset('eps_iq_G1', data=eps_iq_G1)
        subgroup.create_dataset('eps_iq_G2', data=eps_iq_G2)
        subgroup.create_dataset('eps_iq_S', data=eps_iq_S)
        subgroup.create_dataset('p_iq_S', data=p_iq_S)
        
        print('OVER.')
        print('\n\n')

    def cell_run(self) -> None:
        """ Run the cell-dependent analysis.
        Treats each cell independently, assuming that different loci are independent realizations
        of the same cell-dependent process.
        In S phase, since there are two equations and three unknowns, we use an approximation
        for the bias rate: we calculate it using only the early replicating loci, for which
        the replication state is known. The approximations (done for both efficiency and bias)
        can be tested for G1 and G2, where the equations can be solved exactly.
        Estimates:
            - eps_c, detection efficiency. shape: (ncells),
            - eps_c_, approximated detection efficiency. shape: (ncells),
            - beta_c, bias rate. shape: (ncells), 
            - beta_c_, approximated bias rate. shape: (ncells).
            - p_c, replication probability. shape: (ncells).
        """
        
        print('CELL-DEPENDENT RUN')
        print('------------------')
        
        # Delete the previous results if they exist
        if 'cell_run' in self.h5:
            del self.h5['cell_run']
        
        # Identify early replicating loci
        RT_early = 0.95
        p_i_S = self.h5['locus_run']['p_i_S'][:]
        early_mask = p_i_S > RT_early
        
        # Delete the previous results if they exist
        if 'cell_run' in self.h5:
            del self.h5['cell_run']
        
        # Calculate the average number of spots and the fraction of zeros per cell
        # using either all autosomic loci or the early replicating autosomic loci.
        n_c = {}
        f0_c = {}
        for loci in ['all', 'early']:
            
            # Create a mask to exclude the X and Y chromosomes
            mask_loci = np.logical_and(self.index.chromstr != 'chrX', self.index.chromstr != 'chrY')
            # Apply the early mask
            if loci == 'early':
                mask_loci = np.logical_and(mask_loci, early_mask)
            
            # Subsample the N matrix for the selected loci
            N_loci = self.N[:, mask_loci, :]
            # Calculate the average number of spots and the fraction of zeros for each cell
            n_c[loci] = np.nanmean(N_loci, axis=(1, 2))  # shape: (ncells)
            f0_c[loci] = np.sum(N_loci == 0, axis=(1, 2)) / np.sum(~np.isnan(N_loci), axis=(1, 2))
        
        # Calculate the approximate efficiency and bias for early loci for G1, S, G2
        # For S cells, we assume that early loci have all replicated, so we can use the G2 equations
        eps_G1_c_, beta_G1_c_ = GMM_solve(n_c['early'][self.G1s], f0_c['early'][self.G1s], p='G1')
        eps_S_c_, beta_S_c_ = GMM_solve(n_c['early'][self.Ss], f0_c['early'][self.Ss], p='G2')
        eps_G2_c_, beta_G2_c_ = GMM_solve(n_c['early'][self.G2s], f0_c['early'][self.G2s], p='G2')
        # Create arrays for all cells and fill them
        eps_c_ = np.full(self.ncells, np.nan)
        eps_c_[self.G1s] = eps_G1_c_
        eps_c_[self.Ss] = eps_S_c_
        eps_c_[self.G2s] = eps_G2_c_
        beta_c_ = np.full(self.ncells, np.nan)
        beta_c_[self.G1s] = beta_G1_c_
        beta_c_[self.Ss] = beta_S_c_
        beta_c_[self.G2s] = beta_G2_c_
        eps_c_ = self.print_n_clip('eps_c_', eps_c_, 0, 1)
        beta_c_ = self.print_n_clip('beta_c_', beta_c_, 0, None)
        
        # Calculate the exact efficiency and bias for G1 and G2
        eps_G1_c, beta_G1_c = GMM_solve(n_c['all'][self.G1s], f0_c['all'][self.G1s], p='G1')
        eps_G2_c, beta_G2_c = GMM_solve(n_c['all'][self.G2s], f0_c['all'][self.G2s], p='G2')
        # Create arrays for all cells and fill them
        eps_c = np.full(self.ncells, np.nan)
        eps_c[self.G1s] = eps_G1_c
        eps_c[self.G2s] = eps_G2_c
        beta_c = np.full(self.ncells, np.nan)
        beta_c[self.G1s] = beta_G1_c
        beta_c[self.G2s] = beta_G2_c
        # Use the approximate beta for S cells
        beta_c[self.Ss] = beta_S_c_
        beta_c = self.print_n_clip('beta_c', beta_c, 0, None)
        
        # Calculate the probability of replication and efficiency for S cells
        p_S_c, eps_S_c = GMM_solve(n_c['all'][self.Ss], f0_c['all'][self.Ss], beta=beta_c[self.Ss])
        
        # Assign the efficiency for S
        eps_c[self.Ss] = eps_S_c
        eps_c = self.print_n_clip('eps_c', eps_c, 0, 1)
        
        # Create the replication probability array for all cells
        p_c = np.full(self.ncells, np.nan)
        p_c[self.G1s] = 0
        p_c[self.G2s] = 1
        p_c[self.Ss] = p_S_c
        p_c = self.print_n_clip('p_c', p_c, 0, 1)
        
        # We haven't used the approximate efficiency besides for calculating the approximate bias
        # However, we want to store it, so that we can compare it to the exact efficiency to assess the approximation.
        # Still, to be more accurate, we have to perform a rescaling: we know that there is a strong correlation between
        # efficiency and Replication Timing: the earliest 5% used to calculate the approximate efficiency actually
        # have a systematic lower detection efficiency.
        # So we can use the results of the locus-dependent analysis to rescale the approximate efficiency.
        for state in ['G1', 'S', 'G2']:
            state_mask = self.states == state
            eps_i_s = self.h5['locus_run'][f'eps_i_{state}'][:]
            eps_c_[state_mask] *= np.nanmean(eps_i_s) / np.nanmean(eps_i_s[early_mask])
        
        # Store the results
        group = self.h5.create_group('cell_run')
        group.create_dataset('eps_c', data=eps_c)
        group.create_dataset('eps_c_', data=eps_c_)
        group.create_dataset('beta_c', data=beta_c)
        group.create_dataset('beta_c_', data=beta_c_)
        group.create_dataset('p_c', data=p_c)
        
        print('OVER.')
        print('\n\n')
    
    def cell_feat_run(self, feat: str) -> None:
        """ Run the cell and feature-dependent analysis.
        Treats each cell and feature quantile independently, combining the data from all loci.
        For S phase, approximates the replication probability using the results of the cell and feature run:
            p_cq_S = p_c_S * p_q_S / mean(p_q_S).
            
        Estimates:
            - eps_cq, detection efficiency. shape: (ncells, nquants),
            - beta_cq, bias rate. shape: (ncells, nquants),
            - p_cq_S, replication probability in S. shape: (ncells, nquants).

        Args:
            feat (str)
        """
        
        print(f'CELL AND FEAT-DEPENDENT RUN ({feat})')
        print('------------------------')
        
        # Delete the previous results if they exist
        if 'cell_feat_run' in self.h5:
            if feat in self.h5['cell_feat_run']:
                del self.h5['cell_feat_run'][feat]
        
        # Initialize the data for the average number of spots and the fraction of zeros
        # for each cell and feature quantile
        n_cq = np.zeros((self.ncells, self.nquants))  # shape: (ncells, nquants)
        f0_cq = np.zeros((self.ncells, self.nquants))
        
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
            
            # Calculate the average number of spots and the fraction of zeros
            n_cq[:, q] = np.nanmean(N_q, axis=(1, 2))  # shape: (ncells)
            f0_cq[:, q] = np.sum(N_q == 0, axis=(1, 2)) / np.sum(~np.isnan(N_q), axis=(1, 2))

        # Calculate efficiency and bias for G1 and G2
        eps_G1_cq, beta_G1_cq = GMM_solve(n_cq[self.G1s, :], f0_cq[self.G1s, :], p='G1')
        eps_G2_cq, beta_G2_cq = GMM_solve(n_cq[self.G2s, :], f0_cq[self.G2s, :], p='G2')
        # Create arrays for all cells and fill them
        eps_cq = np.full((self.ncells, self.nquants), np.nan)  # shape: (ncells, nquants)
        eps_cq[self.G1s, :] = eps_G1_cq
        eps_cq[self.G2s, :] = eps_G2_cq
        beta_cq = np.full((self.ncells, self.nquants), np.nan)  # shape: (ncells, nquants)
        beta_cq[self.G1s, :] = beta_G1_cq
        beta_cq[self.G2s, :] = beta_G2_cq
        
        # For S phase, it would be too much to use the early-replication trick, since we would have too little data.
        # So instead, we approximate the replication probability using our previous results,
        # in particular the cell run and the feature run.
        # We start from the p_c values, and we tile them
        p_c_S = self.h5['cell_run']['p_c'][self.Ss]
        p_c_S = np.tile(p_c_S[:, np.newaxis], (1, self.nquants))  # shape: (ncells_S, nquants)
        # Then we calculate the rescaling factors for each quantile from p_q_S,
        # i.e. the ratio between each p_q value and their average
        p_q_S = self.h5['feat_run'][feat]['p_q_S'][:]
        x_q_S = p_q_S / np.nanmean(p_q_S)
        x_q_S = np.tile(x_q_S[np.newaxis, :], (np.sum(self.Ss), 1))  # shape: (ncells_S, nquants)
        # Finally, we define the cell-and-quantile dependent replication probability as the product of the two
        p_cq_S = p_c_S * x_q_S
        p_cq_S = self.print_n_clip('p_cq_S', p_cq_S, 0, 1)
        # Create a full p_cq matrix to store the results
        p_cq = np.full((self.ncells, self.nquants), np.nan)  # shape: (ncells, nquants)
        p_cq[self.G1s, :] = 0
        p_cq[self.G2s, :] = 1
        p_cq[self.Ss, :] = p_cq_S
        
        # We then calculate the efficiency and bias for S
        eps_cq_S, beta_cq_S = GMM_solve(n_cq[self.Ss, :], f0_cq[self.Ss, :], p=p_cq_S)
        eps_cq[self.Ss, :] = eps_cq_S
        beta_cq[self.Ss, :] = beta_cq_S
        eps_cq = self.print_n_clip('eps_cq', eps_cq, 0, 1)
        beta_cq = self.print_n_clip('beta_cq', beta_cq, 0, None)
        
        # Note that here we do estimate two parameters, differently from the locus-dependent analysis.
        # It's because here we have much more data: each cell has ~100k loci, so ~200k data (two copies).
        # If there are 10 quantiles, we have ~20k data points for each estimation.
        
        # Store the results
        group = self.h5.require_group('cell_feat_run')
        subgroup = group.create_group(feat)
        subgroup.create_dataset('eps_cq', data=eps_cq)
        subgroup.create_dataset('beta_cq', data=beta_cq)
        subgroup.create_dataset('p_cq', data=p_cq)
        
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

    def calculate_repliprob(self, mask: np.ndarray, nrepeat: int = 1) -> list:
        """ 
        TODO: fix with new data structure.
        
        Calculates the replication probability for a given mask.
        
        mask is a boolean numpy array of shape (ncells, ndomains, ncopies),
        indicating for which loci in which cells we have to calculate the
        replication probability.
        
        The function does the following:
            - makes sure that the mask only contains True values for S cells,
            - breaks down the mask into a dictionary containing which loci
              are considered for each cell,
            - randomly creates analogous masks for G1 and G2 cells,
            - estimates eps and beta from G1 and G2,
            - corrects eps and beta with z and cell estimates,
            - calculates the replication probability for the original mask,
            - the process can be repeated multiple times to get a more robust estimate.
            
        Returns a list of length nrepeat containing the replication probabilities
        for each repetition.

        Args:
            mask (np.ndarray): A boolean numpy array of shape (ncells, ndomains, ncopies).
            nrepeat (int): The number of times the process is repeated.

        Returns:
            list: A list of length nrepeat containing the replication probabilities.
        """
        
        # Find the indices of G1 and G2 cells
        # TODO: include G1s and G2s as attributes. I have to re-run though, so I'll do it later.
        G1s = np.where(self.states == 'G1')[0]
        G2s = np.where(self.states == 'G2')[0]
        
        # Get the target dictionary and the target cells
        # tcells is an array of indices of the target cells, i.e. those with at least one True value,
        # tdict is a dictionary containing for each target cell the loci considered and the number of copies.
        tdict, tcells = self._dictionarize_mask(mask)
        
        # Make sure that the target cells are S cells
        tstates = self.states[tcells]
        if not np.all(tstates == 'S'):
            raise ValueError('The target cells must be S cells.')

        # Now we estimate eps and beta in G1 and G2 by random sampling
        # We repeat this process nrepeat times to get a more robust estimate
        tcells_G1s = []
        tcells_G2s = []
        eps_G1s = []
        eps_G2s = []
        beta_G1s = []
        beta_G2s = []
        zq_G1s = []
        zq_G2s = []
        radq_G1s = []
        radq_G2s = []
        for r in range(nrepeat):
            
            tcells_G1, tcells_G2, eps_G1, eps_G2, beta_G1, beta_G2,\
                zq_G1, zq_G2, radq_G1, radq_G2 = self._randomize_G1G2(tdict, tcells, G1s, G2s)
            
            # Store the results
            tcells_G1s.append(tcells_G1)
            tcells_G2s.append(tcells_G2)
            eps_G1s.append(eps_G1)
            eps_G2s.append(eps_G2)
            beta_G1s.append(beta_G1)
            beta_G2s.append(beta_G2)
            zq_G1s.append(zq_G1)
            zq_G2s.append(zq_G2)
            radq_G1s.append(radq_G1)
            radq_G2s.append(radq_G2)
        
        # Calculate the replication probability for the target S cells
        # using the estimates from G1 and G2
        
        # First we need to calculate the average number of spots, zq, radq,
        # eps_c_S, beta_c_S and p_c_S for the target S cells
        n_ic_S = self.n_ic[mask]
        n_S = np.nanmean(n_ic_S)
        f0_S = np.sum(n_ic_S == 0) / np.sum(~np.isnan(n_ic_S))
        print(f'n_S: {n_S}, f0_S: {f0_S}')
        
        eps_c_S = np.nanmean(self.eps_c[tcells])
        beta_c_S = np.nanmean(self.beta_c[tcells])
        p_c_S = np.nanmean(self.p_c[tcells])
        print(f'eps_c_S: {eps_c_S}, beta_c_S: {beta_c_S}, p_c_S: {p_c_S}')
        
        # Find the zq and radq of the target S cells
        zqs_S = self.zq_ic[mask]
        radqs_S = self.radq_ic[mask]
        zq_S = np.nanmean(zqs_S)
        radq_S = np.nanmean(radqs_S)
        # Round to the nearest integer
        zq_S = int(np.round(zq_S))
        radq_S = int(np.round(radq_S))
        # If the number of unique zq values is less or equal than 2,
        # and these two values are consecutive, we don't want to correct for radq
        unq_zqs_S = np.unique(zqs_S)
        print(f'unq_zqs_S: {unq_zqs_S}')
        print(f'np.diff(unq_zqs_S): {np.diff(unq_zqs_S)}')
        if len(unq_zqs_S) <= 2 and np.all(np.abs(np.diff(unq_zqs_S)) == 1):
            radq_S = None
        print(f'zq_S: {zq_S}, radq_S: {radq_S}')
        
        # Compare n_S and the cell average
        n_c_avg = np.nanmean(self.n_ic[tcells, :, :])
        print(f'n_S: {n_S}, n_c_avg: {n_c_avg}')
        
        # Initialize the list of replication probabilities
        p_Ss = []
        
        # Loop over the repetitions
        for r in range(nrepeat):
            
            print(f'Repetition {r}')
            
            tcells_G1 = tcells_G1s[r]
            tcells_G2 = tcells_G2s[r]
            eps_G1, eps_G2 = eps_G1s[r], eps_G2s[r]
            beta_G1, beta_G2 = beta_G1s[r], beta_G2s[r]
            zq_G1, zq_G2 = zq_G1s[r], zq_G2s[r]
            radq_G1, radq_G2 = radq_G1s[r], radq_G2s[r]
            
            print('Before corrections')
            print(f'eps_G1: {eps_G1}, beta_G1: {beta_G1}, zq_G1: {zq_G1}, radq_G1: {radq_G1}')
            print(f'eps_G2: {eps_G2}, beta_G2: {beta_G2}, zq_G2: {zq_G2}, radq_G2: {radq_G2}')
            
            # Correct eps and beta by rad estimates if radq_S is not None
            if radq_S is not None:
                eps_G1 = eps_G1 + self.eps_d_G1[radq_S] - self.eps_d_G1[radq_G1]
                eps_G2 = eps_G2 + self.eps_d_G2[radq_S] - self.eps_d_G2[radq_G2]
                beta_G1 = beta_G1 + self.beta_d_G1[radq_S] - self.beta_d_G1[radq_G1]
                beta_G2 = beta_G2 + self.beta_d_G2[radq_S] - self.beta_d_G2[radq_G2]
                
                print('After rad correction')
                print(f'eps_G1: {eps_G1}, beta_G1: {beta_G1}')
                print(f'eps_G2: {eps_G2}, beta_G2: {beta_G2}')
            
            # Correct eps and beta by cell_n_z estimates
            eps_G1 = eps_G1 + np.nanmean(self.eps_cz[tcells, zq_S]) - np.nanmean(self.eps_cz[tcells_G1, zq_G1])
            eps_G2 = eps_G2 + np.nanmean(self.eps_cz[tcells, zq_S]) - np.nanmean(self.eps_cz[tcells_G2, zq_G2])
            beta_G1 = beta_G1 + np.nanmean(self.beta_cz[tcells, zq_S]) - np.nanmean(self.beta_cz[tcells_G1, zq_G1])
            beta_G2 = beta_G2 + np.nanmean(self.beta_cz[tcells, zq_S]) - np.nanmean(self.beta_cz[tcells_G2, zq_G2])
            
            print('After cell and z correction')
            print(f'eps_G1: {eps_G1}, beta_G1: {beta_G1}')
            print(f'eps_G2: {eps_G2}, beta_G2: {beta_G2}')
            
            # To assign eps and beta in S, do a linear interpolation
            # using p_c_S (the closer to 0, the closer to G1)
            eps_S = eps_G1 + p_c_S * (eps_G2 - eps_G1)
            beta_S = beta_G1 + p_c_S * (beta_G2 - beta_G1)
            
            print(f'eps_S: {eps_S}, beta_S: {beta_S}')
            
            # Calculate the replication probability
            p_S_1 = n_S / (eps_S * beta_S) - 1
            p_S_2 = (1 - eps_S - f0_S) / (eps_S * (1 - eps_S))
            
            def pS_root_func(x: float, n: float, f0: float, eps: float, beta: float) -> float:
                r1 = x - (n / (eps * beta) - 1)
                r2 = x - (1 - eps - f0) / (eps * (1 - eps))
                return np.sqrt(r1 ** 2 + r2 ** 2)
            f = partial(pS_root_func, n=n_S, f0=f0_S, eps=eps_S, beta=beta_S)
            p_S = minimize(f, (p_S_1 + p_S_2) / 2).x[0]
                
            p_Ss.append(p_S)
            
            print(f'p_S_1: {p_S_1}, p_S_2: {p_S_2}, p_S: {p_S}')
            
            # Calculate eps and beta knowing p_c_S
            eps_S_ = (1 + p_c_S - np.sqrt((1 + p_c_S) ** 2 - 4 * p_c_S * (1 - f0_S))) / (2 * p_c_S)
            beta_S_ = n_S / ((1 + p_c_S) * eps_S_)
            
            print(f'eps_S_: {eps_S_}, beta_S_: {beta_S_}')
            
            print('\n\n')
        
        return p_Ss
    
    @staticmethod
    def _dictionarize_mask(mask: np.ndarray) -> dict:
        """ Create a dictionary from a mask of shape (ncells, ndomains, ncopies).
        The dictionary contains for each target cell the loci that True,
        and for each locus the number of copies that are True (either 1 or 2).
        
        For example:
        tdict = {
            cell_10: {locus_1: 1, locus_3: 2, locus_5: 1},
            cell_20: {locus_2: 1, locus_4: 2, locus_6: 1},
              ...
        }

        Args:
            mask (np.ndarray): A boolean numpy array of shape (ncells, ndomains, ncopies).

        Returns:
            tdict (dict): A dictionary containing the target loci for each target cell.
            tcells (np.ndarray): An array containing the indices of the cells that
                                contain at least one True value.
        """
        
        # Find the target cells, i.e. the ones that contain at least one True value
        tcells = np.where(np.sum(mask, axis=(1, 2)) > 0)[0]  # shape: (ntcells), dtype: int
        
        # Initialize the target dictionary
        tdict = {}
        for c in  tcells:
            tdict[c] = {}
            mask_c = mask[c, :, :]  # shape: (ndomains, ncopies)
            
            # Get an array of shape (ndomains) with the number
            # of Trues for each locus for cell c
            locisum_c = np.sum(mask_c, axis=1)  # shape: (ndomains), dtype: int
            locisum_c_mask = locisum_c > 0  # shape: (ndomains), dtype: bool
            for i in np.where(locisum_c_mask)[0]:
                tdict[c][i] = locisum_c[i]
        
        return tdict, tcells
    
    def _randomize_G1G2(
        self, tdict: dict, tcells: np.ndarray, G1s: np.ndarray, G2s: np.ndarray
    ) -> tuple:
        """ Randomly estimate eps and beta with G1 and G2 cells given a target dictionary.
        
        The form of the target dictionary is provided by the _dictionarize_mask function.
        
        This function does the following:
            - randomly selects target cells and loci for G1 and G2,
            - randomly maps the target S cells to the target G1 and G2 cells,
            - randomly assigns copy A or B to the target loci,
            - calculates eps and beta for G1 and G2,
            - calculates the average zq and radq of the cells used.

        Args:
            tdict (dict): A dictionary containing the target loci for each target cell.
            tcells (np.ndarray): An array containing the indices of the target cells.
            G1s (np.ndarray): an array containing the indices of the G1 cells.
            G2s (np.ndarray): an array containing the indices of the G2 cells.

        Returns:
            eps_G1 (float): The efficiency for G1.
            eps_G2 (float): The efficiency for G2.
            beta_G1 (float): The bias for G1.
            beta_G2 (float): The bias for G2.
            zq_G1 (int): The average zq for G1.
            zq_G2 (int): The average zq for G2.
            radq_G1 (int): The average radq for G1.
            radq_G2 (int): The average radq for G2.
        """
        
        # Randomly select target cells from G1 and G2
        tcells_G1 = self._generate_shuffled_indices(len(tcells), G1s)
        tcells_G2 = self._generate_shuffled_indices(len(tcells), G2s)
        
        # Randomly map these cells to the target cells
        map_G1 = {cS: cG1 for cS, cG1 in zip(tcells, tcells_G1)}
        map_G2 = {cS: cG2 for cS, cG2 in zip(tcells, tcells_G2)}
        
        # Initialize the lists to store the G1, G2 randomized data
        ns_G1, ns_G2 = [], []
        zqs_G1, zqs_G2 = [], []
        radqs_G1, radqs_G2 = [], []
        
        # Loop over the target cells and fill the lists
        for c in tcells:
            cG1, cG2 = map_G1[c], map_G2[c]
            for i in tdict[c]:
                
                # Get the number of copies for the current locus
                ncopies_ci = tdict[c][i]
                
                # If the number of copies is 2 there is no randomness
                if ncopies_ci == 2:
                    ns_G1.extend(self.n_ic[cG1, i, :])
                    ns_G2.extend(self.n_ic[cG2, i, :])
                    zqs_G1.extend(self.zq_ic[cG1, i, :])
                    zqs_G2.extend(self.zq_ic[cG2, i, :])
                    radqs_G1.extend(self.radq_ic[cG1, i, :])
                    radqs_G2.extend(self.radq_ic[cG2, i, :])
                    continue
                
                # Otherwise we randomly assign to one of the copies
                h_G1 = np.random.choice([0, 1])
                h_G2 = np.random.choice([0, 1])
                ns_G1.append(self.n_ic[cG1, i, h_G1])
                ns_G2.append(self.n_ic[cG2, i, h_G2])
                zqs_G1.append(self.zq_ic[cG1, i, h_G1])
                zqs_G2.append(self.zq_ic[cG2, i, h_G2])
                radqs_G1.append(self.radq_ic[cG1, i, h_G1])
                radqs_G2.append(self.radq_ic[cG2, i, h_G2])
        
        # Convert the lists to numpy arrays
        ns_G1, ns_G2 = np.array(ns_G1), np.array(ns_G2)
        zqs_G1, zqs_G2 = np.array(zqs_G1), np.array(zqs_G2)
        radqs_G1, radqs_G2 = np.array(radqs_G1), np.array(radqs_G2)
        
        # Calculate the average number of spots and the fraction of zeros
        n_G1, n_G2 = np.nanmean(ns_G1), np.nanmean(ns_G2)
        f0_G1 = np.sum(ns_G1 == 0) / np.sum(~np.isnan(ns_G1))
        f0_G2 = np.sum(ns_G2 == 0) / np.sum(~np.isnan(ns_G2))
        
        # Calculate the average zq and radq of the cells used
        zq_G1, zq_G2 = np.nanmean(zqs_G1), np.nanmean(zqs_G2)
        radq_G1, radq_G2 = np.nanmean(radqs_G1), np.nanmean(radqs_G2)
        # Round to the nearest integer
        zq_G1, zq_G2 = int(np.round(zq_G1)), int(np.round(zq_G2))
        radq_G1, radq_G2 = int(np.round(radq_G1)), int(np.round(radq_G2))
        
        # Calculate the efficiency and bias
        eps_G1 = 1 - f0_G1
        eps_G2 = 1 - f0_G2 ** 0.5
        beta_G1 = n_G1 / eps_G1
        beta_G2 = n_G2 / (2 * eps_G2)
        eps_G1 = self.print_n_clip('eps_G1', eps_G1, 0, 1)
        eps_G2 = self.print_n_clip('eps_G2', eps_G2, 0, 1)
        beta_G1 = self.print_n_clip('beta_G1', beta_G1, 0, None)
        beta_G2 = self.print_n_clip('beta_G2', beta_G2, 0, None)
        
        return tcells_G1, tcells_G2, eps_G1, eps_G2, beta_G1, beta_G2, zq_G1, zq_G2, radq_G1, radq_G2
    
    @staticmethod
    def _generate_shuffled_indices(n: int, idx: np.ndarray) -> np.ndarray:
        """ Given an array of indices, generates a shuffled array of size n
        sampled from the input one.
        
        The number of sampled indices n can either be equal, less than or more than
        the length of the input array:
            - if n is equal to the length of idx, the output array is a shuffled version of idx,
            - if n is less than the length of idx, the output array is sampled from the index
              without replacement,
            - if n is more than the length of idx, the output array is created by first tiling
              the index array as much as possible, and then randomly selecting the rest without
              replacement. This ensures that each different index is used as much as possible.

        Args:
            n (int): The number of indices to sample.
            idx (np.ndarray): The array of indices to sample from.

        Returns:
            np.ndarray: A shuffled array of indices of size n.
        """
        
        # Calculate the number of repetitions and the remainder: n = a * len(idx) + b
        # For example, if n = 430 and len(idx) = 200, then a = 2 and b = 30
        a = n // len(idx)
        b = n % len(idx)

        # Create the repeated part and the remainder part
        out_a = np.tile(idx, a)
        out_b = np.random.choice(idx, b, replace=False)

        # Concatenate the repeated and remainder parts
        out = np.concatenate([out_a, out_b])

        # Shuffle the final array to ensure randomness
        np.random.shuffle(out)

        return out
    
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
            states = self.h5['states'][:]
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
        delta = 10 * (np.max(volumes) + np.max(p_c))
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
        states = self.h5['states'][:]
        
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

    @staticmethod
    def smooth_n_correlate(
        x: np.ndarray, y: np.ndarray,
        x_name: str, y_name: str,
        index: Index, window: int = 12
    ) -> None:
        x_ = smooth(x, index.chromstr, window)
        y_ = smooth(y, index.chromstr, window)
        r = clean_pearsonr(x_, y_)
        print(f"Pearson r between {x_name} and {y_name} after smoothing: {r}")


def GMM_solve(n, f, p = None, eps = None, beta = None):
    """ Implements the solutions of the Generalized Method of Moments (GMM)
    for the statistical model underlying the SimulatedRepliSeq class.
    
    Depending on the input parameters, it uses different equations.
    
    The shape of the output will match the input one.

    Args:
        n: average number of spots.
        f: fraction of zeros.
        p: replication probability.
        eps: detection efficiency.
        beta: overcounting bias.

    Returns:
        Depending on the input parameters, it returns:
            - eps, beta: if p is provided,
            - p, beta: if eps is provided,
            - p, eps: if beta is provided,
            - p: if eps and beta are provided.
    """
    
    # 1) KNOWN: P, GET: EPS, BETA
    if p is not None and eps is None and beta is None:
        
        # Get eps (depends on G1, G2 or S)
        if p == 'G1':
            p = 0
            eps = 1 - f
        elif p == 'G2':
            p = 1
            eps = 1 - f ** 0.5
        else:  # S
            eps = (1 + p - np.sqrt((1 + p) ** 2 - 4 * p * (1 - f))) / (2 * p)
        
        # Get beta
        beta = n / ((1 + p) * eps)
        
        return eps, beta

    # 2) KNOWN: EPS, GET: P, BETA
    elif eps is not None and p is None and beta is None:
        
        # Get p
        p = (1 - eps - f) / (eps * (1 - eps))
        
        # Get beta
        beta = n / ((1 + p) * eps)
        
        return p, beta
    
    # 3) KNOWN: BETA, GET: P, EPS
    elif beta is not None and p is None and eps is None:
        
        # Get eps
        d = n / beta
        eps = (d / 2) * (1 + np.sqrt(1 - 4 * (f + d - 1) / d ** 2))
        
        # We can see from equations that the argument of the square root
        # becomes negative for G2 cells. So for these cases we use the G2 formula
        eps = np.where(np.isnan(eps), 1 - f ** 0.5, eps)
        
        # Get p
        p = n / (eps * beta) - 1
        
        return p, eps
    
    # 4) KNOWN: EPS, BETA, GET: P
    elif eps is not None and beta is not None and p is None:
        
        # The system is overdetermined, so there are two solutions
        p1 = n / (eps * beta) - 1
        p2 = (1 - eps - f) / (eps * (1 - eps))
        
        # If the type is a numpy array, just return the average of the two
        if isinstance(n, np.ndarray):
            return (p1 + p2) / 2
        
        # Otherwise, use a numerical method to minimize the error from both solutions
        def root_func(x: float, p1: float, p2: float) -> float:
            return np.sqrt((x - p1) ** 2 + (x - p2) ** 2)
        f = partial(root_func, p1=p1, p2=p2)
        p = minimize(f, (p1 + p2) / 2).x[0]
        return p


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
