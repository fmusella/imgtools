import os
import numpy as np
import h5py
from alabtools.utils import Index
from ..scf import SingleCellFeature
from ..scf import scf_utils
from ..utils import resample_array, clip_array
from .GMM_solver import GMM_solve
from . import parallelize_features


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
        4. Cell-dependent analysis.
    By feature run, we mean that we calculate the average p, eps, beta for each quantized interval of a feature,
    for example Speckle distance.
    
    The object can be saved and loaded with an HDF5 file.
    
    The object has two methods to perfmorm additional analyses with more advanded biological interpretations,
    'calculate_repliprob_by_feat_loci' and 'calculate_repliprob_by_bootstrap'.
    
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
            - ncells (int): number of cells.
            - nloci (int): number of loci.
            - ncopies (int): number of copies.
            - G1s (np.ndarray): mask for G1 cells. shape: (ncells,).
            - G2s (np.ndarray): mask for G2 cells. shape: (ncells,).
            - Ss (np.ndarray): mask for S cells. shape: (ncells,).
            - loaded (bool): whether the data has been loaded.
                This boolean is used to avoid loading the data again.
    """
    
    
    # INITIALIZATION / INPUT-OUTPUT METHODS
    
    def __init__(self, h5_name: str, mode: str = 'r', scf: SingleCellFeature = None) -> None:
        """ Initialize the SimulatedRepliSeqExperiment object.
        
        There are two ways to initialize the object:
            - from a previously saved HDF5 file,
            - from a SingleCellFeature object. A new HDF5 file is created.

        Args:
            h5_name (str)
            mode (str, optional): Mode to open the HDF5 file. Defaults to 'r'.
            scf (SingleCellFeature, optional)
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
            self.h5 = self.from_scf(scf)
            return
        
        # Otherwise, read the HDF5 file
        self.h5 = h5py.File(h5_name, mode=mode)
    
    def from_scf(self, scf: SingleCellFeature) -> h5py.File:
        """ Creates a new HDF5 file from a SingleCellFeature object.
        
        The data stored are:
            - the index of the SCF,
            - the cell states,
            - the cell volumes,
            - the spotcount data (N),
        
        We curate the missing chromosomes in the spotcount data.

        Args:
            scf (SingleCellFeature)

        Returns:
            h5py.File: initialized HDF5 file for the SimulatedRepliSeqExperiment.
        """
        
        # Check that the SCF file is consistent
        self._check_scf(scf)
        
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
        scf_utils.curate_missing_chromosomes(N, scf.index)
        # Save the spotcount data
        h5.create_dataset('N', data=scf.get_feature('spotcount'))
        
        return h5
    
    @staticmethod
    def _check_scf(scf: SingleCellFeature) -> None:
        """ Check the input SingleCellFeature object.
        
        It checks that:
         - the input is a SingleCellFeature object,
         - the SCF contains the 'spotcount' feature,
         - the SCF contains the 'cell_states' feature,
         - the 'cell_states' feature only contains 'G1', 'S' and 'G2',
         - the SCF contains the 'volumes' feature,

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
    
    def close(self) -> None:
        """ Close the HDF5 file. """
        self.h5.close()
    

    # RUN METHODS
    
    def run(self, overwrite: bool = False, schedule: list = ['#'], scf: SingleCellFeature = None, nquants: int = 20) -> None:
        """ Run the simulated Repli-Seq experiment.
        
        Perform the analysis in the following steps:
            1. Population-wide analysis.
            2. Feature-dependent analyses.
            3. Locus-dependent analysis.
            4. Cell-dependent analysis.
        
        If the key 'overwrite' is True, the previous results are deleted,
        otherwise previously-done runs are skipped.
        
        The argument 'schedule' is a list specifying which runs to perform
        and which to skip ('#' performs all the runs).
        
        The results are stored in the object's HDF5 file.
        
        Args:
            overwrite (bool, optional): whether to overwrite previous results. Defaults to False.
            schedule (list, optional): list of runs to perform. Defaults to ['#'].
            scf (SingleCellFeature, optional): SCF object. Required for the feature-dependent analysis. Defaults to None.
            nquants (int, optional): number of quantiles for the feature-dependent analysis. Defaults to 20.
        """
        
        # Check the schedule. The accepted runs are:
        accepted_schedule = [
            'population_run',
            'feat_run',
            'locus_run',
            'cell_run',
        ]
        # If the schedule only contains '#', perform all runs
        if schedule == ['#']:
            schedule = accepted_schedule
        # Check that runs in the schedule are accepted
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
            if scf is None:
                raise ValueError("The SCF object must be provided for the feature-dependent analysis.")
            self.feat_run(scf, nquants, overwrite)
            
        # Locus-dependent analysis
        if 'locus_run' in schedule:
            if 'locus_run' not in self.h5 or overwrite:
                self.locus_run()
        
        # Cell-dependent analysis
        if 'cell_run' in schedule:
            if 'cell_run' not in self.h5 or overwrite:
                self.cell_run()
    
    def _load_to_memory(self):
        """ Load the data from the HDF5 file to memory.
        It's done to avoid reading the file every time we access the data.
        Adds also an attribute 'loaded' so that, if the data is already loaded,
        we don't do it again.
        """
        
        # Check if the data has already been loaded
        if hasattr(self, 'loaded') and self.loaded:
            return
        
        # Add the loaded attribute, so that we don't load the data again
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
    
    def population_run(self) -> None:
        """ Run the population-wide analysis.
        Separately for G1, S, G2, it combines the data from all cells and loci to estimate average values.
        In S phase, since there are two equations and three unknowns, we assume that the efficiency is the average
        of G1 and G2.
        Estimates:
            - nsamples_G1, number of samples in G1. int,
            - eps_G1, detection efficiency in G1. float,
            - eps_G1_err, error in eps_G1. float,
            - beta_G1, bias rate in G1. float,
            - beta_G1_err, error in beta_G1. float,
            
            - nsamples_G2, number of samples in G2. int,
            - eps_G2, detection efficiency in G2. float,
            - eps_G2_err, error in eps_G2. float,
            - beta_G2, bias rate in G2. float,
            - beta_G2_err, error in beta_G2. float,
            
            - nsamples_S, number of samples in S. int,
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
        # G1
        group.create_dataset('nsamples_G1', data=stat['G1']['nsamples'])
        group.create_dataset('eps_G1', data=eps_G1)
        group.create_dataset('eps_G1_err', data=eps_G1_err)
        group.create_dataset('beta_G1', data=beta_G1)
        group.create_dataset('beta_G1_err', data=beta_G1_err)
        # G2
        group.create_dataset('nsamples_G2', data=stat['G2']['nsamples'])
        group.create_dataset('eps_G2', data=eps_G2)
        group.create_dataset('eps_G2_err', data=eps_G2_err)
        group.create_dataset('beta_G2', data=beta_G2)
        group.create_dataset('beta_G2_err', data=beta_G2_err)
        # S
        group.create_dataset('nsamples_S', data=stat['S']['nsamples'])
        group.create_dataset('eps_S', data=eps_S)
        group.create_dataset('eps_S_err', data=eps_S_err)
        group.create_dataset('beta_S', data=beta_S)
        group.create_dataset('beta_S_err', data=beta_S_err)
        group.create_dataset('p_S', data=p_S)
        group.create_dataset('p_S_err', data=p_S_err)
        
        print('OVER.')
        print('\n\n')
    
    def feat_run(self, scf: SingleCellFeature, nquants: int, overwrite: bool) -> None:
        """ Run the feature-dependent analysis in parallel.
        
        The parallelization code is in the 'parallelize_features' module,
        where each feature in the SCF file is run in parallel.
        
        Results are stored in the HDF5 file:
          - group 'feat_run' contains a subgroup for each feature,
          - each subgroup 'feat' contains a subgroup for each nquants,
          - each nquant subgroup contains the results of the feature-dependent analysis:
                - nsamples_q_G1, number of samples in G1. shape: (nquants),
                - eps_q_G1, detection efficiency in G1. shape: (nquants),
                - eps_q_G1_err, error in eps_q_G1. shape: (nquants),
                - beta_q_G1, bias rate in G1. shape: (nquants),
                - beta_q_G1_err, error in beta_q_G1. shape: (nquants),
                
                - nsamples_q_G2, number of samples in G2. shape: (nquants),
                - eps_q_G2, detection efficiency in G2. shape: (nquants),
                - eps_q_G2_err, error in eps_q_G2. shape: (nquants),
                - beta_q_G2, bias rate in G2. shape: (nquants),
                - beta_q_G2_err, error in beta_q_G2. shape: (nquants),
                
                - nsamples_q_S, number of samples in S. shape: (nquants),
                - eps_q_S, detection efficiency in S. shape: (nquants),
                - eps_q_S_err, error in eps_q_S. shape: (nquants),
                - beta_q_S, bias rate in S. shape: (nquants),
                - beta_q_S_err, error in beta_q_S. shape: (nquants),
                - p_q_S, replication probability in S. shape: (nquants),
                - p_q_S_err, error in p_q_S. shape: (nquants).

        Args:
            scf (SingleCellFeature)
            nquants (int): number of quantiles for the feature.
            overwrite (bool): whether to overwrite previous results.
        """
        
        print('LOCUS-DEPENDENT RUN')
        print('-------------------')
        
        print(f'   Number of features: {len(scf.feature_list)}')
        print('   Submitting the calculation in parallel...')
        
        # Set the configuration for the parallelization
        config = {
            'nquants': nquants,
            'parallel': {'controller': 'ipyparallel'}
        }
        
        # Run the calculation in parallel for all features
        # result is a dictionary with eps, beta, p, their errors, for G1, G2 and S
        result = parallelize_features.control_func(scf, config)
        
        # Store the results in the HDF5 file
        # Create a group for the feature run
        group = self.h5.require_group('feat_run')
        
        # Loop over the features
        for feat, feat_result in result.items():
            
            # Create a subgroup for the feature
            feat_subgroup = group.require_group(feat)
            
            # If the feat subgroup already has a subsubgroup for the nquants AND overwrite is False, skip
            if not overwrite and str(nquants) in feat_subgroup:
                continue
            # Otherwise, if the subgroup already exists AND overwrite is True, delete it
            if overwrite and str(nquants) in feat_subgroup:
                del feat_subgroup[str(nquants)]
                
            # Create a subgroup for the nquants
            nquant_subgroup = feat_subgroup.create_group(str(nquants))
            
            # Store the results in the subgroup (eps_q_G1, eps_q_G1_err, ...)
            for key, value in feat_result.items():
                nquant_subgroup.create_dataset(key, data=value)
        
        print('OVER.')
        print('\n\n')
    
    def locus_run(self) -> None:
        """ Run the locus-dependent analysis.
        Treats each locus independently, assuming that different cells are independent realizations
        of the same locus-dependent proces (separately for G1, S and G2).
        In S phase, since there are two equations and three unknowns, we assume that the efficiency
        signal is the locus-dependent average of G1 and G2.
        Estimates:
            - nsamples_i_G1, number of samples in G1. shape: (nloci),
            - eps_i_G1, detection efficiency in G1. shape: (nloci),
            - eps_i_G1_err, error in eps_i_G1. shape: (nloci),
            - beta_i_G1, bias rate in G1. shape: (nloci),
            - beta_i_G1_err, error in beta_i_G1. shape: (nloci),
            
            - nsamples_i_G2, number of samples in G2. shape: (nloci),
            - eps_i_G2, detection efficiency in G2. shape: (nloci),
            - eps_i_G2_err, error in eps_i_G2. shape: (nloci),
            - beta_i_G2, bias rate in G2. shape: (nloci),
            - beta_i_G2_err, error in beta_i_G2. shape: (nloci),
            
            - nsamples_i_S, number of samples in S. shape: (nloci),
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
        eps_i_G1 = clip_array(eps_i_G1, 0, 1)
        eps_i_G2 = clip_array(eps_i_G2, 0, 1)
        beta_i_G1 = clip_array(beta_i_G1, 0, None)
        beta_i_G2 = clip_array(beta_i_G2, 0, None)

        # Assume that the efficiency in S is the average of G1 and G2
        eps_i_S = (eps_i_G1 + eps_i_G2) / 2
        eps_i_S_err = np.sqrt(eps_i_G1_err ** 2 + eps_i_G2_err ** 2) / 2
        
        # Calculate replication probability and bias in S
        p_i_S, beta_i_S, p_i_S_err, beta_i_S_err = GMM_solve(stat['S'], eps=eps_i_S, eps_err=eps_i_S_err)
        p_i_S = clip_array(p_i_S, 0, 1)
        beta_i_S = clip_array(beta_i_S, 0, None)
        
        # Store the results
        group = self.h5.create_group('locus_run')
        # G1
        group.create_dataset('nsamples_i_G1', data=stat['G1']['nsamples'])
        group.create_dataset('eps_i_G1', data=eps_i_G1)
        group.create_dataset('eps_i_G1_err', data=eps_i_G1_err)
        group.create_dataset('beta_i_G1', data=beta_i_G1)
        group.create_dataset('beta_i_G1_err', data=beta_i_G1_err)
        # G2
        group.create_dataset('nsamples_i_G2', data=stat['G2']['nsamples'])
        group.create_dataset('eps_i_G2', data=eps_i_G2)
        group.create_dataset('eps_i_G2_err', data=eps_i_G2_err)
        group.create_dataset('beta_i_G2', data=beta_i_G2)
        group.create_dataset('beta_i_G2_err', data=beta_i_G2_err)
        # S
        group.create_dataset('nsamples_i_S', data=stat['S']['nsamples'])
        group.create_dataset('eps_i_S', data=eps_i_S)
        group.create_dataset('eps_i_S_err', data=eps_i_S_err)
        group.create_dataset('beta_i_S', data=beta_i_S)
        group.create_dataset('beta_i_S_err', data=beta_i_S_err)
        group.create_dataset('p_i_S', data=p_i_S)
        group.create_dataset('p_i_S_err', data=p_i_S_err)
        
        print('OVER.')
        print('\n\n')

    def cell_run(self) -> None:
        """ Run the cell-dependent analysis.
        Treats each cell independently, assuming that different loci are independent realizations
        of the same cell-dependent process.
        To estimate the single cell efficiency, we first solve for each cell in G1 and G2.
        Then we assume that each cell has the same efficiency (average of cells) and 
        the error is the standard deviation among cells.
        Estimates:
            - nsamples_c, number of samples in each cell. shape: (ncells),
            - eps_c, detection efficiency. shape: (ncells),
            - eps_c_err, error in eps_c. shape: (ncells),
            - eps_c_, detection efficiency for single cells in G1 and G2. shape: (ncells),
            - eps_c_err_, error in eps_c_. shape: (ncells),
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
        
        # Calculate the efficiencies for each single cell in G1 and G2
        eps_c_ = np.full(self.ncells, np.nan)
        eps_c_err_ = np.full(self.ncells, np.nan)
        for state in ['G1', 'G2']:
            mask = self.states == state
            eps_c_s, _, eps_c_s_err, _ = GMM_solve(stat[state], p=state)
            eps_c_[mask] = eps_c_s
            eps_c_err_[mask] = eps_c_s_err
                
        # Mask cells with nuclear volume > 400 um^3
        # Smaller nuclei have a much lower efficiency,
        # and it doesn't generalize well to S-phase
        volumes_mask = self.volumes > 400  # 400 um^3
        volumes_mask_G1 = volumes_mask[self.G1s]
        volumes_mask_G2 = volumes_mask[self.G2s]
        
        # Calculate average and standard deviation for G1 and G2
        eps_G1 = np.mean(eps_c_[self.G1s][volumes_mask_G1])
        eps_G2 = np.mean(eps_c_[self.G2s][volumes_mask_G2])
        eps_S = (eps_G1 + eps_G2) / 2
        eps_G1_err = np.std(eps_c_[self.G1s][volumes_mask_G1], ddof=1)
        eps_G2_err = np.std(eps_c_[self.G2s][volumes_mask_G2], ddof=1)
        eps_S_err = np.sqrt(eps_G1_err ** 2 + eps_G2_err ** 2) / 2
        
        # Now construct the efficiency array with the averages
        eps_c = np.full(self.ncells, np.nan)
        eps_c[self.G1s] = eps_G1
        eps_c[self.G2s] = eps_G2
        eps_c[self.Ss] = eps_S
        # And we use the standard deviations as errors
        eps_c_err = np.full(self.ncells, np.nan)
        eps_c_err[self.G1s] = eps_G1_err
        eps_c_err[self.G2s] = eps_G2_err
        eps_c_err[self.Ss] = eps_S_err
        
        # Calculate the replication probability and bias per cell
        p_c, beta_c, p_c_err, beta_c_err = GMM_solve(stat['G1SG2'], eps=eps_c, eps_err=eps_c_err)
        
        # Clip arrays
        eps_c = clip_array(eps_c, 0, 1)
        beta_c = clip_array(beta_c, 0, None)
        p_c = clip_array(p_c, 0, 1)
         
        # Store the results
        group = self.h5.create_group('cell_run')
        group.create_dataset('nsamples_c', nsamples)
        group.create_dataset('eps_c', data=eps_c)
        group.create_dataset('eps_c_err', data=eps_c_err)
        group.create_dataset('eps_c_', data=eps_c_)
        group.create_dataset('eps_c_err_', data=eps_c_err_)
        group.create_dataset('beta_c', data=beta_c)
        group.create_dataset('beta_c_err', data=beta_c_err)
        group.create_dataset('p_c', data=p_c)
        group.create_dataset('p_c_err', data=p_c_err)
        
        print('OVER.')
        print('\n\n')
    
    
    def calculate_repliprob_by_mask(self, M: np.ndarray, S_stage: tuple = None) -> dict:
        """ Calculate the replication probability for a given mask of loci / cells / copies.
        
        If there are no cells in the state, the function returns None.

        Args:
            M (np.ndarray): boolean mask array. shape: (ncells, nloci, ncopies).
            S_stage (tuple, optional): tuple with the start and end of the S stage p_c to use.
                Defaults to None.

        Returns:
            (If there are no cells in the state, returns None.)
            dict: dictionary with the keys:
                - 'nsamples_G1', number of samples in G1. int,
                - 'eps_G1', detection efficiency in G1. float,
                - 'eps_G1_err', error in eps_G1. float,
                - 'beta_G1', bias rate in G1. float,
                - 'beta_G1_err', error in beta_G1. float,
                
                - 'nsamples_G2', number of samples in G2. int,
                - 'eps_G2', detection efficiency in G2. float,
                - 'eps_G2_err', error in eps_G2. float,
                - 'beta_G2', bias rate in G2. float,
                - 'beta_G2_err', error in beta_G2. float,
                
                - 'nsamples_S', number of samples in S. int,
                - 'eps_S', detection efficiency in S. float,
                - 'eps_S_err', error in eps_S. float,
                - 'beta_S', bias rate in S. float,
                - 'beta_S_err', error in beta_S. float,
                - 'p_S', replication probability in S. float,
                - 'p_S_err', error in p_S. float,
                - 't_S', average pseudo-time of S-phase cells selected. float,
                - 't_S_std', standard deviation of pseudo-time of S-phase cells selected. float.
        """
        
        # Load the data to memory
        self._load_to_memory()
        
        # Check that the M mask array has the same shape as N and it's boolean
        if not M.shape == self.N.shape:
            raise ValueError("The mask M must have the same shape as N.")
        if not np.issubdtype(M.dtype, np.bool_):
            raise ValueError("The mask M must be boolean.")
               
        # If the cell run hasn't been performed yet, raise an error
        if 'cell_run' not in self.h5:
            raise RuntimeError("The cell_run must be performed before calculating the replication probability by mask.")
        # Get the cell pseud-time
        p_c = self.h5['cell_run']['p_c'][:]  # shape: (ncells,)
        
        # Initialize the summary statistics dictionary
        stat = {}
        
        # Loop over the states
        for s in ['G1', 'S', 'G2']:
            
            # Get the mask for the state
            mask_state = self.states == s
            
            # If the state is S AND the S_stage is provided, filter the S cells in the S_stage
            if s == 'S' and S_stage is not None:
                mask_state = np.logical_and(mask_state, np.logical_and(p_c > S_stage[0], p_c < S_stage[1]))
            
            # If there are no cells in the state, exit the function returning None
            if not np.any(mask_state):
                return None
            
            # Mask for the state
            N_s = self.N[mask_state, :, :]  # shape: (ncells_s, nloci, ncopies)
            M_s = M[mask_state, :, :]  # shape: (ncells_s, nloci, ncopies)
            
            # Get the data in the mask
            N_s = N_s[M_s]  # shape: (1D_collapsed,)
            
            # Create a zero-indicator version of N_s: 1 if n = 0, 0 otherwise
            B_s = (N_s == 0).astype(float)
            B_s[np.isnan(N_s)] = np.nan
            
            # Calculate average/std
            nsamples = np.sum(~np.isnan(N_s))  # int
            stat[s] = {
                'nsamples': nsamples,
                'n': np.nanmean(N_s),
                'n_var': np.nanvar(N_s, ddof=1) / nsamples,
                'f': np.nanmean(B_s),
                'f_var': np.nanvar(B_s, ddof=1) / nsamples,
                'nf_cov': - np.nanmean(N_s) * np.nanmean(B_s) / nsamples
            }
            
            # If the state is S, also store the average/std of the pseudo-times of the S cells
            if s == 'S':
                M_c_s = np.any(M_s, axis=(1, 2))  # shape: (ncells_state,)
                p_c_s = p_c[mask_state]  # shape: (ncells_state,)
                stat[s]['t_S'] = np.nanmean(p_c_s[M_c_s])  # avg pseudo-time
                stat[s]['t_S_std'] = np.nanstd(p_c_s[M_c_s], ddof=1)  # std pseudo-time
            
        # Calculate efficiency and bias in G1 and G2
        eps_G1, beta_G1, eps_G1_err, beta_G1_err = GMM_solve(stat['G1'], p='G1')
        eps_G2, beta_G2, eps_G2_err, beta_G2_err = GMM_solve(stat['G2'], p='G2')
        eps_G1 = clip_array(eps_G1, 0, 1)
        eps_G2 = clip_array(eps_G2, 0, 1)
        beta_G1 = clip_array(beta_G1, 0, None)
        beta_G2 = clip_array(beta_G2, 0, None)
        
        # We assume that the efficiency in S is the average of G1 and G2
        eps_S = (eps_G1 + eps_G2) / 2
        eps_S_err = np.sqrt(eps_G1_err ** 2 + eps_G2_err ** 2) / 2
        
        # Calculate replication probability and bias in S
        p_S, beta_S, p_S_err, beta_S_err = GMM_solve(stat['S'], eps=eps_S, eps_err=eps_S_err)
        p_S = clip_array(p_S, 0, 1)
        beta_S = clip_array(beta_S, 0, None)
        
        results = {
            # G1
            'nsamples_G1': stat['G1']['nsamples'],
            'eps_G1': eps_G1, 'eps_G1_err': eps_G1_err,
            'beta_G1': beta_G1, 'beta_G1_err': beta_G1_err,
            # G2
            'nsamples_G2': stat['G2']['nsamples'],
            'eps_G2': eps_G2, 'eps_G2_err': eps_G2_err,
            'beta_G2': beta_G2, 'beta_G2_err': beta_G2_err,
            # S
            'nsamples_S': stat['S']['nsamples'],
            'eps_S': eps_S, 'eps_S_err': eps_S_err,
            'beta_S': beta_S, 'beta_S_err': beta_S_err,
            'p_S': p_S, 'p_S_err': p_S_err,
            't_S': stat['S']['t_S'], 't_S_std': stat['S']['t_S_std']
        }
        return results
    
    
    def calculate_repliprob_by_feat_loci(
        self, scf: SingleCellFeature, nquants: int, S_stage: tuple, loci: np.ndarray, reweighting: bool = False
    ) -> dict:
        """ Calculate the replication probability for a given mask of loci, stratified by quantiles of a feature.
        
        The calculation is done in parallel for all features in the SCF file, same as 'feat_run'.
        
        Results are returned in a dictionary format.

        Args:
            scf (SingleCellFeature): SCF object.
            nquants (int): number of quantiles for the feature.
            S_stage (tuple, optional): minimum and maximum cell progression probabilities for S-phase. Defaults to (0., 1.).
            loci (np.ndarray): array of shape (nloci,) with the loci to calculate the replication probability.
            reweighting (bool, optional): whether to re-weight the G1/G2 efficiencies when calculating the S-phase efficiency.
                    If True, the S-phase efficiency is calculated by weighting the G1 and G2 efficiencies by the S-phase
                    replication probability obtained from the non-weighted approach. Defaults to False.

        Returns:
            dict: results of the calculation, with a key for each feature and the following sub keys
                  (each sub-key has an array of shape (nquants,) as value):
            
                    - 'nsamples_G1': number of samples in G1,
                    - 'eps_q_G1': efficiency in G1,
                    - 'eps_q_G1_err': error in eps_q_G1,
                    - 'beta_q_G1': bias in G1,
                    - 'beta_q_G1_err': error in beta_q_G1,
                    
                    - 'nsamples_G2': number of samples in G2,
                    - 'eps_q_G2': efficiency in G2,
                    - 'eps_q_G2_err': error in eps_q_G2,
                    - 'beta_q_G2': bias in G2,
                    - 'beta_q_G2_err': error in beta_q_G2,
                    
                    - 'nsamples_S': number of samples in S,
                    - 'eps_q_S': efficiency in S,
                    - 'eps_q_S_err': error in eps_q_S,
                    - 'beta_q_S': bias in S,
                    - 'beta_q_S_err': error in beta_q_S,
                    - 'p_q_S': replication probability in S,
                    - 'p_q_S_err': error in p_q_S.
        """
        
        # Define the configuration and the arrays to store in the parallel temporary directory
        config = {
            'nquants': nquants,
            'S_stage': S_stage,
            're-weighting': reweighting,
            'parallel': {'controller': 'ipyparallel'}
        }
        arrays = {'loci': loci, 'p_c': self.h5['cell_run']['p_c'][:]}
        
        # Run the calculation in parallel for all features
        result = parallelize_features.control_func(scf, config, arrays)
        
        # The results are stored in a dictionary, we just return it
        return result
    

    def calculate_repliprob_by_bootstrap(self, mask: np.ndarray, nrepeat: int = 1) -> tuple:
        """
        Calculates the replication probability for a given loci mask using a bootstrap approach.
        
        mask is a boolean numpy array of shape (ncells, ndomains, ncopies),
        indicating for which loci in which cells we have to calculate the
        replication probability.
        
        The function does the following:
            - makes sure that the mask only contains True values for S cells,
            - estimates eps from G1 and G2 by bootstrapping,
            - corrects eps with cell-dependent estimates,
            - calculates the replication probability for the original mask,
            - the process can be repeated multiple times to get a more robust estimate.
        
        The input feature specifies which feature to use for the correction.
            
        Returns two lists, one containing the replication probabilities and the other
        containing the errors in the replication probabilities.

        Args:
            mask (np.ndarray): A boolean numpy array of shape (ncells, ndomains, ncopies).
            nrepeat (int): The number of times the process is repeated.

        Returns:
            p_Ss (list): A list of length nrepeat containing the replication probabilities.
            p_S_errs (list): A list of length nrepeat containing the errors in the replication probabilities.
        """
        
        # Load the data from the HDF5 file into memory
        self._load_to_memory()
        
        # Get the target cells, i.e. those with at least one locus present in the mask
        tcells = np.where(np.sum(mask, axis=(1, 2)) > 0)[0]  # shape: (ntcells), dtype: int
        
        # Make sure that the target cells are all S
        if not np.all(self.states[tcells] == 'S'):
            raise ValueError('The target cells must be all in S phase.')

        # Now we estimate eps in G1 and G2 by bootstrapping
        # We repeat this process nrepeat times to get a more robust estimate
        G1G2_results = {
            'tcells_G1': [],
            'tcells_G2': [],
            'eps_G1': [],
            'eps_G2': [],
            'eps_G1_err': [],
            'eps_G2_err': [],
        }
        for r in range(nrepeat):
            G1G2_results = self.bootstrap_G1G2(tcells, mask, G1G2_results)
        
        # Calculate the replication probability for the target S cells
        # using the estimates from G1 and G2
        
        # First we need to calculate the summary statistics for the target S cells
        N_S = self.N[mask]
        B_S = (N_S == 0).astype(float)
        B_S[np.isnan(N_S)] = np.nan
        nsamples = np.sum(~np.isnan(N_S))  # int
        n = np.nanmean(N_S)  # float
        f = np.nanmean(B_S)
        stat_S = {
            'nsamples': nsamples,
            'n': n,
            'n_var': np.nanvar(N_S, ddof=1) / nsamples,
            'f': f,
            'f_var': np.nanvar(B_S, ddof=1) / nsamples,
            'nf_cov': - n * f / nsamples
        }
        
        # Initialize the list of inferred replication probabilities and errors
        p_Ss, p_S_errs = [], []
        
        # Loop over the repetitions
        for r in range(nrepeat):
            
            # Get the G1G2 randomization results for the current repetition
            tcells_G1 = G1G2_results['tcells_G1'][r]
            tcells_G2 = G1G2_results['tcells_G2'][r]
            eps_G1 = G1G2_results['eps_G1'][r]
            eps_G2 = G1G2_results['eps_G2'][r]
            eps_G1_err = G1G2_results['eps_G1_err'][r]
            eps_G2_err = G1G2_results['eps_G2_err'][r]
            
            # Correct the efficiency and bias with the cell-dependent estimates
            eps_c = self.h5['cell_run']['eps_c'][:]
            eps_G1 = eps_G1 + np.nanmean(eps_c[tcells]) - np.nanmean(eps_c[tcells_G1])
            eps_G2 = eps_G2 + np.nanmean(eps_c[tcells]) - np.nanmean(eps_c[tcells_G2])
            
            # Assign eps_S and beta_S as the average of G1 and G2
            eps_S = (eps_G1 + eps_G2) / 2
            eps_S_err = np.sqrt(eps_G1_err**2 + eps_G2_err**2) / 2
            
            # Calculate the replication probability in S
            p_S, _, p_S_err, _ = GMM_solve(stat_S, eps=eps_S, eps_err=eps_S_err)
            p_Ss.append(p_S)
            p_S_errs.append(p_S_err)
        
        return p_Ss, p_S_errs
    
    def bootstrap_G1G2(self, tcells: np.ndarray, mask: np.ndarray, G1G2_results: dict) -> tuple:
        """ Estimate the efficiency in G1/G2 given the current S-phase mask by
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
                - eps_G1_err (list): A list of the errors in the estimated efficiencies for G1.
                - eps_G2_err (list): A list of the errors in the estimated efficiencies for G2.

        Returns:
            G1G2_results (dict): The updated dictionary containing the results of the current randomization.
        """
        
        # We ignore cells with a nuclear volume less than a threshold
        # These cells have a very small efficiency that doesn't
        # generalize well to S cells
        vols_mask = self.volumes > 400
        
        # Randomly select target cells from G1 and G2,
        # resampling them to have the same number of cells as S
        tcells_G1 = resample_array(len(tcells), np.where(np.logical_and(self.G1s, vols_mask))[0])
        tcells_G2 = resample_array(len(tcells), np.where(np.logical_and(self.G2s, vols_mask))[0])
        
        # Map these cells to the target S cells
        map_G1 = {cS: cG1 for cS, cG1 in zip(tcells, tcells_G1)}
        map_G2 = {cS: cG2 for cS, cG2 in zip(tcells, tcells_G2)}
        
        # Initialize the flat arrays to store the G1, G2 randomized data
        N_G1, N_G2 = np.array([]), np.array([])
        
        # Loop over the target cells
        for c in tcells:
            cG1, cG2 = map_G1[c], map_G2[c]
            
            # Get the mask to apply from the S cell
            mask_c = mask[c, :, :]  # shape: (ndomains, ncopies)
            
            # Apply the mask to the N matrices for cG1 and cG2
            N_G1 = np.concatenate((N_G1, self.N[cG1, mask_c]))
            N_G2 = np.concatenate((N_G2, self.N[cG2, mask_c]))
        
        # Calculate the zero-indicator matrices
        B_G1 = (N_G1 == 0).astype(float)
        B_G2 = (N_G2 == 0).astype(float)
        B_G1[np.isnan(N_G1)] = np.nan
        B_G2[np.isnan(N_G2)] = np.nan
        
        # Calculate the summary statistics
        nsamples_G1 = np.sum(~np.isnan(N_G1))
        n_G1 = np.nanmean(N_G1)
        f_G1 = np.nanmean(B_G1)
        stat_G1 = {
            'nsamples': nsamples_G1,
            'n': n_G1,
            'n_var': np.nanvar(N_G1, ddof=1) / nsamples_G1,
            'f': f_G1,
            'f_var': np.nanvar(B_G1, ddof=1) / nsamples_G1,
            'nf_cov': - n_G1 * f_G1 / nsamples_G1
        }
        nsamples_G2 = np.sum(~np.isnan(N_G2))
        n_G2 = np.nanmean(N_G2)
        f_G2 = np.nanmean(B_G2)
        stat_G2 = {
            'nsamples': nsamples_G2,
            'n': n_G2,
            'n_var': np.nanvar(N_G2, ddof=1) / nsamples_G2,
            'f': f_G2,
            'f_var': np.nanvar(B_G2, ddof=1) / nsamples_G2,
            'nf_cov': - n_G2 * f_G2 / nsamples_G2
        }
        
        # Calculate the efficiency in G1 and G2
        eps_G1, _, eps_G1_err, _ = GMM_solve(stat_G1, p='G1')
        eps_G2, _, eps_G2_err, _ = GMM_solve(stat_G2, p='G2')
        eps_G1 = np.clip(eps_G1, 0, 1)
        eps_G2 = np.clip(eps_G2, 0, 1)
        
        # Append the results to the dictionary
        G1G2_results['tcells_G1'].append(tcells_G1)
        G1G2_results['tcells_G2'].append(tcells_G2)
        G1G2_results['eps_G1'].append(eps_G1)
        G1G2_results['eps_G2'].append(eps_G2)
        G1G2_results['eps_G1_err'].append(eps_G1_err)
        G1G2_results['eps_G2_err'].append(eps_G2_err)
        
        return G1G2_results
    
    
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
    
    def group_by_cellcycle(self, ngroups: int) -> np.ndarray:
        """ Group the S-phase cells into ngroups with equal number of cells.
        
        The groups are defined by increasing replication probability intervals.
        
        The function returns the quantiles that divide the S cells into ngroups.

        Args:
            ngroups (int): Number of S-phase groups to create.

        Returns:
            np.ndarray: Array of shape (ngroups, 2) with the quantiles that divide the S cells into ngroups.
        """
        
        try:
            states = self.h5['states'][:].astype(str)
        except KeyError:
            raise KeyError('The states array is not available.')
        try:
            p_c = self.h5['cell_run']['p_c'][:]
        except KeyError:
            raise KeyError('The p_c array is not available.')
        
        # Get p_c for S cells
        p_c = p_c[states == 'S']
        
        # Get the quantiles that divide the S cells into ngroups
        q = np.linspace(0, 1, ngroups + 1)
        p_c_quants = np.quantile(p_c, q)
        
        # Reshape the quantiles to a 2D array of shape (ngroups, 2)
        p_c_quants_reshape = []
        for i in range(len(p_c_quants) - 1):
            p_c_quants_reshape.append(p_c_quants[i:i + 2])
        p_c_quants_reshape = np.array(p_c_quants_reshape)
        
        return p_c_quants_reshape



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
