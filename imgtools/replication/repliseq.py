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
    
    This class aims to solve the equations, estimating the parameters p, eps and beta.
    The solution is done in several steps:
        1. Population-wide analysis.
        2. Z-dependent analysis.
        3. Locus-dependent analysis.
        4. Locus and z-dependent analysis.
        5. Cell-dependent analysis.
        6. Cell and z-dependent analysis.
        7. Sliding window analysis.
    These four steps introduce increasing complexity in the model, whereby the parameters
    are made dependent on locus, cell, z quantile, or combinations of these.
    
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
        G1s (np.ndarray): mask for G1 cells. shape: (ncells).
        G2s (np.ndarray): mask for G2 cells. shape: (ncells).
        Ss (np.ndarray): mask for S cells. shape: (ncells).
        volumes (np.ndarray): cell nuclear volumes of the SCF data. shape: (ncells).
        n_ic (np.ndarray): spotcount of the SCF, i.e. number of spots per cell and per locus. shape: (ncells, nloci, ncopies).
        z_ic (np.ndarray): z coordinate of the SCF. shape: (ncells, nloci, ncopies).
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
        self.G1s = None
        self.G2s = None
        self.Ss = None
        self.volumes = None
        self.n_ic = None
        self.z_ic = None
        self.rad_ic = None  # distance to the nuclear envelope
    
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
        obj.G1s = obj.states == 'G1'
        obj.G2s = obj.states == 'G2'
        obj.Ss = obj.states == 'S'
        obj.volumes = scf.volumes
        obj.n_ic = scf.get_feature('spotcount')
        if 'z' in scf:
            obj.z_ic = scf.get_feature('z')
        elif 'z_imputed' in scf:
            obj.z_ic = scf.get_feature('z_imputed')
        if 'envsurf' in scf:
            obj.rad_ic = scf.get_feature('envsurf')
        elif 'envsurf_imputed' in scf:
            obj.rad_ic = scf.get_feature('envsurf_imputed')
        obj.ncells, obj.nloci, obj.ncopies = obj.n_ic.shape
        
        return obj
    
    @staticmethod
    def _check_scf(scf: SingleCellFeature) -> None:
        """ Check the input SingleCellFeature object.
        
        It checks that:
         - the input is a SingleCellFeature object,
         - the SCF contains the 'spotcount' feature,
         - the SCF contains the 'z' feature or the 'z_imputed' feature,
         - the SCF contains the 'envsurf' feature or the 'envsurf_imputed' feature,
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
        if 'z' not in scf.feature_list and 'z_imputed' not in scf.feature_list:
            raise ValueError("The input scf must contain the 'z' feature.")
        if 'envsurf' not in scf.feature_list and 'envsurf_imputed' not in scf.feature_list:
            raise ValueError("The input scf must contain the 'envsurf' feature.")
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
                # If the values is a number (int or float), save as an attribute
                if isinstance(value, (int, float)):
                    f.attrs[key] = value
                # If it's an array, save as a dataset
                elif isinstance(value, np.ndarray):
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
            
            # Loop over the attributes of the file
            for key in f.attrs.keys():
                
                # Load the attributes as integers or floats
                self.__dict__[key] = f.attrs[key]
            
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
        """ Run the simulated Repli-Seq experiment.
        
        Perform the analysis in the following steps:
            1. Population-wide analysis.
            2. Z-dependent analysis.
            3. Radial-dependent analysis.
            4. Locus-dependent analysis.
            5. Locus and z-dependent analysis.
            6. Cell-dependent analysis.
            7. Cell and z-dependent analysis.
            8. Cell and radial-dependent analysis.
            9. Sliding window analysis.
        
        The results are stored in the object's attributes.
        
        Args:
            config (dict): configuration dictionary. Must contain the following keys:
                - sex (whether it's a male or a female cell),
                - nz (number of quantiles for the z coordinates),
                - nrad (number of quantiles for the radial distances),
                - sliding_window_size (size of the sliding window for the sliding window analysis).
        """
        # Set the config
        self._check_config(config)
        self.config = config
        # Prepare the data
        self.curate_missing_chromosomes()
        self.quantize_zcoords()
        self.quantize_rad()
        # Run the analysis
        self.population_run()
        self.z_run()
        self.rad_run()
        self.locus_run()
        self.locus_n_z_run()
        self.locus_n_rad_run()
        self.cell_run()
        self.cell_n_z_run()
        self.cell_n_rad_run()
        """self.complete_eps_beta()
        self.sliding_window_run()"""
        
    
    @staticmethod
    def _check_config(config: dict) -> None:
        """ Check the input config dictionary.
        
        It checks that the input is a dictionary and that it contains the required keys:
         - sex,
         - nz,
         - nrad,
         - sliding_window_size
         
        It also checks that the 'sex' key is a string and that it is either 'male' or 'female'.

        Args:
            config (dict)
        """
            
        if not isinstance(config, dict):
            raise TypeError("The input config must be a dictionary.")
        
        required_keys = [
            'sex',
            'nz',
            'nrad',
            'sliding_window_size',
        ]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing key '{key}' in the input config.")
        
        if not isinstance(config['sex'], str):
            raise TypeError(f"Input sex in config must be str. Got type {type(config['sex'])} instead.")
        if not config['sex'] in ['male', 'female']:
            raise ValueError(f"Input sex in config must be either 'male' or 'female'")
    
    def curate_missing_chromosomes(self) -> None:
        """ Set the spotcount matrices for missing chromosomal copies (i.e. all 0s) as NaN.
        """
        
        # Loop over cells
        for cellnum in range(self.ncells):
        
            # Loop over the chromosomes and mask them
            for chrom in self.index.genome.chroms:
                mask_chrom = self.index.chromstr == chrom  # shape: (nloci)
                
                # Loop over the copies
                for copynum in range(self.n_ic.shape[2]):
                    
                    # If the matrix of the cell/chrom/copy is made of only 0s, set it as NaN in the object
                    if np.all(self.n_ic[cellnum, mask_chrom, copynum] == 0):
                        self.n_ic[cellnum, mask_chrom, copynum] = np.nan
    
    def quantize_zcoords(self) -> None:
        """ Quantize the z coordinates of the SCF data.
        In each cell, the z coordinates are quantized into a fixed number of slices (given in the config).
        The quantized z coordinates are stored in the 'zq_ic' attribute.
        We also store the quantiles of the z coordinates in the 'zquants' attribute.
        """
        
        # Get the number of quantiles for the z coordinates
        nquants = self.config['nz']
        # Initialize the quantized z coordinates
        # We initialize with -1: the NaN values in z_ic will remain as -1
        zq_ic = np.full(self.z_ic.shape, -1)  # shape: (ncells, nloci, ncopies)
        
        # Loop over the cells
        for c in range(self.ncells):
            
            # Get the z coordinates for the cell
            z_c = self.z_ic[c, :, :]  # shape: (nloci, ncopies)
            
            # Initialize the quantized z coordinates for the cell
            zq_c = np.full(z_c.shape, -1)  # shape: (nloci, ncopies)
            
            # Get the z quantiles of the cell
            quants_c = np.nanquantile(z_c, np.linspace(0, 1, nquants + 1))  # shape: (nquants + 1)
            
            # Loop over the quantiles
            for q in range(nquants):
                # Get the mask for the quantile
                if q == nquants - 1:
                    mask_q = z_c >= quants_c[q]  # include the last value if it's the last quantile
                else:
                    mask_q = np.logical_and(z_c >= quants_c[q], z_c < quants_c[q + 1])
                # Assign the quantile to the quantized z coordinates
                zq_c[mask_q] = q
            
            # Store the quantized z coordinates for the cell
            zq_ic[c, :, :] = zq_c
        
        # Get the quantiles as an array
        zquants = np.arange(nquants)
        
        # Store the data
        self.zquants = zquants
        self.zq_ic = zq_ic
    
    def quantize_rad(self) -> None:
        """ Quantize the radial distances of the SCF data.
        In each cell, the radial distances are quantized into a fixed number of slices (given in the config).
        The quantized radial distances are stored in the 'radq_ic' attribute.
        We also store the quantiles of the radial distances in the 'radquants' attribute.
        """
        
        # Get the number of quantiles for the radial distances
        nquants = self.config['nrad']
        # Initialize the quantized radial distances
        # We initialize with -1: the NaN values in  will remain as -1
        radq_ic = np.full(self.rad_ic.shape, -1)  # shape: (ncells, nloci, ncopies)
        
        # Loop over the cells
        for c in range(self.ncells):
            
            # Get the radial distances for the cell
            rad_c = self.rad_ic[c, :, :]  # shape: (nloci, ncopies)
            
            # Initialize the quantized radial distances for the cell
            radq_c = np.full(rad_c.shape, -1)  # shape: (nloci, ncopies)
            
            # Get the radial quantiles of the cell
            quants_c = np.nanquantile(rad_c, np.linspace(0, 1, nquants + 1))  # shape: (nquants + 1)
            
            # Loop over the quantiles
            for q in range(nquants):
                # Get the mask for the quantile
                if q == nquants - 1:
                    mask_q = rad_c >= quants_c[q]  # include the last value if it's the last quantile
                else:
                    mask_q = np.logical_and(rad_c >= quants_c[q], rad_c < quants_c[q + 1])
                # Assign the quantile to the quantized radial distances
                radq_c[mask_q] = q
            
            # Store the quantized radial distances for the cell
            radq_ic[c, :, :] = radq_c
        
        # Get the quantiles as an array
        radquants = np.arange(nquants)
        
        # Store the data
        self.radquants = radquants
        self.radq_ic = radq_ic
    
    
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
        
        # Calculate the average number of spots for G1, S and G2, and their fractions of zeros
        n = {}
        f0 = {}
        for s in ['G1', 'S', 'G2']:
            
            # Create the state mask
            mask_state = self.states == s
            
            # Create a mask for the X and Y chromosomes (to be ignored)
            if self.config['sex'] == 'male':
                mask_XY = np.logical_or(self.index.chromstr == 'chrX', self.index.chromstr == 'chrY')
            else:
                mask_XY = np.zeros(self.nloci, dtype=bool)
            
            # Subsample the n_ic matrix
            n_ic_s = self.n_ic[mask_state, :, :]
            n_ic_s = n_ic_s[:, ~mask_XY, :]
            
            # Calculate quantities
            n[s] = np.nanmean(n_ic_s)  # float
            f0[s] = np.sum(n_ic_s == 0) / np.sum(~np.isnan(n_ic_s))
        
        # Calculate efficiency and bias in G1 and G2
        eps_G1, beta_G1 = GMM_solve(n['G1'], f0['G1'], p='G1')
        eps_G2, beta_G2 = GMM_solve(n['G2'], f0['G2'], p='G2')
        
        # We assume that the efficiency in S is the average of G1 and G2
        eps_S = (eps_G1 + eps_G2) / 2
        
        # Calculate replication probability and bias in S
        p_S, beta_S = GMM_solve(n['S'], f0['S'], eps=eps_S)
        
        # Store the results
        self.eps_G1 = eps_G1
        self.beta_G1 = beta_G1
        self.eps_G2 = eps_G2
        self.beta_G2 = beta_G2
        self.eps_S = eps_S
        self.beta_S = beta_S
        self.p_S = p_S
        
        print('OVER.')
        print('\n\n')
    
    def z_run(self) -> None:
        """ Run the z-dependent analysis.
        Treats each z quantile independently, combining the data from all cells and loci
        to estimate average values (separately for G1, S and G2).
        As in the population run, in S phase we assume that the efficiency is the average of G1 and G2.
        Estimates:
            - eps_z_G1, detection efficiency in G1. shape: (nquants),
            - beta_z_G1, bias rate in G1. shape: (nquants),
            - eps_z_G2, detection efficiency in G2. shape: (nquants),
            - beta_z_G2, bias rate in G2. shape: (nquants),
            - eps_z_S, detection efficiency in S. shape: (nquants),
            - beta_z_S, bias rate in S. shape: (nquants),
            - p_z_S, replication probability in S. shape: (nquants).
        """
        
        print('Z-DEPENDENT RUN')
        print('---------------')
        
        # Calculate the average number of spots and the fraction of zeros per z quantile,
        # separately for G1, S and G2
        n = {}
        f0 = {}
        for s in ['G1', 'S', 'G2']:
            
            # Create the state mask
            mask_state = self.states == s   
            # Create a mask for the X and Y chromosomes (to be ignored)
            if self.config['sex'] == 'male':
                mask_XY = np.logical_or(self.index.chromstr == 'chrX', self.index.chromstr == 'chrY')
            else:
                mask_XY = np.zeros(self.nloci, dtype=bool)  
            # Subsample the n_ic and zq_ic matrices
            n_ic_s = self.n_ic[mask_state, :, :][:, ~mask_XY, :]
            zq_ic_s = self.zq_ic[mask_state, :, :][:, ~mask_XY, :]
            
            # Loop over the z quantiles
            n[s] = np.zeros(len(self.zquants))  # shape: (nquants)
            f0[s] = np.zeros(len(self.zquants))  # shape: (nquants)
            for z in self.zquants:
                
                # Create the z mask
                mask_z = zq_ic_s == z
                # Subsample the n_ic matrix
                n_ic_s_z = n_ic_s[mask_z]
                
                # Calculate the average number of spots and the fraction of zeros
                n[s][z] = np.nanmean(n_ic_s_z)
                f0[s][z] = np.sum(n_ic_s_z == 0) / np.sum(~np.isnan(n_ic_s_z))

        # Calculate efficiency and bias in G1 and G2
        eps_z_G1, beta_z_G1 = GMM_solve(n['G1'], f0['G1'], p='G1')
        eps_z_G2, beta_z_G2 = GMM_solve(n['G2'], f0['G2'], p='G2')
        eps_z_G1 = self.print_n_clip('eps_z_G1', eps_z_G1, 0, 1)
        eps_z_G2 = self.print_n_clip('eps_z_G2', eps_z_G2, 0, 1)
        beta_z_G1 = self.print_n_clip('beta_z_G1', beta_z_G1, 0, None)
        beta_z_G2 = self.print_n_clip('beta_z_G2', beta_z_G2, 0, None)
        
        # We assume that the efficiency in S is the average of G1 and G2
        eps_z_S = (eps_z_G1 + eps_z_G2) / 2
        
        # Calculate replication probability and bias in S
        p_z_S, beta_z_S = GMM_solve(n['S'], f0['S'], eps=eps_z_S)
        p_z_S = self.print_n_clip('p_z_S', p_z_S, 0, 1)
        beta_z_S = self.print_n_clip('beta_z_S', beta_z_S, 0, None)
        
        # Store the results
        self.eps_z_G1 = eps_z_G1
        self.beta_z_G1 = beta_z_G1
        self.eps_z_G2 = eps_z_G2
        self.beta_z_G2 = beta_z_G2
        self.eps_z_S = eps_z_S
        self.beta_z_S = beta_z_S
        self.p_z_S = p_z_S
        
        print('OVER.')
        print('\n\n')
    
    def rad_run(self) -> None:
        """ Run the rad-dependent analysis.
        Treats each rad quantile independently, combining the data from all cells and loci
        to estimate average values (separately for G1, S and G2).
        Estimates:
            - eps_d_G1, detection efficiency in G1. shape: (nquants),
            - beta_d_G1, bias rate in G1. shape: (nquants),
            - eps_d_G2, detection efficiency in G2. shape: (nquants),
            - beta_d_G2, bias rate in G2. shape: (nquants),
            - eps_d_S, detection efficiency in S. shape: (nquants),
            - beta_d_S, bias rate in S. shape: (nquants).
            - p_d_S, replication probability in S. shape: (nquants).
        """
        
        print('RAD-DEPENDENT RUN')
        print('---------------')
        
        # Calculate the average number of spots and the fraction of zeros per rad quantile,
        # separately for G1, S and G2
        n = {}
        f0 = {}
        for s in ['G1', 'S', 'G2']:
            
            # Create the state mask
            mask_state = self.states == s   
            # Create a mask for the X and Y chromosomes (to be ignored)
            if self.config['sex'] == 'male':
                mask_XY = np.logical_or(self.index.chromstr == 'chrX', self.index.chromstr == 'chrY')
            else:
                mask_XY = np.zeros(self.nloci, dtype=bool)  
            # Subsample the n_ic and radq_ic matrices
            n_ic_s = self.n_ic[mask_state, :, :][:, ~mask_XY, :]
            radq_ic_s = self.radq_ic[mask_state, :, :][:, ~mask_XY, :]
            
            # Loop over the rad quantiles
            n[s] = np.zeros(len(self.radquants))
            f0[s] = np.zeros(len(self.radquants))
            for d in self.radquants:
                
                # Create the z mask
                mask_d = radq_ic_s == d
                # Subsample the n_ic matrix
                n_ic_s_d = n_ic_s[mask_d]
                
                # Calculate the average number of spots and the fraction of zeros
                n[s][d] = np.nanmean(n_ic_s_d)
                f0[s][d] = np.sum(n_ic_s_d == 0) / np.sum(~np.isnan(n_ic_s_d))

        # Calculate efficiency and bias in G1 and G2
        eps_d_G1, beta_d_G1 = GMM_solve(n['G1'], f0['G1'], p='G1')
        eps_d_G2, beta_d_G2 = GMM_solve(n['G2'], f0['G2'], p='G2')
        eps_d_G1 = self.print_n_clip('eps_d_G1', eps_d_G1, 0, 1)
        eps_d_G2 = self.print_n_clip('eps_d_G2', eps_d_G2, 0, 1)
        beta_d_G1 = self.print_n_clip('beta_d_G1', beta_d_G1, 0, None)
        beta_d_G2 = self.print_n_clip('beta_d_G2', beta_d_G2, 0, None)
        
        # We assume that the efficiency in S is the average of G1 and G2
        eps_d_S = (eps_d_G1 + eps_d_G2) / 2
        
        # Calculate replication probability and bias in S
        p_d_S, beta_d_S = GMM_solve(n['S'], f0['S'], eps=eps_d_S)
        p_d_S = self.print_n_clip('p_d_S', p_d_S, 0, 1)
        beta_d_S = self.print_n_clip('beta_d_S', beta_d_S, 0, None)
        
        # Store the results
        self.eps_d_G1 = eps_d_G1
        self.beta_d_G1 = beta_d_G1
        self.eps_d_G2 = eps_d_G2
        self.beta_d_G2 = beta_d_G2
        self.eps_d_S = eps_d_S
        self.beta_d_S = beta_d_S
        self.p_d_S = p_d_S
        
        print('OVER.')
        print('\n\n')
    
    def locus_run(self) -> None:
        """ Run the locus-dependent analysis.
        Treats each locus independently, assuming that different cells are independent realizations
        of the same locus-dependent proces (separately for G1, S and G2).
        In S phase, since there are two equations and three unknowns, we assume that the efficiency
        signal is the locus-dependent average of G1 and G2.
        Note: the bias rate is not estimated in this analysis, as the statistical power is not enough.
        Estimates:
            - eps_i_G1, detection efficiency in G1. shape: (nloci),
            - eps_i_G2, detection efficiency in G2. shape: (nloci),
            - eps_i_S, detection efficiency in S. shape: (nloci),
            - p_i_S, replication probability in S. shape: (nloci).
        """
        
        print('LOCUS-DEPENDENT RUN')
        print('-------------------')
        
        # Calculate the average number of spots for G1, S and G2, and their fractions of zeros
        n_i = {}
        f0_i = {}
        for s in ['G1', 'S', 'G2']:
            
            # Create the state mask
            mask_state = self.states == s
            n_ic_s = self.n_ic[mask_state, :, :]
            
            # Calculate the average number of spots and the fraction of zeros for each locus
            n_i[s] = np.nanmean(n_ic_s, axis=(0, 2))  # shape: (nloci)
            f0_i[s] = np.sum(n_ic_s == 0, axis=(0, 2)) / np.sum(~np.isnan(n_ic_s), axis=(0, 2))
        
        # Calculate the efficiency in G1 and G2
        # We don't estimate also beta, since I saw that it doesn't improve the results,
        # so we just assume that the bias is uniform across loci.
        eps_i_G1, _ = GMM_solve(n_i['G1'], f0_i['G1'], p='G1')
        eps_i_G2, _ = GMM_solve(n_i['G2'], f0_i['G2'], p='G2')
        eps_i_G1 = self.print_n_clip('eps_i_G1', eps_i_G1, 0, 1)
        eps_i_G2 = self.print_n_clip('eps_i_G2', eps_i_G2, 0, 1)

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
        
        # Calculate the replication probability in S
        p_i_S, _ = GMM_solve(n_i['S'], f0_i['S'], eps=eps_i_S)
        p_i_S = self.print_n_clip('p_i_S', p_i_S, 0, 1)
        
        # Store the results
        self.eps_i_G1 = eps_i_G1
        self.eps_i_G2 = eps_i_G2
        self.eps_i_S = eps_i_S
        self.p_i_S = p_i_S
        
        print('OVER.')
        print('\n\n')
    
    def locus_n_z_run(self) -> None:
        """ Run the locus and z-dependent analysis.
        Treats each locus and z quantile independently, assuming that different cells
        are independent realizations of the same locus-dependent process (separately for G1, S and G2).
        Note: as in the locus-dependent analysis, we assume that the bias rate is uniform across loci,
        so we just use the beta value from the z-dependent analysis.
        Estimates:
            - eps_iz_G1, detection efficiency in G1. shape: (nloci, nquants),
            - eps_iz_G2, detection efficiency in G2. shape: (nloci, nquants),
            - eps_iz_S, detection efficiency in S. shape: (nloci, nquants),
            - p_iz_S, replication probability in S. shape: (nloci, nquants).
        """
        
        print('LOCUS AND Z-DEPENDENT RUN')
        print('---------------')
        
        # Calculate the average number of spots and the fraction of zeros
        # per locus and z quantile, separately for G1, S and G2
        n_iz = {}
        f0_iz = {}
        for s in ['G1', 'S', 'G2']:
            
            # Create the state mask
            mask_state = self.states == s   
            # Subsample the n_ic and zq_ic matrices
            n_ic_s = self.n_ic[mask_state, :, :]
            zq_ic_s = self.zq_ic[mask_state, :, :]
            
            # Loop over the z quantiles
            n_iz[s] = np.zeros((self.nloci, len(self.zquants)))  # shape: (nloci, nquants)
            f0_iz[s] = np.zeros((self.nloci, len(self.zquants)))  # shape: (nloci, nquants)
            for z in self.zquants:
                
                # Create the z mask
                mask_z = zq_ic_s == z
                
                # Set n_ic_s_z to NaN where mask_z is False
                n_ic_s_z = np.where(mask_z, n_ic_s, np.nan)
                
                # Calculate the average number of spots and the fraction of zeros
                n_iz[s][:, z] = np.nanmean(n_ic_s_z, axis=(0, 2))  # shape: (nloci)
                f0_iz[s][:, z] = np.sum(n_ic_s_z == 0, axis=(0, 2)) / np.sum(~np.isnan(n_ic_s_z), axis=(0, 2))
        
        # Calculate the efficiency in G1 and G2
        # Again, we assume that the bias rate is uniform across loci
        eps_iz_G1, _ = GMM_solve(n_iz['G1'], f0_iz['G1'], p='G1')
        eps_iz_G2, _ = GMM_solve(n_iz['G2'], f0_iz['G2'], p='G2')
        eps_iz_G1 = self.print_n_clip('eps_iz_G1', eps_iz_G1, 0, 1)
        eps_iz_G2 = self.print_n_clip('eps_iz_G2', eps_iz_G2, 0, 1)
        
        # Assume that the efficiency in S is the average of G1 and G2
        eps_iz_S = (eps_iz_G1 + eps_iz_G2) / 2
        
        # For S, since we assume that the bias rate is uniform across loci,
        # we can just use the beta value from the z-dependent analysis and tile it
        beta_iz_S = np.tile(self.beta_z_S[np.newaxis, :], (self.nloci, 1))  # shape: (nloci, nquants)
        
        # Calculate the probability of replication in S
        p_iz_S = GMM_solve(n_iz['S'], f0_iz['S'], eps=eps_iz_S, beta=beta_iz_S)
        p_iz_S = self.print_n_clip('p_iz_S', p_iz_S, 0, 1)
        
        # Store the results
        self.eps_iz_G1 = eps_iz_G1
        self.eps_iz_G2 = eps_iz_G2
        self.eps_iz_S = eps_iz_S
        self.p_iz_S = p_iz_S
        
        print('OVER.')
        print('\n\n')
    
    def locus_n_rad_run(self) -> None:
        """ Run the locus and rad-dependent analysis.
        Treats each locus and rad quantile independently, assuming that different cells
        are independent realizations of the same locus-dependent process (separately for G1, S and G2).
        Note: as in the locus-dependent analysis, the bias rate is assumed to be uniform across loci,
        and we just use the beta value from the rad-dependent analysis.
        Estimates:
            - eps_id_G1, detection efficiency in G1. shape: (nloci, nquants),
            - eps_id_G2, detection efficiency in G2. shape: (nloci, nquants),
            - eps_id_S, detection efficiency in S. shape: (nloci, nquants),
            - p_id_S, replication probability in S. shape: (nloci, nquants).
        """
        
        print('LOCUS AND RAD-DEPENDENT RUN')
        print('---------------')
        
        # Calculate the average number of spots and the fraction of zeros
        # per locus and rad quantile, separately for G1, S and G2
        n_id = {}
        f0_id = {}
        for s in ['G1', 'S', 'G2']:
            
            # Create the state mask
            mask_state = self.states == s   
            # Subsample the n_ic and radq_ic matrices
            n_ic_s = self.n_ic[mask_state, :, :]
            radq_ic_s = self.radq_ic[mask_state, :, :]
            
            # Loop over the rad quantiles
            n_id[s] = np.zeros((self.nloci, len(self.radquants)))  # shape: (nloci, nquants)
            f0_id[s] = np.zeros((self.nloci, len(self.radquants)))  # shape: (nloci, nquants)
            for d in self.radquants:
                
                # Create the rad mask
                mask_d = radq_ic_s == d
                
                # Set n_ic_s_d to NaN where mask_d is False
                n_ic_s_d = np.where(mask_d, n_ic_s, np.nan)
                
                # Calculate the average number of spots and the fraction of zeros
                n_id[s][:, d] = np.nanmean(n_ic_s_d, axis=(0, 2))  # shape: (nloci)
                f0_id[s][:, d] = np.sum(n_ic_s_d == 0, axis=(0, 2)) / np.sum(~np.isnan(n_ic_s_d), axis=(0, 2))
        
        # Calculate the efficiency in G1 and G2
        # Again, we assume that the bias rate is uniform across loci
        eps_id_G1, _ = GMM_solve(n_id['G1'], f0_id['G1'], p='G1')
        eps_id_G2, _ = GMM_solve(n_id['G2'], f0_id['G2'], p='G2')
        eps_id_G1 = self.print_n_clip('eps_id_G1', eps_id_G1, 0, 1)
        eps_id_G2 = self.print_n_clip('eps_id_G2', eps_id_G2, 0, 1)
        
        # Assume that the efficiency in S is the average of G1 and G2
        eps_id_S = (eps_id_G1 + eps_id_G2) / 2
        
        # For S, since we assume that the bias rate is uniform across loci,
        # we can just use the beta value from the z-dependent analysis and tile it
        beta_id_S = np.tile(self.beta_d_S[np.newaxis, :], (self.nloci, 1))  # shape: (nloci, nquants)
        
        # Calculate the probability of replication in S
        p_id_S = GMM_solve(n_id['S'], f0_id['S'], eps=eps_id_S, beta=beta_id_S)
        p_id_S = self.print_n_clip('p_id_S', p_id_S, 0, 1)
 
        # Store the results
        self.eps_id_G1 = eps_id_G1
        self.eps_id_G2 = eps_id_G2
        self.eps_id_S = eps_id_S
        self.p_id_S = p_id_S
        
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
        
        # Identify early replicating loci
        RT_early = 0.95
        early_mask = self.p_i_S > RT_early
        
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
                    early_mask
                )
            n_ic_loci = self.n_ic[:, mask_loci, :]
            # Calculate the average number of spots and the fraction of zeros for each cell
            n_c[loci] = np.nanmean(n_ic_loci, axis=(1, 2))  # shape: (ncells)
            f0_c[loci] = np.sum(n_ic_loci == 0, axis=(1, 2)) / np.sum(~np.isnan(n_ic_loci), axis=(1, 2))
        
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
            state_mask = getattr(self, f'{state}s')
            eps_i_s = getattr(self, f'eps_i_{state}')
            eps_c_[state_mask] *= np.nanmean(eps_i_s) / np.nanmean(eps_i_s[early_mask])
        
        # Store the results
        self.eps_c = eps_c
        self.eps_c_ = eps_c_
        self.beta_c = beta_c
        self.beta_c_ = beta_c_
        self.p_c = p_c
        
        print('OVER.')
        print('\n\n')
    
    def cell_n_z_run(self) -> None:
        """ Run the cell and z-dependent analysis.
        Treats each cell and z quantile independently, assuming that different loci are independent realizations
        of the same cell-and-z-dependent process.
        Uses an approximation for the replication probability of cells in S phase, combining the cell and z runs.
        Estimates:
            - eps_cz, detection efficiency. shape: (ncells, nquants),
            - beta_cz, bias rate. shape: (ncells, nquants),
            - p_cz, replication probability. shape: (ncells, nquants).
        """
        
        print('CELL AND Z-DEPENDENT RUN')
        print('------------------------')
        
        # Initialize the data for the average number of spots and the fraction of zeros
        # for each cell and z quantile
        n_cz = np.zeros((self.ncells, len(self.zquants)))  # shape: (ncells, nquants)
        f0_cz = np.zeros((self.ncells, len(self.zquants)))  # shape: (ncells, nquants)
        
        # Remove the X and Y chromosomes if sex is male
        if self.config['sex'] == 'male':
            mask_XY = np.logical_or(self.index.chromstr == 'chrX', self.index.chromstr == 'chrY')
        else:
            mask_XY = np.zeros(self.nloci, dtype=bool)
        n_ic = self.n_ic[:, ~mask_XY, :]
        zq_ic = self.zq_ic[:, ~mask_XY, :]
        
        # Loop over the z quantiles
        for z in range(len(self.zquants)):
            
            # Create the z mask
            mask_z = zq_ic == z
            
            # Set n_ic_z to NaN where mask_z is False
            n_ic_z = np.where(mask_z, n_ic, np.nan)
            
            # Calculate the average number of spots and the fraction of zeros
            n_cz[:, z] = np.nanmean(n_ic_z, axis=(1, 2))  # shape: (ncells)
            f0_cz[:, z] = np.sum(n_ic_z == 0, axis=(1, 2)) / np.sum(~np.isnan(n_ic_z), axis=(1, 2))

        # Calculate efficiency and bias for G1 and G2
        eps_G1_cz, beta_G1_cz = GMM_solve(n_cz[self.G1s, :], f0_cz[self.G1s, :], p='G1')
        eps_G2_cz, beta_G2_cz = GMM_solve(n_cz[self.G2s, :], f0_cz[self.G2s, :], p='G2')
        # Create arrays for all cells and fill them
        eps_cz = np.full((self.ncells, len(self.zquants)), np.nan)  # shape: (ncells, nquants)
        eps_cz[self.G1s, :] = eps_G1_cz
        eps_cz[self.G2s, :] = eps_G2_cz
        beta_cz = np.full((self.ncells, len(self.zquants)), np.nan)  # shape: (ncells, nquants)
        beta_cz[self.G1s, :] = beta_G1_cz
        beta_cz[self.G2s, :] = beta_G2_cz
        
        # For S phase, it would be too much to use the early trick, since we would have too little data.
        # So instead, we approximate the replication probability using our previous results,
        # in particular the cell run and the z-dependent run.
        # We start from the p_c values, and we tile them
        p_c_S = self.p_c[self.Ss]
        p_c_S = np.tile(p_c_S[:, np.newaxis], (1, len(self.zquants)))  # shape: (ncells_S, nquants)
        # Then we calculate the rescaling factors for each quantile from p_z_S,
        # i.e. the ratio between each p_z value and their average
        x_z_S = self.p_z_S / np.nanmean(self.p_z_S)
        x_z_S = np.tile(x_z_S[np.newaxis, :], (np.sum(self.Ss), 1))  # shape: (ncells_S, nquants)
        # Finally, we define the cell-and-z dependent replication probability as the product of the two
        p_cz_S = p_c_S * x_z_S
        p_cz_S = self.print_n_clip('p_cz_S', p_cz_S, 0, 1)
        # Create a full p_cz matrix to store the results
        p_cz = np.full((self.ncells, len(self.zquants)), np.nan)  # shape: (ncells, nquants)
        p_cz[self.G1s, :] = 0
        p_cz[self.G2s, :] = 1
        p_cz[self.Ss, :] = p_cz_S
        
        # We then calculate the efficiency and bias for S
        eps_cz_S, beta_cz_S = GMM_solve(n_cz[self.Ss, :], f0_cz[self.Ss, :], p=p_cz_S)
        eps_cz[self.Ss, :] = eps_cz_S
        beta_cz[self.Ss, :] = beta_cz_S
        eps_cz = self.print_n_clip('eps_cz', eps_cz, 0, 1)
        beta_cz = self.print_n_clip('beta_cz', beta_cz, 0, None)
        
        # Note that here we do estimate two parameters, differently from the locus-dependent analysis.
        # It's because here we have much more data: each cell has ~100k loci, so ~200k data (two copies).
        # If there are 10 z quantiles, we have ~20k data points for each estimation.
        
        # Store the results
        self.eps_cz = eps_cz
        self.beta_cz = beta_cz
        self.p_cz = p_cz
        
        print('OVER.')
        print('\n\n')
    
    def cell_n_rad_run(self) -> None:
        
        print('CELL AND RAD-DEPENDENT RUN')
        print('------------------------')
        
        # Initialize the data for the average number of spots and the fraction of zeros
        # for each cell and rad quantile
        n_cd = np.zeros((self.ncells, len(self.radquants)))  # shape: (ncells, nquants)
        f0_cd = np.zeros((self.ncells, len(self.radquants)))  # shape: (ncells, nquants)
        
        # Remove the X and Y chromosomes if sex is male
        if self.config['sex'] == 'male':
            mask_XY = np.logical_or(self.index.chromstr == 'chrX', self.index.chromstr == 'chrY')
        else:
            mask_XY = np.zeros(self.nloci, dtype=bool)
        n_ic = self.n_ic[:, ~mask_XY, :]
        radq_ic = self.radq_ic[:, ~mask_XY, :]
        
        # Loop over the rad quantiles
        for d in range(len(self.radquants)):
            
            # Create the rad mask
            mask_d = radq_ic == d
            
            # Set n_ic to NaN where the mask is False
            n_ic_d = np.where(mask_d, n_ic, np.nan)
            
            # Calculate the average number of spots and the fraction of zeros
            n_cd[:, d] = np.nanmean(n_ic_d, axis=(1, 2))  # shape: (ncells)
            f0_cd[:, d] = np.sum(n_ic_d == 0, axis=(1, 2)) / np.sum(~np.isnan(n_ic_d), axis=(1, 2))

        # Calculate efficiency and bias for G1 and G2
        eps_G1_cd, beta_G1_cd = GMM_solve(n_cd[self.G1s, :], f0_cd[self.G1s, :], p='G1')
        eps_G2_cd, beta_G2_cd = GMM_solve(n_cd[self.G2s, :], f0_cd[self.G2s, :], p='G2')
        # Create arrays for all cells and fill them
        eps_cd = np.full((self.ncells, len(self.radquants)), np.nan)  # shape: (ncells, nquants)
        eps_cd[self.G1s, :] = eps_G1_cd
        eps_cd[self.G2s, :] = eps_G2_cd
        beta_cd = np.full((self.ncells, len(self.radquants)), np.nan)  # shape: (ncells, nquants)
        beta_cd[self.G1s, :] = beta_G1_cd
        beta_cd[self.G2s, :] = beta_G2_cd
        
        # For S phase, it would be too much to use the early trick, since we would have too little data.
        # So instead, we approximate the replication probability using our previous results,
        # in particular the cell run and the rad-dependent run.
        # We start from the p_c values, and we tile them
        p_c_S = self.p_c[self.Ss]
        p_c_S = np.tile(p_c_S[:, np.newaxis], (1, len(self.radquants)))  # shape: (ncells_S, nquants)
        # Then we calculate the rescaling factors for each quantile from p_d_S,
        # i.e. the ratio between each p_d value and their average
        x_d_S = self.p_d_S / np.nanmean(self.p_d_S)
        x_d_S = np.tile(x_d_S[np.newaxis, :], (np.sum(self.Ss), 1))  # shape: (ncells_S, nquants)
        # Finally, we define the cell-and-rad dependent replication probability as the product of the two
        p_cd_S = p_c_S * x_d_S
        p_cd_S = self.print_n_clip('p_cd_S', p_cd_S, 0, 1)
        # Create a full p_cd matrix to store the results
        p_cd = np.full((self.ncells, len(self.radquants)), np.nan)  # shape: (ncells, nquants)
        p_cd[self.G1s, :] = 0
        p_cd[self.G2s, :] = 1
        p_cd[self.Ss, :] = p_cd_S
        
        # We then calculate the efficiency and bias for S
        eps_cd_S, beta_cd_S = GMM_solve(n_cd[self.Ss, :], f0_cd[self.Ss, :], p=p_cd_S)
        eps_cd[self.Ss, :] = eps_cd_S
        beta_cd[self.Ss, :] = beta_cd_S
        eps_cd = self.print_n_clip('eps_cd', eps_cd, 0, 1)
        beta_cd = self.print_n_clip('beta_cd', beta_cd, 0, None)
        
        # Note that here we do estimate two parameters, differently from the locus-dependent analysis.
        # It's because here we have much more data: each cell has ~100k loci, so ~200k data (two copies).
        # If there are 10 z quantiles, we have ~20k data points for each estimation.
        
        # Store the results
        self.eps_cd = eps_cd
        self.beta_cd = beta_cd
        self.p_cd = p_cd
        
        print('OVER.')
        print('\n\n')
    
    def complete_eps_beta(self) -> None:
        
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
        """ Calculates the replication probability for a given mask.
        
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
        nG1 = np.nansum(self.states == 'G1')
        nS = np.nansum(self.states == 'S')
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
