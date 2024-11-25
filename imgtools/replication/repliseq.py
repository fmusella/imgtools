import os
import numpy as np
import h5py
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
        P(N = 0) = 1 - (1 + p) * eps + p^2 * eps^2.
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
        self.volumes = None
        self.n_ic = None
        self.z_ic = None
    
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
        obj.z_ic = scf.get_feature('z')
        obj.ncells, obj.nloci, obj.ncopies = obj.n_ic.shape
        
        return obj
    
    @staticmethod
    def _check_scf(scf: SingleCellFeature) -> None:
        """ Check the input SingleCellFeature object.
        
        It checks that:
         - the input is a SingleCellFeature object,
         - the SCF contains the 'spotcount' feature,
         - the SCF contains the 'z' feature,
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
        if 'z' not in scf.feature_list:
            raise ValueError("The input scf must contain the 'z' feature.")
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
            3. Locus-dependent analysis.
            4. Locus and z-dependent analysis.
            5. Cell-dependent analysis.
            6. Cell and z-dependent analysis.
            7. Sliding window analysis.
        
        The results are stored in the object's attributes.
        
        Args:
            config (dict): configuration dictionary. Must contain the following keys:
                - sex,
                - sliding_window_size,
                - sliding_window_f0_threshold,
                - sliding_window_efficiency_threshold.
        """
        # Set the config
        self._check_config(config)
        self.config = config
        # Prepare the data
        self.curate_missing_chromosomes()
        self.quantize_zcoords()
        # Run the analysis
        self.population_run()
        self.z_run()
        self.locus_run()
        self.locus_n_z_run()
        self.cell_run()
        self.cell_n_z_run()
        self.sliding_window_run()
    
    @staticmethod
    def _check_config(config: dict) -> None:
        """ Check the input config dictionary.
        
        It checks that the input is a dictionary and that it contains the required keys:
         - sex,
         - nslices,
         - sliding_window_size
         
        It also checks that the 'sex' key is a string and that it is either 'male' or 'female'.

        Args:
            config (dict)
        """
            
        if not isinstance(config, dict):
            raise TypeError("The input config must be a dictionary.")
        
        required_keys = [
            'sex',
            'nslices',
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
        
        # Get the number of slices to quantize the z coordinates
        nquants = self.config['nslices']
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
                mask_q = np.logical_and(z_c >= quants_c[q], z_c <= quants_c[q + 1])
                # Assign the quantile to the quantized z coordinates
                zq_c[mask_q] = q
            
            # Store the quantized z coordinates for the cell
            zq_ic[c, :, :] = zq_c
        
        # Get the quantiles as an array
        zquants = np.arange(nquants)
        
        # Store the data
        self.zquants = zquants
        self.zq_ic = zq_ic
    
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
        
        # Calculate the efficiency in G1 and G2
        eps_G1 = 1 - f0['G1']
        eps_G2 = 1 - f0['G2'] ** 0.5
        
        # Calculate the bias in G1 and G2
        beta_G1 = n['G1'] / eps_G1 
        beta_G2 = n['G2'] / (2 * eps_G2)
        
        # We assume that the efficiency in S is the average of G1 and G2
        eps_S = (eps_G1 + eps_G2) / 2
        
        # Calculate the bias and the replication probability in S
        p_S = (1 - eps_S - f0['S']) / (eps_S * (1 - eps_S))
        beta_S = n['S'] / ((1 + p_S) * eps_S)
        
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
            - p_z_S, replication probability in S. float.
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

        # Calculate the efficiency in G1 and G2
        eps_z_G1 = 1 - f0['G1']
        eps_z_G2 = 1 - f0['G2'] ** 0.5
        eps_z_G1 = self.print_n_clip('eps_z_G1', eps_z_G1, 0, 1)
        eps_z_G2 = self.print_n_clip('eps_z_G2', eps_z_G2, 0, 1)
        
        # Calculate the bias in G1 and G2
        beta_z_G1 = n['G1'] / eps_z_G1
        beta_z_G2 = n['G2'] / (2 * eps_z_G2)
        beta_z_G1 = self.print_n_clip('beta_z_G1', beta_z_G1, 0, None)
        beta_z_G2 = self.print_n_clip('beta_z_G2', beta_z_G2, 0, None)
        
        # We assume that the efficiency in S is the average of G1 and G2
        eps_z_S = (eps_z_G1 + eps_z_G2) / 2
        eps_z_S = self.print_n_clip('eps_z_S', eps_z_S, 0, 1)
        
        # Use the population-run S-phase replication probability to get the bias,
        # since pS is the same for all z quantiles
        beta_z_S = n['S'] / ((1 + self.p_S) * eps_z_S)
        beta_z_S = self.print_n_clip('beta_z_S', beta_z_S, 0, None)
        
        # Store the results
        self.eps_z_G1 = eps_z_G1
        self.beta_z_G1 = beta_z_G1
        self.eps_z_G2 = eps_z_G2
        self.beta_z_G2 = beta_z_G2
        self.eps_z_S = eps_z_S
        self.beta_z_S = beta_z_S
        
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
        eps_i_G1 = 1 - f0_i['G1']
        eps_i_G2 = 1 - f0_i['G2'] ** 0.5
        eps_i_G1 = self.print_n_clip('eps_i_G1', eps_i_G1, 0, 1)
        eps_i_G2 = self.print_n_clip('eps_i_G2', eps_i_G2, 0, 1)

        # Assume that the efficiency in S is the average of G1 and G2
        eps_i_S = (eps_i_G1 + eps_i_G2) / 2
        eps_i_S = self.print_n_clip('eps_i_S', eps_i_S, 0, 1)
        # Note: since we assume the bias not to depend on i,
        # we could use beta_S to estimate both eps_i_S and p_i_S
        # However, I think that the statistical power is not good enough to
        # estimate two parameters. Indeed, the results looked bad.
        # Note that here, for the locus-dependent analysis, we don't have
        # a lot of data for the estimation: there are ~250 cells in G1,
        # ~250 cells in G2, and ~500 cells in S. Since there are two copies
        # we multiply these number by 2, but it's still very little.
        
        # Calculate the replication probability in S
        p_i_S = (1 - eps_i_S - f0_i['S']) / (eps_i_S * (1 - eps_i_S))
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
        Note: as in the locus-dependent analysis, the bias rate is not estimated in this analysis.
        Also we don't need to estimate p_iz_S, since it doesn't depend on z.
        Estimates:
            - eps_iz_G1, detection efficiency in G1. shape: (nloci, nquants),
            - eps_iz_G2, detection efficiency in G2. shape: (nloci, nquants),
            - eps_iz_S, detection efficiency in S. shape: (nloci, nquants),
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
        eps_iz_G1 = 1 - f0_iz['G1']
        eps_iz_G2 = 1 - f0_iz['G2'] ** 0.5
        eps_iz_G1 = self.print_n_clip('eps_iz_G1', eps_iz_G1, 0, 1)
        eps_iz_G2 = self.print_n_clip('eps_iz_G2', eps_iz_G2, 0, 1)
        
        # Calculate the efficiency for S
        p_iz_S = np.tile(self.p_i_S[:, np.newaxis], (1, len(self.zquants)))  # shape: (nloci, nquants)
        beta_iz_S = np.tile(self.beta_z_S[np.newaxis, :], (self.nloci, 1))  # shape: (nloci, nquants)
        eps_iz_S = n_iz['S'] / ((1 + p_iz_S) * beta_iz_S)
        eps_iz_S = self.print_n_clip('eps_iz_S', eps_iz_S, 0, 1)
        # Note: here we just have to estimate one parameter (eps_iz_S),
        # since p_iz_S doesn't depend on z. That's why here we can estimate eps_iz_S
        # However, I checked that using eps_iz_S = (eps_iz_G1 + eps_iz_G2) / 2
        # also gives good results.
 
        # Store the results
        self.eps_iz_G1 = eps_iz_G1
        self.eps_iz_G2 = eps_iz_G2
        self.eps_iz_S = eps_iz_S
        
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
        
        # Get the masks for G1, S and G2
        G1s = self.states == 'G1'
        G2s = self.states == 'G2'
        Ss = self.states == 'S'
        
        # Calculate the approximate efficiency for G1, S, G2,
        # using only the early replicating loci (whose replication state is known)
        eps_c_ = np.full(self.ncells, np.nan)
        eps_c_[G1s] = 1 - f0_c['early'][G1s]
        eps_c_[Ss] = 1 - f0_c['early'][Ss] ** 0.5
        eps_c_[G2s] = 1 - f0_c['early'][G2s] ** 0.5
        eps_c_ = self.print_n_clip('eps_c_', eps_c_, 0, 1)
        
        # Calculate the approximate bias for G1, S, G2
        beta_c_ = np.full(self.ncells, np.nan)
        beta_c_[G1s] = n_c['early'][G1s] / eps_c_[G1s]
        beta_c_[Ss] = n_c['early'][Ss] / (2 * eps_c_[Ss])
        beta_c_[G2s] = n_c['early'][G2s] / (2 * eps_c_[G2s])
        beta_c_ = self.print_n_clip('beta_c_', beta_c_, 0, None)
        
        # Correct the approximate efficiency
        # We know that early loci have on average a lower detection efficiency
        # We can use the locus-dependent efficiency to correct the approximate efficiency
        # Separately for G1, S and G2, we calculate the correction factor as
        # the ratio between the average locus-dependent efficiency genome-wide
        # divided by the average locus-dependent efficiency for early replicating loci
        print('Average efficiencies before correction:')
        print(f"G1: {np.nanmean(eps_c_[G1s])}")
        print(f"S: {np.nanmean(eps_c_[Ss])}")
        print(f"G2: {np.nanmean(eps_c_[G2s])}")
        correction_G1 = np.nanmean(self.eps_i_G1) / np.nanmean(self.eps_i_G1[early_mask])
        correction_S = np.nanmean(self.eps_i_S) / np.nanmean(self.eps_i_S[early_mask])
        correction_G2 = np.nanmean(self.eps_i_G2) / np.nanmean(self.eps_i_G2[early_mask])
        print('Correction factors:')
        print(f"G1: {correction_G1}")
        print(f"S: {correction_S}")
        print(f"G2: {correction_G2}")
        eps_c_[G1s] = eps_c_[G1s] * correction_G1
        eps_c_[Ss] = eps_c_[Ss] * correction_S
        eps_c_[G2s] = eps_c_[G2s] * correction_G2
        print('Average efficiencies after correction:')
        print(f"G1: {np.nanmean(eps_c_[G1s])}")
        print(f"S: {np.nanmean(eps_c_[Ss])}")
        print(f"G2: {np.nanmean(eps_c_[G2s])}")
        
        # Calculate the full efficiency for G1 and G2
        eps_c = np.full(self.ncells, np.nan)
        eps_c[G1s] = 1 - f0_c['all'][G1s]
        eps_c[G2s] = 1 - f0_c['all'][G2s] ** 0.5
        
        # Calculate b_c for G1 and G2
        beta_c = np.full(self.ncells, np.nan)
        beta_c[G1s] = n_c['all'][G1s] / eps_c[G1s]
        beta_c[G2s] = n_c['all'][G2s] / (2 * eps_c[G2s])
        # Use the approximate b for S
        beta_c[Ss] = beta_c_[Ss]
        beta_c = self.print_n_clip('beta_c', beta_c, 0, None)
        
        # Calculate the efficiency for S
        d_S_c = n_c['all'][Ss] / beta_c[Ss]
        eps_S_c = (d_S_c / 2) * (1 + np.sqrt(1 - 4 * (f0_c['all'][Ss] + d_S_c - 1) / d_S_c ** 2))
        # Correct the efficiency for NaN values
        # They arise when the square root is negative,
        # and we can show this happens for cells at the end of S phase, close to G2
        # For these cases we can just use the G2 efficiency
        eps_S_c[np.isnan(eps_S_c)] = 1 - f0_c['all'][Ss][np.isnan(eps_S_c)] ** 0.5
        # Assign the efficiency for S
        eps_c[Ss] = eps_S_c
        eps_c = self.print_n_clip('eps_c', eps_c, 0, 1)
        
        # Calculate the replication probability
        p_c = np.full(self.ncells, np.nan)
        p_c[G1s] = 0
        p_c[G2s] = 1
        p_c[Ss] = n_c['all'][Ss] / (eps_c[Ss] * beta_c[Ss]) - 1
        p_c = self.print_n_clip('p_c', p_c, 0, 1)
        
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
        As in the cell-dependent analysis, we use the early-loci approximation to estimate the bias rate in S.
        Estimates:
            - eps_cz, detection efficiency. shape: (ncells, nquants),
            - eps_cz_, approximated detection efficiency. shape: (ncells, nquants),
            - beta_cz, bias rate. shape: (ncells, nquants),
            - beta_cz_, approximated for bias rate. shape: (ncells, nquants),
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
        
        # Get the masks for G1, S and G2
        G1s = self.states == 'G1'
        G2s = self.states == 'G2'
        Ss = self.states == 'S'

        # Calculate the efficiency for G1 and G2
        eps_cz = np.full((self.ncells, len(self.zquants)), np.nan)  # shape: (ncells, nquants)
        eps_cz[G1s, :] = 1 - f0_cz[G1s, :]
        eps_cz[G2s, :] = 1 - f0_cz[G2s, :] ** 0.5
        # Calculate the efficiency for S using the results from the cell-dependent analysis
        # Note that here we do estimate two parameters, differently from the locus-dependent analysis.
        # It's because here we have much more data: each cell has ~100k loci, so ~200k data (two copies).
        # Since there are 10 z quantiles, we have ~20k data points for each estimation.
        p_c_S = self.p_c[Ss]
        p_cz_S = np.tile(p_c_S[:, np.newaxis], (1, len(self.zquants)))  # shape: (ncells, nquants)
        eps_cz_S = (1 + p_cz_S - np.sqrt((1 + p_cz_S) ** 2 - 4 * p_cz_S * (1 - f0_cz[Ss, :]))) / (2 * p_cz_S)
        eps_cz[Ss, :] = eps_cz_S
        eps_cz = self.print_n_clip('eps_cz', eps_cz, 0, 1)
        
        # Calculate the bias for G1 and G2
        beta_cz = np.full((self.ncells, len(self.zquants)), np.nan)  # shape: (ncells, nquants)
        beta_cz[G1s, :] = n_cz[G1s, :] / eps_cz[G1s, :]
        beta_cz[G2s, :] = n_cz[G2s, :] / (2 * eps_cz[G2s, :])
        # Calculate the bias for S
        beta_cz_S = n_cz[Ss, :] / ((1 + p_cz_S) * eps_cz_S)
        beta_cz[Ss, :] = beta_cz_S
        beta_cz = self.print_n_clip('beta_cz', beta_cz, 0, None)
        
        # Store the results
        self.eps_cz = eps_cz
        self.beta_cz = beta_cz
        
        print('OVER.')
        print('\n\n')
    
    def sliding_window_run(self) -> None:
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
        
        # Create a tiled efficiency tensor of shape (ncells, nloci, nquants, ncopies)
        eps_icz = np.zeros((self.ncells, self.nloci, len(self.zquants), self.ncopies), dtype=float)
        for cellnum, state in enumerate(self.states):
            for copynum in range(self.ncopies):
                if state == 'G1':
                    eps_icz[cellnum, :, :, copynum] = self.eps_iz_G1
                elif state == 'S':
                    eps_icz[cellnum, :, :, copynum] = self.eps_iz_S
                elif state == 'G2':
                    eps_icz[cellnum, :, :, copynum] = self.eps_iz_G2

        # Create a tiled bias tensor of shape (ncells, nloci, nquants, ncopies)
        beta_icz = np.tile(self.beta_cz[:, np.newaxis, :, np.newaxis], (1, self.nloci, 1, self.ncopies))

        # Create a locus, cell, copy dependent tensors by selecting the actual z values
        eps_ic = np.full((self.ncells, self.nloci, self.ncopies), np.nan)
        beta_ic = np.full((self.ncells, self.nloci, self.ncopies), np.nan)
        for z in self.zquants:
            mask_z = self.zq_ic == z
            eps_ic[mask_z] = eps_icz[:, :, z, :][mask_z]
            beta_ic[mask_z] = beta_icz[:, :, z, :][mask_z]
        # Loop over cells and z to correct eps_ic and beta_ic
        for cellnum in range(self.ncells):
            for copynum in range(self.ncopies):
                for z in self.zquants:
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
        n_ic = self.print_n_clip('n_ic', self.n_ic, 0, 4)
        
        # Get the window size in units of loci
        window = int(np.ceil(self.config['sliding_window_size'] / self.index.resolution()))
        
        # Calculate the sliding window averages
        n_ic_SW = scf_utils.sliding_matrix(n_ic, self.index, window=window, method='mean')
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
