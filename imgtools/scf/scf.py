import os
import numpy as np
from scipy import stats
import h5py
from alabtools.utils import Index
from statsmodels.stats.multitest import fdrcorrection
from . import scf_utils
from ..cte import ChromatinTracingExperiment


class SingleCellFeature:
    """ A class to store feature data from single-cell DNA experiments, e.g. DNA counts,
    where the data are organized as a matrix of shape ncells x ndomains x ncopies.
    
    The data structure describes the chromosomal domains with the Index object,
    and the cells with a cell label (e.g. cell ID) and a cell state (e.g. cell cycle phase).
    
    ----------
    Attributes:
        h5_name (str): path and name of the HDF5 file.
        h5 (h5py.File): HDF5 file to store the data.
                        Contains the following groups:
                        - index: Index object.
                        - attrs: attributes.
                        - cell_labels: array with the cell IDs.
                        - cell_states: array with the cell states.
                        - volumes: array with the cell volumes.
                        - feature_list: list of feature matrices.
                        - [feature]: feature matrix. (saved with a particular name)
    ---------- 
    Properties (from h5 file):
        index (Index): Index object.
        attrs (dict): attributes.
        cell_labels (np.ndarray): array with the cell IDs.
        cell_states (np.ndarray): array with the cell states.
        volumes (np.ndarray): array with the cell volumes.
        feature_list (list): list of feature matrices.
    """
    
    def __init__(self, h5_name: str, mode: str = 'r') -> None:
        """ Initialize the SingleCellFeature object.
        
        A HDF5 file is created to store the data.
        
        The file is opened in the specified mode, that should match the use case,
        e.g. a file cannot be created if the mode is 'r'.

        Args:
            h5_name (str): path and name of the HDF5 file.
            mode (str): 'r', 'r+', 'w', 'w-', 'x', 'a'. Defaults to 'r'.
        """
        
        # Extend the name with its absolute path
        h5_name = os.path.abspath(h5_name)
        
        # Check that h5_name has a valid path
        if not os.path.exists(os.path.dirname(h5_name)):
            raise FileNotFoundError("The path of the HDF5 file does not exist.")
        
        # Check that mode is valid
        if not mode in ['r', 'r+', 'w', 'w-', 'x', 'a']:
            raise ValueError("mode must be one of 'r', 'r+', 'w', 'w-', 'x', 'a'.")
        
        # If the file doesn't exists, make sure that mode is write (w, w-, x, r+, a)
        if not os.path.exists(h5_name) and mode not in ['w', 'w-', 'x', 'r+', 'a']:
            raise FileNotFoundError("The HDF5 file does not exist. Use mode 'w', 'w-', 'x', 'r+', 'a' to create it.")
        
        # Open the HDF5 file
        self.h5_name = h5_name
        self.h5 = h5py.File(h5_name, mode)
    
    
    # CONTAIN METHOD
    def __contains__(self, name: str) -> bool:
        """ Check if a dataset exists in the h5 file."""
        return name in self.h5
    
     
    # SETTER FUNCTIONS
    
    def set_index(self, index: Index) -> None:
        """ Set the Index object in the h5 file."""
        index.save(self.h5)
    
    def set_attrs(self, attrs: dict) -> None:
        """ Save the attributes in the h5 file.
        Attributes are stored in the root of the h5 file."""
        for key in attrs:
            self.h5.attrs[key] = attrs[key]
    
    def set_cell_labels(self, cell_labels: np.ndarray) -> None:
        """ Save the cell labels in the h5 file.
        Cell labels are string, so they must be converted to 'S' type."""
        self.h5.create_dataset('cell_labels', data=np.array(cell_labels).astype('S'))
    
    def set_cell_states(self, cell_states: np.ndarray) -> None:
        """ Save the cell states in the h5 file.
        Cell states are string, so they must be converted to 'S' type."""
        self.h5.create_dataset('cell_states', data=np.array(cell_states).astype('S'))
    
    def set_volumes(self, volumes: np.ndarray) -> None:
        """ Save the cell volumes in the h5 file."""
        self.h5.create_dataset('volumes', data=volumes, dtype='float32')
    
    def set_feature(self, matrix: np.ndarray, feature: str, doc: str) -> None:
        """ Save the feature matrix in the h5 file."""
        # Check that the feature is not already in the h5 file
        if feature in self:
            raise ValueError("The feature matrix '{}' already exists in the h5 file.".format(feature))
        # Add the matrix to the h5 file
        self.h5.create_dataset(feature, data=matrix, dtype=matrix.dtype)
        # Add the documentation to the matrix
        self.h5[feature].attrs['doc'] = doc
        
    
    
    # GETTER FUNCTIONS
    
    def get_index(self) -> Index:
        """ Get the Index object from the h5 file."""
        return Index(self.h5)
    
    def get_attrs(self) -> dict:
        """ Get the attributes from the h5 file."""
        attrs = {}
        for key in self.h5.attrs:
            attrs[key] = self.h5.attrs[key]
        return attrs
    
    def get_cell_labels(self) -> np.ndarray:
        """ Get the cell labels from the h5 file.
        Cell labels are string, we retrieve them in 'str' type, i.e. unicode."""
        return self.h5['cell_labels'][:].astype(str)
    
    def get_cellnum(self, cellID: str) -> int:
        """ Get the cell number of the input cellID.
        The number is the index of cellID in the cell_labels array."""
        cell_labels = self.get_cell_labels()
        return np.where(cell_labels == cellID)[0][0]
    
    def get_cellID(self, cellnum: int) -> str:
        """ Get the cellID of the input cell number.
        The cellID is the value of cell_labels at index cellnum."""
        cell_labels = self.get_cell_labels()
        return cell_labels[cellnum]
    
    def get_cell_states(self) -> np.ndarray:
        """ Get the cell states from the h5 file.
        Cell states are string, we retrieve them in 'str' type, i.e. unicode."""
        return self.h5['cell_states'][:].astype(str)
    
    def get_volumes(self) -> np.ndarray:
        """ Get the cell volumes from the h5 file."""
        return self.h5['volumes'][:]
    
    def get_feature(self, feature: str, cellID: str = None) -> np.ndarray:
        """ Get the feature matrix from the h5 file.
        The feature matrix is a 3D array of shape ncells x ndomains x ncopies.
        It can be retrieved for all cells or for a specific cellID.
        
        Args:
            name (str): name of the feature matrix to retrieve.
            cellID (str, optional): cell ID to retrieve the feature matrix. Defaults to None.
        
        Returns:
            np.ndarray: feature matrix of shape ncells x ndomains x ncopies (if cellID is None), otherwise of shape ndomains x ncopies.
        """
        if not feature in self:
            raise ValueError(f"The feature matrix '{feature}' does not exist in the h5 file.")
        if cellID is None:
            return self.h5[feature][:]
        else:
            cellnum = self.get_cellnum(cellID)
            return self.h5[feature][cellnum, :, :]
    
    def get_feature_documentation(self, name: str) -> str:
        """ Get the documentation of the feature matrix from the h5 file."""
        if not name in self:
            raise ValueError(f"The feature matrix '{name}' does not exist in the h5 file.")
        return self.h5[name].attrs['doc']
    
    def get_feature_list(self) -> list:
        """ Get the list of feature matrices in the h5 file."""
        # Get the list of keys in the h5 file
        h5_keys = list(self.h5.keys())
        # Remove the keys that are not feature matrices
        remove_keys = ['index', 'genome', 'cell_labels', 'cell_states', 'volumes']
        for key in remove_keys:
            if key in h5_keys:
                h5_keys.remove(key)
        return h5_keys
    
    
    # DEFINE PROPERTIES (READ ONLY)
    index = property(get_index, doc="Index object.")
    attrs = property(get_attrs, doc="Attributes.")
    cell_labels = property(get_cell_labels, doc="Cell labels.")
    cell_states = property(get_cell_states, doc="Cell states.")
    volumes = property(get_volumes, doc="Cell volumes.")
    feature_list = property(get_feature_list, doc="List of feature matrices.")
    
    
    # INPUT/OUTPUT FUNCTIONS
    
    def close(self) -> None:
        """ Close the HDF5 file."""
        self.h5.close()
    
    def add_key_to_attrs(self, key: str, value) -> None:
        """ Add a key to the attributes in the h5 file."""
        self.h5.attrs[key] = value
    
    def add_index_attrs_cell_labels(self, index: Index, attrs: dict, cell_labels: np.ndarray) -> None:
        """ Add the Index object, the attributes, and the cell labels to the h5 file, checking consistency.

        Args:
            index (Index)
            attrs (dict): attributes of the data.
            cell_labels (np.ndarray, str): array with the cell IDs.
        """
        
        # Check that the Index object is valid
        if not isinstance(index, Index):
            raise TypeError("index must be an Index object.")
        
        # Check that the attributes are valid
        if not isinstance(attrs, dict):
            raise TypeError("attrs must be a dictionary.")
        required_attrs_keys = ['ncell', 'max_ntrace_per_chrom']
        for key in required_attrs_keys:
            if key not in attrs:
                raise ValueError("attrs must have the key '{}'.".format(key))
        
        # Check that the cell labels are valid
        if not isinstance(cell_labels, np.ndarray):
            raise TypeError("cell_labels must be a numpy array.")
        if not issubclass(cell_labels.dtype.type, np.str_):
            raise TypeError("cell_labels must be a numpy array of strings.")
        
        self.set_index(index)
        self.set_attrs(attrs)
        self.set_cell_labels(cell_labels)
    
    def add_feature(self, matrix: np.ndarray, feature: str, doc: str = '') -> None:
        """ Add a feature matrix to the h5 file, checking consistency.

        Args:
            matrix (np.ndarray, int/float): matrix of shape ncells x ndomains x ncopies.
            feature (str): name of the feature associated to the matrix.
            doc (str, optional): documentation of the feature matrix. Defaults to ''.
        """
        
        # Check that the feature name is valid
        if not isinstance(feature, str):
            raise TypeError("The name of the matrix must be a string.")
        
        # Check that the matrix is a numpy array
        if not isinstance(matrix, np.ndarray):
            raise TypeError("The matrix must be a numpy array.")
        # Check that the matrix is either int or float
        if not np.issubdtype(matrix.dtype, np.integer) and not np.issubdtype(matrix.dtype, np.floating):
            raise TypeError("The matrix must be a numpy array of integers or floats.")
        # Check that the matrix has the right shape
        shape_expected = (len(self.cell_labels), len(self.index), self.attrs['max_ntrace_per_chrom'])
        if not matrix.shape == shape_expected:
            raise ValueError("The shape of the matrix is not valid. Expected: {}, Found: {}.".format(shape_expected, matrix.shape))
        
        self.set_feature(matrix, feature, doc)
    
    def add_volumes(self, volumes: np.ndarray) -> None:
        """ Add the cell volumes to the h5 file, checking consistency.

        Args:
            volumes (np.ndarray): numpy array of cell volumes of shape (ncells,)
        """
        
        # Check that volumes is a numpy array
        if not isinstance(volumes, np.ndarray):
            raise TypeError("volumes must be a numpy array.")
        # Check that volumes is a float array
        if not np.issubdtype(volumes.dtype, np.floating):
            raise TypeError("volumes must be a numpy array of floats.")
        # Check that volumes has the right shape (ncells,)
        if len(volumes) != len(self.cell_labels):
            raise TypeError("volumes must have the same number of cells as cell_labels.")
        
        self.set_volumes(volumes)
    
    def add_cell_states(self, cell_states: np.ndarray) -> None:
        """ Add the cell states to the h5 file, checking consistency.

        Args:
            cell_states (np.ndarray): numpy array of cell states of shape (ncells,)
        """
        
        # Check that cell_states is a numpy array
        if not isinstance(cell_states, np.ndarray):
            raise TypeError("volumes must be a numpy array.")
        # Check that cell_states is a string array
        if not np.issubdtype(cell_states.dtype, np.str_):
            raise TypeError("cell_states must be a numpy array of strings.")
        # Check that cell_states has the right shape (ncells,)
        if len(cell_states) != len(self.cell_labels):
            raise TypeError("cell_states must have the same number of cells as cell_labels.")
        
        self.set_cell_states(cell_states)
    
    def add_data_from_cte(self, cte: ChromatinTracingExperiment) -> None:
        """ Add the data from a ChromatinTracingExperiment object to the SCF object.
        
        It checks that the CTE object has the 'index' and 'cell_labels' datasets,
        and if so it adds them to the SCF object, together with the 'attrs' dictionary.
        
        If the CTE object also has the 'alphashapes' and/or 'cell_states' datasets, it adds them to the SCF too.

        Args:
            cte (ChromatinTracingExperiment)
        """
        
        # Check that the ChromatinTracingExperiment object is valid: must contain 'index', 'cell_labels'
        required_data = ['index', 'cell_labels']
        for key in required_data:
            if key not in cte:
                raise ValueError(f"The ChromatinTracingExperiment object must have the key '{key}'.")
        
        # Add the index/attributes/cell_labels
        self.add_index_attrs_cell_labels(cte.index, cte.attrs, cte.cell_labels)
        
        # Add the volumes if present
        if 'alphashapes' in cte:
            volumes = []
            for cellID in cte.cell_labels:
                volumes.append(cte.get_alphashapes(cellID)['mesh'].volume)
            volumes = np.array(volumes, dtype=np.float32)
            self.add_volumes(volumes)
        
        # Add the cell states if present
        if 'cell_states' in cte:
            self.add_cell_states(cte.cell_states)
    
    def pop_cells(self, cellIDs_topop: list) -> None:
        """ Remove cells from the SCF object in place.
        
        It is assumed that the Index doesn't change after the cells are removed.
        
        The attributes also doesn't change, but two additional keys are added:
        - ncells_removed: number of removed cells.
        - ncells_remaining: number of remaining cells.

        Args:
            cellIDs_topop (list): list of cellIDs to remove.
        """
        
        if 'cell_labels' not in self:
            return None
        
        # Create a mask to select the cells to keep
        mask = np.isin(self.cell_labels, cellIDs_topop, invert=True)  # True for cells to keep
        
        # Remove the cellIDs from the cell labels
        cell_labels = self.cell_labels[mask]
        del self.h5['cell_labels']
        self.set_cell_labels(cell_labels)
        del cell_labels
        
        # Remove the cellIDs from the cell states
        if 'cell_states' in self:
            cell_states = self.cell_states[mask]
            del self.h5['cell_states']
            self.set_cell_states(cell_states)
            del cell_states
        
        # Remove the cellIDs from the volumes
        if 'volumes' in self:
            volumes = self.volumes[mask]
            del self.h5['volumes']
            self.set_volumes(volumes)
            del volumes
        
        # Remove the cellIDs from all the feature matrices
        for feature in self.feature_list:
            mat = self.get_feature(feature)[mask, :, :]
            doc = self.get_feature_documentation(feature)
            del self.h5[feature]
            self.set_feature(mat, feature, doc)
            del mat
        
        # Get the new number of cells
        ncell_new = len(self.cell_labels)
        
        # Get the number of removed cells and the remaining cells
        ncell_removed = self.attrs['ncell'] - ncell_new
        
        # Include these numbers in the attributes
        self.add_key_to_attrs('ncell_removed', ncell_removed)
        self.add_key_to_attrs('ncell_remaining', ncell_new)
    
    def get_feature_by_spotIDs(
        self, cellID: str, cte: ChromatinTracingExperiment, feature: str, nquants: int = None
    ) -> np.ndarray:
        """ Get the feature values for a cell as a 1D array of shape (nspots,),
        ordered as the spots in the ChromatinTracingExperiment object.

        Args:
            cellID (str)
            cte (ChromatinTracingExperiment)
            feature (str)
            nquants (int, optional): Number of quantiles to quantize the feature values. Optional

        Returns:
            featvals (np.ndarray, shape (nspots,)): feature values for the cell ordered as the spots in the CTE.
        """
        
        # Get the feature matrix for the cell
        feature_mat = self.get_feature(feature, cellID)  # shape (ndomains, ncopies)
        
        # If the number of quantiles is provided, quantize the feature values
        if nquants is not None:
            if not isinstance(nquants, int):
                raise TypeError("nquants must be an integer.")
            if nquants < 1 or nquants > 100:
                raise ValueError(f"nquants must be between 1 and 100. Got {nquants}.")
            feature_mat = scf_utils.quantize_matrix_cell(feature_mat, nquants)
        
        # Create a hash table for the index
        index_hash = self.index.get_index_hashmap()
        
        # Get the domain info (traceIDs, chroms, starts, ends) of each spot from the CTE
        _, _, _, chroms, starts, ends, _, traceIDs, _ = cte.get_data(cellID, format='numpy')
        # Get the hash table for traceIDs
        traceID_hash = cte.get_trace_hashmap(cellID)
        
        # Collect the feature values following the order of the spots
        featvals = []
        for traceID, chrom, start, end in zip(traceIDs, chroms, starts, ends):
            
            # Get the position of the spot in the array using the hash tables
            i_domain = index_hash[(chrom, start, end)]
            assert len(i_domain) == 1, f"Multiple domains found for {chrom}:{start}-{end} in cell {cellID}."
            i_domain = i_domain[0]
            i_trace = traceID_hash[chrom][traceID]
            
            # Get the feature value
            featval = feature_mat[i_domain, i_trace]
            featvals.append(featval)
        
        return np.array(featvals)
    
    
    # COMPUTATION FUNCTIONS
    
    def haploid_profile(
        self,
        feature: str,
        isolate_state: str = None,
        resolution: int = None,
        norm_by_radii: bool = False,
        norm_by_zscore: bool = False,
        ) -> tuple:
        """ Computes a 1D haploid profile of the required feature matrix, providing the mean and the standard deviation.
        If isolate_state is provided, it is computed only for the cells in that state (e.g. S phase).
        The feature matrix can be normalized by the cell volume and/or z-scored (if both are True, the feature matrix is first normalized by the cell volume and then z-scored).
        If cutoff is provided, the function computes an association frequency signal with the provided cutoff.

        Args:
            feature (str): name of the feature matrix to compute the profile.
            isolate_state (str, optional): cell state to isolate. Defaults to None.
            norm (bool, optional): if True, the feature matrix is normalized by the cell effective radius. Defaults to False.
            zscore (bool, optional): if True, the feature matrix is z-scored. Defaults to False.

        Returns:
            (np.ndarray): 1D haploid mean profile of the data.
            (np.ndarray): 1D haploid standard deviation profile of the data.
        """
        
        # Get the feature matrix
        mat = self.get_feature(feature)
        
        # Get the feature matrix
        # If resolution is provided, get the coarse-grained matrix
        if resolution is not None:
            method = 'consensus' if 'association' in feature else 'average'
            mat, _ = scf_utils.coarsegrain_matrix(mat, self.index, resolution, method)
        
        # If requested, normalize the feature matrix
        if norm_by_radii:
            radii = (3 / (4 * np.pi) * self.volumes) ** (1/3)  # effective radius of the cells
            mat = scf_utils.normalize_matrix(mat, norm_arr=radii)
        if norm_by_zscore:
            mat = scf_utils.normalize_matrix(mat, by_zscore=True)
        
        # Select only cells in the specified state if isolate_state is provided
        if isolate_state is not None:
            if not 'cell_states' in self:
                raise ValueError("Cell states are not defined. Cannot isolate state.")
            if not isolate_state in self.cell_states:
                raise ValueError("State {} is not defined. Cannot isolate state.".format(isolate_state))
            mask = self.cell_states == isolate_state
        # Otherwise, select all cells
        else:
            mask = np.ones(len(self.cell_labels), dtype=bool)
        
        # Compute the mean and standard deviation
        mean = np.nanmean(mat[mask, :, :], axis=(0, 2))  # np.array of shape (ndomains,)
        std = np.nanstd(mat[mask, :, :], axis=(0, 2))
        
        return mean, std
    
    def perform_ttest(self, feature: str, states: list, resolution: int, norm_by_radii: bool = False, norm_by_zscore: bool = False, correct_fdr: bool = True) -> (np.ndarray, np.ndarray, Index):
        """ Performs a two-sample t-test on the feature matrix between the two specified states.
        The p-values are computed for each bin of the index, at the specified resolution.
        The feature matrix can be normalized by the cell volume and/or z-scored (if both are True, the feature matrix is first normalized by the cell volume and then z-scored).
        The function also returns a sign for each bin, indicating whether the first state is up-regulated (1) or down-regulated (-1) compared to the second state.
        

        Args:
            feature (str): name of the feature matrix to perform the t-test.
            states (list): list of two states to compare.
            resolution (int): resolution of the index to perform the t-test.
            norm_by_radii (bool, optional): if True, the feature matrix is normalized by the cell effective radius. Defaults to False.
            norm_by_zscore (bool, optional): if True, the feature matrix is z-scored. Defaults to False.
            correct_fdr (bool, optional): if True, the p-values are corrected for multiple testing using the Benjamini-Hochberg procedure. Defaults to True.

        Returns:
            pvals (np.ndarray): array of p-values of the t-test.
            signs (np.ndarray): array of signs of the difference (1 if state 1 > state 2, -1 if state 1 < state 2).
            index_coarse (Index): coarse-grained index at the specified resolution.
        """
        
        if not len(states) == 2:
            raise ValueError("The states list must contain exactly two states.")
        if states[0] not in self.cell_states or states[1] not in self.cell_states:
            raise ValueError("One or both states are not defined in the cell_states array.")
        
        # Get the feature matrix
        mat = self.get_feature(feature)
        
        # If requested, normalize the feature matrix
        if norm_by_radii:
            radii = (3 / (4 * np.pi) * self.volumes) ** (1/3)  # effective radius of the cells
            mat = scf_utils.normalize_matrix(mat, norm_arr=radii)
        if norm_by_zscore:
            mat = scf_utils.normalize_matrix(mat, by_zscore=True)
        
        # Get the feature matrix for the two states
        mat_1 = mat[self.cell_states == states[0], :, :]
        mat_2 = mat[self.cell_states == states[1], :, :]
        
        # Coarse-grain the index to the required resolution
        index_coarse = self.index.coarsegrain(resolution)
        
        # Initialize the array of p-values and sign (wheather it's up or down-regulated)
        pvals = np.zeros(len(index_coarse)).astype('float32')
        signs = np.zeros(len(index_coarse)).astype('int32')
        
        # Get mappings to coarse-grain the signals in the index
        _, _, bmap = None, None, None  # TODO: previous function was bugged, need to fix it
        
        # Loop over the bins of the coarse index
        for i in range(len(index_coarse)):
            
            # Get the indices of the fine-grain bins that are included in the coarse-grain bin i
            indices = bmap[i]
            
            # Get the data of both states for these indices
            data_1 = mat_1[:, indices, :].flatten()
            data_2 = mat_2[:, indices, :].flatten()
            # Remove NaNs
            data_1 = data_1[~np.isnan(data_1)]
            data_2 = data_2[~np.isnan(data_2)]
            
            # Compute the p-value
            _, pval = stats.ttest_ind(data_1, data_2, equal_var=False)
            
            # Store the p-value
            pvals[i] = pval
            
            # Compute the sign of the difference
            sign = np.sign(np.nanmean(data_1) - np.nanmean(data_2))  # positive if data_1 > data_2
            signs[i] = sign
        
        # Correct the p-values for multiple testing
        if correct_fdr:
            pvals = fdrcorrection(pvals)[1]
        
        return pvals, signs, index_coarse

    def identify_ds_regions_by_association(
        self,
        feature: str,
        states: list,
        resolution: int,
        top_percentile: int = 90,
        bottom_percentile: int = 50
    ) -> tuple:
        """ Identifies differentially structured regions between the two specified states based on the association profile.

        Args:
            feature (str): name of the feature matrix to analyze.
            states (list): list of two states to compare.
            resolution (int): coarse-grained resolution to perform the analysis.
            top_percentile (int, optional): percentile to define the top regions. Defaults to 90.
            bottom_percentile (int, optional): percentile to define the bottom regions. Defaults to 50.

        Returns:
            (np.ndarray): array of scores of the differentially structured regions.
            (np.ndarray): array of signs of the differentially structured regions (1 if state 1 > state 2, -1 if state 1 < state 2).
            (Index): coarse-grained index at the specified resolution.
        """
        
        if not len(states) == 2:
            raise ValueError("The states list must contain exactly two states.")
        if states[0] not in self.cell_states or states[1] not in self.cell_states:
            raise ValueError("One or both states are not defined in the cell_states array.")
        
        # Get the association profiles for the two states at the specified resolution
        profile_avg_1, _ = self.haploid_profile(feature, isolate_state=states[0], resolution=resolution)
        profile_avg_2, _ = self.haploid_profile(feature, isolate_state=states[1], resolution=resolution)
        
        # Identify the regions that are in the top% with the highest association in both states
        top_1 = profile_avg_1 > np.percentile(profile_avg_1, top_percentile)
        top_2 = profile_avg_2 > np.percentile(profile_avg_2, top_percentile)
        
        # Identify the regions that are in the bottom% in both states
        bottom_1 = profile_avg_1 < np.percentile(profile_avg_1, bottom_percentile)
        bottom_2 = profile_avg_2 < np.percentile(profile_avg_2, bottom_percentile)
        
        # Identify the regions that are in the top 10% in state 1 and in the bottom 50% in state 2 and vice versa
        ds_regions_1 = np.logical_and(top_1, bottom_2)
        ds_regions_2 = np.logical_and(top_2, bottom_1)
        
        # As score, calculate the log2 ratio of the average associations in the two states
        scores = np.log2(profile_avg_1 / profile_avg_2)

        # Set the score as NaN in the regions that are not differentially structured
        scores[~np.logical_or(ds_regions_1, ds_regions_2)] = np.nan
        
        # Define as sign +1 the ds_regions_1 and -1 the ds_regions_2, and 0 the rest
        signs = np.zeros(len(profile_avg_1)).astype('int32')
        signs[ds_regions_1] = 1
        signs[ds_regions_2] = -1
        
        # Finally, calculate the coars-grained index
        index_coarse = self.index.coarsegrain(resolution)
        
        return scores, signs, index_coarse
    
    
    def stack_n_sort_matrix(self, feature: str, isolate_state: str = None, sorter: np.ndarray = None, resolution: int = None, norm_by_zscore: bool = False) -> tuple:
        """ Transform the feature matrix into a 2D array of shape (ncell * ncopy_max, ndomain).
        Each row corresponds to a cell and a copy of the feature matrix, and copies of the same cell are stacked.
        The cells are sorted by the sorter array. If sorter is not provided, the cells are sorted by volume.

        Args:
            feature (str): name of the feature matrix to transform.
            isolate_state (str, optional): cell state to isolate. Defaults to None.
            sorter (np.ndarray, optional): array to sort the cells. Defaults to None (sort by volume).

        Returns:
            (np.ndarray): stacked and sorted matrix, 2D array of shape (ncell * ncopy_max, ndomain).
            (np.ndarray): sorted array of the sorter, 1D array of shape (ncell * ncopy_max).
        """
        
        # Get the feature matrix
        mat = self.get_feature(feature)
        
        # Coarse-grain the matrix if resolution is provided
        if resolution is not None:
            method = 'consensus' if 'association' in feature else 'average'
            mat, _ = scf_utils.coarsegrain_matrix(mat, self.index, resolution, method)
        
        # If requested, normalize the feature matrix
        if norm_by_zscore:
            mat = scf_utils.normalize_matrix(mat, by_zscore=True)
        
        # If sorter is not provided, sort the cells by volume
        if sorter is None:
            sorter = self.volumes
        if not len(sorter) == len(self.cell_labels):
            raise ValueError("The sorter array must have the same length as the number of cells.")
        
        # Select only cells in the specified state if isolate_state is provided
        if isolate_state is not None:
            if not 'cell_states' in self:
                raise ValueError("Cell states are not defined. Cannot isolate state.")
            if not isolate_state in self.cell_states:
                raise ValueError("State {} is not defined. Cannot isolate state.".format(isolate_state))
            mask = self.cell_states == isolate_state
        # Otherwise, select all cells
        else:
            mask = np.ones(len(self.cell_labels), dtype=bool)
        mat = mat[mask, :, :]
        sorter = sorter[mask]
        
        # Sort the cells by the sorter array
        mat_srt = mat[np.argsort(sorter), :, :]
        sorter_srt = sorter[np.argsort(sorter)]
        
        # Reshape the matrix to a 2D array (ncell * ncopy_max, ndomain)
        ncell, ndomain, ncopy_max = mat_srt.shape
        mat_srt_stack = np.zeros((ncell * ncopy_max, ndomain), dtype=mat_srt.dtype)
        for i_cell in range(ncell):
            for i_copy in range(ncopy_max):
                mat_srt_stack[i_cell * ncopy_max + i_copy, :] = mat_srt[i_cell, :, i_copy]
        
        return mat_srt_stack, sorter_srt
    