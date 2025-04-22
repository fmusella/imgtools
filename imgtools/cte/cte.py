import os
import h5py
import numpy as np
from .fofct import read_fofct
from alabtools.utils import Index, get_index_from_bed
from .validator import CTEData
from . import cte_utils
from . import cte_io


class ChromatinTracingExperiment:
    """ A class to store and manipulate data from a Chromatin Tracing (CT) Experiment, like DNAseqFISH+.
    
    The chromosomal domains are described with the Index object, while the imaging data are organized
    in a nested dictionary with a specific structure, defined in the validator module.
    The rational is that the experiment is divided in cells, chromosomes, traces (or copies) and spots.
    The data structure allows to easily parse the data among these levels.
    
    ----------
    Attributes:
        h5_name (str): path and name of the HDF5 file.
        h5 (h5py.File): HDF5 file to store the data.
                        Contains the following groups:
                        - index: Index object.
                        - attrs: attributes.
                        - cell_labels: array with the cell IDs.
                        - data: nested dictionary (saved as arrays) with the data.
                        (optional)
                        - cell_states: array with the cell states.
                        - alphashapes: dictionary with the alpha shapes (saved as arrays).
        ---------- 
    Properties (from h5 file):
        index (Index): Index object.
        attrs (dict): attributes.
        cell_labels (np.ndarray): array with the cell IDs.
        cell_states (np.ndarray): array with the cell states.
    --------------------
    """
    
    def __init__(self, h5_name: str, mode: str = 'r') -> None:
        """ Initialize the ChromatinTracingExperiment object.
        
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
        
        # If the file doesn't exists, make sure that mode is 'w' or 'w-'
        if not os.path.exists(h5_name) and mode not in ['w', 'w-', 'x']:
            raise FileNotFoundError("The HDF5 file does not exist. Use mode 'w', 'w-', or 'x'.")
        
        # Open the HDF5 file
        self.h5_name = h5_name
        self.h5 = h5py.File(h5_name, mode)
    
    
    # CONTAINS FUNCTION
    def __contains__(self, name: str) -> bool:
        """ Check if a group or dataset exists in the HDF5 file."""
        return name in self.h5
    
    
    # SETTER FUNCTIONS
    
    def set_index(self, index: Index) -> None:
        """ Set the index in the HDF5 file."""
        cte_io.save_index_to_hdf5(index, self.h5)
    
    def set_attrs(self, attrs: dict) -> None:
        """ Set the attributes in the HDF5 file."""
        cte_io.save_attrs_to_hdf5(attrs, self.h5)
    
    def set_cell_labels(self, cell_labels: np.ndarray) -> None:
        """ Set the cell labels in the HDF5 file."""
        cte_io.save_cell_labels_to_hdf5(cell_labels, self.h5)
    
    def set_cell_states(self, cell_states: np.ndarray) -> None:
        """ Set the cell states in the HDF5 file."""
        # Check that cell_states and cell_labels have the same length
        if len(cell_states) != len(self.cell_labels):
            raise ValueError("cell_states and cell_labels must have the same length.")
        cte_io.save_cell_states_to_hdf5(cell_states, self.h5)
    
    def set_triad_labels(self, triad_labels: np.ndarray = None) -> None:
        """ Set the triad labels in the HDF5 file, i.e. cellID / chrom / traceID triads.
        If triad_labels is None, it is calculated from the data."""
        if triad_labels is None:
            triad_labels = self.calculate_triad_labels()
        cte_io.save_triad_labels_to_hdf5(triad_labels, self.h5)
    
    def set_cell_data(self, cellID: str, cell_data: dict) -> None:
        """ Set the data for a cell in the HDF5 file."""
        cte_io.save_cell_data_to_hdf5(cellID, cell_data, self.h5)
    
    def set_data(self, data: dict) -> None:
        """ Set the data in the HDF5 file."""
        cte_io.save_data_to_hdf5(data, self.h5)
    
    def set_alphashapes(self, alphashapes: dict) -> None:
        """ Set the alphashapes in the HDF5 file."""
        # Check that alphashapes and cell_labels have the same length
        if len(alphashapes) != len(self.cell_labels):
            raise ValueError("alphashapes and cell_labels must have the same length.")
        # Check that all cellIDs in alphashapes are in cell_labels
        for cellID in alphashapes:
            if cellID not in self.cell_labels:
                raise ValueError("cellID {} not in cell labels.".format(cellID))
        cte_io.save_alphashapes_to_hdf5(alphashapes, self.h5)
    
    def set_data_attrs_index(
        self,
        data: dict,
        assembly: str = None,
        index: Index = None,
        attrs: dict = None,
        check_data: bool = False
    ) -> None:
        """ Set data, assembly, index and attributes.
        Checks that the data (dict) is in the correct format if check_data is True.
        Derives the Index and attributes from the data, if not provided.

        Args:
            data (dict): data in dictionary format.
            assembly (str, optional): assembly name. Defaults to None.
            index (Index, optional): Index object. Defaults to None.
            attrs (dict, optional): attributes. Defaults to None.
            check_data (bool, optional): check that the data is in the correct format. Defaults to True.
        """
        
        # Check that either index or assembly is provided
        if index is None and assembly is None:
            raise IOError("Either index or assembly must be provided.")
        
        # Get the assembly from the index if it is not provided
        if assembly is None:
            assembly = index.genome.assembly
        
        # If check_data, use pydantic to check that the data is in the correct format
        # This might slow down the code, so it can be turned off
        if check_data:
            checker = CTEData(root=data)
            del checker
        
        # Get the Index and the attributes from the data, if they haven't been provided
        if index is None or attrs is None:
            index_inferred, attrs_inferred = cte_utils.get_index_and_attrs(data, assembly)
        # Use the inferred Index and attributes if they haven't been provided
        if index is None:
            index = index_inferred
        if attrs is None:
            attrs = attrs_inferred
        
        # Set the index, attributes, cell labels, and data
        self.set_index(index)
        self.set_attrs(attrs)
        cell_labels = np.array([cellID for cellID in data.keys()]).astype(str)
        self.set_cell_labels(cell_labels)
        self.set_data(data)
    
    
    # GETTER FUNCTIONS
    
    def get_index(self) -> Index:
        """ Get the index."""
        return Index(self.h5)
    
    def get_attrs(self) -> dict:
        """ Get the attributes."""
        return cte_io.load_attrs_from_hdf5(self.h5)
    
    def get_cell_labels(self) -> np.ndarray:
        """ Get the cell labels."""
        return self.h5['cell_labels'][:].astype(str)
    
    def get_cellID(self, cellnum: int) -> str:
        """ Get the cellID corresponding to a cell number."""
        cell_labels = cte_io.load_cell_labels_from_hdf5(self.h5)
        return cell_labels[cellnum]
    
    def get_cellnum(self, cellID: str) -> int:
        """ Get the cell number corresponding to a cellID. """
        cell_labels = cte_io.load_cell_labels_from_hdf5(self.h5)
        if cellID not in cell_labels:
            raise ValueError("cellID {} not in cell labels.".format(cellID))
        return np.where(np.array(cell_labels) == cellID)[0][0]

    def get_cell_states(self) -> np.ndarray:
        """ Get the cell states."""
        return cte_io.load_cell_states_from_hdf5(self.h5)
    
    def get_triad_labels(self) -> np.ndarray:
        """ Get the triad labels (i.e. cellID / chrom / traceID triads)."""
        return cte_io.load_triad_labels_from_hdf5(self.h5)
    
    def get_data(self, cellID: str, chrom: str = None, traceID: str = None, format: str = 'dict'):
        """ Get the data for a cell, a chromosome in a cell, or a trace in a chromosome in a cell.

        Args:
            cellID (str)
            chrom (str, optional)
            traceID (str, optional)
            format (str, optional): 'dict' or 'numpy'. Defaults to 'dict'.

        Returns:
            If chrom is None and traceID is None:
                - format=='dict': data is a dictionary with the format:
                    data[chrom][traceID][spotID] = {'x': float, 'y': float, 'z': float, 'chrom': str, 'start': int, 'end': int, 'lum': float}.
                - format=='numpy': data is a tuple of numpy arrays:
                    (xs, ys, zs, chroms, starts, ends, lums, traceIDs, spotIDs)
            If chrom is not None and traceID is None:
                - format=='dict': data is a dictionary with the format:
                    data[traceID][spotID] = {'x': float, 'y': float, 'z': float, 'chrom': str, 'start': int, 'end': int, 'lum': float}.
                - format=='numpy': data is a tuple of numpy arrays:
                    (xs, ys, zs, starts, ends, lums, traceIDs, spotIDs)
            If chrom is not None and traceID is not None:
                - format=='dict': data is a dictionary with the format:
                    data[spotID] = {'x': float, 'y': float, 'z': float, 'chrom': str, 'start': int, 'end': int, 'lum': float}.
                - format=='numpy': data is a tuple of numpy arrays:
                    (xs, ys, zs, starts, ends, lums, spotIDs)
        """
        if chrom is None and traceID is None:
            return cte_io.load_cell_data_from_hdf5(cellID, self.h5, format)
        elif chrom is not None and traceID is None:
            return cte_io.load_chrom_data_from_hdf5(cellID, chrom, self.h5, format)
        elif chrom is not None and traceID is not None:
            return cte_io.load_trace_data_from_hdf5(cellID, chrom, traceID, self.h5, format)
    
    def get_alphashapes(self, cellID: str = None) -> dict:
        """ Get the alphashapes for a cell (if cellID is provided) or for all cells, in the format:
                alphashapes[cellID] = alphashapes[cellID] = {'alpha': float, 'mesh': trimesh.Trimesh}. """
        # If cellID is provided, return the alphashape for that cell
        if cellID is not None:
            return cte_io.load_cell_alphashape_from_hdf5(cellID, self.h5)
        # Otherwise, return all alphashapes as a dictionary
        alphashapes = {}
        for cellID in self.cell_labels:
            alphashapes[cellID] = cte_io.load_cell_alphashape_from_hdf5(cellID, self.h5)
        return alphashapes
    
    
    # DEFINE PROPERTIES (READ ONLY)
    index = property(get_index, doc="Index object.")
    attrs = property(get_attrs, doc="Attributes.")
    cell_labels = property(get_cell_labels, doc="Cell labels.")
    cell_states = property(get_cell_states, doc="Cell states.")
    
    
    # INPUT/OUTPUT FUNCTIONS
    
    def close(self) -> None:
        """ Close the HDF5 file."""
        self.h5.close()
    
    def check_consistency(self) -> None:
        """ Checks the consistency of the HDF5 file."""
        cte_io.check_consistency(self.h5)
       
    def read_from_fofct(
        self, filename: str, assembly: str, coord_scaling: tuple = (1, 1, 1), check_data: bool = False
    ) -> None:
        """ Read data from a fofct file.
        Data is stored in the data attribute of the ChromatinTracingExperiment object.
        
        The X, Y, Z coordinates can be rescaled with the coord_scaling parameter:
           X = X * coord_scaling[0], Y = Y * coord_scaling[1], Z = Z * coord_scaling[2].

        Args:
            filename (str): path to the fofct file.
            assembly (str): assembly name.
            coord_scaling (tuple, optional): tuple with the scaling factors for the X, Y and Z coordinates.
            check_data (bool, optional): if True, check that the data is in the correct format.
        """
        
        # Check that filename is a string, a .csv file, and that it exists
        if not isinstance(filename, str):
            raise TypeError("filename must be a string.")
        if not filename.endswith('.csv'):
            raise ValueError("filename must be a .csv file.")
        if not os.path.exists(filename):
            raise FileNotFoundError("File {} does not exist.".format(filename))
        
        # Check that coord_scaling is a tuple of three numbers
        if not isinstance(coord_scaling, tuple):
            raise TypeError("coord_scaling must be a tuple.")
        if len(coord_scaling) != 3:
            raise ValueError("coord_scaling must be a tuple of three floats.")
        if not all(isinstance(x, (int, float)) for x in coord_scaling):
            raise TypeError("coord_scaling must be a tuple of three floats.")

        data = read_fofct(filename, coord_scaling)
        
        index, attrs = cte_utils.get_index_and_attrs(data, assembly)

        self.set_data_attrs_index(data, assembly, index, attrs, check_data)
    
    def merge(self, other, filename: str, tag1: str = None, tag2: str = None) -> None:
        """ Merge two ChromatinTracingExperiment objects.
        If there is an overlap between the cell labels, tag1 and tag2 must be provided to distinguish the cells.

        Args:
            other (ChromatinTracingExperiment): the other ChromatinTracingExperiment object to merge.
            filename (str): name of the file (with path) where the merged object will be saved.
            tag1 (str, optional): string to distinguish the cells in the first ChromatinTracingExperiment object.
                                  Defaults to None, in which case the cell labels must be different.
            tag2 (str, optional): string to distinguish the cells in the second ChromatinTracingExperiment object.
                                  Defaults to None, in which case the cell labels must be different.
        """
        
        # Check that other is a ChromatinTracingExperiment object
        if not isinstance(other, ChromatinTracingExperiment):
            raise TypeError("other must be a ChromatinTracingExperiment object.")
        
        # Check that the index are the same
        if self.index != other.index:
            raise ValueError("Cannot merge ChromatinTracingExperiment objects with different indices.")
        
        # Initialize the merged ChromatinTracingExperiment object
        merged = ChromatinTracingExperiment(filename, 'w')
        
        # Set the index of the new object
        merged.set_index(self.index)
        
        # Get the attributes of the merged data
        attrs_merged = cte_utils.get_merged_attrs(self.attrs, other.attrs)
        # Set the attributes of the new object
        merged.set_attrs(attrs_merged)
        
        # Get the cell labels of the merged data
        cell_labels_1 = self.cell_labels
        cell_labels_2 = other.cell_labels
        # Add the tags to the cell labels
        if tag1 is not None:
            cell_labels_1 = np.array([cellID + '_' + tag1 for cellID in cell_labels_1]).astype(str)
        if tag2 is not None:
            cell_labels_2 = np.array([cellID + '_' + tag2 for cellID in cell_labels_2]).astype(str)
        # Check that there is no overlap between the cell labels
        if len(set(cell_labels_1).intersection(set(cell_labels_2))) > 0:
            raise ValueError("There is an overlap between the cell labels. Provide tags to distinguish the cells.")
        # Merge the cell labels
        cell_labels_merged = np.concatenate((cell_labels_1, cell_labels_2)).astype(str)
        # Set the cell labels of the new object
        merged.set_cell_labels(cell_labels_merged)
        
        # Set the data of the merged object
        cte_io.merge_group_from_hdf5('data', self.h5, other.h5, merged.h5, tag1, tag2)
        
        # If the cell states are present in both objects, merge them
        if 'cell_states' in self.h5 and 'cell_states' in other.h5:
            cell_states_1 = self.cell_states
            cell_states_2 = other.cell_states
            cell_states_merged = np.concatenate((cell_states_1, cell_states_2)).astype(str)
            merged.set_cell_states(cell_states_merged)
        
        # If the alphashapes are present in both objects, merge them
        if 'alphashapes' in self.h5 and 'alphashapes' in other.h5:
            cte_io.merge_group_from_hdf5('alphashapes', self.h5, other.h5, merged.h5, tag1, tag2)
        
        # Check the consistency of the merged object
        merged.check_consistency()
        
        merged.close()
    
    def pop_cells(self, cellIDs_topop: list) -> None:
        """ Remove cells from the CTE object in place.
        
        It is assumed that the Index doesn't change after the cells are removed.
        
        The attributes also doesn't change, but two additional keys are added:
        - ncells_removed: number of removed cells.
        - ncells_remaining: number of remaining cells.
        
        Args:
            cellIDs_topop (list): list of cellIDs to remove.
        """
        
        # We have to remove cell_labels at the end,
        # because it is used to remove the cells from all other groups.
        
        # Remove the cells from the data
        cte_io.pop_cell_data_from_hdf5(self.h5, cellIDs_topop)
        
        # Remove the cells from the cell_states
        cte_io.pop_cell_states_from_hdf5(self.h5, cellIDs_topop)
        
        # Remove the cells from the alphashapes
        cte_io.pop_cell_alphashape_from_hdf5(self.h5, cellIDs_topop)
        
        # Remove the cells from the cell_labels
        cte_io.pop_cell_labels_from_hdf5(self.h5, cellIDs_topop)
        
        # Get the new number of cells
        ncell_new = len(self.cell_labels)
        
        # Get the number of removed cells and the remaining cells
        ncell_removed = self.attrs['ncell'] - ncell_new
        
        # Include these numbers in the attributes
        cte_io.add_key_to_attrs_in_hdf5('ncell_removed', ncell_removed, self.h5)
        cte_io.add_key_to_attrs_in_hdf5('ncell_remaining', ncell_new, self.h5)
    
    def pop_spots(self, spotIDs_topop: dict) -> None:
        """ Remove spots from the CTE object in place.
        
        Spots are removed from the data in the HDF5 file.
        
        It is assumed that everything else but the data doesn't change,
        i.e. the Index, the attributes, the cell_labels, the cell_states, and the alphashapes.
        
        The attributes also doesn't change, but two additional keys are added:
        - nspots_removed: number of removed spots.
        - nspots_remaining: number of remaining spots.
        
        Args:
            spotIDs_topop (dict): dictionary with the spotIDs to remove, with the format:
                                  spotIDs_topop[cellID][chrom][traceID] = [spotID1, spotID2, ...]
        """
        
        # Remove the spots from the data (in place, returns the number of removed spots)
        nspot_popped = cte_io.pop_spot_data_from_hdf5(self.h5, spotIDs_topop)
        
        # Get the new number of spots
        nspot_new = self.attrs['nspot'] - nspot_popped
        # Include these numbers in the attributes
        cte_io.add_key_to_attrs_in_hdf5('nspot_removed', nspot_popped, self.h5)
        cte_io.add_key_to_attrs_in_hdf5('nspot_remaining', nspot_new, self.h5)
    
    
    # MISCELLANEOUS FUNCTIONS
    
    def get_trace_hashmap(self, cellID: str) -> dict:
        """ Create a hashmap for the traces in a cell.
        
        This function gives a unique and consistent integer to each traceID in the cell,
        so that it can be used as an index in a matrix.
        
        The hashmap is a dictionary with the format:
            traceID_hash = {
                'chr1': {
                    'traceID1': 0,
                    'traceID2': 1,
                    ...
                },
                ...
            },
        meaning that traceID_has[chrom][traceID] gives an integer (from 0 to ntraces-1)
        that can be used to construct a matrix.
        
        The consistency is guaranteed by sorting the traceIDs in each chromosome.

        Args:
            cellID (str)

        Returns:
            dict: _description_
        """
        
        # Get the data for the cell in dictionary format
        cell_data = self.get_data(cellID)
        
        # Initialize the hashmap
        traceID_hash = {}
        
        # Loop over the chromosomes and fill the hashmap
        for chrom in cell_data:
            # Get the traceIDs for the chromosome
            traceIDs = list(cell_data[chrom].keys())
            # Sort the traceIDs to ensure that the hashmap is consistent
            traceIDs.sort()
            # Add the chromosome to the hashmap
            traceID_hash[chrom] = {traceID: i for i, traceID in enumerate(traceIDs)}
        
        return traceID_hash
    
    def calculate_triad_labels(self) -> np.ndarray:
        """ Calculate all the cellID / chrom / traceID triads.
        i.e. [[cellID1, chrom1, traceID1], [cellID1, chrom1, traceID2], ...]

        Returns:
            np.ndarray: array with the cellID / chrom / traceID triads. shape=(ntriads, 3)
        """
        # Initialize the list of triads
        triadIDs = []
        # Loop over the cellIDs
        for cellID in self.cell_labels:
            # Get the data for the cell in dictionary format
            cell_data = self.get_data(cellID)
            # Loop over the chromosomes and traces
            for chrom in cell_data:
                for traceID in cell_data[chrom]:
                    # Add the triad to the list
                    triadIDs.append([cellID, chrom, traceID])
        # Convert the list to a numpy array
        triadIDs = np.array(triadIDs).astype(str)
        return triadIDs
    
    def get_bed_values_by_spotIDs(self, cellID: str, bedfile: str) -> np.ndarray:
        """ Get the values from a bed file for each spot in a cell.
        
        The BED file should be in the format:
            chrom   start   end    value
            'chr1'  1000    2000    1
            'chr1'  2000    3000    2
            ...
        
        This function reads the BED file and returns an array of values
        sorted by the appearance of the spotIDs in the CTE.
        
        Args:
            cellID (str): cell ID
            bedfile (str): path to the BED file

        Returns:
            (np.ndarray): array of values sorted by the appearance of the spotIDs in the CTE.
        """
        
        # Read the bed file as Index
        index = self.index  # get the index from the CTE
        try:
            bed = get_index_from_bed(bedfile, genome=index.genome)
        except Exception as e:
            raise ValueError("Could not read the bed file as Index.") from e
        if bed != index:
            raise ValueError("The bed file does not match the CTE index.")
        
        # Try getting the values from the bed file
        try:
            bed_values = bed.track0
        except Exception as e:
            raise ValueError("Could not get labels from the bed file.") from e
        
        # Get the domain info (chroms, starts, ends) of each spot from the CTE
        _, _, _, chroms, starts, ends, _, _, _ = self.get_data(cellID, format='numpy')
        # Get the index hashmap
        index_hashmap = index.get_index_hashmap()
        
        # Convert the values into an array sorted by the spotIDs in the CTE
        values = []
        for chrom, start, end in zip(chroms, starts, ends):
            i_domain = index_hashmap[(chrom, start, end)]
            assert len(i_domain) == 1, f"Multiple domains found for {chrom}:{start}-{end}."
            i_domain = i_domain[0]
            values.append(bed_values[i_domain])
        
        return np.array(values)
    
    @staticmethod
    def look_for_noisy_trace(traceID):
        """ Check if a trace is noisy.
            If traceID is an integer, it is considered noisy if it is negative.
            If traceID is a string, it is considered noisy if it contains a '-'.
            Warning: if a valid traceID contains a '-', it will be considered noisy.

        Args:
            traceID (str or int): The trace ID to check.

        Raises:
            Exception: If traceID is not an integer or a string.

        Returns:
            is_noise (bool): True if the trace is noisy, False otherwise.
        """
        
        is_noise = False
        
        if isinstance(traceID, int):
            if traceID < 0:
                is_noise = True
        
        elif isinstance(traceID, str):
            if '-' in traceID:
                is_noise = True
        
        else:
            raise Exception("traceID must be an integer or string.")
        
        return is_noise

    def sort_by_start(self):
        """ Sort the data by start position: in each trace, spotIDs are sorted by start position.

        Returns:
            other (ChromatinTracingExperiment): a new ChromatinTracingExperiment object with the sorted data.
        """
        
        sorted_data = {}
        
        for cellID in self.data:
            if cellID not in sorted_data:
                sorted_data[cellID] = {}
            
            for chrom in self.data[cellID]:
                if chrom not in sorted_data[cellID]:
                    sorted_data[cellID][chrom] = {}
                
                for traceID in self.data[cellID][chrom]:
                    
                    trace_data = self.data[cellID][chrom][traceID]
                    sorted_trace_data = sorted(trace_data.items(), key=lambda x: x[1]['start'])  # TODO: check that this works
                    sorted_data[cellID][chrom][traceID] = dict(sorted_trace_data)
        
        # Create a new ChromatinTracingExperiment object
        other = ChromatinTracingExperiment()
        
        # Set the attributes of the new object
        other.set_data_attrs_index(data=sorted_data, assembly=self.assembly, index=self.index)
        other.set_cell_states(self.cell_states)
        other.set_alphashapes(self.alphashapes)
        
        del sorted_data
        
        return other
