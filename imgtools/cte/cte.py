import os
import pickle
import json
import h5py
import numpy as np
from .fofct import read_fofct
from alabtools.utils import Index
from .validator import CTEData
from . import cte_utils
from . import io
from ..scmatrix import SingleCellMatrix


class ChromatinTracingExperiment:
    """ A class to store and manipulate data from a Chromatin Tracing (CT) Experiment, like DNAseqFISH+.
    
    The data is stored in a nested dictionary whose structure is defined by the pydantic models: 
        SpotData,
        TraceData,
        ChromData,
        CellData.
    (see below for details).
    
    --------------------
    Attributes:
        assembly (str): assembly name.
        index (Index): Index object.
        data (dict): data in dictionary format.
        attrs (dict): attributes of the data.
    
    Attributes that can be added later:
        cell_states (dict): dictionary of cell states.
        alphashapes (dict): dictionary of alpha shapes.
    --------------------
    """
    
    def __init__(self, h5_name: str) -> None:
        
        # Load the HDF5 file in read+write mode
        self.h5 = h5py.File(h5_name, 'a')
    
    
    # SETTER FUNCTIONS
    
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
        
        # Save the data, attributes and index to the HDF5 file
        io.save_data_to_hdf5(data, self.h5)
        io.save_attrs_to_hdf5(attrs, self.h5)
        index.save(self.h5)
    
    def set_cell_states(self, cell_states: dict) -> None:
        """ Set the cell states.
        Checks that the cell_states is a dictionary and that the keys are cellIDs in the data.

        Args:
            cell_states (dict): dictionary of cell states.
        """
        
        # Check that data is not None
        if self.data is None:
            raise ValueError("data must be set before cell_states.")
        
        # Check that cell_states is a dictionary
        if not isinstance(cell_states, dict):
            raise TypeError("cell_states must be a dictionary.")
        
        # Check that the keys of cell_states are cellIDs in the data
        for cellID in cell_states:
            if cellID not in self.data:
                raise ValueError("cellID {} not in data.".format(cellID))
        
        self.cell_states = cell_states
    
    def set_alphashapes(self, alphashapes: dict) -> None:
        """ Set the alpha shapes.
        Checks that the alphashapes is a dictionary and that the keys are cellIDs in the data.

        Args:
            alphashapes (dict): dictionary of alpha shapes.
                                alphashapes[cellID] = {'mesh': trimesh.Trimesh, 'volume': float}
        """
        
        # Check that alphashapes is a dictionary
        if not isinstance(alphashapes, dict):
            raise TypeError("alphashapes must be a dictionary.")
        
        # Check that the keys of alphashapes are cellIDs in the data
        for cellID in alphashapes:
            if cellID not in self.data:
                raise ValueError("cellID {} not in data.".format(cellID))
        
        self.alphashapes = alphashapes
    
    
    # GETTER FUNCTIONS
    
    def get_cellID(self, cellnum: int) -> str:
        """ Get the cellID corresponding to a cell number."""
        cellIDs = self.f['cellIDs'][:].astype('U20')
        return cellIDs[cellnum]
    
    def get_cellnum(self, cellID: str) -> int:
        """Get the cell number corresponding to a cellID."""
        cellIDs = self.f['cellIDs'][:].astype('U20')
        if cellID not in cellIDs:
            raise ValueError("cellID {} not in cell labels.".format(cellID))
        return np.where(np.array(cellIDs) == cellID)[0][0] + 1
    
    def get_cell_data(self, cellID: str, format: str = 'dict'):
        """ Get the data for a cell.
        If format is 'dict', the data is returned as a dictionary,
        if format is 'numpy', the data is returned as tuples of numpy arrays.

        Args:
            cellID (str)
            format (str, optional): 'dict' or 'numpy'. Defaults to 'dict'.

        Returns:
            cell_data (dict or tuple of numpy arrays)
        """
        cell_data = io.load_cell_data_from_hdf5(cellID, self.h5, format)
        return cell_data
    
    def get_chrom_data(self, cellID: str, chrom: str, format: str = 'dict'):
        """ Get the data for a chromosome in a cell.
        If format is 'dict', the data is returned as a dictionary,
        if format is 'numpy', the data is returned as tuples of numpy arrays.

        Args:
            cellID (str)
            chrom (str)
            format (str, optional): 'dict' or 'numpy'. Defaults to 'dict'.

        Returns:
            chrom_data (dict or tuple of numpy arrays)
        """
        chrom_data = io.load_chrom_data_from_hdf5(cellID, chrom, self.h5, format)
        return chrom_data
    
    def get_trace_data(self, cellID: str, chrom: str, traceID: str, format: str = 'dict'):
        """ Get the data for a trace in a chromosome in a cell.
        If format is 'dict', the data is returned as a dictionary,
        if format is 'numpy', the data is returned as tuples of numpy arrays.

        Args:
            cellID (str)
            chrom (str)
            traceID (str)
            format (str, optional): 'dict' or 'numpy'. Defaults to 'dict'.

        Returns:
            trace_data (dict or tuple of numpy arrays)
        """
        trace_data = io.load_trace_data_from_hdf5(cellID, chrom, traceID, self.h5, format)
        return trace_data
    
    
    # INPUT/OUTPUT FUNCTIONS
    
    def save_as_pickle(self, filename: str, protocol: int = 4):
        """Saves the object to a pickle file.

        Args:
            filename (str): name of the directory where the object will be saved.
            protocol (int, optional): pickle protocol. Defaults to 4.

        Raises:
            TypeError: filename is not a string.
            FileNotFoundError: filename is not a valid directory.
        """

        # Check that filename is a string and that the directory exists
        if not isinstance(filename, str):
            raise TypeError("filename must be a string.")
        if not os.path.exists(os.path.dirname(filename)):
            raise NotADirectoryError("Directory {} does not exist.".format(os.path.dirname(filename)))

        # Save the object to a pickle file
        with open(filename, 'wb') as f:
            pickle.dump(self, f, protocol=protocol)
    
    def load_from_pickle(self, filename: str, check_data: bool = False):
        """Loads a ChromatinTracingExperiment object from a pickle file.

        Args:
            filename (str): path and name of the pickle file.

        Raises:
            TypeError: filename is not a string.
            FileNotFoundError: filename is not a valid file.
            Exception: the object could not be loaded from the file.
            TypeError: the loaded object is not a ChromatinTracingExperiment object.
            Exception: the loaded object does not have data.
        """

        # Check that filename is a string and that the file exists
        if not isinstance(filename, str):
            raise TypeError("filename must be a string.")
        if not os.path.exists(filename):
            raise FileNotFoundError("File {} does not exist.".format(filename))

        # Try to load the object from the pickle file
        try:
            with open(filename, 'rb') as f:
                loaded_object = pickle.load(f)
        except:
            raise Exception("Could not load object from file {}.".format(filename))

        # Check that the loaded object is a ChromatinTracingExperiment object and that it has data
        if not isinstance(loaded_object, ChromatinTracingExperiment):
            raise TypeError("Loaded object is not a ChromatinTracingExperiment object.")
        if loaded_object.data == {}:
            raise Exception("Loaded object does not have data.")
        
        # Check that the data is in the correct format if requested
        if check_data:
            checker = CTEData(root=loaded_object.data)
            del checker

        # Update the attributes of the current ChromatinTracingExperiment object
        self.__dict__.update(loaded_object.__dict__)
        
        del loaded_object
    
    def save_as_json(self, dirname: str) -> None:
        """ Save the data of the ChromatinTracingExperiment object to a json file.
        
        Args:
            dirname (str): name of the directory where the object will be saved.
        """
        
        # Check that dirname is a string and that the directory exists
        if not isinstance(dirname, str):
            raise TypeError("filename must be a string.")
        if not os.path.exists(dirname):
            raise NotADirectoryError("Directory {} does not exist.".format(dirname))
        
        # Save the data to a json file
        with open(os.path.join(dirname, 'data.json'), 'w') as f:
            json.dump(self.data, f)
        
        # Save the attributes to a json file
        with open(os.path.join(dirname, 'attrs.json'), 'w') as f:
            json.dump(self.attrs, f)
        
        # Save the index to a hdf5 file
        with h5py.File(os.path.join(dirname, 'index.h5'), 'w') as f:
            self.index.save(f)
        
        # If cell_states is not None, save it to a json file
        if self.cell_states is not None:
            with open(os.path.join(dirname, 'cell_states.json'), 'w') as f:
                json.dump(self.cell_states, f)
        
        # If alphashapes is not None, save it to a pickle file
        if self.alphashapes is not None:
            with open(os.path.join(dirname, 'alphashapes.pkl'), 'wb') as f:
                pickle.dump(self.alphashapes, f)
    
    def load_as_json(self, dirname: str, check_data: bool = False) -> None:
        """ Load data from a directory.
        The directory must contain the following files: 'data.json', 'attrs.json', 'index.h5'.
        If the files 'cell_states.json' and 'alphashapes.pkl' are present, they are also loaded.
        
        Args:
            dirname (str): directory where the data is stored.
            check_data (bool, optional): check that the data is in the correct format. Defaults to False.
        """
        
        # Check that dirname is a string and that the directory exists
        if not isinstance(dirname, str):
            raise TypeError("filename must be a string.")
        if not os.path.exists(dirname):
            raise NotADirectoryError("Directory {} does not exist.".format(dirname))
        
        # Load the data from the json file
        if not os.path.exists(os.path.join(dirname, 'data.json')):
            raise FileNotFoundError("File data.json does not exist in directory {}.".format(dirname))
        with open(os.path.join(dirname, 'data.json'), 'r') as f:
            data = json.load(f)
        
        # Load the attributes from the json file
        if not os.path.exists(os.path.join(dirname, 'attrs.json')):
            raise FileNotFoundError("File attrs.json does not exist in directory {}.".format(dirname))
        with open(os.path.join(dirname, 'attrs.json'), 'r') as f:
            attrs = json.load(f)
        
        # Load the index from the hdf5 file
        if not os.path.exists(os.path.join(dirname, 'index.h5')):
            raise FileNotFoundError("File index.h5 does not exist in directory {}.".format(dirname))
        with h5py.File(os.path.join(dirname, 'index.h5'), 'r') as f:
            index = Index(f)
        
        # Load the cell_states from the json file if it exists
        if os.path.exists(os.path.join(dirname, 'cell_states.json')):
            with open(os.path.join(dirname, 'cell_states.json'), 'r') as f:
                cell_states = json.load(f)
        else:
            cell_states = None
        
        # Load the alphashapes from the pickle file if it exists
        if os.path.exists(os.path.join(dirname, 'alphashapes.pkl')):
            with open(os.path.join(dirname, 'alphashapes.pkl'), 'rb') as f:
                alphashapes = pickle.load(f)
        else:
            alphashapes = None
        
        # Set the data, index, and attributes
        self.set_data_attrs_index(data, index=index, attrs=attrs, check_data=check_data)
        # Set the cell_states and alphashapes if present
        if cell_states is not None:
            self.set_cell_states(cell_states)
        if alphashapes is not None:
            self.set_alphashapes(alphashapes)
       
    def read_from_fofct(self, filename: str, assembly: str, check_data: bool = False) -> None:
        """ Read data from a fofct file.
        Data is stored in the data attribute of the ChromatinTracingExperiment object.
        Args:
            filename (str): path to the fofct file. """
        
        # Check that filename is a string, a .csv file, and that it exists
        if not isinstance(filename, str):
            raise TypeError("filename must be a string.")
        if not filename.endswith('.csv'):
            raise ValueError("filename must be a .csv file.")
        if not os.path.exists(filename):
            raise FileNotFoundError("File {} does not exist.".format(filename))

        data = read_fofct(filename)
        
        index, attrs = cte_utils.get_index_and_attrs(data, assembly)

        self.set_data_attrs_index(data, assembly, index, attrs, check_data)
    
    
    # MISCELLANEOUS FUNCTIONS
    
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
    
    
    def merge(self, other, tag1: str = None, tag2: str = None, check_data: bool = False):
        """ Merge two ChromatinTracingExperiment objects.
        If there is an overlap between the cell labels, tag1 and tag2 must be provided to distinguish the cells.

        Args:
            other (ChromatinTracingExperiment): the other ChromatinTracingExperiment object to merge.
            tag1 (str, optional): string to distinguish the cells in the first ChromatinTracingExperiment object.
                                  Defaults to None, in which case the cell labels must be different.
            tag2 (str, optional): string to distinguish the cells in the second ChromatinTracingExperiment object.
                                  Defaults to None, in which case the cell labels must be different.
            check_data (bool, optional): check that the data is in the correct format. Defaults to False.

        Returns:
            merged (ChromatinTracingExperiment): a new ChromatinTracingExperiment object with the merged data.
        """
        
        # Check that other is a ChromatinTracingExperiment object
        if not isinstance(other, ChromatinTracingExperiment):
            raise TypeError("other must be a ChromatinTracingExperiment object.")
        
        # Check that the index are the same
        if self.index != other.index:
            raise ValueError("Cannot merge ChromatinTracingExperiment objects with different indices.")
        
        # If there is an overlap between the cell labels, check that tag1 and tag2 are provided and different
        if len(set(self.data.keys()).intersection(set(other.data.keys()))) > 0 and (tag1 is None or tag2 is None or tag1 == tag2):
            raise ValueError("There is an overlap between the cell labels. tag1 and tag2 must be provided.")
        
        # Create a data dictionary for the merged data
        merged_data = {}
        # First ChromatinTracingExperiment object (self)
        if tag1 is None:  # no tag provided
            merged_data.update(self.data)
        else:  # tag provided, must be appended to each cellID
            if not isinstance(tag1, str):
                raise TypeError("tag1 must be a string.")
            merged_data.update({cellID + '_' + tag1: self.data[cellID] for cellID in self.data})
        # Second ChromatinTracingExperiment object (other)
        if tag2 is None:  # no tag provided
            merged_data.update(other.data)
        else:  # tag provided, must be appended to each cellID
            if not isinstance(tag2, str):
                raise TypeError("tag2 must be a string.")
            merged_data.update({cellID + '_' + tag2: other.data[cellID] for cellID in other.data})
        
        # Get the attributes of the merged data
        merged_attrs = cte_utils.get_merged_attrs(self.attrs, other.attrs)
        
        # Create a new ChromatinTracingExperiment object
        merged = ChromatinTracingExperiment()
        merged.set_data_attrs_index(data=merged_data, index=self.index, attrs=merged_attrs, check_data=check_data)
        
        return merged

    
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
    
    
    
    def create_count_matrix(self) -> SingleCellMatrix:
        """ Create a count matrix from the data, i.e. counts the number of spots each domain (chrom, start, end)
            is present in each cell/trace.
            
            The count matrix is returned as a SingleCellMatrix object, with the following attributes:
                index: Index object.
                cell_labels: list of cell labels.
                matrix: np.array of shape (n_cells, n_domains, max_ntrace_per_chrom).
                spot_hash: dictionary of the position of each spot in the count matrix.

        Returns:
            SingleCellMatrix: count matrix.
        """
        # MOVE IT IN THE ANALYSIS MODULE
        
        # Create a hash table for the cellIDs
        cellIDs = list(self.data.keys())
        cellID_hash = {cellID: i for i, cellID in enumerate(cellIDs)}
        
        # Create a hash table for the index
        index_hash = self.index.get_index_hashmap()
        
        # Initialize the count array, with shape (n_cells, n_domains, max_ntrace_per_chrom)
        count = np.zeros(
            (len(cellID_hash), len(index_hash), self.attrs['max_ntrace_per_chrom']),
            dtype=np.int32
        )
        
        # Initialize the spotID hash table, needed to retrieve the position of a spot in the count array
        spotID_hash = {}
        
        # Fill the count array, looping over all spots
        for cellID in self.data:
            for chrom in self.data[cellID]:
                
                # Get the traces in the chromosome and hash them
                traceIDs = list(self.data[cellID][chrom].keys())
                traceID_hash = {traceID: i for i, traceID in enumerate(traceIDs)}
                
                for traceID in self.data[cellID][chrom]:
                    for spotID in self.data[cellID][chrom][traceID]:
                        
                        spot_data = self.data[cellID][chrom][traceID][spotID]
                        start, end = spot_data['start'], spot_data['end']
                        
                        # Get the position of the spot in the count matrix using the hash tables
                        i_cell = cellID_hash[cellID]
                        i_domain = index_hash[(chrom, start, end)]
                        i_trace = traceID_hash[traceID]
                        
                        # Increment the count array
                        count[i_cell, i_domain, i_trace] += 1
                        
                        # Add spotID to the hash table
                        spotID_hash[spotID] = (i_cell, i_domain, i_trace)
        
        # Creates the volumes array if the alphashapes are present
        volumes = None
        if hasattr(self, 'alphashapes'):
            volumes = [self.alphashapes[cellID]['volume'] for cellID in cellIDs]
            volumes = np.array(volumes, dtype=np.float32)
        
        # Create a SingleCellMatrix object and add the count data
        sc_count_matrix = SingleCellMatrix()
        sc_count_matrix.add_data(
            index = self.index,
            cell_labels = np.array(cellIDs, dtype='U10'),
            volumes=volumes,
            matrix = count,
            spot_hash = spotID_hash
        )
        
        return sc_count_matrix
