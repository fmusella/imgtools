import os
import pickle
import json
import h5py
import numpy as np
from .fofct import read_fofct
from alabtools.utils import Index
from .validator import CTEData
from . import cte_utils
from . import cte_io


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
    
    
    # SETTER FUNCTIONS
    
    def set_index(self, index: Index) -> None:
        """ Set the index in the HDF5 file."""
        cte_io.save_index_to_hdf5(index, self.h5)
    
    def set_attrs(self, attrs: dict) -> None:
        """ Set the attributes in the HDF5 file."""
        cte_io.save_attrs_to_hdf5(attrs, self.h5)
    
    def set_cellIDs(self, cellIDs: list) -> None:
        """ Set the cell labels in the HDF5 file."""
        cte_io.save_cellIDs_to_hdf5(cellIDs, self.h5)
    
    def set_data(self, data: dict) -> None:
        """ Set the data in the HDF5 file."""
        cte_io.save_data_to_hdf5(data, self.h5)
    
    def set_alphashapes(self, alphashapes: dict) -> None:
        """ Set the alphashapes in the HDF5 file."""
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
        self.set_cellIDs(list(data.keys()))
        self.set_data(data)
    
    
    def set_cell_states(self, cell_states: dict) -> None:
        # Placeholder
        pass
    
    
    # GETTER FUNCTIONS
    
    def get_index(self) -> Index:
        """ Get the index."""
        return Index(self.h5)
    
    def get_attrs(self) -> dict:
        """ Get the attributes."""
        return cte_io.load_attrs_from_hdf5(self.h5)
    
    def get_cellIDs(self) -> np.ndarray:
        """ Get the cell labels."""
        return self.h5['cellIDs'][:].astype('U20')
    
    def get_cellID(self, cellnum: int) -> str:
        """ Get the cellID corresponding to a cell number."""
        cellIDs = cte_io.load_cellIDs_from_hdf5(self.h5)
        return cellIDs[cellnum]
    
    def get_cellnum(self, cellID: str) -> int:
        """ Get the cell number corresponding to a cellID. """
        cellIDs = cte_io.load_cellIDs_from_hdf5(self.h5)
        if cellID not in cellIDs:
            raise ValueError("cellID {} not in cell labels.".format(cellID))
        return np.where(np.array(cellIDs) == cellID)[0][0]
    
    def get_data(self, cellID: str, chrom: str = None, traceID: str = None, format: str = 'dict'):
        """ Get the data for a cell, a chromosome in a cell, or a trace in a chromosome in a cell."""
        if chrom is None and traceID is None:
            return cte_io.load_cell_data_from_hdf5(cellID, self.h5, format)
        elif chrom is not None and traceID is None:
            return cte_io.load_chrom_data_from_hdf5(cellID, chrom, self.h5, format)
        elif chrom is not None and traceID is not None:
            return cte_io.load_trace_data_from_hdf5(cellID, chrom, traceID, self.h5, format)
    
    def get_alphashapes(self, cellID: str) -> dict:
        """ Get the alphashapes for a cell."""
        return cte_io.load_cell_alphashape_from_hdf5(cellID, self.h5)
    
    
    # INPUT/OUTPUT FUNCTIONS
    
    def close(self) -> None:
        """ Close the HDF5 file."""
        self.h5.close()
    
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
    
    
    def merge(self, other, filename: str, tag1: str = None, tag2: str = None, check_data: bool = False):
        """ Merge two ChromatinTracingExperiment objects.
        If there is an overlap between the cell labels, tag1 and tag2 must be provided to distinguish the cells.

        Args:
            other (ChromatinTracingExperiment): the other ChromatinTracingExperiment object to merge.
            filename (str): name of the file (with path) where the merged object will be saved.
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
        index_1 = self.get_index()
        index_2 = other.get_index()
        if index_1 != index_2:
            raise ValueError("Cannot merge ChromatinTracingExperiment objects with different indices.")
        
        # Get the attributes of the merged data
        attrs_1 = self.get_attrs()
        attrs_2 = other.get_attrs()
        attrs_merged = cte_utils.get_merged_attrs(attrs_1, attrs_2)

        # Get the data of the merged data
        # TODO: it can be optimized: we don't need to convert the data to a dict.
        #       We would need to code, in cte_io, a way to save data to the hdf5 file directly from numpy arrays.
        data_merged = {}
        # Get the data of the first ChromatinTracingExperiment object
        for cellID in self.get_cellIDs():
            cell_data = self.get_data(cellID, format='dict')
            cellID_w_tag = cellID + '_' + tag1 if tag1 is not None else cellID
            data_merged[cellID_w_tag] = cell_data
        # Get the data of the second ChromatinTracingExperiment object
        for cellID in other.get_cellIDs():
            cell_data = other.get_data(cellID, format='dict')
            cellID_w_tag = cellID + '_' + tag2 if tag2 is not None else cellID
            if cellID_w_tag in data_merged:
                raise ValueError("cellID {} already in data_merged. tag1 and tag2 must be provided to distinguish the cells.".format(cellID_w_tag))
            data_merged[cellID_w_tag] = cell_data
        
        # Create a new ChromatinTracingExperiment object
        merged = ChromatinTracingExperiment(filename, 'w')
        merged.set_data_attrs_index(data=data_merged, index=index_1, attrs=attrs_merged, check_data=check_data)
        
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
