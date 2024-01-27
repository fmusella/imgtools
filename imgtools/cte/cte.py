import os
import sys
import pickle
from collections import defaultdict
from functools import partial
import tempfile
import numpy as np
from matplotlib import pyplot as plt
from .fofct import read_fofct
from alabtools.utils import Index
from alabtools.parallel import Controller
from alabtools.plots import write_pdb
from .validator import CTEData
from . import utils
from . import parallelization
from . import visualization
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
    
    def __init__(self):
        self.assembly = None
        self.index = None
        self.data = {}
        self.attrs = {}
        self.cell_states = None
        self.alphashapes = None
    
    
    # INPUT/OUTPUT FUNCTIONS
    
    def save(self, filename: str):
        """Saves the object to a pickle file.

        Args:
            filename (str): name of the directory where the object will be saved.

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
            pickle.dump(self, f)
    
    def load(self, filename: str, check_data: bool = False):
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
    
    def add_data(self,
                 data: dict,
                 assembly: str = None,
                 index: Index = None,
                 attrs: dict = None,
                 check_data: bool = False):
        """ Add data to the ChromatinTracingExperiment object.
        
        Checks that the data (dict) is in the correct format.
        
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
        
        # If check_data, use pydantic to check that the data is in the correct format
        # This might slow down the code, so it can be turned off
        if check_data:
            checker = CTEData(root=data)
            del checker
        
        # Get the Index and the attributes from the data, if they haven't been provided
        if index is None or attrs is None:
            index_inferred, attrs_inferred = utils.get_index_and_attrs(data, assembly)
        # Use the inferred Index and attributes if they haven't been provided
        if index is None:
            index = index_inferred
        if attrs is None:
            attrs = attrs_inferred
        
        # Update the attributes of the ChromatinTracingExperiment object
        self.data = data
        self.index = index
        self.attrs = attrs
    
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
        merged_attrs = utils.get_merged_attrs(self.attrs, other.attrs)
        
        # Create a new ChromatinTracingExperiment object
        merged = ChromatinTracingExperiment()
        merged.add_data(data=merged_data, index=self.index, attrs=merged_attrs, check_data=check_data)
        
        return merged
    
    def read_from_fofct(self, filename: str, assembly: str, check_data: bool = False):
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
        
        index, attrs = utils.get_index_and_attrs(data, assembly)

        self.add_data(data, assembly, index, attrs, check_data)
    
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
        
        # Add the sorted data to the new ChromatinTracingExperiment object
        other.add_data(data=sorted_data, assembly=self.assembly, index=self.index)
        
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
    
    def add_cell_states(self, cell_states: dict):
        """ Add cell states to the ChromatinTracingExperiment object.
        
        Args:
            cell_states (dict): dictionary of cell states.
        """
        
        # Check that cell_states is a dictionary
        if not isinstance(cell_states, dict):
            raise TypeError("cell_states must be a dictionary.")
        # Check that the keys of cell_states are in the data
        for cellID in cell_states:
            if cellID not in self.data:
                raise ValueError("cellID {} not in data.".format(cellID))
        # Add cell_states as an attribute
        self.cell_states = cell_states
    
    # DATA RETRIEVAL FUNCTIONS
    
    def get_cellnum(self, cellID):
        """Get the cell number corresponding to a cellID."""
        assert len(self.cell_labels) > 0, "No cell labels."
        assert cellID in self.cell_labels, "cellID {} not in cell labels.".format(cellID)
        return np.where(np.array(self.cell_labels) == cellID)[0][0] + 1
    
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
    
    
    def save_cell_pdb(self, cellID: str, path: str, filename: str = None):
        # MOVE TO VISUALIZATION.PY
        """Write a pdb file for a cell.
        The noise traces are not written."""
        
        # Check that cellID is a string and that it is in the data
        if not isinstance(cellID, str):
            raise TypeError("cellID must be a string.")
        if not cellID in self.data:
            raise ValueError("cellID {} not in data.".format(cellID))
        
        if not isinstance(path, str):
            raise TypeError("path must be a string.")
        if not os.path.exists(path):
            raise NotADirectoryError("Directory {} does not exist.".format(path))
        
        # Get data for cell in numpy array format
        xs, ys, zs, chroms, starts, _, lums, traceIDs, _ = utils.cell_to_numpy(self.data[cellID])
        
        # Convert chroms to chromnums, e.g. 'chr1' --> '1', 'chrX' --> 'X'
        chromnums = []
        for c in chroms:
            chromnums.append(c.replace('chr', ''))
        chromnums = np.array(chromnums).astype('U20')

        # Convert traceIDs to trace ranks within each chromosome, and then to strings
        # e.g. traceID: '12_1' --> trace_rank: 1 ---> tracenum: 'A'
        tranks = self.get_trace_ranks_for_cell(cellID)  # ranks of each trace in each chromosome of the cell
        tracenums = []
        for chrom, traceID in zip(chroms, traceIDs):
            t = tranks[chrom][traceID]  # rank of traceID in chrom
            if t > 0:
                # Valid traces (positive integers) are converted like this:
                #   1 --> 'A', 2 --> 'B', ...
                tracenums.append(chr(t + 64))
            elif t < 0:
                # Noisy traces (negative integers) are converted like this:
                #   -1 --> 'Z', -2 --> 'Y', ...
                tracenums.append(chr(t + 91))
            else:
                raise Exception("Trace number cannot be 0.")
        tracenums = np.array(tracenums).astype('U20')
        
        # Convert start to units of 100000 bp, so that it fits in the occupancy field of the pdb file
        # i.e. 200000000 bp --> 2000.00
        starts = starts / 100000
        
        # Convert lums so that they fit in the beta field of the pdb file
        lums = lums - np.min(lums)
        lums = lums / np.max(lums)
        lums = lums * 1000
        
        # Write dictionary for pdb file
        celldata_for_pdb = {'x': xs,
                            'y': ys,
                            'z': zs,
                            'residue_name': chromnums,
                            'chain_id': tracenums,
                            'occupancy': starts,
                            'beta': lums}
        
        # Write pdb file
        if filename is None:
            filename = os.path.join(path, cellID + '.pdb')
        
        write_pdb(filename, celldata_for_pdb)
    
    def save_all_pdbs(self, path):
        # MOVE TO VISUALIZATION.PY
        """Write pdb files for all cells."""
        
        assert isinstance(path, str), "path must be a string."
        assert os.path.exists(path), "path does not exist."
        
        for cellID in self.data:
            self.save_cell_pdb(cellID, path)
    
    def save_cell_pyplot(self, cellID: str, path: str, filename: str = None, plot_params: dict = {}):
        # MOVE TO VISUALIZATION.PY
        """ Plot a cell using matplotlib.

        Args:
            cellID (str)
            path (str)
            filename (str, optional): name of the file where the plot will be saved. If None, use cellID.
            plot_params (dict, optional): parameters for the plot. Defaults to {}.
        """
        
        # Check that cellID is a string and that it is in the data
        if not isinstance(cellID, str):
            raise TypeError("cellID must be a string.")
        if not cellID in self.data:
            raise ValueError("cellID {} not in data.".format(cellID))
        
        # Check that path is a string and that it exists
        if not isinstance(path, str):
            raise TypeError("path must be a string.")
        if not os.path.exists(path):
            raise NotADirectoryError("Directory {} does not exist.".format(path))
        
        # If filename is not provided, use cellID
        if filename is None:
            filename = os.path.join(path, cellID + '.png')
        # Check that filename is a string
        if not isinstance(filename, str):
            raise TypeError("filename must be a string.")
        
        # Check that plot_params is a dictionary
        if not isinstance(plot_params, dict):
            raise TypeError("plot_params must be a dictionary.")
        
        # Get data for cell in numpy array format
        xs, ys, zs, chroms, _, _, _, _, _ = utils.cell_to_numpy(self.data[cellID])
        data_for_pyplot = {
            'x': xs,
            'y': ys,
            'z': zs,
            'chrom': chroms
        }
        
        # Plot cell
        visualization.cell_pyplot(filename, cellID, data_for_pyplot, plot_params)
    
    def save_all_pyplots(self, path: str, plot_params: dict = {}):
        # MOVE TO VISUALIZATION.PY
        """ Save pyplots for all cells."""
        
        # Check that path is a valid directory
        if not isinstance(path, str):
            raise TypeError("path must be a string.")
        if not os.path.exists(path):
            raise NotADirectoryError("Directory {} does not exist.".format(path))
        
        for cellID in self.data:
            self.save_cell_pyplot(cellID, path, plot_params=plot_params)
    
    
    def save_cell_cmm(self, cellID: str, path: str, radius: float):
        # MOVE TO VISUALIZATION.PY
        """ Write a cmm file for a cell.
        
        Each trace is written in a separate cmm file.

        Args:
            cellID (str)
            path (str): directory where the cmm files will be saved.
        """
        
        if cellID not in self.data:
            raise ValueError("cellID {} not in data.".format(cellID))
        
        if not os.path.exists(path):
            raise NotADirectoryError("Directory {} does not exist.".format(path))
        
        for chrom in self.data[cellID]:
            for traceID in self.data[cellID][chrom]:
                
                xs, ys, zs, _, _, _, _, _ = utils.trace_dict_to_numpy(self.data[cellID][chrom][traceID])
                
                visualization.write_cmm(
                    filename = os.path.join(path, '{}_{}_{}.cmm'.format(cellID, chrom, traceID)),
                    marker_str = 'cellID: {}, chrom: {}, traceID: {}'.format(cellID, chrom, traceID),
                    coord = np.array([xs, ys, zs]).T,
                    radius = radius,
                )

    
    # DATA MANIPULATION FUNCTIONS
    
    
    def run_mrc(self, config: dict):
        # MOVE TO VOLUMES.PY
        """ Performs the mrc file creation task on the population.
        
        The mrc files (volumes and surfaces) are stored in the path specified in config.
        
        The function also saves - in this path - a pickle file with the origins and shapes
        of each cell volume.
        
        Args:
            config (dict): configuration dictionary for the mrc file creation.
        """
        
        # Create a temporary directory
        tempdir = tempfile.mkdtemp(dir=os.getcwd())
        sys.stdout.write("Temporary directory for nodes' results: {}\n".format(tempdir))
        
        # Save the data of each cell separately in the temporary directory as a pickle file
        for cellID in self.alphashapes:
            filename = os.path.join(tempdir, '{}_mesh.pickle'.format(cellID))
            with open(filename, 'wb') as f:
                pickle.dump(self.alphashapes[cellID]['mesh'], f)
        
        # set the parallel and reduce tasks
        parallel_task = partial(parallelization.mrc_parallel, config=config, tempdir=tempdir)
        reduce_task = partial(parallelization.mrc_reduce, config=config, tempdir=tempdir)
        
        # create a Controller
        controller = Controller(config)

        # run the parallel task
        controller.map_reduce(parallel_task, reduce_task, args=list(self.alphashapes.keys()))
        
        # Delete the non-empty temporary directory
        os.system('rm -r {}'.format(tempdir))
        
        del controller
    
    def run_mrc_single_cell(self, cellID: str, params: dict):
        # MOVE TO VOLUMES.PY
        """ Performs the mrc file creation task on a single cell.
        
        The mrc files (volume and surface) are stored in the path
        specified in params.
        
        The function returns the origin and shape of the volume mrc file,
        necessary for aligning the mrc files in 3D space.

        Args:
            cellID (str): cell ID.
            params (dict): configuration dictionary for the mrc file creation.

        Returns:
            origin (tuple): origin of the volume mrc file in voxel units.
            shape (tuple): shape of the volume mrc file in voxel units.
        """
        
        # Check that all required keys are present in params
        parallelization.check_config(params, parallelization.required_keys_mrc, parallel=False)
        
        # Perform the mrc file creation
        origin, shape = parallelization.do_cell_mrc(cellID, self.alphashapes[cellID]['mesh'], params)
        
        return origin, shape
