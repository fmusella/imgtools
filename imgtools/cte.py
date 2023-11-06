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
from pydantic import BaseModel, RootModel, StrictFloat, StrictInt, StrictStr, field_validator
from pydantic_core.core_schema import FieldValidationInfo
from typing import Dict
from . import utils
from . import parallelization
from . import visualization


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
    
    --------------------
    """
    
    def __init__(self):
        self.assembly = None
        self.index = None
        self.data = {}
        self.attrs = {}
    
    
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
    
    def load(self, filename: str, check_data: bool = True):
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
                 check_data: bool = True):
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
    
    def read_from_fofct(self, filename: str, assembly: str, check_data: bool = True):
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
    
    def get_trace_ranks_for_chromosome(self, cellID, chrom):
        """ Get the ranks of the traces in a chromosome.
        
            The rank of valid traces is positive:
                rank 1 --> valid trace with the most spots, 2 --> second most, etc.
            The rank of valid traces is negative:
                rank -1 --> noisy trace with the most spots, -2 --> second most, etc.

        Args:
            cellID (str): cell ID.
            chrom (str): chromosome.

        Returns:
            ranks (dict): dictionary of the rank of traces for a chromosome in a cell.
                          ranks[traceID] = rank
        """
        
        # Initialize the counts of spots per trace for valid and noisy traces
        valid_counts, noisy_counts = {}, {}
        
        for traceID in self.data[cellID][chrom]:
            
            if self.look_for_noisy_trace(traceID):
                noisy_counts[traceID] = len(self.data[cellID][chrom][traceID])
            else:
                valid_counts[traceID] = len(self.data[cellID][chrom][traceID])

        # Sort traces by number of spots
        sorted_valid = sorted(valid_counts, key=valid_counts.get, reverse=True)
        sorted_noisy = sorted(noisy_counts, key=noisy_counts.get, reverse=True)
        
        # Assign rank to traces
        ranks = {}
        # Rank valid traces (1, 2, 3, ...)
        for rank, traceID in enumerate(sorted_valid, start=1):
            ranks[traceID] = rank
        # Rank noisy traces (-1, -2, -3, ...)
        for rank, traceID in enumerate(sorted_noisy, start=1):
            ranks[traceID] = -rank

        return ranks
    
    def get_trace_ranks_for_cell(self, cellID):
        """ Get ranks of traces - within each chromosome - for a cell.
        
        Within each chromosome, the rank of valid traces is positive:
            rank 1 --> valid trace with the most spots, 2 --> second most, etc.
        Within each chromosome, the rank of valid traces is negative:
            rank -1 --> noisy trace with the most spots, -2 --> second most, etc.

        Args:
            cellID (str): cell ID.

        Returns:
            ranks (dict): dictionary of the rank of traces for a cell.
                          ranks[chrom][traceID] = rank
        """
        
        ranks = {}
        
        for chrom in self.data[cellID]:
            ranks[chrom] = self.get_trace_ranks_for_chromosome(cellID, chrom)
        
        return ranks
        

    # SUMMARY STATISTICS AND VISUALIZATION FUNCTIONS
    
    def compute_trace_coverage(self, cellID: str, chrom: str, traceID: str):
        """ Computes the coverage of a trace.
        
        The coverage is defined as the number of unique domains divided by the total number of domains.

        Args:
            cellID (str)
            chrom (str)
            traceID (str)

        Returns:
            coverage (float): coverage of the trace.
        """
        
        # Check that cellID, chrom and traceID are in the data
        if cellID not in self.data:
            raise ValueError("cellID {} not in data.".format(cellID))
        if chrom not in self.data[cellID]:
            raise ValueError("chrom {} not in data[{}].".format(chrom, cellID))
        if traceID not in self.data[cellID][chrom]:
            raise ValueError("traceID {} not in data[{}][{}].".format(traceID, cellID, chrom))
        
        # Find unique domains in traceID
        unique_domains = set()
        for spotID in self.data[cellID][chrom][traceID]:
            start = self.data[cellID][chrom][traceID][spotID]['start']
            end = self.data[cellID][chrom][traceID][spotID]['end']
            unique_domains.add((start, end))
        
        # The coverage is the number of unique domains divided by the number of domains
        coverage = len(unique_domains) / np.sum(self.index.chromstr == chrom)
        
        return coverage
    
    def compute_trace_neighbor_distances(self, cellID: str, chrom: str, traceID: str):
        """ Computes the genomic and spatial distances between neighboring spots in a trace.

        Args:
            cellID (str)
            chrom (str)
            traceID (str)

        Returns:
            gdist (np.array): array of the genomic distances between neighboring spots in the trace.
            sdist (np.array): array of the spatial distances between neighboring spots in the trace.
        """
        
        # Check that cellID, chrom and traceID are in the data
        if cellID not in self.data:
            raise ValueError("cellID {} not in data.".format(cellID))
        if chrom not in self.data[cellID]:
            raise ValueError("chrom {} not in data[{}].".format(chrom, cellID))
        if traceID not in self.data[cellID][chrom]:
            raise ValueError("traceID {} not in data[{}][{}].".format(traceID, cellID, chrom))
        
        # get the data in numpy array format
        xs, ys, zs, chroms, starts, ends, lums, spotIDs = utils.trace_dict_to_numpy(self.data[cellID][chrom][traceID])
        crds = np.array([xs, ys, zs]).T
        
        # If there is only one spot, skip
        if len(crds) == 1:
            return None, None
        
        # Sort by genomic start position
        crds = crds[np.argsort(starts)]
        starts = starts[np.argsort(starts)]
        
        # Compute genomic distances between neighboring spots
        gdist = np.diff(starts)
        
        # Compute spatial distances between neighboring spots
        sdist = np.linalg.norm(np.diff(crds, axis=0), axis=1)
        
        return gdist, sdist
    
    def distribution_ntrace_per_chromosome(self, ignore_noisy_trace: bool = True):
        """Computes the distribution of the number of traces per chromosome across cells.

        Returns:
            ntrace_per_chrom (list): list of the number of traces per chromosome across cells."""
        
        ntrace_per_chrom = []  # list of the number of traces per chromosome across cells

        for cellID in self.data:
            for chrom in self.data[cellID]:
                
                ntrace_chrom_cell = 0
                
                for traceID in self.data[cellID][chrom]:
                    
                    if ignore_noisy_trace and self.look_for_noisy_trace(traceID):
                        continue
                    
                    ntrace_chrom_cell += 1
                    
                ntrace_per_chrom.append(ntrace_chrom_cell)
                
        ntrace_per_chrom = np.array(ntrace_per_chrom)
        
        return ntrace_per_chrom
    
    def distirbution_avg_spot_per_tracerank(self):
        """ Computes the average number of spots per trace rank.
        
        Within each chromosome, the rank of valid traces is positive:
            rank 1 --> valid trace with the most spots, 2 --> second most, etc.
        Within each chromosome, the rank of valid traces is negative:
            rank -1 --> noisy trace with the most spots, -2 --> second most, etc.

        Returns:
            nspot_per_trank (dict): dictionary of the average number of spots per trace rank."""
        
        # Initialize dictionary for the average number of spots per trace rank
        # We initialize the default element to an empty list, so that we can append to it without checking if it exists
        nspot_per_rank = defaultdict(list)

        for cellID in self.data:
            for chrom in self.data[cellID]:
                
                # Get ranks of traces in the chromosome
                trace_ranks = self.get_trace_ranks_for_chromosome(cellID, chrom)
                
                for traceID in self.data[cellID][chrom]:
                    
                    # rank of traceID
                    r = trace_ranks[traceID]
                    # Number of spots in traceID
                    nspot = len(self.data[cellID][chrom][traceID])
                    # Add nspot_t to the list of spots for rank t
                    nspot_per_rank[r].append(nspot)
        
        # Compute average number of spots per trace rank
        for r in nspot_per_rank:
            nspot_per_rank[r] = np.mean(np.array(nspot_per_rank[r]))

        return nspot_per_rank
    
    def distribution_nspot_per_trace(self, ignore_noisy_trace: bool = True):
        """ Computes the distribution of the number of spots per trace across cells.

        Args:
            ignore_noisy_trace (bool, optional): ignore noisy traces. Defaults to True.

        Returns:
            nspot_per_trace (np.array): array of the number of spots per trace across cells.
        """
        
        nspot_per_trace = []
        
        for cellID in self.data:
            for chrom in self.data[cellID]:
                for traceID in self.data[cellID][chrom]:
                    
                    if ignore_noisy_trace and self.look_for_noisy_trace(traceID):
                        continue
                    
                    nspot = len(self.data[cellID][chrom][traceID])
                    nspot_per_trace.append(nspot)
        
        nspot_per_trace = np.array(nspot_per_trace)
        
        return nspot_per_trace
        
    def distribution_coverage_per_trace(self, ignore_noisy_traces: bool = True):
        """ Compute the distribution of the coverage of each trace.
        
        Args:
            ignore_noisy_traces (bool, optional): ignore noisy traces. Defaults to True.

        Returns:
            coverage_distribution (np.array): array of the coverage of each trace.
        """

        coverage_distribution = []
        
        for cellID in self.data: 
            for chrom in self.data[cellID]:
                for traceID in self.data[cellID][chrom]:
                    
                    # ignore noisy traces if requested
                    if ignore_noisy_traces and self.look_for_noisy_trace(traceID):
                        continue
                    
                    coverage = self.compute_trace_coverage(cellID, chrom, traceID)
                    
                    # add coverage to list
                    coverage_distribution.append(coverage)
        
        coverage_distribution = np.array(coverage_distribution)
        
        return coverage_distribution
    
    def distribution_neighbor_distances(self, ignore_noisy_traces: bool = True):
        """ Compute the average spatial and genomic distance between neighboring spots in each trace.

        Args:
            ignore_noisy_traces (bool, optional): ignore noisy traces. Defaults to True.

        Returns:
            genomic_distances (list): list of the genomic distances between neighboring spots in each trace.
            spatial_distances (list): list of the spatial distances between neighboring spots in each trace.
        """
        
        # Initialize lists
        avg_genomic_distances = []
        max_genomic_distances = []
        min_genomic_distances = []
        
        avg_spatial_distances = []
        max_spatial_distances = []
        min_spatial_distances = []
        
        # Loop over cells, chromosomes and traces and fill lists
        for cellID in self.data:
            for chrom in self.data[cellID]:
                for traceID in self.data[cellID][chrom]:
                    
                    # ignore noisy traces if requested
                    if ignore_noisy_traces and self.look_for_noisy_trace(traceID):
                        continue
                    
                    # get the genomic and spatial distances between neighboring spots in the trace
                    gdist, sdist = self.compute_trace_neighbor_distances(cellID, chrom, traceID)
                    
                    # Add to lists
                    avg_genomic_distances.append(np.mean(gdist))
                    max_genomic_distances.append(np.max(gdist))
                    min_genomic_distances.append(np.min(gdist))
                    
                    avg_spatial_distances.append(np.mean(sdist))
                    max_spatial_distances.append(np.max(sdist))
                    min_spatial_distances.append(np.min(sdist))
        
        # Return lists (cast to numpy arrays) in dictionary
        distance_distributions = {
            'avg_genomic_distances': np.array(avg_genomic_distances),
            'max_genomic_distances': np.array(max_genomic_distances),
            'min_genomic_distances': np.array(min_genomic_distances),
            'avg_spatial_distances': np.array(avg_spatial_distances),
            'max_spatial_distances': np.array(max_spatial_distances),
            'min_spatial_distances': np.array(min_spatial_distances)
        }
        
        return distance_distributions
    
    def save_cell_pdb(self, cellID: str, path: str, filename: str = None):
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
        
        # Convert start to kbp
        starts = starts / 1000
        
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
        """Write pdb files for all cells."""
        
        assert isinstance(path, str), "path must be a string."
        assert os.path.exists(path), "path does not exist."
        
        for cellID in self.data:
            self.save_cell_pdb(cellID, path)
    
    def save_cell_pyplot(self, cellID: str, path: str, filename: str = None, plot_params: dict = {}):
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
        """ Save pyplots for all cells."""
        
        # Check that path is a valid directory
        if not isinstance(path, str):
            raise TypeError("path must be a string.")
        if not os.path.exists(path):
            raise NotADirectoryError("Directory {} does not exist.".format(path))
        
        for cellID in self.data:
            self.save_cell_pyplot(cellID, path, plot_params=plot_params)
    
    
    def save_cell_cmm(self, cellID: str, path: str, radius: float):
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
    
    def run_tracing(self, config):
        """ Performs the Genomic Iterative DBSCAN tracing on the population.
        
        Accepts either serial or parallel computation, as specified by the alabtools.parallel.Controller class.

        Args:
            config (dict): configuration dictionary for the Genomic Iterative DBSCAN algorithm.
                           Required keys: 'dbscan_eps', 'dbscan_min_samples', 'window_size', 'delta', 'max_missing_windows', 'parallel'.

        Returns:
            other (ChromatinTracingExperiment): a new ChromatinTracingExperiment object with the traced data.
        """
        
        # Create a temporary directory
        tempdir = tempfile.mkdtemp(dir=os.getcwd())
        sys.stdout.write("Temporary directory for nodes' results: {}\n".format(tempdir))
        
        # Save the data of each cell separately in the temporary directory as a pickle file
        for cellID in self.data:
            filename = os.path.join(tempdir, '{}_data.pickle'.format(cellID))
            with open(filename, 'wb') as f:
                pickle.dump(self.data[cellID], f)
        
        # set the parallel and reduce tasks
        parallel_task = partial(parallelization.tracing_parallel, config=config, tempdir=tempdir)
        reduce_task = partial(parallelization.tracing_reduce, tempdir=tempdir)
        
        # create a Controller
        controller = Controller(config)

        # run the parallel and reduce tasks
        traced_data = controller.map_reduce(parallel_task, reduce_task, args=list(self.data.keys()))
        
        # Delete the non-empty temporary directory
        os.system('rm -r {}'.format(tempdir))
        
        # Create a new ChromatinTracingExperiment object
        other = ChromatinTracingExperiment()

        # Add the traced data to the new ChromatinTracingExperiment object
        other.add_data(data=traced_data, assembly=self.assembly, index=self.index)
        
        del controller, traced_data
        
        return other
    
    def do_tracing_single_chrom(self, cellID, chrom, params):
        """Performs the Genomic Iterative DBSCAN tracing on a single chromosome of a single cell.

        Args:
            cellID (str): cell ID.
            chrom (str): chromosome.
            params (dict): configuration dictionary for the Genomic Iterative DBSCAN algorithm.
                           required keys: 'dbscan_eps', 'dbscan_min_samples', 'window_size', 'delta', 'max_missing_windows'.
        
        Returns:
            other (ChromatinTracingExperiment): a new ChromatinTracingExperiment object with the traced data.
        """
        
        # Check that all required keys are present in params
        parallelization.check_config(params, parallelization.required_keys_tracing, parallel=False)
        
        # Perform the tracing
        traced_chrom_data = parallelization.do_chromosome_tracing(chrom, self.data[cellID][chrom], params)
        
        # Create a new ChromatinTracingExperiment object
        other = ChromatinTracingExperiment()
        
        # Add the traced data to the new ChromatinTracingExperiment object
        other.add_data(data={cellID: {chrom: traced_chrom_data}}, assembly=self.assembly, index=self.index)
        
        del traced_chrom_data
        
        return other
    
    
    def run_alphashape(self, config: dict):
        """ Performs the alphashape computation on the population.

        Args:
            config (dict): configuration dictionary for the alphashape computation.
        """
        
        # Create a temporary directory
        tempdir = tempfile.mkdtemp(dir=os.getcwd())
        sys.stdout.write("Temporary directory for nodes' results: {}\n".format(tempdir))
        
        # Save the data of each cell separately in the temporary directory as a pickle file
        for cellID in self.data:
            filename = os.path.join(tempdir, '{}_data.pickle'.format(cellID))
            with open(filename, 'wb') as f:
                pickle.dump(self.data[cellID], f)
        
        # set the parallel and reduce tasks
        parallel_task = partial(parallelization.alphashape_parallel, config=config, tempdir=tempdir)
        reduce_task = partial(parallelization.alphashape_reduce, tempdir=tempdir)
        
        # create a Controller
        controller = Controller(config)

        # run the parallel and reduce tasks
        alphashapes = controller.map_reduce(parallel_task, reduce_task, args=list(self.data.keys()))
        
        # Delete the non-empty temporary directory
        os.system('rm -r {}'.format(tempdir))
        
        # Store the alphashape in the ChromatinTracingExperiment object
        self.alphashapes = alphashapes
        
        del controller, alphashapes
    
    def run_alphashape_single_cell(self, cellID: str, params: dict):
        """ Performs the alphashape computation on a single cell.

        Args:
            cellID (str): cell ID.
            params (dict): configuration dictionary for the alphashape computation.

        Returns:
            alpha (float): alpha parameter of the alphashape.
            mesh (trimesh.Trimesh): mesh of the alphashape.
        """
        
        # Check that all required keys are present in params
        parallelization.check_config(params, parallelization.required_keys_alphashape, parallel=False)
        
        # Perform the alphashape computation
        alpha, mesh = parallelization.do_cell_alphashape(self.data[cellID], params)
        
        return alpha, mesh
    
    
    def run_mrc(self, config: dict):
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
        
    
    def run_cleaning(self, coverage_threshold: float, gendist_threshold: float):
        """ Performs the cleaning of the traced data.
        
        Creates a new ChromatinTracingExperiment object with the cleaned data, i.e. without:
            - noisy traces
            - traces with a too-low coverage (less than coverage_threshold)
            - traces with a too-large minimum genomic distance between neighbors

        Args:
            coverage_threshold (float): minimum coverage for a trace to be kept.
            gendist_threshold (float): maximum threshold for the minimum genomic distance between neighbors for a trace to be kept.

        Returns:
            other (ChromatinTracingExperiment): a new ChromatinTracingExperiment object with the cleaned data.
        """
        
        # Initialize the cleaned data
        clean_data = {}
        
        # Loop over cells, chromosomes and traces and fill lists
        for cellID in self.data:
            clean_data[cellID] = {}  # initialize dictionary for cellID
            
            for chrom in self.data[cellID]:
                clean_data[cellID][chrom] = {}  # initialize dictionary for chrom
                
                for traceID in self.data[cellID][chrom]:
                        
                        # Ignore noisy traces
                        if self.look_for_noisy_trace(traceID):
                            continue
                        
                        # Ignore traces with low coverage
                        coverage = self.compute_trace_coverage(cellID, chrom, traceID)
                        if coverage < coverage_threshold:
                            continue
                        
                        # Compute the minimum genomic distance between neighboring spots
                        gdist, _ = self.compute_trace_neighbor_distances(cellID, chrom, traceID)
                        min_gdist = np.min(gdist)
                        if min_gdist > gendist_threshold:
                            continue
                        
                        # If everything is ok, add the trace to the cleaned data
                        clean_data[cellID][chrom][traceID] = self.data[cellID][chrom][traceID]
                
                # If chrom data is empty, delete it
                if clean_data[cellID][chrom] == {}:
                    del clean_data[cellID][chrom]
            
            # If cell data is empty, delete it
            if clean_data[cellID] == {}:
                del clean_data[cellID]
        
        # Create a new ChromatinTracingExperiment object
        other = ChromatinTracingExperiment()
        
        # Add the traced data to the new ChromatinTracingExperiment object
        other.add_data(data=clean_data, assembly=self.assembly, index=self.index)
        
        del clean_data
        
        return other
    
    def trim_trace_data(self, cellID: str, chrom: str, traceID: str):
        """ Remove multiple spots associated with the same domain in a trace.
        
        It uses the spots_3d_median function to choose a spot among the repeated ones:
            - If there are two spots, it chooses the one with closest distance to the trace's Center of Mass.
            - If there are more than two spots, it chooses the one with minimum average distance to the other spots.

        Args:
            cellID (str)
            chrom (str)
            traceID (str)
        
        Returns:
            trimmed_trace_data (dict): dictionary of the trimmed trace data.
        """
               
        # Take the trace data
        try:
            trace_data = self.data[cellID][chrom][traceID]
        except KeyError:
            raise KeyError("CellID {}, chrom {} and traceID {} not in data.".format(cellID, chrom, traceID))
        
        # Convert the trace data to numpy array format
        xs, ys, zs, chroms, starts, ends, lums, spotIDs = utils.trace_dict_to_numpy(trace_data)
        
        # Compute the Center of Mass of the trace
        com = np.array([np.mean(xs), np.mean(ys), np.mean(zs)])
        
        # Identify the domains as the (start, end) pairs (chrom is the same for all spots in the trace)
        domains = np.array([starts, ends]).T
        
        # Identify the unique domains
        unique_domains = np.unique(domains, axis=0)
        
        # If there are no repeated domains, return the original trace data
        if np.array_equal(domains, unique_domains):
            return trace_data
        
        # If there are repeated domains, trim them according to the 3D median procedure
        
        # Initialize the trimmed trace data
        trimmed_trace_data = {}
        
        for domain in unique_domains:
            
            # Find the indices associated with the domain
            indices = np.where(np.all(domains == domain, axis=1))[0]
            
            # Get the coordinates of the spots associated with the domain
            points = np.array([xs[indices], ys[indices], zs[indices]]).T
            
            # Compute the spots 3D median, getting the index - among points - of the 3D median spot
            median_idx = utils.spots_3d_median(points, com)
            
            # Get the index of the median spot in the indices array
            median_idx = indices[median_idx]
            
            assert median_idx in indices, "Median index not in indices. Something went wrong."
            
            trimmed_trace_data[spotIDs[median_idx]] = {
                    'x': xs[median_idx],
                    'y': ys[median_idx],
                    'z': zs[median_idx],
                    'chrom': chroms[median_idx],
                    'start': starts[median_idx],
                    'end': ends[median_idx],
                    'lum': lums[median_idx]
                }
        
        return trimmed_trace_data

    def run_trim(self):
        """ Trim the data, removing multiple spots associated with the same domain in each trace.

        Returns:
            other (ChromatinTracingExperiment): a new ChromatinTracingExperiment object with the trimmed data.
        """
        
        trimmed_data = {}
        
        # Loop over cells, chromosomes and traces and trim the trace data
        for cellID in self.data:
            if cellID not in trimmed_data:
                trimmed_data[cellID] = {}
            for chrom in self.data[cellID]:
                if chrom not in trimmed_data[cellID]:
                    trimmed_data[cellID][chrom] = {}
                for traceID in self.data[cellID][chrom]:
                    trimmed_data[cellID][chrom][traceID] = self.trim_trace_data(cellID, chrom, traceID)
                    
        # Create a new ChromatinTracingExperiment object
        other = ChromatinTracingExperiment()
        
        # Add the traced data to the new ChromatinTracingExperiment object
        other.add_data(data=trimmed_data, assembly=self.assembly, index=self.index)
        
        del trimmed_data
        
        return other


# Functions to validate the data format, that is a nested Dictionary of the form:
#        data[cellID][chrom][traceID][spotID] = {'x': float,
#                                                'y': float,
#                                                'z': float,
#                                                'chrom': str,
#                                                'start': int,
#                                                'end': int,
#                                                'lum': float}
# where cellID, chrom, traceID and spotID are strings, and chrom must start with 'chr'

class SpotData(BaseModel):
    """ Validate the format of the data for a single spot:
               spot_data = {'x': float,
                            'y': float,
                            'z': float,
                            'chrom': str,
                            'start': int,
                            'end': int,
                            'lum': float}
    """
    x: StrictFloat
    y: StrictFloat
    z: StrictFloat
    chrom: StrictStr
    start: StrictInt
    end: StrictInt
    lum: StrictFloat
    # Check that chromosome name starts with 'chr'
    @field_validator('chrom')
    def check_chrom(cls, v: str):
        if not v.startswith('chr'):
            raise ValueError('Chromosome name must start with "chr"')
        return v
    # Check that start > 0
    @field_validator('start')
    def check_start(cls, v: int):
        if v < 0:
            raise ValueError('Start position must be positive')
        return v
    # Check that end > start
    @field_validator('end')
    def check_end(cls, v: int, info: FieldValidationInfo):
        # Check that start has been validated
        if 'start' not in info.data:
            raise ValueError('Start position has not been validated yet')
        # Check that end > start
        if v < info.data['start']:
            raise ValueError('End must be greater than start')

class TraceData(RootModel):
    """ Validate the format of the data for a single trace:
            trace_data[traceID][spotID] = {'x': float,
                                           'y': float,
                                           'z': float,
                                           'chrom': str,
                                           'start': int,
                                           'end': int,
                                           'lum': float}
    """
    root: Dict[StrictStr, SpotData]
    # Check that the dictionary is not empty
    @field_validator('root')
    def check_root(cls, v: dict):
        if len(v) == 0:
            raise ValueError('Dictionary of TraceData cannot be empty')

class ChromData(RootModel):
    """ Validate the format of the data for a single chromosome:
            chrom_data[chrom][traceID][spotID] = {'x': float,
                                                  'y': float,
                                                  'z': float,
                                                  'chrom': str,
                                                  'start': int,
                                                  'end': int,
                                                  'lum': float}
    """
    root: Dict[StrictStr, TraceData]
    # Check that the dictionary is not empty
    @field_validator('root')
    def check_root(cls, v: dict):
        if len(v) == 0:
            raise ValueError('Dictionary of ChromData cannot be empty')

class CellData(RootModel):
    """ Validate the format of the data for a single cell:
            cell_data[cellID][chrom][traceID][spotID] = {'x': float,
                                                         'y': float,
                                                         'z': float,
                                                         'chrom': str,
                                                         'start': int,
                                                         'end': int,
                                                         'lum': float}
    """
    root: Dict[StrictStr, ChromData]
    @field_validator('root')
    def check_root(cls, v: dict):
        # Check that the dictionay is not empty
        if len(v) == 0:
            raise ValueError('Dictionary of CellData cannot be empty')
        # Check that the keys are valid chromosome names
        for k in v.keys():
            if not k.startswith('chr'):
                raise ValueError('Chromosome name must start with "chr"')

class CTEData(RootModel):
    """ Validate the format of the data for a ChromatinTracingExperiment data attribute:
            cte_data[cellID][chrom][traceID][spotID] = { 'x': float,
                                                         'y': float,
                                                         'z': float,
                                                         'chrom': str,
                                                         'start': int,
                                                         'end': int,
                                                         'lum': float}
    """
    root: Dict[StrictStr, CellData]
    # Check that the dictionay is not empty
    @field_validator('root')
    def check_root(cls, v: dict):
        if len(v) == 0:
            raise ValueError('Dictionary of CTEData cannot be empty')
        