import os
import sys
import pickle
from collections import defaultdict
from functools import partial
import tempfile
import numpy as np
from .fofct import read_fofct
from alabtools.utils import Index
from alabtools.parallel import Controller
from alabtools.plots import write_pdb
from pydantic import BaseModel, RootModel, StrictFloat, StrictInt, StrictStr, field_validator
from pydantic_core.core_schema import FieldValidationInfo
from typing import Dict
from . import utils
from . import parallelization


class ChromatinTracingExperiment:
    """ A class to store and manipulate data from a Chromatin Tracing (CT) Experiment, like DNAseqFISH+. """
    
    def __init__(self):
        self.assembly = None
        self.index = None
        self.data = {}
        self.summary_metrics = {}
    
    
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
                 summary_metrics: dict = None,
                 check_data: bool = True):
        """ Add data to the ChromatinTracingExperiment object.
        
        Checks that the data (dict) is in the correct format.
        
        Derives the Index and Summary Metrics from the data, if not provided.

        Args:
            data (dict): data in dictionary format.
            assembly (str, optional): assembly name. Defaults to None.
            index (Index, optional): Index object. Defaults to None.
            summary_metrics (dict, optional): summary metrics. Defaults to None.
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
        
        # Get the Index and the Summary Metrices from the data, if they haven't been provided
        if index is None or summary_metrics is None:
            index_inferred, summary_metrics_inferred = utils.get_index_and_summary_metrics(data, assembly)
        # Use the inferred Index and Summary Metrics if they haven't been provided
        if index is None:
            index = index_inferred
        if summary_metrics is None:
            summary_metrics = summary_metrics_inferred
        
        # Update the attributes of the ChromatinTracingExperiment object
        self.data = data
        self.index = index
        self.summary_metrics = summary_metrics
    
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
        
        index, summary_metrics = utils.get_index_and_summary_metrics(data, assembly)

        self.add_data(data, assembly, index, summary_metrics, check_data)
    
    
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
    
    def ntrace_summary_stats(self):
        """Computes the distribution of the number of traces per chromosome across cells.

        Returns:
            ntrace_per_chrom (list): list of the number of traces per chromosome across cells."""
        ntrace_per_chrom = []  # list of the number of traces per chromosome across cells

        for cellID in self.data:
            for chrom in self.data[cellID]:
                ntrace_chrom_cell = 0
                
                for traceID in self.data[cellID][chrom]:
                    if self.look_for_noisy_trace(traceID):
                        continue
                    ntrace_chrom_cell += 1
                ntrace_per_chrom.append(ntrace_chrom_cell)
                
        ntrace_per_chrom = np.array(ntrace_per_chrom)
        
        return ntrace_per_chrom
    
    def avg_spot_per_tracerank(self):
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
    
    def save_cell_pdb(self, cellID: str, path: str):
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
        filename = os.path.join(path, cellID + '.pdb')
        write_pdb(filename, celldata_for_pdb)
    
    def save_all_pdbs(self, path):
        """Write pdb files for all cells."""
        
        assert isinstance(path, str), "path must be a string."
        assert os.path.exists(path), "path does not exist."
        
        for cellID in self.data:
            self.save_cell_pdb(cellID, path)

    
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
        parallelization.check_config_tracing(params, parallel=False)
        
        # Perform the tracing
        traced_chrom_data = parallelization.do_chromosome_tracing(chrom, self.data[cellID][chrom], params)
        
        # Create a new ChromatinTracingExperiment object
        other = ChromatinTracingExperiment()
        
        # Add the traced data to the new ChromatinTracingExperiment object
        other.add_data(data={cellID: {chrom: traced_chrom_data}}, assembly=self.assembly, index=self.index)
        
        del traced_chrom_data
        
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
    def check_start(cls, v: int, info: FieldValidationInfo):
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
        