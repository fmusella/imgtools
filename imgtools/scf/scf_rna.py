import numpy as np
from .scf import SingleCellFeature

class SingleCellFeatureRNA(SingleCellFeature):
    """ A class to store feature data from single-cell intron RNA imaging experiments,
    where the data are organized as a matrix of shape ncells x ngenes x ncopies.
    
    The data structure describes the chromosomal domains with the Index object,
    and the cells with a cell label (e.g. cell ID) and a cell state (e.g. cell cycle phase).
    
    In addition, it describes the genes information, i.e. the labels, chrom:start-end positions.
    
    Inherits from SingleCellFeature, which provides the basic structure for single-cell features.
    
    ----------
    Attributes:
        h5_name (str): path and name of the HDF5 file.
        h5 (h5py.File): HDF5 file to store the data.
                        Contains the following groups:
                        Inhereits from SingleCellFeature:
                            - index: Index object.
                            - attrs: attributes.
                            - cell_labels: array with the cell IDs.
                            - cell_states: array with the cell states.
                            - volumes: array with the cell volumes.
                            - feature_list: list of feature matrices.
                            - [feature]: feature matrix. (saved with a particular name)
                        Specific to SingleCellFeatureRNA:
                            - genes: group containing gene information:
                                - gene_labels: string array of shape (ngenes,).
                                - gene_chromstr: string array of shape (ngenes,).
                                - gene_start: integer array of shape (ngenes,).
                                - gene_end: integer array of shape (ngenes,).
    ---------- 
    Properties (from h5 file):
        Inherited from SingleCellFeature:
            index (Index): Index object.
            attrs (dict): attributes.
            cell_labels (np.ndarray): array with the cell IDs.
            cell_states (np.ndarray): array with the cell states.
            volumes (np.ndarray): array with the cell volumes.
            feature_list (list): list of feature matrices.
        Specific to SingleCellFeatureRNA:
            gene_labels (np.ndarray): gene labels.
            gene_chromstr (np.ndarray): gene chromosome strings.
            gene_start (np.ndarray): gene start positions.
            gene_end (np.ndarray): gene end positions.
    """
    
    def __init__(self, h5_name: str, mode: str = 'r') -> None:
        
        # Call the parent class constructor
        super().__init__(h5_name, mode)
        # Set the data type to RNA
        self.DATA_TYPE = 'RNA'
        # Require the 'genes' group in the HDF5 file
        # If the mode is 'a' or 'w', require it (create if not exists)
        if mode in ['a', 'w']:
            self.h5.require_group('genes')
        # If the mode is 'r', check if the 'genes' group exists
        else:
            if 'genes' not in self.h5:
                raise KeyError("The 'genes' group is required for RNA data but not found in the HDF5 file.")
    
    
    # SETTER FUNCTIONS
    
    def set_genes(
        self,
        gene_labels: np.ndarray,
        gene_chromstr: np.ndarray, gene_start: np.ndarray, gene_end: np.ndarray
    ) -> None:
        """ Save the gene information in the h5 file:
          - Stores everything in the 'genes' group.
          - gene_labels: string array of shape (ngenes,).
          - gene_chromstr: string array of shape (ngenes,).
          - gene_start: integer array of shape (ngenes,).
          - gene_end: integer array of shape (ngenes,).
        """
        genes_group = self.h5.require_group('genes')
        genes_group.create_dataset('gene_labels', data=np.array(gene_labels).astype('S'))
        genes_group.create_dataset('gene_chromstr', data=np.array(gene_chromstr).astype('S'))
        genes_group.create_dataset('gene_start', data=np.array(gene_start).astype(int))
        genes_group.create_dataset('gene_end', data=np.array(gene_end).astype(int))
    
    
    # GETTER FUNCTIONS
    
    def get_gene_labels(self) -> np.ndarray:
        """ Get the gene labels from the h5 file.
        Gene labels are string, we retrieve them in 'str' type, i.e. unicode."""
        return self.h5['genes/gene_labels'][:].astype(str)
    
    def get_gene_chromstr(self) -> np.ndarray:
        """ Get the gene chromosome strings from the h5 file.
        Gene chromosome strings are string, we retrieve them in 'str' type, i.e. unicode."""
        return self.h5['genes/gene_chromstr'][:].astype(str)
    
    def get_gene_start(self) -> np.ndarray:
        """ Get the gene start positions from the h5 file.
        Gene start positions are integers, we retrieve them as int."""
        return self.h5['genes/gene_start'][:].astype(int)
    
    def get_gene_end(self) -> np.ndarray:
        """ Get the gene end positions from the h5 file.
        Gene end positions are integers, we retrieve them as int."""
        return self.h5['genes/gene_end'][:].astype(int)
    
    def get_cell_states(self) -> np.ndarray:
        """ Get the cell states from the h5 file.
        Cell states are string, we retrieve them in 'str' type, i.e. unicode."""
        return self.h5['cell_states'][:].astype(str)
    
    
    # DEFINE PROPERTIES (READ ONLY)
    gene_labels = property(get_gene_labels, doc="Gene labels.")
    gene_chromstr = property(get_gene_chromstr, doc="Gene chromosome strings.")
    gene_start = property(get_gene_start, doc="Gene start positions.")
    gene_end = property(get_gene_end, doc="Gene end positions.")
    
    
    # INPUT/OUTPUT FUNCTIONS
    
    def get_expected_shape(self) -> tuple:
        """ Get the expected shape of the feature matrices.
        For RNA data, we expect the shape to be (ncells, ngenes, ncopies)
        """
        return len(self.cell_labels), len(self.gene_labels), self.attrs['max_ntrace_per_chrom']
    