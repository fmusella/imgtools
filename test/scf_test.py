import unittest
import random
import numpy as np
from imgtools.scf import SingleCellFeature
from alabtools.utils import Genome, Index

class TestSingleCellFeature(unittest.TestCase):
    
    def setUp(self) -> None:
        return super().setUp()
    
    def tearDown(self) -> None:
        return super().tearDown()
    
    def test_save_load(self) -> None:
        """ Test that the data can be saved and loaded correctly."""
        
        # Create the data
        index, attrs, mat = create_random_data()
        cell_labels = np.arange(attrs['ncell']).astype('U20')
        
        # Create a SCF object
        filename = './test.scf.h5'
        scf = SingleCellFeature(filename, 'w')
        
        scf.add_index_attrs_cell_labels(index, attrs, cell_labels)
        scf.add_matrix(mat, 'test')
        
        # Check that the properties of the SCF object are correct
        self.assertEqual(scf.index, index)
        self.assertEqual(scf.attrs, attrs)
        self.assertEqual(scf.feature_list, ['test'])
        np.testing.assert_array_equal(scf.cell_labels, cell_labels)
        np.testing.assert_array_equal(scf.get_matrix('test'), mat)


def create_index():
    """ Create a Genome and Index object for testing."""
    
    # Create the Genome object
    chroms = np.array(['chr1', 'chr2', 'chr7', 'chrX'])
    lengths = np.array([400, 400, 200, 600])
    origins = np.array([0, 200, 0, 200])
    genome = Genome(assembly='mm10', chroms=chroms, lengths=lengths, origins=origins)
    
    # Create the Index object
    resolution = 100
    index = genome.bininfo_optimized(resolution=resolution)
    
    return index

def create_random_data():
    """ Create a random data structure for testing."""
    
    # Get the index
    index = create_index()
    
    # Choose the attributes (n. of cells, ...)
    ncell = 4
    max_ntrace_per_chrom = 2
    attrs = {'ncell': ncell, 'max_ntrace_per_chrom': max_ntrace_per_chrom}
    
    # Create the data: a matrix of random values of shape (ncell, len(index), max_ntrace_per_chrom)
    mat = np.random.rand(ncell, len(index), max_ntrace_per_chrom).astype(np.float32)
    
    return index, attrs, mat


if __name__ == '__main__':
    unittest.main()
