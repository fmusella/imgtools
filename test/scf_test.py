import unittest
import random
import numpy as np
import copy
from imgtools.scf import SingleCellFeature
from alabtools.utils import Genome, Index

class TestSingleCellFeature(unittest.TestCase):
    
    def setUp(self) -> None:
        return super().setUp()
    
    def tearDown(self) -> None:
        return super().tearDown()
    
    def test_save_load(self) -> None:
        """ Test that the data can be saved and loaded correctly."""
        
        # Create the test data
        data = create_random_data()
        index = data['index']
        attrs = data['attrs']
        cell_labels = data['cell_labels']
        cell_states = data['cell_states']
        volumes = data['volumes']
        mat = data['mat']
        
        # Create a SCF object
        filename = './test.scf.h5'
        scf = SingleCellFeature(filename, 'w')
        
        # Add the data to the SCF object
        scf.add_index_attrs_cell_labels(index, attrs, cell_labels)
        scf.add_cell_states(cell_states)
        scf.add_volumes(volumes)
        scf.add_matrix(mat, 'test')
        
        # Check that the data has been added correctly
        self.assertEqual(scf.index, index)
        self.assertEqual(scf.attrs, attrs)
        self.assertEqual(scf.feature_list, ['test'])
        np.testing.assert_array_equal(scf.cell_labels, cell_labels)
        np.testing.assert_array_equal(scf.cell_states, cell_states)
        np.testing.assert_array_equal(scf.volumes, volumes)
        np.testing.assert_array_equal(scf.get_matrix('test'), mat)
    
    def test_pop_cells(self) -> None:
        """ Test the pop_cells method."""
        
        # Create the test data
        data = create_random_data()
        index = data['index']
        attrs = data['attrs']
        cell_labels = data['cell_labels']
        cell_states = data['cell_states']
        volumes = data['volumes']
        mat = data['mat']
        
        # Create a SCF object
        filename = './test.scf.h5'
        scf = SingleCellFeature(filename, 'w')
        
        # Add the data to the SCF object
        scf.add_index_attrs_cell_labels(index, attrs, cell_labels)
        scf.add_cell_states(cell_states)
        scf.add_volumes(volumes)
        scf.add_matrix(mat, 'test')
        
        # Pop the cells
        cellIDs_topop = np.random.choice(cell_labels, 2, replace=False)
        attrs_popped = copy.deepcopy(attrs)
        attrs_popped['ncell'] -= 2
        scf_pop = scf.pop_cells(cellIDs_topop, index, attrs_popped, './test.scf.popped.h5')  # assuming index is the same
        
        # Check that the data has been popped correctly
        self.assertEqual(scf_pop.index, index)
        self.assertEqual(scf_pop.attrs, attrs_popped)
        self.assertEqual(scf_pop.feature_list, ['test'])
        assert len(scf_pop.cell_labels) == 3
        assert len(scf_pop.cell_states) == 3
        assert scf_pop.get_matrix('test').shape == (3, len(index), 2), "Shape of matrix, {}, is wrong.".format(scf.get_matrix('test').shape)
        


def create_index() -> Index:
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

def create_random_data() -> dict:
    """ Create a random data structure for testing."""
    
    # Get the index
    index = create_index()
    
    # Choose the attributes (n. of cells, ...)
    ncell = 5
    max_ntrace_per_chrom = 2
    attrs = {'ncell': ncell, 'max_ntrace_per_chrom': max_ntrace_per_chrom}
    cell_labels = np.arange(ncell).astype('U20')
    cell_states = np.array([random.choice(['G1', 'G2', 'S']) for _ in range(ncell)])
    volumes = np.random.rand(ncell).astype(np.float32)
    
    # Create the data: a matrix of random values of shape (ncell, len(index), max_ntrace_per_chrom)
    mat = np.random.rand(ncell, len(index), max_ntrace_per_chrom).astype(np.float32)
    
    data = {
        'index': index,
        'attrs': attrs,
        'cell_labels': cell_labels,
        'cell_states': cell_states,
        'volumes': volumes,
        'mat': mat
    }
    
    return data


if __name__ == '__main__':
    unittest.main()
