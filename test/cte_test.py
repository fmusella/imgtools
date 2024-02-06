import unittest
import random
import numpy as np
from imgtools.cte import ChromatinTracingExperiment
from imgtools.cte import validator
from alabtools.utils import Genome, Index

class TestChromatinTracingExperiment(unittest.TestCase):
    
    def setUp(self) -> None:
        return super().setUp()
    
    def tearDown(self) -> None:
        return super().tearDown()

class TestCTEData(unittest.TestCase):
        
        def setUp(self) -> None:
            return super().setUp()
        
        def tearDown(self) -> None:
            return super().tearDown()
        
        def test_pydantic_validation(self) -> None:
            
            index, data = create_random_data()
            checker = validator.CTEData(root=data)
        
        def test_save_load(self) -> None:
            """ Test that the data can be saved and loaded correctly."""
            
            # Create the data
            index, data = create_random_data()
            
            # Create a CTE object
            filename = './test.cte.h5'
            cte = ChromatinTracingExperiment(filename, 'w')
            
            cte.set_data_attrs_index(data=data, index=index, check_data=True)
            
            # Check that the index object is the same
            self.assertEqual(cte.get_index(), index)
            
            # For each cell, check that the data is the same
            for cellID in data:
                self.assertEqual(cte.get_data(cellID), data[cellID])


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
    
    # Create the labels of cells, chromosomes, traces and spots
    cellIDs = ['cell1', 'cell2', 'cell3']
    traceIDs = ['trace1', 'trace2']
    
    # Create the data
    data = {}
    
    # Initialize the spotID, increment it for each spot added
    spotID = 1
    
    # Loop over the cells, chromosomes and traces to create the data
    for cellID in cellIDs:
        data[cellID] = {}
        
        for chrom in index.genome.chroms:
            data[cellID][chrom] = {}
            
            for traceID in traceIDs:
                data[cellID][chrom][traceID] = {}
                
                # Get the arrays of start and end for the chromosome
                start = index.start[index.chromstr == chrom]
                end = index.end[index.chromstr == chrom]
                
                # Loop over each start/end pair to create the spots
                for i in range(len(start)):
                    
                    spot_data = {
                        'x': 100 * random.random(),
                        'y': 100 * random.random(),
                        'z': 100 * random.random(),
                        'chrom': chrom,
                        'start': int(start[i]),
                        'end': int(end[i]),
                        'lum': 1000 * random.random()
                    }
                    
                    # Add the spot data to the data structure
                    data[cellID][chrom][traceID][str(spotID)] = spot_data
                    
                    # Increment the spotID
                    spotID += 1
    
    return index, data


if __name__ == '__main__':
    unittest.main()
