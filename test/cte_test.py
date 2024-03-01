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
            
            # Check the consistency of the CTE object
            cte.check_consistency()
            
            # Check that the index object is the same
            self.assertEqual(cte.index, index)
            
            # Check that cell_labels is the same
            np.testing.assert_array_equal(cte.cell_labels, np.array([cellID for cellID in data]).astype('U20'))
            
            # For each cell, check that the data is the same
            for cellID in data:
                self.assertEqual(cte.get_data(cellID), data[cellID])
        
        def test_pop_cells(self) -> None:
            
            # Create the data
            index, data = create_random_data()
            
            # Create a CTE object
            filename = './test.cte.h5'
            cte = ChromatinTracingExperiment(filename, 'w')
            cte.set_data_attrs_index(data=data, index=index, check_data=True)
            
            # Check the consistency of the CTE object
            cte.check_consistency()
            
            # Pop the cells
            cellIDs_topop = ['cell1', 'cell2']
            cte.pop_cells(cellIDs_topop)
            
            # Check again the consistency of the CTE object
            cte.check_consistency()
            
            # Check the number of cells in the attribiutes
            self.assertEqual(cte.attrs['ncell'], 5)
            self.assertEqual(cte.attrs['ncell_removed'], 2)
            self.assertEqual(cte.attrs['ncell_remaining'], 3)
            
            # Check that the cell is no longer in the data
            for cellID in cellIDs_topop:
                
                self.assertNotIn(cellID, cte.cell_labels)
                
                # Try to get the data
                try:
                    cte.get_data(cellID)
                    raise ValueError('The cell should not be in the data.')
                except KeyError:
                    pass
                
        def test_pop_spots(self) -> None:
            """ Test the pop_spots method."""
            
            # Create the data
            index, data = create_random_data()
            
            # Create a CTE object
            filename = './test.cte.h5'
            cte = ChromatinTracingExperiment(filename, 'w')
            cte.set_data_attrs_index(data=data, index=index, check_data=True)
            
            # Check the consistency of the CTE object
            cte.check_consistency()
            
            # Get the spotIDs to pop
            # Since it has to be given as a dictionary (spots_topop[cellID][chrom][traceID] = [spotID1, spotID2, ...]),
            # I have to create a dictionary with the spotIDs to pop
            # I first write a list and then I convert it to a dictionary
            spots_topop_list = ['1', '4', '5', '9']
            spots_topop = {}
            for cellID in data:
                for chrom in data[cellID]:
                    for traceID in data[cellID][chrom]:
                        for spotID in data[cellID][chrom][traceID]:
                            if spotID in spots_topop_list:
                                if cellID not in spots_topop:
                                    spots_topop[cellID] = {}
                                if chrom not in spots_topop[cellID]:
                                    spots_topop[cellID][chrom] = {}
                                if traceID not in spots_topop[cellID][chrom]:
                                    spots_topop[cellID][chrom][traceID] = []
                                spots_topop[cellID][chrom][traceID].append(spotID)
            
            # Pop the spots
            cte.pop_spots(spots_topop)
            
            # Check again the consistency of the CTE object
            cte.check_consistency()
            
            # Check the number of spots in the attributes
            self.assertEqual(cte.attrs['nspot_removed'], 4)
            self.assertEqual(cte.attrs['nspot_remaining'], cte.attrs['nspot'] - 4)
            
            # Check that the spots are no longer in the data
            for cellID in data:
                for chrom in data[cellID]:
                    for traceID in data[cellID][chrom]:
                        for spotID in data[cellID][chrom][traceID]:
                            if spotID in spots_topop_list:
                                self.assertNotIn(spotID, cte.get_data(cellID)[chrom][traceID])
            

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
    cellIDs = ['cell1', 'cell2', 'cell3', 'cell4', 'cell5']
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
