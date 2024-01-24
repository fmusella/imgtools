import unittest
import random
from imgtools.cte import ChromatinTracingExperiment
from imgtools.cte.validator import CTEData

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
        
        def test_pydantic_validation(self):
            
            data = create_random_data()
            
            checker = CTEData(root=data)
            
            return None


def create_random_data():
    
    # Create the labels of cells, chromosomes, traces and spots
    cellIDs = ['cell1', 'cell2', 'cell3']
    chroms = ['chr1', 'chr2', 'chr3', 'chr4']
    traceIDs = ['trace1', 'trace2']
    spotIDs = ['spot1', 'spot2', 'spot3', 'spot4', 'spot5']
    
    # Randomly shuffle the labels
    random.shuffle(cellIDs)
    random.shuffle(chroms)
    random.shuffle(traceIDs)
    random.shuffle(spotIDs)
    
    # Create the data
    data = {}
    
    for cellID in cellIDs:
        
        data[cellID] = {}
        
        for chrom in chroms:
            
            data[cellID][chrom] = {}
            
            for traceID in traceIDs:
                
                data[cellID][chrom][traceID] = {}
                
                for spot in spotIDs:
                    
                    
                    spot_data = {'x': 100 * random.random(),
                                 'y': 100 * random.random(),
                                 'z': 100 * random.random(),
                                 'chrom': chrom,
                                 'start': random.randint(1000, 2000),
                                 'end': random.randint(2000, 3000),
                                 'lum': 1000 * random.random()
                    }
                    
                    data[cellID][chrom][traceID][spot] = spot_data
    
    return data


if __name__ == '__main__':
    unittest.main()
