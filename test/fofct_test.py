import unittest
import os
import random
import numpy as np
from alabtools.utils import Genome, Index
from imgtools.cte.fofct import read_fofct
from imgtools.cte.cte_utils import get_index_and_attrs

# Set the parameters for the test
TEST_PARAMS = {'ncell': 10,
               'assembly': 'hg38',
               'chroms': np.array(['chr1', 'chr2', 'chr7', 'chrX']),
               'lengths': np.array([400, 400, 200, 600]),
               'origins': np.array([0, 200, 0, 200]),
               'resolution': 100}


class TestFofct(unittest.TestCase):
    """Test class for the FoFCT package.
    """
    
    def setUp(self):
        super().setUp()
        # set file directory as the working directory
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        # create the test data
        self.index = createTestIndex()
        self.data, self.attrs = createTestData(self.index)
        # write the fofct file
        self.fofct_file = 'test.fofct.csv'
        writeFofctFile(self.fofct_file, self.data)
    
    def tearDown(self):
        super().tearDown()
        # remove the fofct file
        os.remove(self.fofct_file)
    
    def test_read_fofct(self):
        """Test the reading of FoFCT files."""
        # Read the FoFCT file
        data = read_fofct(self.fofct_file)
        # Get the index and attrs
        index, attrs = get_index_and_attrs(data, TEST_PARAMS['assembly'])
        # Check that the data matches the original data
        self.assertEqual(index, self.index)
        self.assertEqual(data, self.data)
        self.assertEqual(attrs, self.attrs)  # <--- FIX HERE THE max_nspot_per_domain!

def createTestIndex() -> Index:
    """Create a test Index."""
    # Generate the Genome
    genome = Genome(assembly=TEST_PARAMS['assembly'],
                    chroms=TEST_PARAMS['chroms'],
                    lengths=TEST_PARAMS['lengths'],
                    origins=TEST_PARAMS['origins'])
    # Generate the Index
    index = genome.bininfo_optimized(TEST_PARAMS['resolution'])
    return index

def createTestData(index: Index) -> (dict, dict):
    """Create random data for the test."""
    
    random.seed(0)
    
    # Initialize data and summary metrics variables
    data = {}
    ncell = TEST_PARAMS['ncell']
    nchrom = len(TEST_PARAMS['chroms'])
    nspot = 0
    max_ntrace_per_chrom = 0
    max_nspot_per_trace = 0
    max_nspot_per_domain = 0
    
    # Loop over and fill in random data
    spotID = 1
    for cellID in range(ncell):
        
        # Add cell entry
        data[str(cellID)] = {}
        
        for chrom in index.genome.chroms:
            
            # Add chrom entry
            data[str(cellID)][chrom] = {}
            
            # Generate random number of traces
            ntrace = random.randint(1, 5)
            
            # Update the maximum number of traces per chromosome
            max_ntrace_per_chrom = max(max_ntrace_per_chrom, ntrace)
            
            for traceID in range(ntrace):
                
                # Add trace entry
                data[str(cellID)][chrom][str(traceID)] = {}
                
                # Initialize the number of spots in the trace
                nspot_trace = 0
                    
                # Loop over domains
                for s, e in zip(index.start[index.chromstr == chrom], index.end[index.chromstr == chrom]):
                    
                    # Generate number of spots in the domain
                    nspot_domain = random.randint(1, 5)
                    
                    # Update the number of spot counts
                    nspot += nspot_domain
                    nspot_trace += nspot_domain
                    max_nspot_per_domain = max(max_nspot_per_domain, nspot_domain)
                    
                    # Loop over spots of the domain
                    for _ in range(nspot_domain):
                    
                        x, y, z = 100. * np.random.rand(3)
                        lum = 1000. * np.random.rand()
                        spot_data = {'x': x,
                                     'y': y,
                                     'z': z,
                                     'chrom': chrom,
                                     'start': s,
                                     'end': e,
                                     'lum': lum
                                     }
                        data[str(cellID)][chrom][str(traceID)][str(spotID)] = spot_data
                        spotID += 1
            
            # Update the maximum number of spots per trace
            max_nspot_per_trace = max(max_nspot_per_trace, nspot_trace)
            
    # Create the attrs dictionary
    attrs = {
        'ncell': ncell,
        'nchrom': nchrom,
        'nspot': nspot,
        'max_ntrace_per_chrom': max_ntrace_per_chrom,
        'max_nspot_per_trace': max_nspot_per_trace,
        'max_nspot_per_domain': max_nspot_per_domain
    }
    
    return data, attrs

def writeFofctFile(filename: os.path, data: dict) -> None:
    """Write a FoF-CT file (csv format) from the data.
    """
    # create the FoF-CT file
    fofct_file = open(filename, 'w')
    # write the header
    fofct_file.write('#FOF-CT_version=v0.1,\n')
    fofct_file.write('#genome_assembly={},\n'.format(TEST_PARAMS['assembly']))
    fofct_file.write('##XYZ_unit=micron,\n')
    fofct_file.write('"#experimenter_name: Francesco Musella,,,\n')
    fofct_file.write('###lab_name: Frank Alber,\n')
    fofct_file.write('#""description: test FOFCT file for CtFile reading,\n')
    fofct_file.write('##columns=(Spot_ID,Trace_ID,X,Y,Z,Intensity,' +
                        'Chrom,Chrom_Start,Chrom_End,Cell_ID,' +
                        'Extra_Cell_ROI_ID,Additional_Feature)\n')
    # write the data
    for cellID in data:
        for chrom in data[cellID]:
            for traceID in data[cellID][chrom]:
                for spotID in data[cellID][chrom][traceID]:
                    spot_data = data[cellID][chrom][traceID][spotID]
                    fofct_file.write('{},'.format(spotID))
                    fofct_file.write('{},'.format(traceID))
                    fofct_file.write('{},{},{},'.format(spot_data['x'], spot_data['y'], spot_data['z']))
                    fofct_file.write('{},'.format(spot_data['lum']))
                    fofct_file.write('{},{},{},'.format(chrom, spot_data['start'], spot_data['end']))
                    fofct_file.write('{},'.format(cellID))
                    fofct_file.write('{},'.format('EC'))  # useless
                    fofct_file.write('{}'.format('AF'))  # useless
                    fofct_file.write('\n')
    # close the file
    fofct_file.close()
            