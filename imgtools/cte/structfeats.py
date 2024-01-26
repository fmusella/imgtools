import numpy as np
import tempfile
import os
import sys
from functools import partial
import pickle
from alabtools.parallel import Controller
from . import ChromatinTracingExperiment
from . import utils


# CHROMOSOME VOLUMES

def get_chromosome_volumes(cte: ChromatinTracingExperiment, config: dict) -> dict:
    
    # Create a temporary directory
    tempdir = tempfile.mkdtemp(dir=os.getcwd())
    sys.stdout.write("Temporary directory for nodes' results: {}\n".format(tempdir))
    
    # Save the data of each cell separately in the temporary directory as a pickle file
    for cellID in cte.data:
        filename = os.path.join(tempdir, '{}_data.pickle'.format(cellID))
        with open(filename, 'wb') as f:
            pickle.dump(cte.data[cellID], f)
    
    # set the parallel and reduce tasks
    parallel_task = partial(parallel_chromosome_volumes, config=config, tempdir=tempdir)
    reduce_task = partial(reduce_chromosome_volumes, tempdir=tempdir)
    
    # create a Controller
    controller = Controller(config)

    # run the parallel and reduce tasks
    chrom_vols = controller.map_reduce(parallel_task, reduce_task, args=list(cte.data.keys()))
    
    # Delete the non-empty temporary directory
    os.system('rm -r {}'.format(tempdir))
    
    del controller
    
    return chrom_vols

def parallel_chromosome_volumes(cellID: str, config: dict, tempdir: str) -> str:
    
    # Check the config
    if not 'alpha' in config:
        raise ValueError("config should contain the key 'alpha'.")
    if not 'force' in config:
        raise ValueError("config should contain the key 'force'.")
    
    assert isinstance(cellID, str), "cellID should be a string. Got type: {}".format(type(cellID))
    
    assert isinstance(tempdir, str), "tempdir should be a string. Got type: {}".format(type(tempdir))
    assert os.path.isdir(tempdir), "tempdir should be a directory. Got: {}".format(tempdir)
    
    # Try to load the data for the cell with pickle
    in_filename = os.path.join(tempdir, '{}_data.pickle'.format(cellID))
    assert os.path.isfile(in_filename), "Data for cell {} not found.".format(cellID)
    with open(in_filename, 'rb') as f:
        cell_data = pickle.load(f)
    
    # Initialize the dictionary that will contain the chromosomal volumes
    chrom_vols = {}
    
    # Perform tracing on each chromosome
    for chrom in cell_data:
        for traceID in cell_data[chrom]:
            
            # Get the data of the chromosomal trace and fit an alpha shape
            xs, ys, zs, _, _, _, _, _ = utils.trace_dict_to_numpy(cell_data[chrom][traceID])
            points = np.array([xs, ys, zs]).T
            
            _, mesh = utils.fit_alphashape(points, config['alpha'], config['force'])
            
            # Calculate the volume of the alpha shape and save it
            if chrom not in chrom_vols:
                chrom_vols[chrom] = []
            chrom_vols[chrom].append(mesh.volume)
        
            del xs, ys, zs, _, points, mesh
        
    # Save the chromosomal volumes for the cell
    out_filename = os.path.join(tempdir, '{}_chrom_vols.pickle'.format(cellID))
    with open(out_filename, 'wb') as f:
        pickle.dump(chrom_vols, f)
    
    del cell_data, chrom_vols
    
    return cellID

def reduce_chromosome_volumes(cellIDs: list, tempdir: str) -> dict:
    
     # Check cellIDs
    assert isinstance(cellIDs, list), "cellIDs should be a list. Got type: {}".format(type(cellIDs))
    assert len(cellIDs) > 0, "cellIDs should not be empty."
    
    # Initialize the output, which is a dictionary of chromosomal volumes
    chrom_vols = {}

    for cellID in cellIDs:
        
        # Get the filename for the temporary chromosomal volumes of the cell
        filename = os.path.join(tempdir, '{}_chrom_vols.pickle'.format(cellID))
        
        # Check that the file exists
        assert os.path.isfile(filename), "Chrom volumes file for cell {} not found.".format(cellID)

        # Load the cell file
        try:
            with open(filename, 'rb') as f:
                cell_chrom_vols = pickle.load(f)
        except:
            raise ValueError("Chrom volumes file for cell {} is not a valid pickle file.".format(cellID))
        
        # Add the chromosomal volumes of the cell to the output
        chrom_vols[cellID] = cell_chrom_vols
    
    return chrom_vols

