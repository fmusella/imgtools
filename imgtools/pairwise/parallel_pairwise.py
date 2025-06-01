import os
import sys
import pickle
import tempfile
from functools import partial
import typing
from alabtools.parallel import Controller
from ..cte import ChromatinTracingExperiment

def control_func(
    cte: ChromatinTracingExperiment, config: dict, func_node: typing.Callable,
    reduce_initialization: typing.Callable, reduce_update: typing.Callable
) -> object:
    
    # Create a temporary directory
    tempdir = tempfile.mkdtemp(dir=os.getcwd())
    sys.stdout.write(f"Temporary directory for nodes' results: {tempdir}\n")
    
    # create a Controller
    controller = Controller(config)
    
    # Get names of the chromosomes
    chroms = cte.index.genome.chroms  # shape: (nchroms,)
    # Get a list with all pairs of chromosomes
    chrom_pairs = []
    for i in range(len(chroms)):
        for j in range(i, len(chroms)):
            chrom_pairs.append((chroms[i], chroms[j]))

    # run the parallel and reduce tasks
    parallel_task = partial(
        parallel_general,
        cte_name = cte.h5_name,
        config=config,
        tempdir=tempdir,
        func_node=func_node
    )
    reduce_task = partial(
        reduce_general,
        cte_name = cte.h5_name,
        config=config,
        tempdir=tempdir,
        reduce_initialization=reduce_initialization,
        reduce_update=reduce_update
    )
    result = controller.map_reduce(
        parallel_task,
        reduce_task,
        args = chrom_pairs,
    )
    
    # Delete the non-empty temporary directory
    os.system(f'rm -r {tempdir}')
    
    del controller
    
    return result

def parallel_general(
    chrom_pair: tuple, cte_name: str, config: dict, tempdir: str, func_node: typing.Callable
) -> tuple:
    
    # Get the chromosomes for the current pair
    chrom1, chrom2 = chrom_pair
    
    # Perform the task for the pair on the node with the 'func_node' function
    pair_result = func_node(chrom1, chrom2, cte_name, config, tempdir)
    
    # Save the pair results in the temporary directory as a pickle file
    out_filename = os.path.join(tempdir, f'{chrom1}_{chrom2}_result.pickle')
    with open(out_filename, 'wb') as f:
        pickle.dump(pair_result, f)
    
    del pair_result
    
    return chrom_pair

def reduce_general(
    chrom_pairs: list, cte_name: str, config: dict, tempdir: str,
    reduce_initialization: typing.Callable, reduce_update: typing.Callable
) -> object:
    
    assert isinstance(chrom_pairs, list), f"chrom_pairs should be a list. Got {type(chrom_pairs)}."
    assert len(chrom_pairs) > 0, "chrom_pairs should not be empty."
    
    # Initialize the result using the 'reduce_initialization' function
    result = reduce_initialization(chrom_pairs, cte_name, config)
    
    # Iterate over the chrom pairs and update the result using the 'reduce_update' function
    for chrom_pair in chrom_pairs:
        
        # Get the chromosomes for the current pair
        chrom1, chrom2 = chrom_pair
        
        # Get the filename for the current pair
        filename = os.path.join(tempdir, f'{chrom1}_{chrom2}_result.pickle')
        assert os.path.isfile(filename), f"Parallel result file for chromosome pair {chrom_pair} does not exist: {filename}"
        
        # Load the result for the current pair
        with open(filename, 'rb') as f:
            pair_result = pickle.load(f)
        
        # Update the result
        result = reduce_update(chrom1, chrom2, result, pair_result, cte_name, config, tempdir)
        
        del pair_result
    
    return result
