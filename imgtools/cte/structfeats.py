import numpy as np
from .cte import ChromatinTracingExperiment
from . import utils
from . import _parallelization


# CHROMOSOME VOLUMES

def _chromvols_pfunc(cell_data: dict, data_attrs, index, config: dict) -> dict:
    """ Parallel function for calculating the volumes of the chromosomes of a cell."""
    
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
    
    return chrom_vols

def get_chromvols(cte: ChromatinTracingExperiment, config: dict) -> dict:
    
    required_keys = {
        'alpha': {'type': float, 'positive': True},
        'force': {'type': bool}
    }
        
    def rfunc_init(cellIDs, data_attrs, index, config) -> dict:
        """ Initialization for the reduction function."""
        return {}
    
    def rfunc_update(cellID: str, result: dict, cell_result: dict, cellIDs, data_attrs, index, config) -> dict:
        """ Update for the reduction function."""
        result[cellID] = cell_result
        return result
    
    chrom_vols = _parallelization.control_func(
        cte.data,
        cte.attrs,
        cte.index,
        config,
        required_keys,
        _chromvols_pfunc,
        rfunc_init,
        rfunc_update
    )
    
    return chrom_vols
