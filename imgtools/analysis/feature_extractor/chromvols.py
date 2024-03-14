import numpy as np
import h5py
from ...cte import ChromatinTracingExperiment
from ...cte import cte_utils
from ...cte import cte_io
from ...cte import cte_parallel
from ... import utils


# CHROMOSOME VOLUMES

chromvols_required_keys = {
    'alpha': {'type': float, 'positive': True},
    'force': {'type': bool},
    'reducing_factor': {'type': float, 'positive': True}
}

def run_chromvols(cte: ChromatinTracingExperiment, config: dict) -> dict:
    """ Calculate in parallel the volumes of the chromosomes of the cells in the experiment with alpha shapes.

    Args:
        cte (ChromatinTracingExperiment)
        config (dict): configuration dictionary.

    Returns:
        chrom_vols (dict): dictionary of the type {cellID: {chrom: {traceID: {alpha: float, volume: float, area: float}}}
    """
        
    def rfunc_init(_1, _2, _3) -> dict:
        """ Initialize the result dictionary for the reduction function.

        Args:
            _*: not used, just to match the signature of the function.

        Returns:
            dict: empty dictionary.
        """
        return {}
    
    def rfunc_update(cellID: str, chrom_vols: dict, cell_chrom_vols: dict, _1, _2) -> dict:
        """ Update the population-level dictionary with the data of a cell in the reduction function.

        Args:
            cellID (str)
            chrom_vols (dict): population-level dictionary.
            cell_chrom_vols (dict): data of the cell.
            _*: not used, just to match the signature of the function.

        Returns:
            chrom_vols (dict): updated population-level dictionary, of type
                               {cellID: {chrom: {traceID: {alpha: float, volume: float, area: float}}}
        """
        chrom_vols[cellID] = cell_chrom_vols
        return chrom_vols
    
    chrom_vols = cte_parallel.control_func(
        cte,
        config,
        chromvols_required_keys,
        _chromvols_nfunc,
        rfunc_init,
        rfunc_update
    )
    
    return chrom_vols

def _chromvols_nfunc(cellID: str, cte_name: str, config: dict) -> dict:
    """ Node function for calculating the volumes of the chromosomes of a cell.
    Returns a dictionary of the type: {chrom: {traceID: {alpha: float, volume: float, area: float}}}."""
    
    # Read the data of the cell from the HDF5 file
    with h5py.File(cte_name, 'r') as f:
        cell_data = cte_io.load_cell_data_from_hdf5(cellID, f, format='dict')
    
    # Initialize the dictionary to store the data of each chromosomal trace
    chrom_vols = {}

    # Perform tracing on each chromosomal trace
    for chrom in cell_data:
        chrom_vols[chrom] = {}
        
        for traceID in cell_data[chrom]:
            
            # Get the data of the chromosomal trace
            xs, ys, zs, _, _, _, _ = cte_utils.trace_dict_to_numpy(cell_data[chrom][traceID])
            points = np.array([xs, ys, zs]).T
            
            # Fit the alpha shape to the points
            alpha, mesh = utils.fit_alphashape(points, config['alpha'], config['force'], reducing_factor=config['reducing_factor'])
            
            # Store the data of the chromosomal trace (alpha, volume, area)
            chrom_vols[chrom][traceID] = {
                'alpha': alpha,
                'volume': mesh.volume,
                'area': mesh.area
            }
        
            del xs, ys, zs, _, points, mesh
    
    del cell_data
    
    return chrom_vols
