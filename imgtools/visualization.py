import os
import sys
import pickle
import numpy as np
from scipy.stats import gaussian_kde
from scipy.ndimage import binary_dilation, binary_erosion
from matplotlib import pyplot as plt
from matplotlib import colors as plt_colors
from matplotlib import cm
import trimesh
from alabtools.utils import get_index_from_bed
from alabtools.plots import write_pdb
from .cte import ChromatinTracingExperiment, cte_utils, cte_parallel
from .cte.metrics import get_trace_ranks_for_cell
from .scf import SingleCellFeature, scf_utils
from . import parallel
from . import utils


# PDB functions

def save_cell_pdb(
    path: str,
    cellID: str,
    cte: ChromatinTracingExperiment,
    scf: SingleCellFeature = None,
    feature: str = None,
    feature_nquants: int = None,
    bedfile: str = None,
    exclude_imputed: bool = False
) -> None:
    """ Save a PDB file for a cell.
    The PDB file will contain the 3D coordinates of the spots in the cell, with the following columns:
    - x: x-coordinate of the spot
    - y: y-coordinate of the spot
    - z: z-coordinate of the spot
    - residue_name: chromosome number
    - chain_id: trace number
    - occupancy: start position of the spot in bp
    - beta: feature value (luminescence or other feature)
    - element_symbol: 'Na' if the feature value is NaN, 'O' otherwise
    - atom_name: domain-labels of each spot. Optional
    
    If a SingleCellFeature object is provided, the feature values will be used as the beta factor.
    Otherwise, the luminescence values will be used.

    Args:
        path (str): folder to save the pdb file
        cellID (str)
        cte (ChromatinTracingExperiment)
        scf (SingleCellFeature or None)
        feature (str or None)
        feature_nquants (int or None): number of quantiles to quantize the feature values. Optional
        bedfile (str or None): path to a BED file with the labels of each domain. Optional
        exclude_imputed (bool): if True, imputed spots are excluded from the PDB file. Default is False.
    """
    
    # Check that the path exists. If not, create it.
    if not isinstance(path, str):
        raise TypeError("path must be a string.")
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Get data for cell in numpy array format
    xs, ys, zs, chroms, starts, ends, lums, traceIDs, spotIDs = cte.get_data(cellID, format='numpy')
    
    # Convert chroms to chromnums, e.g. 'chr1' --> '1', 'chrX' --> 'X'
    chromnums = []
    for c in chroms:
        chromnums.append(c.replace('chr', ''))
    chromnums = np.array(chromnums).astype(str)

    # Convert traceIDs to trace ranks within each chromosome, and then to strings
    # e.g. traceID: '12_1' --> trace_rank: 1 ---> tracenum: 'A'
    tranks = get_trace_ranks_for_cell(cte, cellID)  # ranks of each trace in each chromosome of the cell
    tracenums = []
    for chrom, traceID in zip(chroms, traceIDs):
        t = tranks[chrom][traceID]  # rank of traceID in chrom
        if t > 0:
            # Valid traces (positive integers) are converted like this:
            #   1 --> 'A', 2 --> 'B', ...
            tracenums.append(chr(t + 64))
        elif t < 0:
            # Noisy traces (negative integers) are converted like this:
            #   -1 --> 'Z', -2 --> 'Y', ...
            tracenums.append(chr(t + 91))
        else:
            raise Exception("Trace number cannot be 0.")
    tracenums = np.array(tracenums).astype(str)
    
    # Get the hash table for traceIDs
    traceID_hash = cte.get_trace_hashmap(cellID)
    
    # If a BED file is provided, get the labels for each spot
    if bedfile is not None:
        labels = get_labels_from_bed(bedfile, cte, chroms, starts, ends)
    else:
        labels = np.full(len(xs), '', dtype='U4')
    
    # If a feature is provided, use it as the beta factor
    if scf is not None and feature is not None:
        # If the number of quantiles is provided, check that it is valid
        if feature_nquants is not None:
            if not isinstance(feature_nquants, int):
                raise TypeError("feature_nquants must be an integer.")
            if feature_nquants < 1 or feature_nquants > 999:
                raise ValueError(f"feature_nquants must be between 1 and 999. Got {feature_nquants}.")
        # Get the feature values for the spots
        featvals = get_feature_for_pdb(cellID, scf, feature, traceID_hash, traceIDs, chroms, starts, ends, feature_nquants)
    # Otherwise, use the luminescence as the beta factor
    else:
        featvals = lums
    
    # Create a 2-string-valued array that is 'Na' where the feature value is NaN, and 'O' where it is not
    featsnan = np.where(np.isnan(featvals), 'Na', 'O')
    featsnan = featsnan.astype('U2')
    
    # If all values are NaN, set the feature values to 0
    if np.all(np.isnan(featvals)):
        featvals = np.zeros(featvals.shape)
    # Otherwise, replace the NaNs with the minimum value of the feature
    else:
        featvals[np.isnan(featvals)] = np.nanmin(featvals)
    
    # Clip featvals to 5% and 95% percentiles to remove outliers (if not quantized)
    if feature_nquants is None:
        featvals = np.clip(featvals, np.percentile(featvals, 5), np.percentile(featvals, 95))
    # If the feature values are constant (min == max), set them to 0
    if np.min(featvals) == np.max(featvals):
        featvals = np.zeros(featvals.shape)
    # Otherwise, min-max normalize to [0, 999] (if not quantized)
    else:
        if feature_nquants is None:
            featvals = (featvals - np.min(featvals)) / (np.max(featvals) - np.min(featvals)) * 999
    # Truncate to 2 decimal places
    featvals = np.round(featvals, 2)
    
    # Convert starts to units in bp such that the maximum values has 3 digits above the decimal point (i.e. < 1000)
    while np.max(starts) >= 1000:
        starts = starts / 10
    # Truncate to 2 decimal places
    starts = np.round(starts, 2)
    
    # If exclude_imputed is True, create a mask to exclude imputed spots
    # (If exclude_imputed is False, the mask is all True)
    mask_spots = np.ones(len(spotIDs), dtype=bool)
    if exclude_imputed:
        for i, spotID in enumerate(spotIDs):
            if 'IMPUTED' in spotID:
                mask_spots[i] = False
    
    # Write dictionary for pdb file
    celldata_for_pdb = {
        'x': xs[mask_spots],
        'y': ys[mask_spots],
        'z': zs[mask_spots],
        'residue_name': chromnums[mask_spots],
        'chain_id': tracenums[mask_spots],
        'occupancy': starts[mask_spots],
        'beta': featvals[mask_spots],
        'element_symbol': featsnan[mask_spots],
        'atom_name': labels[mask_spots],
    }
    
    # Write pdb file
    if feature is None:
        filename = os.path.join(path, f"{cellID}.pdb")
    else:
        filename = os.path.join(path, f"{cellID}_{feature}.pdb")
    
    write_pdb(filename, celldata_for_pdb)

def get_labels_from_bed(
    bedfile: str, cte: ChromatinTracingExperiment,
    chroms: np.ndarray, starts: np.ndarray, ends: np.ndarray
) -> np.ndarray:
    """ Get the labels from a BED file for the spots in the CTE.
    
    The BED file should have the same length of the CTE Index.
    It provides a label for each domain in the Index.
    
    Labels should be <= 4 characters.
    
    This function converts the Index-based labels into
    an array of labels for the spots in the CTE.

    Args:
        bedfile (str): path to the BED file
        cte (ChromatinTracingExperiment)
        chroms (np.ndarray): Array of chromosome names for the spots
        starts (np.ndarray): Array of start positions for the spots
        ends (np.ndarray): Array of end positions for the spots

    Returns:
        (np.ndarray, 'U4' type): Array of symbols-converted labels for the spots of CTE
    """
    
    # Read the bed file as Index
    index = cte.index  # get the index from the CTE
    bed = get_index_from_bed(bedfile, genome=index.genome)
    if bed != index:
        raise ValueError("The bed file does not match the CTE index.")
    
    # Try getting the labels from the bed file
    try:
        labels = bed.track0.astype(str)
    except Exception as e:
        raise ValueError("Could not get labels from the bed file.") from e
    
    # Check that the lengths of the strings are <= 4
    if len(labels[0]) > 4:
        raise ValueError("BED labels should be <= 4 characters.")
    
    # Convert the labels into an array for the spots in the CTE
    labels_cte = []
    index_hashmap = index.get_index_hashmap()
    for chrom, start, end in zip(chroms, starts, ends):
        i_domain = index_hashmap[(chrom, start, end)]
        assert len(i_domain) == 1, f"Multiple domains found for {chrom}:{start}-{end}."
        i_domain = i_domain[0]
        labels_cte.append(labels[i_domain])
    labels_cte = np.array(labels_cte).astype('U4')
    
    return labels_cte

def get_feature_for_pdb(
    cellID: str,
    scf: SingleCellFeature,
    feature: str,
    traceID_hash: dict,
    traceIDs: np.ndarray,
    chroms: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    nquants: int = None
) -> np.ndarray:
    """ Get the feature values for a cell in the same order as the spots in the CTE.

    Args:
        cellID (str)
        scf (SingleCellFeature)
        feature (str)
        traceID_hash (dict): Dictionary that maps traceIDs to numpy array indices, obtained from the CTE
        traceIDs (np.ndarray): Array of traceIDs for the spots
        chroms (np.ndarray): Array of chromosome names for the spots
        starts (np.ndarray): Array of start positions for the spots
        ends (np.ndarray): Array of end positions for the spots
        nquants (int, optional): Number of quantiles to quantize the feature values. Optional

    Returns:
        featvals (np.ndarray): Array of feature values for the spots, ordered as the spots in the CTE
    """
    
    # Get the feature matrix
    feature_mat = scf.get_feature(feature, cellID)
    
    # If the number of quantiles is provided, quantize the feature values
    if nquants is not None:
        feature_mat = scf_utils.quantize_matrix_cell(feature_mat, nquants)
        # The NaN values are set as -1 in the quantized matrix
        # We convert them back to NaN
        feature_mat = feature_mat.astype(float)
        feature_mat[feature_mat == -1] = np.nan
    
    # Create a hash table for the index
    index_hash = scf.index.get_index_hashmap()
    
    # Get the feature values for the cell, in the same order as the spots
    featvals = []
    for traceID, chrom, start, end in zip(traceIDs, chroms, starts, ends):
        
        # Get the position of the spot in the array using the hash tables
        i_domain = index_hash[(chrom, start, end)]
        assert len(i_domain) == 1, f"Multiple domains found for {chrom}:{start}-{end} in cell {cellID}."
        i_domain = i_domain[0]
        i_trace = traceID_hash[chrom][traceID]
        
        # Get the feature value
        featval = feature_mat[i_domain, i_trace]
        featvals.append(featval)
    featvals = np.array(featvals).astype(float)
    
    return featvals

def save_all_features_cell_pdbs(
    cellID: str,
    cte: ChromatinTracingExperiment,
    scf: SingleCellFeature,
    path: str,
    bedfile: str = None
) -> None:
    """ Save the PDB files for each feature in a cell.

    Args:
        cellID (str)
        cte (ChromatinTracingExperiment)
        scf (SingleCellFeature)
        path (str): path to save the pdb files
        bedfile (str, optional): path to a BED file with the labels of each domain. Optional
    """
    
    # If the path does not exist, create it
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Create a subfolder for the cell
    cell_path = os.path.join(path, cellID)
    if not os.path.exists(cell_path):
        os.makedirs(cell_path)
    
    sys.stdout.write(f"Saving PDB files for cell {cellID} in {cell_path}...\n")
    
    # Get the list of features in the SingleCellFeature object
    features = scf.feature_list
    
    # Remove the 'replistate' feature from the list
    if 'replistate' in features:
        features.remove('replistate')
    
    sys.stdout.write(f"Features:\n")
    for feature in features:
        sys.stdout.write(f"  - {feature}\n")
    
    # Save the pdb files for each feature
    for feature in features:
        
        sys.stdout.write(f"     ...saving feature {feature}...\n")
        save_cell_pdb(cell_path, cellID, cte, scf, feature, bedfile)
    
    sys.stdout.write(f"Done.\n")


# CMM functions

def save_cell_cmm_bychrom(
    cte: ChromatinTracingExperiment, cellID: str,
    path: str, radius: float, do_link: bool = True,
    colormap: str = 'tab20'
) -> None:
    """ Write a cmm file for a cell.
    
    Each trace is written in a separate cmm file.

    Args:
        cte (ChromatinTracingExperiment)
        cellID (str)
        path (str): directory where the cmm files will be saved
        radius (float): size of the markers (in physical units)
        do_link (bool, optional): if True, links are drawn between consecutive markers. Default is True.
        colormap (str, optional): name of the colormap to use. Default is 'tab20'.
    """
    
    # Check that the path exists. If not, create it.
    if not isinstance(path, str):
        raise TypeError("path must be a string.")
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Get the data for the cell in dictionary format
    cell_data = cte.get_data(cellID)
    
    # Map each chromosome to a different color from the colormap
    cmap = np.array(cm.get_cmap(colormap).colors)
    chrom2color = {chrom: cmap[i % 20] for i, chrom in enumerate(cell_data.keys())}
    
    # Loop over chromosomes and traces, and write each trace to a separate cmm file
    for chrom in cell_data:
        for traceID in cell_data[chrom]:
            
            # Get the data for the trace
            xs, ys, zs, starts, ends, _, _ = cte_utils.trace_dict_to_numpy(cell_data[chrom][traceID])
            
            # If do_link is True, create links between the markers
            if do_link:
                # Sort the data by the start position, so that links are drawn in the correct order
                sort = np.argsort(starts)
                xs, ys, zs, starts, ends = xs[sort], ys[sort], zs[sort], starts[sort], ends[sort]
                # Two spots are linked only if they are consecutive in the sorted array,
                # i.e. the end position of the first spot is the start position of the second spot
                # Create a boolean array of size n-1, where True means that i and i+1 are linked
                links = np.roll(starts, -1) == ends
                links = links[:-1]
            else:
                links = None
            
            utils.write_cmm(
                filename = os.path.join(path, f'{chrom}_{traceID}.cmm'),
                marker_str = f'cellID: {cellID}, chrom: {chrom}, traceID: {traceID}',
                coord = np.array([xs, ys, zs]).T,
                radius = radius,
                color = chrom2color[chrom],
                links = links
            )

def save_cell_cmm_bybed(
    cte: ChromatinTracingExperiment,
    cellID: str, path: str, radius: float,
    bedfile: str,
    scf: SingleCellFeature = None, feature: str = None,
    pmin: float = None, pmax: float = None,
    colormap: str = 'Reds'
) -> None:
    """ Write a cmm file for a cell in different files,
    where each file corresponds to a different label in the BED file.
    
    E.g. if the BED file has labels 'E', 'M', 'L', 'NA' for early, mid, late Initiation and not assigned,
    the function will create 4 cmm files, one for each label, e.g. cellID_E.cmm, cellID_M.cmm, cellID_L.cmm, cellID_NA.cmm.
    
    The color of each marker in each file is either:
        - If no SCF and feature are provided, a different color from the tab20 colormap for each label
        - If SCF and feature are provided, the color is mapped to the single-cell feature values using the selected colormap

    Args:
        cte (ChromatinTracingExperiment)
        cellID (str)
        path (str): directory where the cmm files will be saved
        radius (float): size of the markers (in physical units)
        bedfile (str): path to the BED file with the labels
        scf (SingleCellFeature, optional): Used together with feature to map the feature values to colors. Defaults to None.
        feature (str, optional): Used together with scf to map the feature values to colors. Defaults to None.
        pmin (float, optional): If SCF and feature are provided, the minimum percentile to use for saturation. Defaults to None.
        pmax (float, optional): If SCF and feature are provided, the maximum percentile to use for saturation. Defaults to None.
        colormap (str, optional): name of the colormap to use if SCF and feature are provided. Defaults to 'Reds'.
    """
    
    # Check that the path exists. If not, create it.
    if not isinstance(path, str):
        raise TypeError("path must be a string.")
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Get the data for the cell in dictionary format
    xs, ys, zs, chroms, starts, ends, _, traceIDs, _ = cte.get_data(cellID, format='numpy')
    
    # Get the labels for the spots
    labels = get_labels_from_bed(bedfile, cte, chroms, starts, ends)
    unique_labels = np.unique(labels)
    
    # If a SCF and feature are provided, get the feature values for the spots
    # and map them to the selected colormap
    if scf is not None and feature is not None:
        # Get the feature values for the spots
        traceID_hash = cte.get_trace_hashmap(cellID)
        featvals = get_feature_for_pdb(cellID, scf, feature, traceID_hash, traceIDs, chroms, starts, ends)
        # Get the colormap for the feature values
        cmap = cm.get_cmap(colormap)
        # Interpolate the feature values to the colormap
        pmin = 5 if pmin is None else pmin
        pmax = 95 if pmax is None else pmax
        fmin = np.percentile(featvals, pmin)
        fmax = np.percentile(featvals, pmax)
        norm = plt_colors.Normalize(vmin=fmin, vmax=fmax)
        # Map each feature value to a color from the colormap
        colors = cmap(norm(featvals))[:, :3]
    # Otherwise, map each label to a different color from the tab20 colormap
    else:
        tab20 = np.array(cm.get_cmap('tab20').colors)
        label2color = {label: tab20[i % 20] for i, label in enumerate(unique_labels)}
        colors = np.array([label2color[label] for label in labels])
    
    # Create a CMM file for each unique label
    for label in unique_labels:
            
            # Get the indices of the spots with the label
            idx = np.where(labels == label)
            
            # Write filename and marker string
            filename = os.path.join(path, f'{cellID}_{label}.cmm')
            marker_str = f'cellID: {cellID}, label: {label}'
            if scf is not None and feature is not None:
                filename = filename.replace('.cmm', f'_{feature}.cmm')
                marker_str = marker_str + f', feature: {feature}'
            
            # Write the CMM file
            utils.write_cmm(
                filename = filename,
                marker_str = marker_str,
                coord = np.array([xs[idx], ys[idx], zs[idx]]).T,
                radius = radius,
                color = colors[idx],
            )


# MRC functions

def run_mrc(cte: ChromatinTracingExperiment, config: dict) -> None:
    """ Creates the mrc files for all cells in the experiment in parallel.
    The files are created in a folder specified in config, together with a pickle file
    containing the origin and shape of each MRC file.

    Args:
        cte (ChromatinTracingExperiment)
        config (dict): configuration dictionary for the mrc file creation
    """
    
    def _rfunc_init(_1, _2, _3) -> dict:
        """ Initialize the mrc parameters dictionary for the reduce function.

        Args:
            _*: not used, just to match the signature of the function

        Returns:
            mrc_params (dict): empty dictionary
        """
        mrc_params = {}
        return mrc_params
    
    def _rfunc_update(cellID: str, mrc_params: dict, cell_mrc_params: dict, _1, _2) -> dict:
        """ Update the mrc parameters dictionary for the reduce function.

        Args:
            cellID (str)
            mrc_params (dict): mrc parameters dictionary for the entire population
            cell_mrc_params (dict): mrc parameters dictionary for the cell
            _*: not used, just to match the signature of the function

        Returns:
            mrc_params (dict): updated mrc parameters dictionary for the entire population
        """
        mrc_params[cellID] = cell_mrc_params
        return mrc_params
    
    # If the CTE doesn't have alphashapes, raise an error
    if 'alphashapes' not in cte:
        raise KeyError("Alphashapes not present in the ChromatinTracingExperiment.")
    
    # Check that the path is present in config
    if 'mrc_path' not in config:
        raise KeyError("mrc_path not present in config.")
    # Check that the path is a valid path-like string
    if not isinstance(config['mrc_path'], str):
        raise TypeError("mrc_path must be a string.")
    # Transform the path to an absolute path
    config['mrc_path'] = os.path.abspath(config['mrc_path'])
    # Create the path if it does not exist
    if not os.path.exists(config['mrc_path']):
        os.makedirs(config['mrc_path'])
    
    # Run the MRC calculation in parallel
    # The MRC files are saved in the folder specified in config,
    # and here we return the origin and shape of each cell
    mrc_params = cte_parallel.control_func(
        cte,
        config,
        mrc_required_keys,
        _mrc_nfunc,
        _rfunc_init,
        _rfunc_update
    )
    
    # Save the mrc parameters as a pickle file in the folder specified in config
    out_filename = os.path.join(config['mrc_path'], 'mrc_params.pickle')
    with open(out_filename, 'wb') as f:
        pickle.dump(mrc_params, f)
    
    del mrc_params

def run_mrc_single_cell(cte: ChromatinTracingExperiment, cellID: str, config: dict) -> tuple:
    """ Performs the mrc file creation task on a single cell.
    
    The mrc file is stored in the path
    specified in config.
    
    The function returns the origin and shape of the volume mrc file,
    necessary for aligning the mrc files in 3D space.

    Args:
        cellID (str): cell ID.
        config (dict): configuration dictionary for the mrc file creation.

    Returns:
        origin (tuple): origin of the volume mrc file in voxel units.
        shape (tuple): shape of the volume mrc file in voxel units.
    """
    
    # Check that all required keys are present in config
    cte_parallel.check_config(config, mrc_required_keys, parallel=False)
    
    # Transform the path to an absolute path
    config['mrc_path'] = os.path.abspath(config['mrc_path'])
    # Create the path if it does not exist
    if not os.path.exists(config['mrc_path']):
        os.makedirs(config['mrc_path'])
    
    # Perform the mrc file creation
    origin, shape = _mrc_nfunc(cellID, cte.h5_name, config)
    
    return origin, shape

mrc_required_keys = {
    'resolution': {'type': float, 'positive': True},
    'border': {'type': int, 'positive': True},
    'mrc_path': {'type': str}
}

def _mrc_nfunc(cellID: str, cte_name: str, config: dict) -> dict:
    """ Node function to save the cell MRC file.
    Saves the MRC file for the cell and returns the origin and shape of the file.

    Args:
        cellID (str)
        cte_name (str)
        config (dict): configuration dictionary for the mrc file creation

    Returns:
        cell_mrc_params (dict): dictionary with the origin and shape of the cell MRC file
                                cell_mrc_params['origin']: tuple, origin of the cell MRC file in voxel units
                                cell_mrc_params['shape']: tuple, shape of the cell MRC file in voxel units
    """
    
    # Open the ChromatinTracingExperiment and get the alphashape for the cell
    cte = ChromatinTracingExperiment(cte_name, 'r')
    alphashape = cte.get_alphashapes(cellID)
    cte.close()
    
    # If config has a key 'ndilation', read it and check that it is a positive integer
    if 'ndilation' in config:
        if not isinstance(config['ndilation'], int):
            raise TypeError("ndilation must be an integer.")
        if config['ndilation'] < 1:
            raise ValueError("ndilation must be a positive integer.")
        ndilation = config['ndilation']
    # Otherwise, set ndilation to None
    else:
        ndilation = None
    
    # Save the mrc file for the cell and return the origin and shape of the file
    origin, shape = utils.mesh_to_mrc(
        path = config['mrc_path'],
        name_prefix = cellID,
        mesh = alphashape['mesh'],
        resolution = config['resolution'],
        border = config['border'],
        ndilation=ndilation
    )
    
    cell_mrc_params = {'origin': origin, 'shape': shape}
    
    return cell_mrc_params


# PYPLOT functions

def save_cell_pyplot(cte: ChromatinTracingExperiment, cellID: str, path: str, filename: str = None) -> None:
    """ Save a 3D plot (using matplotlib) of the cell.

    Args:
        cte (ChromatinTracingExperiment)
        cellID (str)
        path (str): directory where the plot will be saved
        filename (str, optional): name of the file to save the plot. If None, the file is named 'cellID.png'
    """
    
    # Check that the path exists. If not, create it.
    if not isinstance(path, str):
        raise TypeError("path must be a string.")
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Get the data for the cell in numpy format
    xs, ys, zs, chroms, _, _, _, _, _ = cte.get_data(cellID, format='numpy')
    
    # Create the 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    # Plot the data, coloring by chromosome
    for chrom in np.unique(chroms):
        idx = np.where(chroms == chrom)
        ax.scatter(xs[idx], ys[idx], zs[idx], s=3)
    # Remove the axes labels, ticks, and grid
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    
    # If filename is None, save the plot as 'cellID.png'
    if filename is None:
        filename = cellID
    # Remove '.png' from filename if present
    if filename.endswith('.png'):
        filename = filename[:-4]
        
    # Save figure in 3 different angles: parallel to xy, parallel to xz and parallel to yz
    ax.view_init(0, 0)
    plt.savefig(os.path.join(path, filename + '_xy.png'))
    ax.view_init(90, 0)
    plt.savefig(os.path.join(path, filename + '_xz.png'))
    ax.view_init(0, 90)
    plt.savefig(os.path.join(path, filename + '_yz.png'))
    
    plt.close(fig)    

def save_all_pyplots(cte: ChromatinTracingExperiment, path: str) -> None:
    """ Save 3D plots for all cells in the experiment.
    
    Args:
        cte (ChromatinTracingExperiment)
        path (str): directory where the plots will be saved
    """
    
    for cellID in cte.cell_labels:
        save_cell_pyplot(cte, cellID, path)

def plot_chrom_alphashape(cell_data: dict, cell_mesh: trimesh.Trimesh, cellID: str, chrom: str, alpha: float, force: bool = False) -> tuple:
    """ Plot the mesh of a cell and the alphashapes of the chromosomal copies.

    Args:
        cell_data (dict): data of the cell in dictionary format
        cell_mesh (trimesh.Trimesh): mesh of the cell
        cellID (str)
        chrom (str)
        alpha (float): alpha parameter for the alphashape to be fitted for each chromosomal copy
        force (bool, optional): if False, the alpha parameter is going to be changed until the alphashape is closed. Default is False.

    Returns:
        fig (matplotlib.figure.Figure): figure object
        ax (matplotlib.axes._subplots.Axes3DSubplot): axes object
    """

    # Initialize the figure
    figsize = (8, 8)
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the mesh of the cell
    ax.plot_trisurf(*zip(*cell_mesh.vertices), triangles=cell_mesh.faces, color='yellow', alpha=0.5)
    
    # Loop over the copies of the chromosome
    for traceID in cell_data[chrom]:
        
        # Get the data of the chromosomal copy and fit an alphashape
        xs, ys, zs, _, _, _, _, _ = cte_utils.trace_dict_to_numpy(cell_data[chrom][traceID])
        points = np.array([xs, ys, zs]).T
        alpha, mesh = utils.fit_alphashape(points, alpha, force)
        print('Alpha: {}'.format(alpha))
        
        # Plot the alphashape
        ax.plot_trisurf(*zip(*mesh.vertices), triangles=mesh.faces, color='red', alpha=0.8)
        
        # Plot the points
        ax.scatter(xs, ys, zs, color='red', s=0.8)
    
    return fig, ax


# BODY MRC functions

def run_body_mrc(cte: ChromatinTracingExperiment, scf: SingleCellFeature, config: dict) -> None:
    """ Creates the mrc files for nuclear bodies in each cell of the experiment.
    
    Required keys in config:
        - 'resolution': float, positive
        - 'border': int, positive
        - 'mrc_path': str
        - 'kde_alpha': float, positive
        - 'bodies_feats': dict
    
    The 'bodies_feats' specifies the nuclear bodies and their features. For examples:
        {
            'Nucleoli': {
                'features': ['Fibrillarin', 'rDNA', 'Rnu3b_RNA', 'ITS1_RNA'],
                'threshold': 99.5,
                'ndilation': 2,
            },
            'Centromeres-Telomeres': {
                'features': ['MajSat', 'MinSat', 'Telomere'],
                'threshold': None
            }
        }
    'threshold' (which is a percentile) and 'ndilation' are optional, and if provided they are used
    to create a binary MRC file.
    
    Args:
        cte (ChromatinTracingExperiment)
        scf (SingleCellFeature)
        config (dict)
    """
    
    def _rfunc_init(_1, _2, _3, _4) -> dict:
        """ Initialize the body mrc parameters dictionary for the reduce function.

        Args:
            _*: not used, just to match the signature of the function

        Returns:
            body_mrc_params (dict): empty dictionary
        """
        return {}

    def _rfunc_update(cellID: str, body_mrc_params: dict, cell_body_mrc_params: dict, _1, _2, _3) -> dict:
        """ Update the mrc parameters dictionary for the reduce function.

        Args:
            cellID (str)
            body_mrc_params (dict): mrc parameters dictionary for the bodies of each cell
            cell_body_mrc_params (dict): mrc parameters dictionary for the bodies of the cell
            _*: not used, just to match the signature of the function

        Returns:
            body_mrc_params (dict): updated mrc parameters dictionary for the bodies of each cell
        """
        body_mrc_params[cellID] = cell_body_mrc_params
        return body_mrc_params
    
    # If the CTE doesn't have alphashapes, raise an error
    if 'alphashapes' not in cte:
        raise KeyError("Alphashapes not present in the ChromatinTracingExperiment.")
    
    # Check that the path is present in config
    if 'mrc_path' not in config:
        raise KeyError("mrc_path not present in config.")
    # Check that the path is a valid path-like string
    if not isinstance(config['mrc_path'], str):
        raise TypeError("mrc_path must be a string.")
    # Transform the path to an absolute path
    config['mrc_path'] = os.path.abspath(config['mrc_path'])
    # Create the path if it does not exist
    if not os.path.exists(config['mrc_path']):
        os.makedirs(config['mrc_path'])
    
    # Run the bodies MRC calculation in parallel
    # The MRC files are saved in the folder specified in config,
    # and here we return the origin and shape of each cell
    mrc_params = parallel.control_func(
        cte,
        scf,
        config,
        body_mrc_required_keys,
        _body_mrc_nfunc,
        _rfunc_init,
        _rfunc_update
    )
    
    # Save the mrc parameters as a pickle file in the folder specified in config
    out_filename = os.path.join(config['mrc_path'], 'mrc_params.pickle')
    with open(out_filename, 'wb') as f:
        pickle.dump(mrc_params, f)

body_mrc_required_keys = {
    'resolution': {'type': float, 'positive': True},
    'border': {'type': int, 'positive': True},
    'mrc_path': {'type': str},
    'kde_alpha': {'type': float, 'positive': True},
    'bodies_feats': {'type': dict}
}

def _body_mrc_nfunc(cellID: str, cte_name: str, scf_name: str, config: dict) -> dict:
    """ Node function to save the body MRC files for a single cell.
    
    - Creates a KDE density for the cell spots, inverting it so that the "missing volume" is the density.
    - For each nuclear body, creates a KDE using only spots with a high feature value.
    - For each nuclear body, combines the previous two KDEs: uses the "missing volume" KDE, but only
      when the KDE of the body is higher than the KDE of the other bodies.
    - If a threshold is provided, creates a binary MRC file using the threshold.

    Args:
        cellID (str)
        cte_name (str)
        scf_name (str)
        config (dict)

    Returns:
        dict: dictionary with the origin and shape of the body MRC files
    """

    # Open the CTE and SCF objects and the relevant data
    cte = ChromatinTracingExperiment(cte_name, 'r')
    scf = SingleCellFeature(scf_name, 'r')
    traceID_hash = cte.get_trace_hashmap(cellID)
    mesh = cte.get_alphashapes(cellID)['mesh']

    # Get the coordinates of the spots of the cell
    xs, ys, zs, chroms, starts, ends, _, traceIDs, _ = cte.get_data(cellID, format='numpy')
    crd = np.array([xs, ys, zs]).T

    # Calculate the Gaussian Kernel Density Estimate
    kde = gaussian_kde(crd.T, bw_method=config['kde_alpha'])

    # Get the bounding box of the mesh
    bbox = mesh.bounding_box.bounds  # np.array of shape (2, 3)
    # Quantize the bounding box by the resolution
    bbox = config['resolution'] * np.round(bbox / config['resolution'])
    # Add the border (multiplied by the resolution) to the bounding box
    bbox[0] -= config['border'] * config['resolution']
    bbox[1] += config['border'] * config['resolution']
    # Calculate the origin of the MRC in voxel units
    origin_mrc_vx = np.round(bbox[0] / config['resolution']).astype(int)

    # Create 3D grid
    XYZ, shape = utils.create_grid(bbox, config['resolution'])

    # Calculate the gaussian KDE on the 3D grid
    k_3d = kde(XYZ.T).reshape(shape)
    # Invert the values of the KDE to get the "missing volume" density
    k_3d = np.max(k_3d) - k_3d

    # Calculate the distance of each point to the mesh
    surface_dists = trimesh.proximity.signed_distance(mesh, XYZ).reshape(shape)
    # Where the absolute distance is less than a threshold, set the value to 0
    threshold = 0.75  # 750 nm
    k_3d[surface_dists < threshold] = 0

    # Create new KDE densities, only considering the spots with a feature value above a threshold
    k_3d_bodies = {}
    for body in config['bodies_feats']:
        
        # Initialize the list of feature values, to be converted later to a numpy array
        bodyvals = []
        
        # Loop over the features of the body
        for feature in config['bodies_feats'][body]['features']:
            
            # Get the feature values for the marker, shape (n_spots,)
            featvals = get_feature_for_pdb(cellID, scf, feature, traceID_hash, traceIDs, chroms, starts, ends)
            bodyvals.append(featvals)
        
        # Convert the list to array of shape (n_features, n_spots)
        bodyvals = np.array(bodyvals)
        # Take the maximum value among the features for each spot
        bodyvals = np.nanmax(bodyvals, axis=0)  # shape (n_spots,)

        # Select the spots with a feature value above a X percentile
        percentile = 80
        threshold = np.nanpercentile(featvals, percentile)
        xs_topfeat = xs[featvals > threshold]
        ys_topfeat = ys[featvals > threshold]
        zs_topfeat = zs[featvals > threshold]
        crd_topfeat = np.array([xs_topfeat, ys_topfeat, zs_topfeat]).T

        # Calculate the Gaussian Kernel Density Estimate from the selected spots
        kde = gaussian_kde(crd_topfeat.T, bw_method=config['kde_alpha'])
        
        # Calculate the KDE on the 3D grid
        k_3d_bodies[body] = kde(XYZ.T).reshape(shape)

    # Write the MRC files for the body, selecting for each voxel the body with the highest value
    for body in config['bodies_feats']:
        k_3d_ = np.copy(k_3d)
        for body2 in config['bodies_feats']:
            if body == body2:
                continue
            k_3d_[k_3d_bodies[body] < k_3d_bodies[body2]] = 0
        utils.write_mrc(
            filename = os.path.join(config['mrc_path'], f'{cellID}_{body}_KDE.mrc'),
            data = k_3d_,
            origin = tuple(origin_mrc_vx),
            voxel_size = (config['resolution'], config['resolution'], config['resolution']),
        )
        # If a threshold is not provided, skip the binary MRC
        if config['bodies_feats'][body]['threshold'] is None:
            continue
        # Otherwise, create a binary MRC using the threshold
        threshold = np.nanpercentile(k_3d_, config['bodies_feats'][body]['threshold'])
        k_3d_b_ = (k_3d_ >= threshold).astype(int)
        # Perform erosion and dilation to remove small objects and fill holes
        k_3d_b_ = binary_erosion(k_3d_b_, iterations=1)
        k_3d_b_ = binary_dilation(k_3d_b_, iterations=config['bodies_feats'][body]['ndilation'])
        # Save the binary MRC
        utils.write_mrc(
            filename = os.path.join(config['mrc_path'], f'{cellID}_{body}_KDE_binary.mrc'),
            data = k_3d_b_,
            origin = tuple(origin_mrc_vx),
            voxel_size = (config['resolution'], config['resolution'], config['resolution']),
        )
    
    return {'origin': origin_mrc_vx, 'shape': shape}

def run_body_mrc_single_cell(cte: ChromatinTracingExperiment, scf: SingleCellFeature, cellID: str, config: dict) -> tuple:
    """ Performs the body mrc file creation task on a single cell.

    Args:
        cte (ChromatinTracingExperiment)
        scf (SingleCellFeature)
        cellID (str)
        config (dict)

    Returns:
        origin (tuple): origin of the volume mrc files in voxel units
        shape (tuple): shape of the volume mrc files in voxel units
    """

    # Check that all required keys are present in config
    parallel.check_config(config, body_mrc_required_keys, parallel=False)
    
    # Transform the path to an absolute path
    config['mrc_path'] = os.path.abspath(config['mrc_path'])
    # Create the path if it does not exist
    if not os.path.exists(config['mrc_path']):
        os.makedirs(config['mrc_path'])
    
    # Perform the mrc file creation
    origin, shape = _body_mrc_nfunc(cellID, cte.h5_name, scf.h5_name, config)
    
    return origin, shape
