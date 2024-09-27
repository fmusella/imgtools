import os
import sys
import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as plt_colors
from matplotlib import cm
import trimesh
from alabtools.utils import get_index_from_bed
from alabtools.plots import write_pdb
from .cte import ChromatinTracingExperiment
from .scf import SingleCellFeature
from .cte import cte_utils
from .cte import cte_parallel
from .cte.metrics import get_trace_ranks_for_cell
from . import utils


# PDB functions

def save_cell_pdb(
    path: str,
    cellID: str,
    cte: ChromatinTracingExperiment,
    scf: SingleCellFeature = None,
    feature: str = None,
    bedfile: str = None
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
        bedfile (str or None): path to a BED file with the labels of each domain. Optional
    """
    
    # Check that the path exists. If not, create it.
    if not isinstance(path, str):
        raise TypeError("path must be a string.")
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Get data for cell in numpy array format
    xs, ys, zs, chroms, starts, ends, lums, traceIDs, _ = cte.get_data(cellID, format='numpy')
    
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
        featvals = get_feature_for_pdb(cellID, scf, feature, traceID_hash, traceIDs, chroms, starts, ends)
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
    
    # Clip featvals to 5% and 95% percentiles to remove outliers
    featvals = np.clip(featvals, np.percentile(featvals, 5), np.percentile(featvals, 95))
    # If the feature values are constant (min == max), set them to 0
    if np.min(featvals) == np.max(featvals):
        featvals = np.zeros(featvals.shape)
    # Otherwise, min-max normalize to [0, 999]
    else:
        featvals = (featvals - np.min(featvals)) / (np.max(featvals) - np.min(featvals)) * 999
    # Truncate to 2 decimal places
    featvals = np.round(featvals, 2)
    
    # Convert starts to units in bp such that the maximum values has 3 digits above the decimal point (i.e. < 1000)
    while np.max(starts) >= 1000:
        starts = starts / 10
    # Truncate to 2 decimal places
    starts = np.round(starts, 2)
    
    # Write dictionary for pdb file
    celldata_for_pdb = {
        'x': xs,
        'y': ys,
        'z': zs,
        'residue_name': chromnums,
        'chain_id': tracenums,
        'occupancy': starts,
        'beta': featvals,
        'element_symbol': featsnan,
        'atom_name': labels,
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

    Returns:
        featvals (np.ndarray): Array of feature values for the spots, ordered as the spots in the CTE
    """
    
    # Get the feature matrix
    feature_mat = scf.get_feature(feature, cellID)
    
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
    path: str, radius: float, do_link: bool = True
) -> None:
    """ Write a cmm file for a cell.
    
    Each trace is written in a separate cmm file.

    Args:
        cte (ChromatinTracingExperiment)
        cellID (str)
        path (str): directory where the cmm files will be saved
        radius (float): size of the markers (in physical units)
        do_link (bool, optional): if True, links are drawn between consecutive markers. Default is True.
    """
    
    # Check that the path exists. If not, create it.
    if not isinstance(path, str):
        raise TypeError("path must be a string.")
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Get the data for the cell in dictionary format
    cell_data = cte.get_data(cellID)
    
    # Map each chromosome to a different color from the tab20 colormap
    tab20 = np.array(cm.get_cmap('tab20').colors)
    chrom2color = {chrom: tab20[i % 20] for i, chrom in enumerate(cell_data.keys())}
    
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
                filename = os.path.join(path, '{}_{}_{}.cmm'.format(cellID, chrom, traceID)),
                marker_str = 'cellID: {}, chrom: {}, traceID: {}'.format(cellID, chrom, traceID),
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
    pmin: float = None, pmax: float = None
) -> None:
    
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
        cmap = cm.get_cmap('seismic')
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
            filename = os.path.join(path, '{}_{}.cmm'.format(cellID, label))
            marker_str = 'cellID: {}, label: {}'.format(cellID, label)
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
    
    # Save the mrc file for the cell and return the origin and shape of the file
    origin, shape = utils.mesh_to_mrc(
        path = config['mrc_path'],
        name_prefix = cellID,
        mesh = alphashape['mesh'],
        resolution = config['resolution'],
        border = config['border']
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
