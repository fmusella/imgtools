import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as plt_colors
from matplotlib import cm
import trimesh
import mrcfile
from alabtools.plots import write_pdb
from .cte import ChromatinTracingExperiment, cte_utils
from .cte.metrics import get_trace_ranks_for_cell
from .scf import SingleCellFeature
from . import utils


# PDB functions

def save_cell_pdb(
    path: str,
    cellID: str,
    cte: ChromatinTracingExperiment,
    scf: SingleCellFeature = None,
    feature: str = None,
    feature_nquants: int = None,
    adjust_featvals: bool = True,
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
        adjust_featvals (bool): if True, the feature values are adjusted for better visualization. Default is True.
        bedfile (str or None): path to a BED file with the labels of each domain. Optional
        exclude_imputed (bool): if True, imputed spots are excluded from the PDB file. Default is False.
    """
    
    # Check that the path exists. If not, create it.
    if not isinstance(path, str):
        raise TypeError("path must be a string.")
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Get data for cell in numpy array format
    d = cte.get_data(cellID, format='numpy')
    xs, ys, zs, chroms, starts, lums, traceIDs, spotIDs = d['xs'], d['ys'], d['zs'], d['chroms'], d['starts'], d['lums'], d['traceIDs'], d['spotIDs']
    if 'geneIDs' in d:
        geneIDs = d['geneIDs']
    else:
        geneIDs = None
    
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
    
    # If a BED file is provided, get the labels as (must be <= 4 characters long)
    if bedfile is not None:
        labels = cte.get_bed_values_by_spotIDs(cellID, bedfile).astype('U4')
    # Or, if the geneIDs are provided, use them as labels
    elif geneIDs is not None:
        labels = geneIDs.astype('U4')  # Convert to Unicode string of length 4
    else:
        labels = np.full(len(xs), '', dtype='U4')
    
    # If a feature is provided, use it as the beta factor
    if scf is not None and feature is not None:
        # Get the feature values for the spots
        featvals = scf.get_feature_by_spotIDs(cellID, cte, feature, feature_nquants).astype(float)
        if feature_nquants is not None:
            featvals[featvals == -1] = np.nan
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
    
    # If the feature is not quantized, adjust the values for better visualization if adjust_featvals is True
    if feature_nquants is None and adjust_featvals:
        # If all values are the same, set the feature values to 0
        if np.all(featvals == featvals[0]):
            featvals = np.zeros(featvals.shape)
        # Otherwise, clip the values to the 5th and 95th percentiles and normalize them from 0 to 999
        # (This is because the beta factor in PDB files is a float between 0 and 999)
        else:
            featvals = np.clip(featvals, np.percentile(featvals, 5), np.percentile(featvals, 95))
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

def write_cmm(
    filename: str, marker_str: str, coord: np.ndarray, radius: float,
    color: np.ndarray = np.array([0, 0, 0]), links: np.ndarray = None
) -> None:
    """ Write a CMM file.
    
    Only works for a single marker set. Colors all markers and links with the same color.

    Args:
        filename (str): name of the file to be written
        marker_str (str): string to identify the marker set
        coord (np.ndarray): numpy array of shape (n_markers, 3)
                containing the coordinates of the markers
        radius (float): size of the markers (in physical units)
        color (np.ndarray, optional): numpy array of shape with the colors of markers and links.
                Can be either (3,) or (n_markers, 3). Defaults to [0, 0, 0]
        links (np.ndarray, optional): numpy array of shape (n-1,),
                True if there is a link between i and i+1. Defaults to None (no links)
    """

    with open(filename,'w') as f:
        
        if color.shape == (3,):
            color = np.tile(color, (len(coord), 1))
        
        f.write('<marker_set name="marker set %s">\n' % marker_str)
        
        # Write markers
        for i in range(len(coord)):
            f.write(
                '<marker id="%d" x="%.3f" y="%.3f" z="%.3f" r="%.3f" g="%.3f" b="%.3f" radius="%.3f" note="" nr="%.3f" ng="%.3f" nb="%.3f"/>\n'
                    % (i + 1, coord[i, 0], coord[i, 1], coord[i, 2],
                       color[i, 0], color[i, 1], color[i, 2],
                       radius, color[i, 0], color[i, 1], color[i, 2])
            )
        
        if links is None:
            f.write('</marker_set>\n')
            return None
        
        # Write links
        for i in range(len(coord) - 1):
            # Skip if there is no link between i and i+1
            if not links[i]:
                continue
            # Otherwise, write the link
            f.write(
                '<link id1="%d" id2="%d" r="%.3f" g="%.3f" b="%.3f" radius="%.3f" />\n'
                    % (i + 1, i + 2, color[i, 0], color[i, 1], color[i, 2], radius / 4)
            )
        
        f.write('</marker_set>\n')

def save_cell_cmm_byfeatquant(
    cellID: str, cte: ChromatinTracingExperiment, scf: SingleCellFeature,
    feature: str, nquants: int, path: str, radius: float,
    colormap: str = 'seismic', exclude_imputed: bool = True, flip: bool = False
) -> None:
    
    # Check that the path exists. If not, create it.
    if not isinstance(path, str):
        raise TypeError("path must be a string.")
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Get the data for the cell in numpy format
    d = cte.get_data(cellID, format='numpy')
    xs, ys, zs, spotIDs = d['xs'], d['ys'], d['zs'], d['spotIDs']
    
    # Get the feature values for the spots
    featvals = scf.get_feature_by_spotIDs(cellID, cte, feature, nquants)
    
    # Flip the feature values if flip is True
    if flip:
        featvals_ = np.copy(featvals)
        featvals = nquants - 1 - featvals_
        featvals[featvals == -1] = -1
    
    # If exclude_imputed is True, create a mask to exclude imputed spots
    mask_spots = np.ones(len(spotIDs), dtype=bool)
    if exclude_imputed:
        for i, spotID in enumerate(spotIDs):
            if 'IMPUTED' in spotID:
                mask_spots[i] = False
    xs = xs[mask_spots]
    ys = ys[mask_spots]
    zs = zs[mask_spots]
    featvals = featvals[mask_spots]
    
    # Create colors by interpolating the colormap to the feature values
    # The -1 values are excluded from the color mapping and set to black
    fmin = np.min(featvals[featvals >= 0])
    fmax = np.max(featvals[featvals >= 0])
    norm = plt_colors.Normalize(vmin=fmin, vmax=fmax)
    cmap = cm.get_cmap(colormap)
    colors = cmap(norm(featvals))[:, :3]
    colors[featvals == -1] = [0, 0, 0]
    
    # Create a CMM file for each quantile
    for q in np.unique(featvals):
            
        # Get the indices of the quantile
        idx = np.where(featvals == q)
        
        # Write filename and marker string
        filename = os.path.join(path, f'{cellID}_{feature}_q={q}.cmm')
        marker_str = f'cellID: {cellID}, feature: {feature}, quantile: {q}'
        
        # Write the CMM file
        write_cmm(
            filename = filename,
            marker_str = marker_str,
            coord = np.array([xs[idx], ys[idx], zs[idx]]).T,
            radius = radius,
            color = colors[idx],
        )

def save_cell_cmm_bychrom(
    cte: ChromatinTracingExperiment, cellID: str,
    path: str, radius: float, do_link: bool = True,
    color_by: str = 'chromosome', colormap: str = 'tab20',
    exclude_imputed: bool = True
) -> None:
    """ Write a cmm file for a cell.
    Each chrom / trace is written in a separate cmm file.
    Markers can be colored either by simple chromosome ID, or by genomic position within the chromosome.

    Args:
        cte (ChromatinTracingExperiment)
        cellID (str)
        path (str): directory where the cmm files will be saved
        radius (float): size of the markers (in physical units)
        do_link (bool, optional): if True, links are drawn between consecutive markers. Default is True.
        color_by (str, optional): how to color the markers.
            Available options: 'chromosome', 'genomic_start'.
                - 'chromosome': each chromosome is colored with a different color from the colormap,
                - 'genomic_start': each marker is colored by its genomic start position.
        colormap (str, optional): name of the colormap to use. Default is 'tab20'.
        exclude_imputed (bool, optional): if True, imputed spots are excluded from the CMM file. Defaults to True.
    """
    
    # Check that the path exists. If not, create it.
    if not isinstance(path, str):
        raise TypeError("path must be a string.")
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Check that the color_by parameter is valid
    available_color_by = ['chromosome', 'genomic_start']
    if color_by not in available_color_by:
        raise ValueError(f"color_by must be one of {available_color_by}. Got {color_by}.")
    
    # Get the data for the cell in dictionary format
    cell_data = cte.get_data(cellID)
    
    # If color_by is 'chromosome', map each chromosome to a different color from the colormap
    if color_by == 'chromosome':
        cmap = np.array(cm.get_cmap(colormap).colors)
        chrom2color = {chrom: cmap[i % 20] for i, chrom in enumerate(cell_data.keys())}
    # Otherwise, if color_by is 'genomic_start',
    # we need to get the genomic start positions of each chromosomes
    elif color_by == 'genomic_start':
        # Get the Genome from the CTE
        genome = cte.index.genome
        # Get the minimum and maximum genomic start positions for each chromosome
        chrom2start = {}
        for chrom, origin, length in zip(genome.chroms, genome.origins, genome.lengths):
            chrom2start[chrom] = (origin, origin + length)
    
    # Loop over chromosomes and traces, and write each trace to a separate cmm file
    for chrom in cell_data:
        for traceID in cell_data[chrom]:
            
            # Get the data for the trace
            d = cte_utils.trace_dict_to_numpy(cell_data[chrom][traceID])
            xs, ys, zs, starts, ends, spotIDs = d['xs'], d['ys'], d['zs'], d['starts'], d['ends'], d['spotIDs']
            
            # Exclude imputed spots if exclude_imputed is True
            if exclude_imputed:
                mask_spots = np.ones(len(spotIDs), dtype=bool)
                for i, spotID in enumerate(spotIDs):
                    if 'IMPUTED' in spotID:
                        mask_spots[i] = False
                xs = xs[mask_spots]
                ys = ys[mask_spots]
                zs = zs[mask_spots]
                starts = starts[mask_spots]
                ends = ends[mask_spots]
            
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
            
            # If color_by is 'chromosome', use the chrom2color mapping
            if color_by == 'chromosome':
                colors = chrom2color[chrom]
            # Otherwise, if color_by is 'genomic_start',
            # map the genomic start positions to colors using the colormap
            elif color_by == 'genomic_start':
                # Normalize the color values between the minimum and maximum genomic start positions
                # of this chromosome
                norm = plt_colors.Normalize(vmin=chrom2start[chrom][0], vmax=chrom2start[chrom][1])
                cmap = cm.get_cmap(colormap)
                colors = cmap(norm(starts))[:, :3]
            
            write_cmm(
                filename = os.path.join(path, f'{chrom}_{traceID}.cmm'),
                marker_str = f'cellID: {cellID}, chrom: {chrom}, traceID: {traceID}',
                coord = np.array([xs, ys, zs]).T,
                radius = radius,
                color = colors,
                links = links
            )

def save_cell_cmm_bybed(
    cte: ChromatinTracingExperiment,
    cellID: str, path: str, radius: float,
    bedfile: str,
    scf: SingleCellFeature = None, feature: str = None,
    pmin: float = None, pmax: float = None,
    colormap: str = 'Reds',
    exclude_imputed: bool = True
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
        exclude_imputed (bool, optional): if True, imputed spots are excluded from the CMM file. Defaults to True.
    """
    
    # Check that the path exists. If not, create it.
    if not isinstance(path, str):
        raise TypeError("path must be a string.")
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Get the data for the cell in dictionary format
    d = cte.get_data(cellID, format='numpy')
    xs, ys, zs, spotIDs = d['xs'], d['ys'], d['zs'], d['spotIDs']
    
    # Get the labels for the spots
    labels = cte.get_bed_values_by_spotIDs(cellID, bedfile).astype(str)
    if len(labels[0]) > 4:
        raise ValueError("BED labels should be <= 4 characters.")
    unique_labels = np.unique(labels)
    
    # If a SCF and feature are provided, get the feature values for the spots
    # and map them to the selected colormap
    if scf is not None and feature is not None:
        
        # Get the feature values for the spots
        featvals = scf.get_feature_by_spotIDs(cellID, cte, feature).astype(float)
        
        # If there are only two non-NaN unique values, e.g. 0 and 1,
        # do a binary color mapping: 0 --> light color, 1 --> dark color
        if len(np.unique(featvals[~np.isnan(featvals)])) == 2:
            colors = np.full((len(xs), 3), [0.75, 0.75, 0.75])  # light gray
            colors[featvals == 0] = [1, 1, 1]  # white
            colors[featvals == 1] = [1, 0, 0]  # red
        
        # Otherwise, do a continuous color mapping
        else:
            # Get the colormap for the feature values
            cmap = cm.get_cmap(colormap)
            # Interpolate the feature values to the colormap
            pmin = 5 if pmin is None else pmin
            pmax = 95 if pmax is None else pmax
            fmin = np.nanpercentile(featvals, pmin)
            fmax = np.nanpercentile(featvals, pmax)
            norm = plt_colors.Normalize(vmin=fmin, vmax=fmax)
            # Map each feature value to a color from the colormap
            colors = cmap(norm(featvals))[:, :3]
    # Otherwise, map each label to a different color from the tab20 colormap
    else:
        tab20 = np.array(cm.get_cmap('tab20').colors)
        label2color = {label: tab20[i % 20] for i, label in enumerate(unique_labels)}
        colors = np.array([label2color[label] for label in labels])
    
    # If exclude_imputed is True, create a mask to exclude imputed spots
    mask_spots = np.ones(len(spotIDs), dtype=bool)
    if exclude_imputed:
        for i, spotID in enumerate(spotIDs):
            if 'IMPUTED' in spotID:
                mask_spots[i] = False
    xs = xs[mask_spots]
    ys = ys[mask_spots]
    zs = zs[mask_spots]
    labels = labels[mask_spots]
    colors = colors[mask_spots]
    
    # Remove NaN feature values
    if scf is not None and feature is not None:
        mask_spots = ~np.isnan(featvals)
        xs = xs[mask_spots]
        ys = ys[mask_spots]
        zs = zs[mask_spots]
        labels = labels[mask_spots]
        colors = colors[mask_spots]
    
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
        write_cmm(
            filename = filename,
            marker_str = marker_str,
            coord = np.array([xs[idx], ys[idx], zs[idx]]).T,
            radius = radius,
            color = colors[idx],
        )


# MRC functions

def write_mrc(
    filename: str,
    data: np.ndarray,
    origin: tuple = (0, 0, 0),
    voxel_size: tuple = (1, 1, 1)
) -> None:
    """Write a MRC file from a numpy array.

    Args:
        filename (str): name of the file to be written.
        data (np.array(shape=(n_x_grid, n_y_grid, n_z_grid))): grid of values (0 or 1)
        origin (tuple, optional): origin of the MRC file in voxel units. Defaults to (0, 0, 0).
        voxel_size (tuple, optional): voxel size of the MRC file in physical units. Defaults to (1, 1, 1).
    """
    
    # Check that the parent directory exists
    if not os.path.exists(os.path.dirname(filename)):
        raise ValueError('The parent directory does not exist')

    # Convert the origin to a tuple
    try:
        origin = tuple(origin)
    except TypeError:
        raise ValueError(f'Origin must be castable to a tuple. Got {origin}')
    
    # Convert the voxel_size to a tuple if it's a number
    if isinstance(voxel_size, (int, float)):
        voxel_size = (voxel_size, voxel_size, voxel_size)
    
    # Swap the axes to match the MRC format
    data = np.swapaxes(data, 0, 2)
    # If the data is boolean or integer, convert it to int8
    if data.dtype == bool or np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.int8)
    # If the data is float, convert it to float32
    elif np.issubdtype(data.dtype, np.floating):
        data = data.astype(np.float32)
    else:
        raise ValueError('The data type is not supported')
    # Create a new MRC file and save the data
    with mrcfile.new(filename, overwrite=True) as mrc:
        mrc.set_data(data)
        # nstart contains the origin in voxel units,
        # but sometimes this is not enough to align the MRC
        # in softwares like ChimeraX
        mrc.nstart = origin
        # So we also set the origin in physical units
        mrc.header.origin = (
            origin[0] * voxel_size[0],
            origin[1] * voxel_size[1],
            origin[2] * voxel_size[2]
        )
        # Set the voxel size (i.e. resolution)
        mrc.voxel_size = voxel_size


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
    d = cte.get_data(cellID, format='numpy')
    xs, ys, zs, chroms = d['xs'], d['ys'], d['zs'], d['chroms']
    
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
        d = cte_utils.trace_dict_to_numpy(cell_data[chrom][traceID])
        xs, ys, zs = d['xs'], d['ys'], d['zs']
        points = np.array([xs, ys, zs]).T
        alpha, mesh = utils.fit_alphashape(points, alpha, force)
        print('Alpha: {}'.format(alpha))
        
        # Plot the alphashape
        ax.plot_trisurf(*zip(*mesh.vertices), triangles=mesh.faces, color='red', alpha=0.8)
        
        # Plot the points
        ax.scatter(xs, ys, zs, color='red', s=0.8)
    
    return fig, ax
