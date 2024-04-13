import os
import sys
import numpy as np
from alabtools.plots import write_pdb
from .cte import ChromatinTracingExperiment
from .cte.metrics import get_trace_ranks_for_cell
from .scf import SingleCellFeature

def save_cell_pdb(
    path: str,
    cellID: str,
    cte: ChromatinTracingExperiment,
    scf: SingleCellFeature = None,
    feature: str = None
) -> None:
    """ Save a PDB file for a cell.
    The PDB file will contain the 3D coordinates of the spots in the cell, with the following columns:
    - x: x-coordinate of the spot
    - y: y-coordinate of the spot
    - z: z-coordinate of the spot
    - atom_name: 'nan' if the feature value is NaN, 'ok' otherwise
    - residue_name: chromosome number
    - chain_id: trace number
    - occupancy: start position of the spot in bp
    - beta: feature value (luminescence or other feature)
    
    If a SingleCellFeature object is provided, the feature values will be used as the beta factor.
    Otherwise, the luminescence values will be used.

    Args:
        path (str): folder to save the pdb file
        cellID (str)
        cte (ChromatinTracingExperiment)
        scf (SingleCellFeature or None)
        feature (str or None)
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
    chromnums = np.array(chromnums).astype('U20')

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
    tracenums = np.array(tracenums).astype('U20')
    
    # If a feature is provided, use it as the beta factor
    if scf is not None and feature is not None:
        traceID_hash = cte.get_trace_hashmap(cellID)
        featvals = get_feature_for_pdb(cellID, scf, feature, traceID_hash, traceIDs, chroms, starts, ends)
    # Otherwise, use the luminescence as the beta factor
    else:
        featvals = lums
    
    # Create a 1-string-valued array that is 'N' where the feature value is NaN, and 'D' where it is not
    featsnan = np.where(np.isnan(featvals), 'nan', 'ok')
    # Replace the NaNs with the minimum value of the feature
    featvals[np.isnan(featvals)] = np.nanmin(featvals)
    
    # Clip featvals to 5% and 95% percentiles to remove outliers
    featvals = np.clip(featvals, np.percentile(featvals, 5), np.percentile(featvals, 95))
    # Min-max normalize lums to [0, 999]
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
        'atom_name': featsnan,
        'residue_name': chromnums,
        'chain_id': tracenums,
        'occupancy': starts,
        'beta': featvals
    }
    
    # Write pdb file
    if feature is None:
        filename = os.path.join(path, f"{cellID}.pdb")
    else:
        filename = os.path.join(path, f"{cellID}_{feature}.pdb")
    
    write_pdb(filename, celldata_for_pdb)

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

def save_cell_pdbs(
    cellID: str,
    cte: ChromatinTracingExperiment,
    scf: SingleCellFeature,
    path: str
) -> None:
    """ Save the PDB files for each feature in a cell.

    Args:
        cellID (str)
        cte (ChromatinTracingExperiment)
        scf (SingleCellFeature)
        path (str): path to save the pdb files
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
    
    sys.stdout.write(f"Features:\n")
    for feature in features:
        sys.stdout.write(f"  - {feature}\n")
    
    # Save the pdb files for each feature
    for feature in features:
        
        sys.stdout.write(f"     ...saving feature {feature}...\n")
        save_cell_pdb(cell_path, cellID, cte, scf, feature)
    
    sys.stdout.write(f"Done.\n")
