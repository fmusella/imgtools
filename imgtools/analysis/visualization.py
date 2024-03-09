import os
import numpy as np
from alabtools.plots import write_pdb
from ..cte import ChromatinTracingExperiment
from ..cte.metrics import get_trace_ranks_for_cell
from ..scf import SingleCellFeature

def save_cell_pdb_with_feature(
    cellID: str,
    feature: str,
    cte: ChromatinTracingExperiment,
    scf: SingleCellFeature,
    path: str
    ) -> None:
    """ Write a pdb file for a cell with a the feature values as beta factors.

    Args:
        cellID (str)
        feature (str)
        cte (ChromatinTracingExperiment)
        scf (SingleCellFeature)
        path (str): folder to save the pdb file
    """
    
    # If the path does not exist, create it
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Get the cell data
    cell_data = cte.get_data(cellID, format='dict')
    cell_feat_arr = scf.get_matrix(feature, cellID)
    
    # Create a hash table for the index
    index_hash = cte.index.get_index_hashmap()
    
    # Retrieve the data and store them in numpy format
    xs, ys, zs, chroms, starts, ends, traceIDs, featvals = [], [], [], [], [], [], [], []
    
    for chrom in cell_data:
            
        # Get the traces in the chromosome and hash them
        unique_chrom_traceIDs = list(cell_data[chrom].keys())
        traceID_hash = {traceID: i for i, traceID in enumerate(unique_chrom_traceIDs)}
        
        for traceID in cell_data[chrom]:

            for spotID in cell_data[chrom][traceID]:
                
                # Unpack the spot data
                spot_data = cell_data[chrom][traceID][spotID]
                x, y, z = spot_data['x'], spot_data['y'], spot_data['z']
                start, end = spot_data['start'], spot_data['end']
                
                # Get the position of the spot in the array using the hash tables
                i_domain = index_hash[(chrom, start, end)]
                assert len(i_domain) == 1, f"Multiple domains found for {chrom}:{start}-{end} in cell {cellID}."
                i_domain = i_domain[0]
                i_trace = traceID_hash[traceID]
                
                # Get the feature value
                featval = cell_feat_arr[i_domain, i_trace]
                
                # Append the data
                xs.append(x)
                ys.append(y)
                zs.append(z)
                chroms.append(chrom)
                starts.append(start)
                ends.append(end)
                traceIDs.append(traceID)
                featvals.append(featval)
    
    # Cast the data to numpy arrays
    xs = np.array(xs).astype(float)
    ys = np.array(ys).astype(float)
    zs = np.array(zs).astype(float)
    starts = np.array(starts).astype(int)
    ends = np.array(ends).astype(int)
    featvals = np.array(featvals).astype(float)
    
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
    
    # Convert starts to units in bp such that the maximum values has 3 digits above the decimal point (i.e. < 1000)
    while np.max(starts) >= 1000:
        starts = starts / 10
    # Truncate to 2 decimal places
    starts = np.round(starts, 2)
    
    # Min/max the feature values to 0/999
    featvals = (featvals - np.min(featvals)) / (np.max(featvals) - np.min(featvals)) * 999
    # Truncate to 2 decimal places
    featvals = np.round(featvals, 2)
    
    # Write dictionary for pdb file
    celldata_for_pdb = {
        'x': xs,
        'y': ys,
        'z': zs,
        'residue_name': chromnums,
        'chain_id': tracenums,
        'occupancy': starts,
        'beta': featvals
    }
    
    # Write pdb file
    filename = os.path.join(path, '{}_{}.pdb'.format(cellID, feature))
    write_pdb(filename, celldata_for_pdb)
