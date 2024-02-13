import numpy as np
import trimesh
from alabtools.utils import Index

required_keys = {
    'cutoff': {'type': float, 'positive': True}
}

def run(cell_arr: np.ndarray, cell_data: dict, cell_alphashape: dict, index: Index, config: dict):
    
    # Initialize the cell association array
    cell_association_arr = np.copy(cell_arr)
    
    # Create a counter array of same shape as cell_arr to store the number of spots per domain (for averaging)
    count_arr = np.zeros(cell_arr.shape, dtype=int)
    
    # Create a hash table for the index
    index_hash = index.get_index_hashmap()
    
    for chrom in cell_data:
            
        # Get the traces in the chromosome and hash them
        traceIDs = list(cell_data[chrom].keys())
        traceID_hash = {traceID: i for i, traceID in enumerate(traceIDs)}
        
        for traceID in cell_data[chrom]:
            
            for spotID in cell_data[chrom][traceID]:
                
                # Unpack the spot data
                spot_data = cell_data[chrom][traceID][spotID]
                x, y, z = spot_data['x'], spot_data['y'], spot_data['z']
                start, end = spot_data['start'], spot_data['end']
                
                # Calculate the distance to the nuclear lamina
                point = np.array([[x, y, z]])
                dist = np.abs(trimesh.proximity.signed_distance(cell_alphashape['mesh'], point)[0])
                
                # Get the position of the spot in the array using the hash tables
                i_domain = index_hash[(chrom, start, end)]
                assert len(i_domain) == 1, f"Error: multiple domains found for {chrom}, {start}, {end}"
                i_domain = i_domain[0]
                i_trace = traceID_hash[traceID]
                
                # Increment the cell array
                cell_arr[i_domain, i_trace] += dist
                count_arr[i_domain, i_trace] += 1
                
                # Increment the cell association array
                if dist <= config['cutoff']:
                    cell_association_arr[i_domain, i_trace] += 1
                    
    
    # Average the distances
    cell_arr = cell_arr / count_arr
    # Set to NaN the values where there are no spots
    cell_arr[count_arr == 0] = np.nan
    cell_association_arr[count_arr == 0] = np.nan
    
    return cell_arr, cell_association_arr
