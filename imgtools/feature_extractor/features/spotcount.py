import numpy as np
from ...cte import ChromatinTracingExperiment

docstring = """Counts the number of spots per domain, seperately for each trace. Missing data are considered as 0."""

required_keys = {
    'indexing': {'type': str, 'optional': True}
}

def run(cellID: str, cte: ChromatinTracingExperiment, config: dict, feat_arr: np.ndarray, _2) -> np.ndarray:
    """ Counts the number of spots per domain, seperately for each trace.
    
    There are two indexing methods available:
    - 'by_domain': uses the domain index to map spots to their respective domains.
    - 'by_gene': uses the gene index to map spots to their respective genes (for intron-RNA data).

    Args:
        cellID (str)
        cte (ChromatinTracingExperiment)
        config (dict): configuration dictionary with the following keys:
            - indexing (str): the indexing method for the Index: ['by_domain' or 'by_gene']. Default is 'by_domain'.
        feat_arr (np.ndarray): feature array of shape (n_domains, n_traces), to be updated with the number of spots
        _*: not used, just to match the function signature
    
    Returns:
        np.ndarray: Updated array of shape (n_domains, n_traces) with the number of spots
    """
    
    # Get the indexing method from the config, default is 'by_domain'
    indexing = config.get('indexing', 'by_domain')
    # Check if the indexing method is valid
    if not indexing in ['by_domain', 'by_gene']:
        raise ValueError(f"Invalid indexing method: {indexing}. Must be one of ['by_domain', 'by_gene'].")
    
    # Get the cell data in dictionary format
    cell_data = cte.get_data(cellID)
    
    # Get the traceID hash table to map traces to their position in the array
    traceID_hash = cte.get_trace_hashmap(cellID)
    
    # Convert the feat_arr to an array of 0s
    feat_arr = np.zeros(feat_arr.shape, dtype=feat_arr.dtype)
    
    # Hash the index, either by domain or by gene
    index = cte.index
    # By domain
    if indexing == 'by_domain':
        index_hash = index.get_index_hashmap()
    # By gene
    else:
        try:
            gene_labels = index.gene_labels
        except Exception as e:
            raise ValueError(f"Error accessing gene labels in index: {e}")
        index_hash = {}
        for i, geneID in enumerate(gene_labels):
            index_hash[geneID] = i
    
    for chrom in cell_data:        
        for traceID in cell_data[chrom]:
            
            # Get the position of the trace in the array using the hash tables
            i_trace = traceID_hash[chrom][traceID]
            
            for spotID in cell_data[chrom][traceID]:
                
                # Unpack the spot data
                spot_data = cell_data[chrom][traceID][spotID]
                start, end = spot_data['start'], spot_data['end']
                geneID = spot_data.get('geneID', None)  # if geneID is not present, it will be None
                
                # Get the position of the spot in the index with the specified indexing method
                if indexing == 'by_domain':
                    i_domain = index_hash[(chrom, start, end)]
                    assert len(i_domain) == 1, f"Error: multiple domains found for {chrom}, {start}, {end}"
                    i_domain = i_domain[0]
                else:
                    if geneID is None:
                        raise ValueError(f"Gene ID is None for spot {spotID} in trace {traceID} of chromosome {chrom}.")
                    i_domain = index_hash[geneID]
                
                # Increment the count
                feat_arr[i_domain, i_trace] += 1
    
    
    return feat_arr
