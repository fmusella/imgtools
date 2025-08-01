import numpy as np
from alabtools.utils import chromstr_to_chromint
from ..cte import ChromatinTracingExperiment

def get_geneIDs_from_cte(cte: ChromatinTracingExperiment) -> tuple:
    """ Extract the gene IDs, together with their chrom:start-end domain,
    from the CTE data.

    Args:
        cte (ChromatinTracingExperiment)

    Returns:
        gene_labels (np.ndarray): Array of gene IDs of shape (ngenes,)
        gene_chromstr (np.ndarray): Array of chromosome strings of shape (ngenes,)
        gene_start (np.ndarray): Array of start positions of shape (ngenes,)
        gene_end (np.ndarray): Array of end positions of shape (ngenes,)
    """
    
    # Initialize the genes dictionary
    genes = {}
    
    # Iterate over each cell ID
    for cellID in cte.cell_labels:
        # Get the cell data
        d = cte.get_data(cellID, format='numpy')
        chroms, starts, ends, geneIDs = d['chroms'], d['starts'], d['ends'], d['geneIDs']
        # Add the genes
        for chrom, start, end, geneID in zip(chroms, starts, ends, geneIDs):
            if geneID not in genes:
                genes[geneID] = []
            genes[geneID] = (chrom, start, end)
    
    # Convert the dictionary to arrays
    gene_labels = np.array([]).astype(str)
    gene_chromstr = np.array([]).astype(str)
    gene_start = np.array([]).astype(int)
    gene_end = np.array([]).astype(int)
    for geneID, (chrom, start, end) in genes.items():
        gene_labels = np.append(gene_labels, geneID)
        gene_chromstr = np.append(gene_chromstr, chrom)
        gene_start = np.append(gene_start, start)
        gene_end = np.append(gene_end, end)
    
    return gene_labels, gene_chromstr, gene_start, gene_end
        
def sort_geneIDs(
    gene_labels: np.ndarray, 
    chromstr: np.ndarray, 
    start: np.ndarray, 
    end: np.ndarray
) -> tuple:
    """ Sort the gene IDs (together with their chr:start-end domains)
    according to chromosomes, and within each chromosome, by start position.

    Args:
        gene_labels (np.ndarray): array of gene IDs of shape (ngenes,)
        chromstr (np.ndarray): array of chromosome strings of shape (ngenes,)
        start (np.ndarray): array of start positions of shape (ngenes,)
        end (np.ndarray): array of end positions of shape (ngenes,)

    Returns:
        gene_labels (np.ndarray): sorted array of gene IDs of shape (ngenes,)
        chromstr (np.ndarray): sorted array of chromosome strings of shape (ngenes,)
        start (np.ndarray): sorted array of start positions of shape (ngenes,)
        end (np.ndarray): sorted array of end positions of shape (ngenes,)
    """
    
    # Order by chromosome
    chromint = chromstr_to_chromint(chromstr)
    order = np.argsort(chromint)
    gene_labels = gene_labels[order]
    chromstr = chromstr[order]
    start = start[order]
    end = end[order]
    
    # Within each chromosome, order by start position
    start_copy = np.copy(start)
    for chrom in np.unique(chromstr):
        mask = chromstr == chrom
        order = np.argsort(start_copy[mask])
        gene_labels[mask] = gene_labels[mask][order]
        chromstr[mask] = chromstr[mask][order]
        start[mask] = start[mask][order]
        end[mask] = end[mask][order]
    
    return gene_labels, chromstr, start, end
    