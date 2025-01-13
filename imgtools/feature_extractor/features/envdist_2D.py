import numpy as np
import trimesh
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, Point
from ...cte import ChromatinTracingExperiment
from ...cte import cte_utils

docstring = """ Measures the 2D distance of each spot to the nuclear envelope at the same z-slice of the spot.
The z-slices are determined as quantiles of the z values of the spots. The distance is calculated as the 2D distance
between the spot and the convex hull of the XY projection of the nuclear envelope at the same z-slice."""

required_keys = {
    'nslices': {'type': int}
}

def run(cellID: str, cte: ChromatinTracingExperiment, config: dict, feat_arr: np.ndarray, _) -> np.ndarray:
    """ Calculate the distance of each spot to 2D section of the nuclear envelope at the same z-slice of the spot.
    
    To make the calculation faster, instead of using a different z-slice for each spot, the cell is divided into
    n z-slices (defined in the configuration), taken as quantiles of the z values. The section is generated for the
    median z value of each slice. Then, each spot is assigned to the z-slice to which it belongs, and the distance
    is calculated from the X-Y projection of the spot in the section.
    
    To generate the section, we take the vertices of the projection of the alphashape at the z value of interest.
    Then, we fit a 2D convex hull to the XY vertices. The distances are then calculated as 2D distances to the
    convex hull.
    
    The nuclear envelope is taken from the alpha shape of the cell.
    
    If there are two or more spots corresponding to the same domain in the trace, the average distance is taken.

    Args:
        cellID (str)
        cte (ChromatinTracingExperiment)
        feat_arr (np.ndarray): initialized nan-valued array of shape (n_domains, n_traces) to store the feature values
        _*: not used, just to match the function signature

    Returns:
        (np.ndarray): updated array of shape (n_domains, n_traces) with the feature values
    """
    
    # Get the parameters from the configuration
    nslices = config['nslices']
    
    # Get the cell data in dictionary format
    cell_data = cte.get_data(cellID)
    
    # Get the alpha shape of the cell
    alphashape = cte.get_alphashapes(cellID)
    
    # Get the traceID hash table to map traces to their position in the array
    traceID_hash = cte.get_trace_hashmap(cellID)
    
    # Convert the cell data in numpy format and get the z coordinates of the spots
    _, _, zs, _, _, _, _, _, _ = cte_utils.cell_dict_to_numpy(cell_data)
    
    # Divide the cell into z-slices defined as quantiles of the z values
    zquants = np.nanquantile(zs, np.linspace(0, 1, nslices + 1))  # shape: (nslices + 1,)
    
    # For each z-slice, get the section of the alpha shape in the median z value of the slice
    zsections = {}
    for q in range(nslices):  # loop over the quantiles
        
        # Get the mask for the quantile
        if q == nslices - 1:  # include the last value if it's the last quantile
            mask_q = zs >= zquants[q]
        else:
            mask_q = np.logical_and(zs >= zquants[q], zs < zquants[q + 1])
        
        # Get the median z value of the slice
        zmed_q = np.nanmedian(zs[mask_q])
        
        # Get the section of the alpha shape at the z value
        section = get_alphashape_zsection(alphashape['mesh'], zmed_q)
        
        # Raise an error if the section is None
        if section is None:
            raise ValueError(f"Error: section is None for cell {cellID}, slice {q}")
        
        # Store the section in the dictionary
        zsections[q] = section
    
    # Get the index and its hash table
    index = cte.index
    index_hash = index.get_index_hashmap()
    
    # Initialize a dictionary to store the feature values for each domain (we will then take the average)
    feat_per_domain = {}
    
    for chrom in cell_data:        
        for traceID in cell_data[chrom]:
            
            # Get the position of the trace in the array
            i_trace = traceID_hash[chrom][traceID]
            
            for spotID in cell_data[chrom][traceID]:
                
                # Unpack the spot data
                spot_data = cell_data[chrom][traceID][spotID]
                x, y, z = spot_data['x'], spot_data['y'], spot_data['z']
                start, end = spot_data['start'], spot_data['end']
                
                # Find the z-slice to which the spot belongs
                zq = np.searchsorted(zquants, z) - 1
                # The searchsorted might fail if the z value is very close to the edges,
                # so we sort it out manually
                if np.isclose(z, zquants[0]):
                    zq = 0
                elif np.isclose(z, zquants[-1]):
                    zq = nslices - 1
                
                # Get the section of the cell at the z-slice
                if zq not in zsections:
                    raise ValueError(f"Error: z-slice {zq} not found in zsections for cell {cellID}")
                section = zsections[zq]
                
                # Calculate the 2D distance between the point and the section of the same z-slice
                point = Point(x, y)
                dist = np.abs(section.exterior.distance(point))
                
                # Get the position of the spot in the Index array using the hash tables
                i_domain = index_hash[(chrom, start, end)]
                assert len(i_domain) == 1, f"Error: multiple domains found for {chrom}, {start}, {end}"
                i_domain = i_domain[0]
                
                # Initialize the list of values for this domain if necessary
                if (i_domain, i_trace) not in feat_per_domain:
                    feat_per_domain[(i_domain, i_trace)] = []
                
                # Add the feature value to the dictionary of values for this domain
                feat_per_domain[(i_domain, i_trace)].append(dist)
                
    
    # Compute the average of the values for each domain and add them to the feature array
    for (i_domain, i_trace), vals in feat_per_domain.items():
        feat_arr[i_domain, i_trace] = np.nanmean(vals)
    
    return feat_arr

def get_alphashape_zsection(mesh: trimesh.Trimesh, z: float) -> Polygon:
    
    # Get the section of the mesh at the given z value
    section = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
    
    # If the section is empty, return None
    if section is None:
        return None

    # Get the vertices of the section
    vertices = section.vertices  # shape (n, 3)

    # Fit a 2D convex hull to the XY vertices
    hull = ConvexHull(vertices[:, :2])

    # Create a polygon representation of the convex hull
    polygon = Polygon(vertices[hull.vertices, :2])
    
    return polygon
