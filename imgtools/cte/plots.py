# Functions that perform general plots for ChromatinTracingExperiment objects.

import os
import numpy as np
from matplotlib import pyplot as plt
from .cte import ChromatinTracingExperiment
from . import cte_utils


def save_cell_pyplot(cte: ChromatinTracingExperiment, cellID: str, path: str, filename: str = None, plot_params: dict = {}) -> None:
    """ Plot a cell using matplotlib. """
    
    # Check that cellID is a string and that it is in the data
    if not isinstance(cellID, str):
        raise TypeError("cellID must be a string.")
    if not cellID in cte.data:
        raise ValueError("cellID {} not in data.".format(cellID))
    
    # Check that path is a string and that it exists
    if not isinstance(path, str):
        raise TypeError("path must be a string.")
    if not os.path.exists(path):
        raise NotADirectoryError("Directory {} does not exist.".format(path))
    
    # If filename is not provided, use cellID
    if filename is None:
        filename = os.path.join(path, cellID + '.png')
    # Check that filename is a string
    if not isinstance(filename, str):
        raise TypeError("filename must be a string.")
    
    # Check that plot_params is a dictionary
    if not isinstance(plot_params, dict):
        raise TypeError("plot_params must be a dictionary.")
    
    # Get data for cell in numpy array format
    xs, ys, zs, chroms, _, _, _, _, _ = cte_utils.cell_to_numpy(cte.data[cellID])
    data_for_pyplot = {
        'x': xs,
        'y': ys,
        'z': zs,
        'chrom': chroms
    }
    
    # Plot cell
    cell_pyplot(filename, cellID, data_for_pyplot, plot_params)

def save_all_pyplots(cte: ChromatinTracingExperiment, path: str, plot_params: dict = {}) -> None:
    """ Save pyplots for all cells. """
    
    # Check that path is a valid directory
    if not isinstance(path, str):
        raise TypeError("path must be a string.")
    if not os.path.exists(path):
        raise NotADirectoryError("Directory {} does not exist.".format(path))
    
    for cellID in cte.data:
        save_cell_pyplot(cellID, path, plot_params=plot_params)

def cell_pyplot_default_params(cellID):
    """ Default parameters for pyplot cell plots.

    Returns:
        params (dict): Default parameters for pyplot cell plots.
    """
    
    params = {
        'figsize': (10, 10),
        'dpi': 200,
        'show_title': False,
        'show_axis': True,
        'show_legend': False,
        'title': 'Cell ' + str(cellID)
    }
    
    return params

def cell_pyplot_complete_params(cellID: str, params: dict):
    """ Complete parameters for pyplot cell plots.

    Args:
        cellID (str)
        params (dict): Incomplete parameters for pyplot cell plots.

    Returns:
        (dict): Complete parameters for pyplot cell plots.
    """
    
    # Get default parameters
    default_params = cell_pyplot_default_params(cellID)
    
    # If params does not contain a key, add it from default_params
    for key in default_params.keys():
        if key not in params.keys():
            params[key] = default_params[key]
    
    return params

def cell_pyplot(filename: str, cellID: str, data: dict, params: dict):
    """ Plot cell data using pyplot.

    Args:
        filename (str): destination filename
        cellID (str): cell ID
        data (dict): data to plot
        params (dict): parameters for pyplot cell plots
    """
    
    # Check input data
    if not isinstance(data, dict):
        raise ValueError('data must be a dict')
    if 'x' not in data.keys():
        raise ValueError('data must contain a key "x"')
    if 'y' not in data.keys():
        raise ValueError('data must contain a key "y"')
    if 'z' not in data.keys():
        raise ValueError('data must contain a key "z"')
    if 'chrom' not in data.keys():
        raise ValueError('data must contain a key "chrom"')
    if len(data['x']) != len(data['y']) or len(data['x']) != len(data['z']) or len(data['x']) != len(data['chrom']):
        raise ValueError('data["x"], data["y"], data["z"] and data["chrom"] must have the same length')
    
    # Complete parameters
    params = cell_pyplot_complete_params(cellID, params)
    
    # Create 3D figure
    fig = plt.figure(figsize=params['figsize'], dpi=params['dpi'])
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot data, each chromosome in a different color
    for i, chrom in enumerate(np.unique(data['chrom'])):
        
        # Get chromosome data
        x_chrom = data['x'][data['chrom'] == chrom]
        y_chrom = data['y'][data['chrom'] == chrom]
        z_chrom = data['z'][data['chrom'] == chrom]

        # Plot chromosome data
        ax.scatter(x_chrom, y_chrom, z_chrom, label=chrom, color='C' + str(i))
    
    # Set legend
    if params['show_legend']:
        ax.legend(loc='best')
    
    # Set title
    if params['show_title']:
        ax.set_title(params['title'])
    
    # Set axis
    if params['show_axis']:
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
    else:
        ax.set_axis_off()
    
    # Save figure in 3 different angles: parallel to xy, parallel to xz and parallel to yz
    ax.view_init(0, 0)
    plt.savefig(filename + '_xy.png')
    ax.view_init(90, 0)
    plt.savefig(filename + '_xz.png')
    ax.view_init(0, 90)
    plt.savefig(filename + '_yz.png')
    
    plt.close(fig)

def plot_chrom_alphashape(data: dict, alphashapes: dict, cellID: str, chrom: str, alpha: float, force: bool = False):

    # Initialize the figure
    figsize = (8, 8)
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    ax = fig.add_subplot(111, projection='3d')
    
    # Get the mesh of the cell
    cell_mesh = alphashapes[cellID]['mesh']
    
    # Plot the mesh of the cell
    ax.plot_trisurf(*zip(*cell_mesh.vertices), triangles=cell_mesh.faces, color='yellow', alpha=0.5)
    
    # Loop over the copies of the chromosome
    for traceID in data[cellID][chrom]:
        
        # Get the data of the chromosomal copy and fit an alphashape
        xs, ys, zs, _, _, _, _, _ = cte_utils.trace_dict_to_numpy(data[cellID][chrom][traceID])
        points = np.array([xs, ys, zs]).T
        alpha, mesh = cte_utils.fit_alphashape(points, alpha, force)
        print('Alpha: {}'.format(alpha))
        
        # Plot the alphashape
        ax.plot_trisurf(*zip(*mesh.vertices), triangles=mesh.faces, color='red', alpha=0.8)
        
        # Plot the points
        ax.scatter(xs, ys, zs, color='red', s=0.8)
    
    return fig, ax
