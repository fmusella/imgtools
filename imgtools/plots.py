import numpy as np
from matplotlib import pyplot as plt


# Pyplot functions

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
