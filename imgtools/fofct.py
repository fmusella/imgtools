# Contains functions to read FOF-CT (4DN Fish Omics Format - Chromatin Tracing) files.

import warnings
import numpy as np
from alabtools.utils import Genome, Index


# Functions to read the header and columns

def read_header(filename):
    """Read the header of the FOF-CT file.
    Returns the list of header lines."""
    # Initialize the header
    header = []
    count, max_count = 0, 200  # count the number of lines read, raise error if too many
    with open(filename, 'r') as csv:
        for line in csv:
            count += 1
            # Too many lines condition
            if count > max_count:
                raise ValueError('The header is either too long or not present.')
            # End of the header conditions
            if line[0] != '#' and line[1] != '#' and 'Trace_ID' not in line:
                break
            # Clean the line
            line = line.strip('#" ,\n()[]{}')
            # Append the line to the header
            header.append(line)
    return header

def read_columns(filename):
    """Read the columns of the FOF-CT file.
    Returns the list of column keys."""
    cols = None
    count, max_count = 0, 200  # count the number of lines read, raise error if too many
    required_keys = ['Spot_ID', 'Cell_ID', 'Trace_ID']
    with open(filename, 'r') as csv:
        for line in csv:
            if cols is not None:
                break
            count += 1
            # Too many lines condition
            if count > max_count:
                raise ValueError('The columns line is not present.')
            # Column line condition
            for key in required_keys:
                if key in line:
                    cols = process_columns_line(line)
    return cols

def process_columns_line(line):
    """Process the columns line of the FOF-CT file."""
    # Check if '=' is present
    if '=' in line:
        line = line.split('=')[1]  # if it is, take the right part
    line = line.replace(' ', '')  # remove spaces
    line = line.strip('#" ,\n()[]{}')  # strip special characters
    cols = line.split(',')  # split by comma
    return cols

def extract_assembly(header):
    """Extract the genome assembly from the header lines of the FOF-CT file."""
    assembly_keys = ['genome_assembly', 'assembly']  # possible keys for the assembly
    assembly = None
    for line in header:
        line = line.replace(' ', '')
        for assembly_key in assembly_keys:
            if assembly_key not in line:
                continue
            # Some datasets have multiple assemblies separated by /
            line = line.split('=')[1]
            assembly = line.split('/')  # take both assemblies if / is present
            # stop looping through the assembly keys if the assembly is found
            break
        if assembly is not None:
            # stop looping through the header lines if the assembly is found
            break
    return assembly


# Function to read the data

def read_data_from_lines(filename, cols):
    """Read the data from the FOF-CT file line by line.
    The domains are returned as a set of tuples (chr, start, end).
    The data is returned as a hashmap with the following structure:
         data[cellID][chrom][traceID][spotID] = {'x': x (float),
                                                 'y': y (float),
                                                 'z': z (float),
                                                 'chrom': chrom (str),
                                                 'start': start (int),
                                                 'end': end (int),
                                                 'lum': lum (float)}
    The reason for using sets and hashmaps is the speed of lookup (O(1)).
    Also returns summary metrics about the data.

    Args:
        filename (str): path to the csv file
        cols (list): list of column keys

    Returns:
        domains (set): set of tuples (chr, start, end)
        data (dict): hashmap of the data
        summary_metrics (dict): summary metrics about the data
    """
    
    # Initialize the data
    data = {}
    
    # Get the indices of each entry in the columns
    cols_to_index = {col: i for i, col in enumerate(cols)}
    
    # Read the data
    with open(filename, 'r') as csv:
        
        for linenum, line in enumerate(csv):
            
            # skip the header
            if line[0] == '#' or line[1] == '#' or 'Trace_ID' in line:
                continue
            
            # unpack the line
            x, y, z, chrom, start, end, spotID, traceID, cellID, lum = unpack_data(line, cols_to_index)
            
            # If spotID is not provided, use the line number
            if spotID is None:
                spotID = str(linenum)
            
            # Process the data
            data = process_data(cellID, chrom, start, end, traceID, spotID, x, y, z, lum, data)
            
    return data

def process_data(cellID, chrom, start, end, traceID, spotID, x, y, z, lum, data):
    """Adds a spotID with its coordinates to the data hashmap if it is not present.
    If the spotID is already present it raises an error."""
    
    # Define the spot data that will be added to the hashmap
    spot_data = {'x': float(x),
                 'y': float(y),
                 'z': float(z),
                 'chrom': str(chrom),
                 'start': int(start),
                 'end': int(end),
                 'lum': float(lum)}
    
    # Case 1: cellID is not yet in the hashmap
    if cellID not in data:
        data[cellID] = {chrom: {traceID: {spotID: spot_data}}}
        
    # Case 2: cellID is in the hashmap, but chrom is not
    elif chrom not in data[cellID]:
        data[cellID][chrom] = {traceID: {spotID: spot_data}}
        
    # Case 3: cellID and chrom are in the hashmap, but traceID is not
    elif traceID not in data[cellID][chrom]:
        data[cellID][chrom][traceID] = {spotID: spot_data}
            
    # Case 4: cellID, chrom and traceID are in the hashmap, but spotID is not
    elif spotID not in data[cellID][chrom][traceID]:
        data[cellID][chrom][traceID][spotID] = spot_data
            
    # Case 5: cellID, chrom, traceID and spotID are in the hashmap
    else:
        raise ValueError('SpotID {} is present twice in the data!').format(spotID)
    
    return data

def unpack_data(line, cols_to_index):
    """Unpacks a line of data from the FOF-CT file.
    Returns data as a tuple."""
    # Separate the line into values
    vals = line.split(',')
    # Check that the number of values is the same as the number of columns
    assert len(vals) == len(cols_to_index)
    # Get the values
    x = float(vals[cols_to_index['X']])
    y = float(vals[cols_to_index['Y']])
    z = float(vals[cols_to_index['Z']])
    chrom = str(vals[cols_to_index['Chrom']])
    start = int(vals[cols_to_index['Chrom_Start']])
    end = int(vals[cols_to_index['Chrom_End']])
    if 'Spot_ID' in cols_to_index:
        spotID = str(vals[cols_to_index['Spot_ID']])
    else:
        spotID = None
    traceID = str(vals[cols_to_index['Trace_ID']])
    if 'Cell_ID' in cols_to_index:
        cellID = str(vals[cols_to_index['Cell_ID']])
    else:
        cellID = traceID
    if 'Intensity' in cols_to_index:
        lum = float(vals[cols_to_index['Intensity']])
    else:
        lum = np.nan
    return x, y, z, chrom, start, end, spotID, traceID, cellID, lum


# Final function to process the FOF-CT file into the desired format

def read_fofct(filename):
    """Reads the FoF-CT file.
    
    The data is returned as an hashmap with the following structure:
        data[cellID][chrom][traceID][spotID] = {'x': x (float),
                                                'y': y (float),
                                                'z': z (float),
                                                'chrom': chrom (str),
                                                'start': start (int),
                                                'end': end (int),
                                                'lum': lum (float)}

    Args:
        filename (str): path to FoF-CT CSV file.

    Returns:
        data (dict): data in dictionary format
    """
    
    # Read the header and the assembly
    header = read_header(filename)
    assembly = extract_assembly(header)
    
    # Read the columns
    cols = read_columns(filename)
    
    # Read the data
    data = read_data_from_lines(filename, cols)
    
    return data
