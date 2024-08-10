# Contains functions to read FOF-CT (4DN Fish Omics Format - Chromatin Tracing) files.

import numpy as np


# Functions to read the header and columns

# NOT USED RIGHT NOW!
def read_header(filename: str) -> list:
    """ Read the header of the FOF-CT file.
    
    We assume that the header is at the beginning of the file and starts with the '#' character.
    We read the lines until we find the first line that does not start with '#',
    and we assume that this is the end of the header.
    
    Returns the list of header lines.

    Args:
        filename (str): path to the FOF-CT file

    Returns:
        list: header lines
    """
    # Initialize the header
    header = []
    # If we read too many lines, raise an error, as the header is either too long or not present
    count, max_count = 0, 200  # initialize counter and set the maximum number of lines to read
    # Read the file
    with open(filename, 'r') as csv:
        # Loop through the lines until we find the end of the header
        for line in csv:
            count += 1  # increment the counter
            # Too many lines condition
            if count > max_count:
                raise ValueError('The header is either too long or not present.')
            # End of the header conditions: the line does not start with '#' and is not empty
            # Also, we skip the line if it contains the 'Trace_ID' string, which is the column names line
            if line[0] != '#' and line[1] != '#' and 'Trace_ID' not in line:
                break
            # Clean the line
            line = line.strip('#" ,\n()[]{}')
            # Append the line to the header
            header.append(line)
    return header

def read_columns(filename: str) -> list:
    """ Read the columns' names of the FOF-CT file.
    
    The columns' names are usually the last line of the header.
    They contain the names of the columns, separated by commas, e.g.:
        "Spot_ID,Cell_ID,Trace_ID,Chrom,Chrom_Start,Chrom_End,X,Y,Z,Intensity"
    
    Returns the list of column keys.

    Args:
        filename (str): path to the FOF-CT file

    Returns:
        list: list of column keys, e.g. ['Spot_ID', 'Cell_ID', 'Trace_ID', 'Chrom', 'Chrom_Start', 'Chrom_End', ...]
    """
    # Initialize the columns as None
    cols = None
    # If we read too many lines, raise an error, as the columns line is either too long or not present
    count, max_count = 0, 200  # initialize counter and set the maximum number of lines to read
    # There are some keys that MUST be present in the columns line
    # Thus, we find the column line by checking if these keys are present
    # (we assume that these keys are present ONLY in the columns line)
    required_keys = ['Cell_ID', 'Trace_ID', 'Chrom', 'Chrom_Start', 'Chrom_End', 'X', 'Y', 'Z']
    # Read the file
    with open(filename, 'r') as csv:
        # Loop through the lines until we find the columns line
        for line in csv:
            # If cols is not None, the columns have been found and we can exit the loop
            if cols is not None:
                break
            # Increment the counter
            count += 1
            # Too many lines condition
            if count > max_count:
                raise ValueError('The columns line is not present.')
            # Check if this line is the columns line, i.e. if it contains the required keys
            # If it does not, continue to the next line
            not_col_line = False  # becomes True if the line does not contain any of the required keys
            for key in required_keys:
                if not_col_line:  # if it is already True, skip the rest of the loop
                    break
                if key not in line:
                    not_col_line = True  # change to True if the key is not present
                    break
            if not_col_line:  # if True, move to the next line
                continue
            # If the line contains the required keys, it is the columns line
            # We process it to get the column keys and exit the loop
            cols = process_columns_line(line)
            break
    return cols

def process_columns_line(line: str) -> list:
    """ Process the columns line of the FOF-CT file.
    
    Removes spaces, special characters, and splits the line by commas, returning a list of column keys.

    Args:
        line (str): raw columns line

    Returns:
        list: list of column keys
    """
    # Some datasets, instead of having the colums line as "Spot_ID,Cell_ID,Trace_ID,...",
    # have it as "Cols=Spot_ID,Cell_ID,Trace_ID,..."
    # We check if the line contains the '=' character, and if it does, we take the right part
    if '=' in line:
        line = line.split('=')[1]  # if it is, take the right part
    line = line.replace(' ', '')  # remove spaces
    line = line.strip('#" ,\n()[]{}')  # strip special characters
    cols = line.split(',')  # split by comma
    return cols

# NOT USED RIGHT NOW!
def extract_assembly(header: list) -> list:
    """ Extract the genome assembly from the header lines of the FOF-CT file.

    Args:
        header (list): list of header lines

    Returns:
        list: list of assemblies read from the header
    """
    # List of possible ways the assembly can be written in the header
    assembly_keys = ['genome_assembly', 'assembly']
    # Initialize the assembly as None
    assembly = None
    # Loop through the header lines
    for line in header:
        line = line.replace(' ', '')  # remove spaces
        # Check if the assembly keys are present in the line
        for assembly_key in assembly_keys:
            # If not, continue to the next line
            if assembly_key not in line:
                continue
            # If it is, take the right part of the line, e.g. "genome_assembly=hg38" -> "hg38"
            line = line.split('=')[1]
            # Some datasets have multiple assemblies separated by /
            assembly = line.split('/')  # take both assemblies if / is present
            # stop looping through the assembly keys if the assembly is found
            break
        if assembly is not None:
            # stop looping through the header lines if the assembly is found
            break
    return assembly


# Function to read the data

def read_data_from_lines(filename: str, cols: list) -> dict:
    """Read the data from the FOF-CT file line by line.
    
    The data is returned as a dictionary with the following structure:
         data[cellID][chrom][traceID][spotID] = {'x': x (float),
                                                 'y': y (float),
                                                 'z': z (float),
                                                 'chrom': chrom (str),
                                                 'start': start (int),
                                                 'end': end (int),
                                                 'lum': lum (float)}

    Args:
        filename (str): path to the FOF-CT file
        cols (list): list of column keys

    Returns:
        dict: data in dictionary format
    """
    
    # Initialize the data
    data = {}
    
    # Initialize the spotID counter dictionary
    # It is updated every time a line is read
    # Its keys are the cellIDs, and its values are the number of spots already read for that cellID
    spotID_counter = {}
    
    # Get a dictionary with the indices of each entry in the columns,
    # e.g. if cols = ['Spot_ID', 'Cell_ID', 'Trace_ID', 'Chrom', 'Chrom_Start', 'Chrom_End', ...],
    # then cols_to_index = {'Spot_ID': 0, 'Cell_ID': 1, 'Trace_ID': 2, 'Chrom': 3, 'Chrom_Start': 4, 'Chrom_End': 5, ...}
    cols_to_index = {col: i for i, col in enumerate(cols)}
    
    # Read the data
    with open(filename, 'r') as csv:
        
        # Loop through the lines
        for line in csv:
            
            # skip the header
            if line[0] == '#' or line[1] == '#' or 'Trace_ID' in line:
                continue
            
            # unpack the line
            x, y, z, chrom, start, end, spotID, traceID, cellID, lum = unpack_data(line, cols_to_index)
            
            # Update the spotID counter
            spotID_counter = update_spotID_counter(spotID_counter, cellID)
            
            # Get the spotID from the counter if it is not present in the line
            # It is the order of appearance of the spot in the cell, starting from 1
            if spotID is None:
                spotID = str(spotID_counter[cellID])
            
            # Update the data with the unpacked line
            data = process_data(cellID, chrom, start, end, traceID, spotID, x, y, z, lum, data)
            
    return data

def process_data(
    cellID: str, chrom: str, start: int, end: int, traceID: str, spotID: str,
    x: float, y: float, z: float, lum: float, data: dict
) -> dict:
    """ Adds the data of a new spot to the data dictionary.
    If the spot is already present, raises an error.

    Args:
        cellID (str)
        chrom (str)
        start (int)
        end (int)
        traceID (str)
        spotID (str)
        x (float)
        y (float)
        z (float)
        lum (float)
        data (dict): data in dictionary format to be updated

    Returns:
        dict: updated data in dictionary format
    """
    
    # Define the spot data that will be added to the data dictionary
    spot_data = {
        'x': float(x),
        'y': float(y),
        'z': float(z),
        'chrom': str(chrom),
        'start': int(start),
        'end': int(end),
        'lum': float(lum)
    }
    
    # Case 1: cellID is not yet in the data dictionary
    if cellID not in data:
        data[cellID] = {chrom: {traceID: {spotID: spot_data}}}
        
    # Case 2: cellID is in the data dictionary, but chrom is not
    elif chrom not in data[cellID]:
        data[cellID][chrom] = {traceID: {spotID: spot_data}}
        
    # Case 3: cellID and chrom are in the data dictionary, but traceID is not
    elif traceID not in data[cellID][chrom]:
        data[cellID][chrom][traceID] = {spotID: spot_data}
            
    # Case 4: cellID, chrom and traceID are in the data dictionary, but spotID is not
    elif spotID not in data[cellID][chrom][traceID]:
        data[cellID][chrom][traceID][spotID] = spot_data
            
    # Case 5: cellID, chrom, traceID and spotID are in the data dictionary
    else:
        raise ValueError('SpotID {} is present twice in the data!').format(spotID)
    
    return data

def unpack_data(line: str, cols_to_index: dict) -> tuple:
    """ Unpacks a line of data from the FOF-CT file,
    extracting the values in the line for each column key.

    Args:
        line (str): line of data from the FOF-CT file, e.g. "1,2,3,chr1,1000,2000,10,20,30,100"
        cols_to_index (dict): dictionary with the indices of each entry, e.g. {'Cell_ID': 0, 'Trace_ID': 1, ...}

    Returns:
        tuple: values in the line for each column key: (x, y, z, chrom, start, end, spotID, traceID, cellID, lum)
    """
    # Separate the line into values (still as strings)
    vals = line.split(',')
    # Check that the number of values is the same as the number of columns
    assert len(vals) == len(cols_to_index), f'Number of values ({len(vals)}) does not match the number of columns ({len(cols_to_index)}).'
    # Get the values
    x = float(vals[cols_to_index['X']])
    y = float(vals[cols_to_index['Y']])
    z = float(vals[cols_to_index['Z']])
    chrom = str(vals[cols_to_index['Chrom']])
    start = int(vals[cols_to_index['Chrom_Start']])
    end = int(vals[cols_to_index['Chrom_End']])
    traceID = str(vals[cols_to_index['Trace_ID']])
    if 'Spot_ID' in cols_to_index:
        spotID = str(vals[cols_to_index['Spot_ID']])
    else:
        spotID = None
    if 'Cell_ID' in cols_to_index:
        cellID = str(vals[cols_to_index['Cell_ID']])
    else:
        cellID = traceID
    if 'Intensity' in cols_to_index:
        lum = float(vals[cols_to_index['Intensity']])
    else:
        lum = np.nan
    return x, y, z, chrom, start, end, spotID, traceID, cellID, lum

def update_spotID_counter(spotID_counter: dict, cellID: str) -> dict:
    """ Updates the spotID counter for the cellID, incrementing it by 1.

    Args:
        spotID_counter (dict): dictionary of the spotID counter
        cellID (str): cellID

    Returns:
        (dict): updated spotID counter
    """
    # If cellID is not present, initialize it to 0
    if cellID not in spotID_counter:
        spotID_counter[cellID] = 0
    # Increment the counter
    # (If the spotID is not present, it is initialized to 0, so the first spotID will be 1)
    spotID_counter[cellID] += 1
    return spotID_counter


# Final function to process the FOF-CT file into the desired format

def read_fofct(filename: str) -> dict:
    """Reads the FoF-CT file.
    
    The data is returned as an dictionary with the following structure:
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
        dict: data in dictionary format
    """
    
    # Read the columns
    cols = read_columns(filename)
    
    # Read the data
    data = read_data_from_lines(filename, cols)
    
    return data
