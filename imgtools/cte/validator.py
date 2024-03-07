from pydantic import BaseModel, RootModel, StrictFloat, StrictInt, StrictStr, field_validator
from pydantic_core.core_schema import FieldValidationInfo
from typing import Dict

# Functions to validate the data format, that is a nested Dictionary of the form:
#        data[cellID][chrom][traceID][spotID] = {'x': float,
#                                                'y': float,
#                                                'z': float,
#                                                'chrom': str,
#                                                'start': int,
#                                                'end': int,
#                                                'lum': float}
# where cellID, chrom, traceID and spotID are strings, and chrom must start with 'chr'

class SpotData(BaseModel):
    """ Validate the format of the data for a single spot:
               spot_data = {'x': float,
                            'y': float,
                            'z': float,
                            'chrom': str,
                            'start': int,
                            'end': int,
                            'lum': float}
    """
    # SpotData is a dictionary with the following keys and types
    x: StrictFloat
    y: StrictFloat
    z: StrictFloat
    chrom: StrictStr
    start: StrictInt
    end: StrictInt
    lum: StrictFloat
    # Check that chromosome name starts with 'chr'
    @field_validator('chrom')
    def check_chrom(cls, v: str):
        if not v.startswith('chr'):
            raise ValueError('Chromosome name must start with "chr"')
        return v
    # Check that start > 0
    @field_validator('start')
    def check_start(cls, v: int):
        if v < 0:
            raise ValueError('Start position must be positive')
        return v
    # Check that end > start
    @field_validator('end')
    def check_end(cls, v: int, info: FieldValidationInfo):
        # Check that start has been validated
        if 'start' not in info.data:
            raise ValueError('Start position has not been validated yet')
        # Check that end > start
        if v < info.data['start']:
            raise ValueError('End must be greater than start')

class TraceData(RootModel):
    """ Validate the format of the data for a single trace:
            trace_data[traceID][spotID] = {'x': float,
                                           'y': float,
                                           'z': float,
                                           'chrom': str,
                                           'start': int,
                                           'end': int,
                                           'lum': float}
    """
    # TraceData is a dictionary where keys are spotID strings (e.g. '14') and values are SpotData
    root: Dict[StrictStr, SpotData]
    # Check that the dictionary is not empty
    @field_validator('root')
    def check_root(cls, v: dict):
        if len(v) == 0:
            raise ValueError('Dictionary of TraceData cannot be empty')

class ChromData(RootModel):
    """ Validate the format of the data for a single chromosome:
            chrom_data[chrom][traceID][spotID] = {'x': float,
                                                  'y': float,
                                                  'z': float,
                                                  'chrom': str,
                                                  'start': int,
                                                  'end': int,
                                                  'lum': float}
    """
    # ChromData is a dictionary where keys are traceID strings (e.g. '0') and values are TraceData
    root: Dict[StrictStr, TraceData]
    # Check that the dictionary is not empty
    @field_validator('root')
    def check_root(cls, v: dict):
        if len(v) == 0:
            raise ValueError('Dictionary of ChromData cannot be empty')

class CellData(RootModel):
    """ Validate the format of the data for a single cell:
            cell_data[cellID][chrom][traceID][spotID] = {'x': float,
                                                         'y': float,
                                                         'z': float,
                                                         'chrom': str,
                                                         'start': int,
                                                         'end': int,
                                                         'lum': float}
    """
    # CellData is a dictionary where keys are chrom strings (e.g. 'chr1') and values are ChromData
    root: Dict[StrictStr, ChromData]
    @field_validator('root')
    def check_root(cls, v: dict):
        # Check that the dictionay is not empty
        if len(v) == 0:
            raise ValueError('Dictionary of CellData cannot be empty')
        # Check that the keys are valid chromosome names
        for k in v.keys():
            if not k.startswith('chr'):
                raise ValueError('Chromosome name must start with "chr"')

class CTEData(RootModel):
    """ Validate the format of the data for a ChromatinTracingExperiment data attribute:
            cte_data[cellID][chrom][traceID][spotID] = { 'x': float,
                                                         'y': float,
                                                         'z': float,
                                                         'chrom': str,
                                                         'start': int,
                                                         'end': int,
                                                         'lum': float}
    """
    # CTEData is a dictionary where keys are cell strings (e.g. '0_1_rep1') and values are CellData
    root: Dict[StrictStr, CellData]
    # Check that the dictionay is not empty
    @field_validator('root')
    def check_root(cls, v: dict):
        if len(v) == 0:
            raise ValueError('Dictionary of CTEData cannot be empty')
        