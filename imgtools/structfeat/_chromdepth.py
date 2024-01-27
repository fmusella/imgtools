import numpy as np
import trimesh
from alabtools.utils import Index

required_keys = {
    'alpha': {'type': float, 'positive': True},
    'force': {'type': bool},
}

def run(cell_data: dict, data_attrs: dict, index: Index, config: dict):
    pass
