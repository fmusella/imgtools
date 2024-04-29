# Features purely from the imaging data
from . import _spotcount
from . import _intensity
from . import _immunof
# Sliding genomic window features
from . import _chaindist
from . import _gyration
from . import _allVSall
from . import _convexhull
# Contact sphere features
from . import _closest
from . import _localcrowd
from . import _longrange
# Alpha shape features
from . import _envsurf
from . import _chromsurf
# Complex features
from . import _kerneldensity
from . import _voronoi

MODULES = {
    """ Dictionary of the feature extraction modules.
    
    Each module has a 'run' function with the following signature:
        run(cellID: str, cte: ChromatinTracingExperiment, config: dict, feat_arr: np.ndarray, feature: str) -> np.ndarray
    
    Each module also has a docstring specifying the feature name and a brief description,
    and a required_keys dictionary specifying the keys that must be present in the config of the run function.
    """
    
    'spotcount': _spotcount,
    'intensity': _intensity,
    'immunof': _immunof,
    
    'chaindist': _chaindist,
    'gyration': _gyration,
    'allVall': _allVSall,
    'convexhull': _convexhull,
    
    'closest': _closest,
    'localcrowd': _localcrowd,
    'longrange': _longrange,
    
    'envsurf': _envsurf,
    'chromsurf': _chromsurf,
    
    'kerneldensity': _kerneldensity,
    'voronoi': _voronoi,
}
