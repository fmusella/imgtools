# Features purely from the imaging data
from . import spotcount
from . import intensity
from . import immunof
# Sliding genomic window features
from . import chaindist
from . import gyration
from . import allVSall
from . import convexhull
# Contact sphere features
from . import closest
from . import localcrowd
from . import longrange
# Alpha shape features
from . import envsurf
from . import chromsurf
# Complex features
from . import kerneldensity
from . import voronoi

MODULES = {
    """ Dictionary of the feature extraction modules.
    
    Each module has a 'run' function with the following signature:
        run(cellID: str, cte: ChromatinTracingExperiment, config: dict, feat_arr: np.ndarray, feature: str) -> np.ndarray
    
    Each module also has a docstring specifying the feature name and a brief description,
    and a required_keys dictionary specifying the keys that must be present in the config of the run function.
    """
    
    'spotcount': spotcount,
    'intensity': intensity,
    'immunof': immunof,
    
    'chaindist': chaindist,
    'gyration': gyration,
    'allVall': allVSall,
    'convexhull': convexhull,
    
    'closest': closest,
    'localcrowd': localcrowd,
    'longrange': longrange,
    
    'envsurf': envsurf,
    'chromsurf': chromsurf,
    
    'kerneldensity': kerneldensity,
    'voronoi': voronoi,
}
