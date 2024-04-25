from . import _envsurf
from . import _chromsurf
from . import _spotcount
from . import _immunof
from . import _intensity
from . import _gyration
from . import _crowd
from . import _neighdist
from . import _longrange

MODULES = {
    """ Dictionary of the feature extraction modules.
    
    Each module has a 'run' function with the following signature:
        run(cellID: str, cte: ChromatinTracingExperiment, config: dict, feat_arr: np.ndarray, feature: str) -> np.ndarray
    
    Each module also has a docstring specifying the feature name and a brief description,
    and a required_keys dictionary specifying the keys that must be present in the config of the run function.
    """
    'spotcount': _spotcount,
    'envsurf': _envsurf,
    'chromsurf': _chromsurf,
    'immunof': _immunof,
    'intensity': _intensity,
    'gyration': _gyration,
    'crowd': _crowd,
    'neighdist': _neighdist,
    'longrange': _longrange,
}
