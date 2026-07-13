import numpy as np
from sklearn.mixture import GaussianMixture
from ...scf import SingleCellFeature
from .synchronizer import CellCycleSynchronizer

class CellCycleGaussianMixture(CellCycleSynchronizer):
    """ Class to perform the Gaussian Mixture model to classify for the cell cycle synchronization.

    The early/late loci used for the bimodal split are supplied as explicit boolean masks
    (config keys 'early_mask' and 'late_mask'), aligned 1:1 with the columns of the feature
    matrix. Typically the early loci are constitutive-early (e.g. top-5% RT) and the late
    loci are constitutive-late (e.g. bottom-5% RT), but any caller-defined selection works.

    We assume that the early replicating regions are instantly replicated in all S cells,
    while the late replicating regions are replicated almost exclusively in G2 cells.

    Thus, we fit two Gaussian Mixture models using the rowsum of the feature matrix for the early and late regions.
    - The first model is fitted to the early regions to separate G1 from S/G2 cells.
    - The second model is fitted to the late regions to separate G2 from G1/S cells.

    This class does NOT use the RT signal, so 'rt_file' is optional (requires_rt = False).

    Inherits from CellCycleSynchronizer.

    --- Attributes (inherit from CellCycleSynchronizer) ---
    scf (SingleCellFeature): SingleCellFeature object.
    index (Index): Index of the SingleCellFeature.
    config (dict): Configuration dictionary for the synchronization method.
    rt_file (str or None): Path to the replication timing file (optional, unused by this class).
    usechroms (list): List of chromosome strings to be used in the synchronization.
    smooth_k (int or None): Smoothing parameter k.
    feature (str): Name of the feature to be used in the synchronization.
    smooth_chromstr (np.array or None): Chromosome strings for the smoothing function.
    matrix (np.array): Matrix of the feature for the chromosomes specified in usechroms.
    rowmean (np.array): Row-wise mean of the matrix.
    states_ (np.array): Array of strings with the states of the cells, e.g. ['G', 'S', 'G', ...], to be updated in the run method.

    --- Attributes (specific to CellCycleGaussianMixture) ---
    early_mask_ (np.array of bool): boolean mask (length == matrix.shape[1]) selecting the early loci.
    late_mask_ (np.array of bool): boolean mask (length == matrix.shape[1]) selecting the late loci.
    loci_chromstr_ (np.array): chrom string of each matrix column (length == matrix.shape[1]).
    loci_start_ (np.array): start bp of each matrix column (length == matrix.shape[1]).
                            early_mask[i]/late_mask[i] correspond to loci_chromstr_[i], loci_start_[i].

    --- Methods (for users) ---
    run: Run the Gaussian Mixture model method.

    Args:
        CellCycleSynchronizer (_type_): _description_
    """

    # This synchronizer uses explicit early/late masks, not the RT signal, so rt_file is optional.
    requires_rt = False

    def __init__(self, scf: SingleCellFeature, config: dict, initial_states: np.array = None) -> None:
        """ Initialize the CellCycleGaussianMixture class.
        Inherits from CellCycleSynchronizer.

        Args:
            scf (SingleCellFeature)
            config (dict): configuration dictionary. Must contain 'early_mask' and 'late_mask'.
            initial_states (np.array, optional): Initial states of the cells, e.g. ['G', 'S', 'G', ...].
                            If None, the states are initialized randomly.
        """

        super().__init__(scf, config, initial_states)

        # Expose the genomic coordinates of the matrix columns, so that callers can
        # build boolean masks (early_mask/late_mask) that are aligned 1:1 with the columns.
        # The base class builds self.matrix by keeping the columns of self.index where
        # self.index.chromstr is in self.usechroms (see CellCycleSynchronizer.prepare_matrix),
        # preserving the original index order. We reproduce the exact same selection here.
        loci_mask = np.isin(self.index.chromstr, self.usechroms)
        self.loci_chromstr_ = np.asarray(self.index.chromstr)[loci_mask]  # (matrix.shape[1],)
        self.loci_start_ = np.asarray(self.index.start)[loci_mask]        # (matrix.shape[1],)

        # Check that the configuration dictionary is correct.
        # This validates and stores self.early_mask_ and self.late_mask_.
        self.check_config()


    def check_config(self):
        """ Check the configuration dictionary for the Gaussian Mixture model.

        The early/late loci used for the bimodal split are supplied as explicit boolean
        masks via the REQUIRED config keys 'early_mask' and 'late_mask'. Each must be a
        1-D boolean (or 0/1) array of length == self.matrix.shape[1], aligned 1:1 with the
        matrix columns / self.loci_chromstr_ / self.loci_start_.

        Stores self.early_mask_ and self.late_mask_.
        """

        # Both masks are required
        for key in ('early_mask', 'late_mask'):
            if key not in self.config or self.config[key] is None:
                raise ValueError(f'Config key "{key}" is required for CellCycleGaussianMixture.')

        # Coerce to boolean arrays and validate their length against the matrix columns
        early_mask = np.asarray(self.config['early_mask']).astype(bool)
        late_mask = np.asarray(self.config['late_mask']).astype(bool)
        ncol = self.matrix.shape[1]
        if early_mask.ndim != 1 or len(early_mask) != ncol:
            raise ValueError(
                f'Config key "early_mask" must be a 1-D boolean array of length '
                f'{ncol} (== self.matrix.shape[1]). Got shape {early_mask.shape} instead.'
            )
        if late_mask.ndim != 1 or len(late_mask) != ncol:
            raise ValueError(
                f'Config key "late_mask" must be a 1-D boolean array of length '
                f'{ncol} (== self.matrix.shape[1]). Got shape {late_mask.shape} instead.'
            )
        self.early_mask_ = early_mask
        self.late_mask_ = late_mask
    
    
    def run(self) -> None:
        """ Run the Gaussian Mixture model to classify the cells into G1, S, and G2 phases.

        The early (top) and late (bottom) loci are selected from the explicit
        early_mask_ / late_mask_ supplied in the config.

        We assume that the early replicating regions are instantly replicated in all S cells,
        while the late replicating regions are replicated almost exclusively in G2 cells.

        Thus, we fit two Gaussian Mixture models using the rowsum of the feature matrix for the early and late regions.
        - The first model is fitted to the early regions to separate G1 from S/G2 cells.
        - The second model is fitted to the late regions to separate G2 from G1/S cells.
        """

        # Select the early (top) and late (bottom) loci used for the bimodal split
        # from the explicit masks (aligned 1:1 with the matrix columns).
        top_idx = np.where(self.early_mask_)[0]
        bottom_idx = np.where(self.late_mask_)[0]
        
        # Calculate the rowsum of the feature matrix for the top and bottom percentiles
        rowsum_top = np.nansum(self.matrix[:, top_idx], axis=1)  # shape: (ncell,)
        rowsum_bottom = np.nansum(self.matrix[:, bottom_idx], axis=1)  # shape: (ncell,)

        # Fit a Gaussian Mixture model to the top signal to separate G1 from S/G2
        X = np.array([rowsum_top]).T
        gm_top = GaussianMixture(n_components=2).fit(X)
        y = gm_top.predict(X)
        # y is a binary array of 0s and 1s. We assign 'G1' to the cluster with the lowest mean
        states_top = np.full(len(y), 'S/G2', dtype='U20')
        mean0, mean1 = gm_top.means_
        if mean0 < mean1:
            states_top[y == 0] = 'G1'
        else:
            states_top[y == 1] = 'G1'

        # Do the same for the bottom signal to separate G1/S from G2
        X = np.array([rowsum_bottom]).T
        gm_bottom = GaussianMixture(n_components=2).fit(X)
        y = gm_bottom.predict(X)
        # Now we assign 'G2' to the cluster with the largest mean
        states_bottom = np.full(len(y), 'G1/S', dtype='U20')
        mean0, mean1 = gm_bottom.means_
        if mean0 > mean1:
            states_bottom[y == 0] = 'G2'
        else:
            states_bottom[y == 1] = 'G2'
        
        # Make sure that the lengths of the states are consistent
        assert len(states_top) == len(states_bottom) == self.matrix.shape[0], 'Shape mismatch between states and matrix.'
        
        # Raise an error if the states are not consistent
        if np.any(np.logical_and(states_top == 'G1', states_bottom == 'G2')):
            n_inconsistent = np.sum(np.logical_and(states_top == 'G1', states_bottom == 'G2'))
            print(f"Warning: {n_inconsistent} cells are classified as both G1 and G2.")
        
        # Combine the states
        states = np.full(len(states_top), 'S', dtype='U20')
        states[states_top == 'G1'] = 'G1'
        states[states_bottom == 'G2'] = 'G2'
        states[np.logical_and(states_top == 'G1', states_bottom == 'G2')] = 'NA'
        
        # Save the results to the object
        self.states_ = states
        self.gm_top_ = gm_top
        self.gm_bottom_ = gm_bottom
