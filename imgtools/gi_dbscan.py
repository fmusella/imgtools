import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist


class GenomicIterativeDBSCAN():
    """Class to perform the Genomic Iterative DBSCAN algorithm.
    
    It segments the chromosome in "windows" and performs DBSCAN on each window.
    Clusters of consecutive windows are merged if they are close enough.
    
    Parameters
    ----------
    dbscan_eps : float
        DBSCAN epsilon.
    dbscan_min_samples : int
        DBSCAN min_samples.
    window_size : int
        window of chromosome to perform DBSCAN on, in bp.
    delta : float
        maximum distance between two consecutive windows
        to be merged in a single trace, in spatial units.
    max_missing_windows : int
        number of missing consecutive windows to mark a trace as inactive.
    
    Attributes
    ----------
    _traces : dict
        dictionary of traces, indexed by trace id, with this structure:
            _traces[traceID] = {'points': np.ndarray of shape (n_points, 3),  # 3D coordinates of the points in the trace
                                'starts': np.ndarray of shape (n_points,),  # start position of the points in the trace
                                'indices': np.ndarray of shape (n_points,),  # indices of the points in the trace
                                'last_window_points': np.ndarray of shape (n_points_last_window, 3),
                                'last_window_starts': np.ndarray of shape (n_points_last_window,),
                                'missing_windows_count': int,  # number of consecutive missing windows
                                'active': bool  # whether the trace is active or not
                                }
    """
    
    def __init__(self, dbscan_eps: float = 1., dbscan_min_samples: int = 5,
                 window_size: int = 5 * 10**6, delta: float = 3., max_missing_windows: int = 10):
        # Parameters
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.window_size = window_size
        self.delta = delta
        self.max_missing_windows = max_missing_windows
        # Internal variables
        self._traces = {}
    
    def fit(self, X: np.ndarray, start: np.ndarray):
        """Perform the Genomic Iterative DBSCAN algorithm.

        Args:
            X (np.ndarray of float): array of shape (n_samples, 3) containing
                the coordinates of the points.
            start (np.ndarray of int): array of shape (n_samples,) containing
                the start position of each point.
        """
        
        self._traces = {}  # reset traces
        
        self._check_input(X, start)  # check that X and start are valid
        
        windows = start // self.window_size  # compute window labels
        
        # Loop over windows to perform Genomic Iterative DBSCAN
        for w in np.unique(windows):
                        
            indices_w = np.where(windows == w)[0]
            X_w = X[windows == w]
            start_w = start[windows == w]
            
            db = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples).fit(X_w)
            clusters_w = db.labels_
            
            self._add_window_to_traces(X_w, start_w, clusters_w, indices_w)
        
        # Reorder traces
        self._reorder_traces()
        
        # Convert traces to a numpy array as in the output of DBSCAN
        self.labels_ = self._traces_to_labels(X)
    
    def _add_window_to_traces(self, X_w, start_w, clusters_w, indices_w):
        """Add points of a window to the traces."""
        
        # list of all clusterIDs in the window
        clusterIDs = np.unique(clusters_w)
        
        # list all traceIDs associated with active traces
        active_traceIDs = np.array([t for t in self._traces.keys() if self._traces[t]['active']])
        
        # Add points of the noise cluster to the _traces dictionary
        if -1 in clusterIDs:
            if -1 in self._traces.keys():
                self._merge_to_trace(X_w[clusters_w == -1], start_w[clusters_w == -1], indices_w[clusters_w == -1], traceID=-1)
            else:
                self._add_new_trace(X_w[clusters_w == -1], start_w[clusters_w == -1], indices_w[clusters_w == -1], traceID=-1)
        
        # Remove noise from clusterIDs and traceIDs    
        clusterIDs = clusterIDs[clusterIDs != -1]
        active_traceIDs = active_traceIDs[active_traceIDs != -1]
        
        # If there are no active traces, create a new trace for each cluster
        if len(active_traceIDs) == 0:
            for c in clusterIDs:
                self._add_new_trace(X_w[clusters_w == c], start_w[clusters_w == c], indices_w[clusters_w == c], traceID=c)
            # exit the function
            return None
        
        # Compute the distance between the center of mass of each cluster and each last-window active trace
        dists = np.zeros((len(clusterIDs), len(active_traceIDs))).astype(float)  # All cluster-trace distances
        for i, c in enumerate(clusterIDs):  # Loop over the clusters
            for j, t in enumerate(active_traceIDs):  # Loop over the traces
                d = np.mean(cdist(X_w[clusters_w == c], self._traces[t]['last_window_points']))
                dists[i, j] = d
        
        # Find the closest cluster-trace pairs iteratively
        while len(clusterIDs) > 0 and len(active_traceIDs) > 0:
            
            # If the minimum distance is larger than delta, stop
            if np.min(dists) > self.delta:
                break

            # Find the best cluster-trace pair
            i, j = np.unravel_index(np.argmin(dists), dists.shape)  # position of the minimum in dists
            d_best = dists[i, j]
            c_best = clusterIDs[i]
            t_best = active_traceIDs[j]
            # Check nothing is wrong
            assert d_best == np.min(dists)
            assert d_best <= self.delta
            assert c_best in clusterIDs
            assert t_best in active_traceIDs
            # Merge the cluster into the trace
            self._merge_to_trace(X_w[clusters_w == c_best], start_w[clusters_w == c_best], indices_w[clusters_w == c_best], traceID=t_best)
            # Remove the cluster and the trace from the scores matrix
            dists = np.delete(dists, i, axis=0)
            dists = np.delete(dists, j, axis=1)
            clusterIDs = np.delete(clusterIDs, i)
            active_traceIDs = np.delete(active_traceIDs, j)
        
        # If there are traces left, increment their missing_windows_count
        for t in active_traceIDs:
            self._traces[t]['missing_windows_count'] += 1
        
        # Mark as inactive traces with missing_windows_count > max_missing_windows
        for t in self._traces:
            if self._traces[t]['missing_windows_count'] == self.max_missing_windows:
                self._traces[t]['active'] = False
        
        # If there are clusters left, create a new trace for each cluster
        for c in clusterIDs:
            self._add_new_trace(X_w[clusters_w == c], start_w[clusters_w == c], indices_w[clusters_w == c], traceID=c)
    
    def _add_new_trace(self, points: np.ndarray, starts: np.ndarray, indices: np.ndarray, traceID: int = None):
        """Add a new trace."""
        self._check_input(points, starts)
        self._check_input(points, indices)
        # If the provided traceID is -1, raise an error if -1 already exists
        if traceID == -1 and -1 in self._traces:
            raise ValueError("TraceID -1 already exists and can't be assigned to a new trace")
        # If the provided traceID is None or already exists, assign a new traceID as the next available positive integer
        if traceID is None or traceID in self._traces:
            traceID = int(max(self._traces.keys())) + 1
        self._traces[traceID] = {'points': points,
                                 'starts': starts,
                                 'indices': indices,
                                 'last_window_points': points,
                                 'last_window_starts': starts,
                                 'missing_windows_count': 0,
                                 'active': True}
    
    def _merge_to_trace(self, points: np.ndarray, starts: np.ndarray, indices: np.ndarray, traceID: int):
        """Merge points to an existing trace."""
        self._check_input(points, starts)
        self._check_input(points, indices)
        assert self._traces[traceID]['active'], "TraceID {} is not active".format(traceID)
        assert traceID in self._traces, "TraceID {} does not exist".format(traceID)
        self._traces[traceID]['points'] = np.concatenate((self._traces[traceID]['points'], points))
        self._traces[traceID]['starts'] = np.concatenate((self._traces[traceID]['starts'], starts))
        self._traces[traceID]['indices'] = np.concatenate((self._traces[traceID]['indices'], indices))
        self._traces[traceID]['last_window_points'] = points
        self._traces[traceID]['last_window_starts'] = starts
        # self._traces[traceID]['missing_windows_count'] = 0
    
    def _reorder_traces(self):
        """Reorder and rename the keys in _traces to be -1, 1, 2, 3, ...
            -1 represents the noise.
            1 represents the valid trace with the largest number of points,
            2 is the second one, and so on."""
        # Extract noise trace if it exists
        noise_trace = None
        if -1 in self._traces:
            noise_trace = self._traces[-1]
            del self._traces[-1]  # remove it, it will be added back later
        # Sort traces based on number of points (in descending order)
        sorted_traces = dict(sorted(self._traces.items(),
                                    key=lambda item: len(item[1]['points']),
                                    reverse=True))
        # Rename the keys to be 1, 2, 3, ...
        reordered_traces = {}
        for idx, (old_id, trace_data) in enumerate(sorted_traces.items(), start=1):
            reordered_traces[idx] = trace_data
        # Add noise trace back with key -1, if it was extracted earlier
        if noise_trace:
            reordered_traces[-1] = noise_trace
        # Update the _traces attribute
        self._traces = reordered_traces
    
    def _traces_to_labels(self, X):
        """Convert the traces to a numpy array as in the output of DBSCAN."""
        # Initialize labels as -1 (noise)
        labels = np.full(X.shape[0], -1)
        # Fill labels with the traceIDs
        for t in self._traces:
            labels[self._traces[t]['indices']] = t
        return labels
    
    
    @staticmethod
    def _check_input(X: np.ndarray, start: np.ndarray):
        """Check that X is a float 2D array of shape (n_samples, 3)
        and that start is an int 1D array of shape (n_samples,)."""
        assert len(X.shape) == 2
        assert len(start.shape) == 1
        assert X.shape[0] == start.shape[0]
        assert X.shape[1] == 3
        assert X.shape[0] > 0
        assert X.dtype == float
        assert start.dtype == int
