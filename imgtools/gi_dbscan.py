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
    merging_separation_threshold : float
        minimum genomic separation score to merge two traces.
    merging_distance_threshold : float
        maximum trace-to-trace spatial distance to merge two traces.
    
    Attributes
    ----------
    _traces : dict
        dictionary of traces, indexed by trace id, with this structure:
            _traces[traceID] = {
                'points': np.ndarray of shape (n_points, 3),  # 3D coordinates of the points in the trace
                'starts': np.ndarray of shape (n_points,),  # start position of the points in the trace
                'indices': np.ndarray of shape (n_points,),  # indices of the points in the trace
                'npoint': int,  # number of points in the trace
                'CoM': np.ndarray of shape (3,),  # center of mass of the points in the trace
                'windowIDs': set of int,  # set of windowIDs in which the trace is present
                'last_window_points': np.ndarray of shape (n_points_last_window, 3),
                'last_window_starts': np.ndarray of shape (n_points_last_window,),
                'last_window_indices': np.ndarray of shape (n_points_last_window,),
                'last_window_npoint': int,
                'last_window_CoM': np.ndarray of shape (3,),
                'missing_windows_count': int,  # number of consecutive missing windows
                'active': bool  # whether the trace is active or not
            }
    """
    
    def __init__(self, dbscan_eps: float = 1.,
                 dbscan_min_samples: int = 5,
                 window_size: int = 5 * 10**6,
                 delta: float = 3.,
                 max_missing_windows: int = 10,
                 merging_separation_threshold: float = 0.5,
                 merging_distance_threshold: float = 1.):
        # Parameters
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.window_size = window_size
        self.delta = delta
        self.max_missing_windows = max_missing_windows
        self.merging_separation_threshold = merging_separation_threshold
        self.merging_distance_threshold = merging_distance_threshold
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
            
            # Take data of the current window
            X_w = X[windows == w]
            start_w = start[windows == w]
            indices_w = np.where(windows == w)[0]
            
            # Apply DBSCAN
            db = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples).fit(X_w)
            
            # Convert the output of DBSCAN to a dictionary of clusters
            clusters = self._convert_to_clusters(X_w, start_w, indices_w, db.labels_)
            
            # Run the code that assigns these new clusters to the existing traces
            self._run_assignment_to_traces(clusters)
        
        # Run the code that recursively merges traces
        self._run_recursive_merging()
        
        # Reorder the traces by number of points
        self._reorder_traces()
        
        # Convert traces to a numpy array as in the output of DBSCAN and save it in the labels_ attribute
        self.labels_ = self._traces_to_labels(X.shape[0])
    
    
    def _convert_to_clusters(self, X: np.ndarray, start: np.ndarray, indices: np.ndarray, db_labels: np.ndarray):
        """Convert the output of DBSCAN to a dictionary of clusters."""
       
        # Initialize clusters dictionary
        clusters = {}
        
        # Take the unique clusterIDs
        clusterIDs = np.unique(db_labels)
        
        for c in clusterIDs:
            
            # Check that the input is valid
            self._check_input(X[db_labels == c], start[db_labels == c])
            self._check_input(X[db_labels == c], indices[db_labels == c])
            
            # Add the cluster to the clusters dictionary
            clusters[c] = {
                'points': X[db_labels == c],
                'starts': start[db_labels == c],
                'indices': indices[db_labels == c],
                'npoint': np.sum(db_labels == c),
                'CoM': np.mean(X[db_labels == c], axis=0),
                'windowIDs': set(start[db_labels == c] // self.window_size)
                }
        
        return clusters
    
    
    def _run_assignment_to_traces(self, clusters: dict):
        """ Assign the clusters to the existing traces, or create new traces if needed. """
        
        # list all clusterIDs
        clusterIDs = np.array(list(clusters.keys())).astype(int)
        # list all traceIDs associated with active traces
        active_traceIDs = np.array([t for t in self._traces.keys() if self._traces[t]['active']])
        
        # Add points of the noise cluster to the _traces dictionary
        if -1 in clusters:
            if -1 in self._traces.keys():
                self._cluster_to_existing_trace(clusters, clusterID=-1, traceID=-1)
            else:
                self._cluster_to_new_trace(clusters, clusterID=-1, traceID=-1)
        
        # Remove noise from clusterIDs and traceIDs    
        clusterIDs = clusterIDs[clusterIDs != -1]
        active_traceIDs = active_traceIDs[active_traceIDs != -1]
        
        # If there are no active traces, create a new trace for each cluster
        if len(active_traceIDs) == 0:
            for c in clusterIDs:
                self._cluster_to_new_trace(clusters, clusterID=c, traceID=c)
            # exit the function
            return None
        
        # Compute the distance between the center of mass of each cluster and each last-window active trace
        dists = np.zeros((len(clusterIDs), len(active_traceIDs))).astype(float)  # All cluster-trace distances
        for i, c in enumerate(clusterIDs):  # Loop over the clusters
            for j, t in enumerate(active_traceIDs):  # Loop over the traces
                d = self._compute_cluster_trace_distance(clusters, clusterID=c, traceID=t)
                dists[i, j] = d
        
        # Find the closest cluster-trace pairs iteratively
        while len(clusterIDs) > 0:
            
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
            self._cluster_to_existing_trace(clusters, clusterID=c_best, traceID=t_best)
            # Remove the cluster and the trace from the scores matrix
            dists = np.delete(dists, i, axis=0)
            clusterIDs = np.delete(clusterIDs, i)
        
        # If there are traces left, increment their missing_windows_count
        for t in active_traceIDs:
            self._traces[t]['missing_windows_count'] += 1
        
        # Mark as inactive traces with missing_windows_count > max_missing_windows
        for t in self._traces:
            if self._traces[t]['missing_windows_count'] == self.max_missing_windows:
                self._traces[t]['active'] = False
        
        # If there are clusters left, create a new trace for each cluster
        for c in clusterIDs:
            self._cluster_to_new_trace(clusters, clusterID=c, traceID=c)        
    
    def _cluster_to_new_trace(self, clusters: dict, clusterID: int, traceID: int = None):
        """Add a new trace."""
        # If the provided traceID is -1, raise an error if -1 already exists
        if traceID == -1 and -1 in self._traces:
            raise ValueError("TraceID -1 already exists and can't be assigned to a new trace")
        # If the provided traceID is None or already exists, assign a new traceID as the next available positive integer
        if traceID is None or traceID in self._traces:
            traceID = int(max(self._traces.keys())) + 1
        # Add the new trace to the _traces dictionary
        self._traces[traceID] = {
            'points': clusters[clusterID]['points'],
            'starts': clusters[clusterID]['starts'],
            'indices': clusters[clusterID]['indices'],
            'npoint': clusters[clusterID]['npoint'],
            'CoM': clusters[clusterID]['CoM'],
            'windowIDs': clusters[clusterID]['windowIDs'],
            'last_window_points': clusters[clusterID]['points'],
            'last_window_starts': clusters[clusterID]['starts'],
            'last_window_indices': clusters[clusterID]['indices'],
            'last_window_npoint': clusters[clusterID]['npoint'],
            'last_window_CoM': clusters[clusterID]['CoM'],
            'missing_windows_count': 0,
            'active': True
            }
    
    def _cluster_to_existing_trace(self, clusters: dict, clusterID: int, traceID: int):
        """Merge points to an existing trace."""
        assert self._traces[traceID]['active'], "TraceID {} is not active".format(traceID)
        assert traceID in self._traces, "TraceID {} does not exist".format(traceID)
        self._traces[traceID]['points'] = np.concatenate((self._traces[traceID]['points'], clusters[clusterID]['points']))
        self._traces[traceID]['starts'] = np.concatenate((self._traces[traceID]['starts'], clusters[clusterID]['starts']))
        self._traces[traceID]['indices'] = np.concatenate((self._traces[traceID]['indices'], clusters[clusterID]['indices']))
        self._traces[traceID]['npoint'] += clusters[clusterID]['npoint']
        self._traces[traceID]['windowIDs'] = self._traces[traceID]['windowIDs'].union(clusters[clusterID]['windowIDs'])
        self._traces[traceID]['CoM'] = np.mean(self._traces[traceID]['points'], axis=0)
        self._traces[traceID]['last_window_points'] = clusters[clusterID]['points']
        self._traces[traceID]['last_window_starts'] = clusters[clusterID]['starts']
        self._traces[traceID]['last_window_indices'] = clusters[clusterID]['indices']
        self._traces[traceID]['last_window_npoint'] = clusters[clusterID]['npoint']
        self._traces[traceID]['last_window_CoM'] = clusters[clusterID]['CoM']
        # If I uncomment the following line, it resets the missing_windows_count every time a trace is merged
        # This means that a trace is marked as inactive only if there are consecutive missing windows
        # Keeping it commented means that a trace is marked as inactive if there a total number of missing windows >= max_missing_windows,
        # even if they are not consecutive
        # self._traces[traceID]['missing_windows_count'] = 0
    
    def _compute_cluster_trace_distance(self, clusters: dict, clusterID: int, traceID: int):
        """Compute the distance between the cluster and the trace."""
        # Make sure the cluster and the trace exist
        assert clusterID in clusters
        assert traceID in self._traces
        # Choose the method to compute the distance
        method = 1
        # Compute the distance in the chosen way
        if method == 1:
            # Distance between the center of mass of the cluster and the center of mass of the trace
            d = np.sqrt(np.sum((clusters[clusterID]['CoM'] - self._traces[traceID]['CoM'])**2))
        elif method == 2:
            # Distance between the center of mass of the cluster and the center of mass of the last window of the trace
            d = np.sqrt(np.sum((clusters[clusterID]['CoM'] - self._traces[traceID]['last_window_CoM'])**2))
        elif method == 3:
            # Minimum distance between the points of the cluster and the points of the trace
            d = np.min(cdist(clusters[clusterID]['points'], self._traces[traceID]['points']))
        elif method == 4:
            # Minimum distance between the points of the cluster and the points of the last window of the trace
            d = np.min(cdist(clusters[clusterID]['points'], self._traces[traceID]['last_window_points']))
        else:
            raise ValueError("Invalid method {}".format(method))
        return d
    
    
    def _run_recursive_merging(self):
        """ Recursively merge traces that satisfy the following conditions:
            1) the separation score is larger than a threshold, i.e. the genomic content of the two traces is very different
            2) the minimum trace-to-trace distance is smaller than a threshold, i.e. the traces are close enough.
        """
        
        while True:
            
            # Initialize pair_found to False: becomes True if a pair of traces is merged
            pair_found = False
            
            for t1 in self._traces:
                for t2 in self._traces:
                    
                    # avoid noise trace
                    if t1 == -1 or t2 == -1:
                        continue
                    # avoid symmetric pairs
                    if t2 <= t1:
                        continue
                    
                    # Compute separation score and min trace-to-trace distance
                    separation = self._get_separation_score(t1, t2)
                    distance = np.min(cdist(self._traces[t1]['points'], self._traces[t2]['points']))
                    
                    # If the separaion score is large enough and the distance is small enough, merge the traces
                    if np.abs(separation) >= self.merging_separation_threshold and distance <= self.merging_distance_threshold:
                        
                        self._merge_traces(t1, t2)
                        pair_found = True
                        break  # exit the t2 loop
                
                if pair_found:
                    break  # exit the t1 loop if a pair was found
                
            if not pair_found:
                break  # exit the while loop if no pair was found
    
    def _get_separation_score(self, traceID1: int, traceID2: int):
        """ Compute the separation score between two traces:
        
            sep_score = sum_ij (sign(start[i] - start[j])) / (n1 * n2)
        
        It is ~0 if the genomic content of the two traces is similar,
        while it ~-1 or ~1 if the genomic content of the two traces is very different,
        e.g. traceID1 is almost completely before traceID2 (score ~-1). """
        
        assert traceID1 in self._traces
        assert traceID2 in self._traces
        
        # Get a matrix of start[i] - start[j] for ever point i in traceID1 and every point j in traceID2
        start_diff = self._traces[traceID1]['starts'][:, np.newaxis] - self._traces[traceID2]['starts'][np.newaxis, :]
        
        # Get the separation score
        n1 = self._traces[traceID1]['npoint']
        n2 = self._traces[traceID2]['npoint']
        sep_score = np.sum(np.sign(start_diff)) / (n1 * n2)
        
        return sep_score
    
    def _merge_traces(self, traceID1: int, traceID2: int):
        """Merge two traces."""
        
        assert traceID1 in self._traces
        assert traceID2 in self._traces
        
        # Add data from traceID2 to traceID1
        self._traces[traceID1]['points'] = np.concatenate((self._traces[traceID1]['points'], self._traces[traceID2]['points']))
        self._traces[traceID1]['starts'] = np.concatenate((self._traces[traceID1]['starts'], self._traces[traceID2]['starts']))
        self._traces[traceID1]['indices'] = np.concatenate((self._traces[traceID1]['indices'], self._traces[traceID2]['indices']))
        self._traces[traceID1]['npoint'] += self._traces[traceID2]['npoint']
        self._traces[traceID1]['windowIDs'] = self._traces[traceID1]['windowIDs'].union(self._traces[traceID2]['windowIDs'])
        self._traces[traceID1]['CoM'] = np.mean(self._traces[traceID1]['points'], axis=0)
        
        # Remove traceID2
        del self._traces[traceID2]


    def _reorder_traces(self):
        """Reorder and rename the keys in _traces to be -1, 1, 2, 3, ...
            -1 represents the noise.
            1 represents the valid trace with the largest number of points,
            2 is the second one, and so on."""
            
        # Extract noise trace if it exists, and remove it from the _traces attribute
        # (it will be added back later)
        noise_trace = None
        if -1 in self._traces:
            noise_trace = self._traces[-1]
            del self._traces[-1]
        
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
    
    
    def _traces_to_labels(self, npoint: int):
        """Convert the traces to a numpy array as in the output of DBSCAN."""
        
        # Initialize labels as -1 (noise)
        labels = np.full(npoint, -1)
        
        for t in self._traces:
            
            if t == -1:
                continue
            
            # Get the indices of the points in the trace
            # and set their label to the traceID
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
