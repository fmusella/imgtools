import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist


class DBSCANClusterCollection:
    """ A class that contains the clusters obtained from DBSCAN.
    This class is used to convert the output of DBSCAN to a dictionary of clusters,
    so that the format matches the one of the traces in TraceCollection.
    
    Attributes
    ----------
    data: dict{
        'points': np.ndarray of shape (n_points, 3) containing the coordinates of the points.
        'starts': np.ndarray of shape (n_points,) containing the start position of each point.
        'indices': np.ndarray of shape (n_points,) containing the indices - in the original data - of each point.
        'npoint': int, number of points in the cluster.
        'CoM': np.ndarray of shape (3,) containing the center of mass of the cluster.
    }
    """
    
    def __init__(self, X: np.ndarray, start: np.ndarray, indices: np.ndarray, db_labels: np.ndarray):
        
        # Check that the input is valid
        check_input(X, start)
        check_input(X, indices)
        check_input(X, db_labels)
        
        # Convert the output of DBSCAN to a dictionary of clusters
        self.data = self._from_numpys_to_dict(X, start, indices, db_labels)
    
    def _from_numpys_to_dict(self, X: np.ndarray, start: np.ndarray, indices: np.ndarray, db_labels: np.ndarray):
        """ Converts the data from the DBSCAN numpy output to a dictionary of clusters.

        Args:
            X (np.ndarray): np.ndarray of shape (n_points, 3) containing the coordinates of the points.
            start (np.ndarray): np.ndarray of shape (n_points,) containing the start position of each point.
            indices (np.ndarray): np.ndarray of shape (n_points,) containing the indices - in the original data - of each point.
            db_labels (np.ndarray): np.ndarray of shape (n_points,) containing the DBSCAN labels of each point.

        Returns:
            data (dict): dictionary of clusters, as defined in the class attributes.
        """
        
        # Initialize data dictionary
        data = {}
        
        # Take the unique clusterIDs
        clusterIDs = np.unique(db_labels)
        
        for c in clusterIDs:
            
            # Check that the input is valid
            check_input(X[db_labels == c], start[db_labels == c])
            check_input(X[db_labels == c], indices[db_labels == c])
            
            # Add the cluster to the clusters dictionary
            data[c] = {
                'points': X[db_labels == c],
                'starts': start[db_labels == c],
                'indices': indices[db_labels == c],
                'npoint': np.sum(db_labels == c),
                'CoM': np.mean(X[db_labels == c], axis=0)
                }
        
        return data
    
    def get_clusterIDs(self):
        """Returns a list of all clusterIDs."""
        return list(self.data.keys())


class TraceCollection:
    """ Manages a collection of traces, i.e. a collection of chromosome 3D points
    that are close enough to be considered a consisted copy of a chromosome.
    
    Attributes
    ----------
    data: dict{
        'points': np.ndarray of shape (n_points, 3) containing the coordinates of the points.
        'starts': np.ndarray of shape (n_points,) containing the start position of each point.
        'indices': np.ndarray of shape (n_points,) containing the indices - in the original data - of each point.
        'npoint': int, number of points in the trace.
        'CoM': np.ndarray of shape (3,) containing the center of mass of the trace.
        'last_window_points': np.ndarray of shape (n_points_last, 3) containing the coordinates of the points in the last window.
        'last_window_starts': np.ndarray of shape (n_points_last,) containing the start position of each point in the last window.
        'last_window_indices': np.ndarray of shape (n_points_last,) containing the indices - in the original data - of each point in the last window.
        'last_window_npoint': int, number of points in the last window.
        'last_window_CoM': np.ndarray of shape (3,) containing the center of mass of the last window.
        'missing_windows_count': int, number of total missing windows.
        'active': bool, True if the trace is active (i.e. it can be merged with new windows)
    }
    """
    
    def __init__(self):
        self.data = {}
    
    def reset(self):
        """ Resets the data to an empty state.
        """
        self.data = {}
    
    def sort(self):
        """ Sorts the traces by number of points (in descending order).
        """
        
        # Extract noise trace if it exists, and remove it from the _traces attribute
        # (it will be added back later)
        noise_trace = None
        if -1 in self.data:
            noise_trace = self.data[-1]
            del self.data[-1]
        
        # Sort traces based on number of points (in descending order)
        sorted_traces = dict(sorted(self.data.items(),
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
        self.data = reordered_traces
    
    def get_traceIDs(self):
        """ Returns a list of all traceIDs.
        """
        return list(self.data.keys())
    
    def get_n_traces(self):
        """ Returns the number of traces.
        """
        return len(self.data)
    
    def get_n_valid_traces(self):
        """ Returns the number of valid traces.
        """
        return len([t for t in self.data if t != -1])
    
    def add_new_trace(self, points: np.ndarray, starts: np.ndarray, indices: np.ndarray, traceID: int = None):
        """ Adds a new trace to the collection.

        Args:
            points (np.ndarray): array of shape (n_points, 3) containing the coordinates of the points.
            starts (np.ndarray): array of shape (n_points,) containing the start position of each point.
            indices (np.ndarray): array of shape (n_points,) containing the indices - in the original data - of each point.
            traceID (int, optional): Desired traceID. If None, the next available positive integer is assigned.
                                     If traceID is -1, an error is raised if -1 already exists.

        Raises:
            ValueError: If traceID is -1 and -1 already exists.
        """
        
        # Check that the input is valid
        check_input(points, starts)
        check_input(points, indices)
        
        # If the provided traceID is -1, raise an error if -1 already exists
        if traceID == -1 and -1 in self.data:
            raise ValueError("TraceID -1 already exists and can't be assigned to a new trace")
        
        # If the provided traceID is None or already exists, assign a new traceID as the next available positive integer
        if traceID is None or traceID in self.data:
            traceID = int(max(self.data.keys())) + 1
        
        # Add the new trace to the data dictionary
        self.data[traceID] = {
            'points': points,
            'starts': starts,
            'indices': indices,
            'npoint': len(points),
            'CoM': np.mean(points, axis=0),
            'last_window_points': points,
            'last_window_starts': starts,
            'last_window_indices': indices,
            'last_window_npoint': len(points),
            'last_window_CoM': np.mean(points, axis=0),
            'missing_windows_count': 0,
            'active': True
        }
    
    def append_to_trace(self, points: np.ndarray, starts: np.ndarray, indices: np.ndarray, traceID: int):
        """ Appends points to an existing trace.

        Args:
            points (np.ndarray): array of shape (n_points, 3) containing the coordinates of the points.
            starts (np.ndarray): array of shape (n_points,) containing the start position of each point.
            indices (np.ndarray): array of shape (n_points,) containing the indices - in the original data - of each point.
            traceID (int): ID of the trace to merge to.

        Raises:
            ValueError: If traceID does not exist.
            ValueError: If traceID is not active.
        """
        
        # Check that the input is valid
        check_input(points, starts)
        check_input(points, indices)
        
        # Check that the trace exists
        if traceID not in self.data:
            raise ValueError("TraceID {} does not exist".format(traceID))
        
        # Check that the trace is active
        if not self.data[traceID]['active']:
            raise ValueError("TraceID {} is not active".format(traceID))
        
        # Append the points to the trace
        self.data[traceID]['points'] = np.concatenate((self.data[traceID]['points'], points))
        self.data[traceID]['starts'] = np.concatenate((self.data[traceID]['starts'], starts))
        self.data[traceID]['indices'] = np.concatenate((self.data[traceID]['indices'], indices))
        self.data[traceID]['npoint'] += len(points)
        self.data[traceID]['CoM'] = np.mean(self.data[traceID]['points'], axis=0)
        self.data[traceID]['last_window_points'] = points
        self.data[traceID]['last_window_starts'] = starts
        self.data[traceID]['last_window_indices'] = indices
        self.data[traceID]['last_window_npoint'] = len(points)
        self.data[traceID]['last_window_CoM'] = np.mean(points, axis=0)

    def add_new_trace_from_cluster(self, clusters: DBSCANClusterCollection, clusterID: int, traceID: int = None):
        """ Adds a new trace to the collection from a cluster.

        Args:
            clusters (DBSCANClusterCollection): clusters obtained from DBSCAN.
            clusterID (int)
            traceID (int, optional): Desired traceID. If None, the next available positive integer is assigned.
                                     If traceID is -1, an error is raised if -1 already exists.

        Raises:
            ValueError: If clusterID does not exist.
        """
        
        # Check that clusterID exists
        if clusterID not in clusters.data:
            raise ValueError("ClusterID {} does not exist".format(clusterID))
        
        # Add the new trace to the data dictionary
        self.add_new_trace(
            clusters.data[clusterID]['points'],
            clusters.data[clusterID]['starts'],
            clusters.data[clusterID]['indices'],
            traceID=traceID
        )
    
    def append_to_trace_from_cluster(self, clusters: DBSCANClusterCollection, clusterID: int, traceID: int):
        """ Appends points to an existing trace from a cluster.

        Args:
            clusters (DBSCANClusterCollection): clusters obtained from DBSCAN.
            clusterID (int)
            traceID (int)

        Raises:
            ValueError: If clusterID does not exist.
        """
        
        # Check that clusterID exists
        if clusterID not in clusters.data:
            raise ValueError("ClusterID {} does not exist".format(clusterID))
        
        # Append the points to the trace
        self.append_to_trace(
            clusters.data[clusterID]['points'],
            clusters.data[clusterID]['starts'],
            clusters.data[clusterID]['indices'],
            traceID=traceID
        )
    
    def merge_traces(self, traceID_1: int, traceID_2: int):
        
        # Check that the traces exist
        if not traceID_1 in self.data:
            raise ValueError("TraceID {} does not exist".format(traceID_1))
        if not traceID_2 in self.data:
            raise ValueError("TraceID {} does not exist".format(traceID_2))

        # Add data from traceID_2 to traceID_1
        self.append_to_trace(
            self.data[traceID_2]['points'],
            self.data[traceID_2]['starts'],
            self.data[traceID_2]['indices'],
            traceID=traceID_1
        )
        
        # Remove traceID2
        del self.data[traceID_2]
    
    def compute_trace_cluster_distance(self, clusters: DBSCANClusterCollection, clusterID: int, traceID: int):
        
        # Check that the cluster and the trace exist
        if not clusterID in clusters.data:
            raise ValueError("ClusterID {} does not exist".format(clusterID))
        if not traceID in self.data:
            raise ValueError("TraceID {} does not exist".format(traceID))
        
        # Choose the method to compute the distance
        method = 1
        
        # Compute the distance in the chosen way
        if method == 1:
            # Distance between the center of mass of the cluster and the center of mass of the trace
            d = np.sqrt(np.sum((clusters.data[clusterID]['CoM'] - self.data[traceID]['CoM'])**2))
        elif method == 2:
            # Distance between the center of mass of the cluster and the center of mass of the last window of the trace
            d = np.sqrt(np.sum((clusters.data[clusterID]['CoM'] - self.data[traceID]['last_window_CoM'])**2))
        elif method == 3:
            # Minimum distance between the points of the cluster and the points of the trace
            d = np.min(cdist(clusters.data[clusterID]['points'], self.data[traceID]['points']))
        elif method == 4:
            # Minimum distance between the points of the cluster and the points of the last window of the trace
            d = np.min(cdist(clusters.data[clusterID]['points'], self.data[traceID]['last_window_points']))
        else:
            raise ValueError("Invalid method {}".format(method))
        
        return d
    
    def iterative_merging(self, proximity_length: int, overlap_threshold: float, distance_threshold: float):
        """ Iteratively merge traces that satisfy the following conditions:
            1) the overlap score is smaller than a threshold, i.e. the genomic content of the two traces is very different
            2) the minimum trace-to-trace distance is smaller than a threshold, i.e. the traces are close enough.

        Args:
            overlap_threshold (float): maximum overlap score to merge two traces.
            distance_threshold (float): maximum trace-to-trace spatial distance to merge two traces.
        """
        
        while True:
            
            # Initialize pair_found to False: becomes True if a pair of traces is merged
            pair_found = False
            
            for t1 in self.data:
                for t2 in self.data:
                    
                    # avoid noise trace
                    if t1 == -1 or t2 == -1:
                        continue
                    # avoid symmetric pairs
                    if t2 <= t1:
                        continue
                    
                    # Compute overlap score
                    overlap = get_overlap_score(self.data[t1]['starts'], self.data[t2]['starts'], proximity_length=proximity_length)
                    # Compute min trace-to-trace distance
                    distance = np.min(cdist(self.data[t1]['points'], self.data[t2]['points']))
                    
                    # If the separaion score is large enough and the distance is small enough, merge the traces
                    if overlap < overlap_threshold and distance <= distance_threshold:
                        
                            self.merge_traces(t1, t2)
                            pair_found = True
                            break  # exit the t2 loop
                
                if pair_found:
                    break  # exit the t1 loop if a pair was found
                
            if not pair_found:
                break  # exit the while loop if no pair was found
    
    def convert_to_labels(self, npoint: int):
        """ Convert the traces to a numpy array as in the output of DBSCAN.

        Args:
            npoint (int): number of points in the original data.

        Returns:
            labels (np.ndarray): array of shape (npoint,) containing the traceID of each point.
        """
        
        # Initialize labels as -1 (noise)
        labels = np.full(npoint, -1)
        
        for t in self.data:
            
            # Get the indices of the points in the trace
            # and set their label to the traceID
            labels[self.data[t]['indices']] = t
        
        return labels
    
    

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
    merging_proximity_length : int
        length of the proximity window for the merging procedure, in bp.
    merging_overlap_threshold : float
        maximum overlap score to merge two traces for the merging procedure.
    merging_distance_threshold : float
        maximum trace-to-trace spatial distance to merge two traces for the merging procedure.
    
    Attributes
    ----------
    _traces : TraceCollection
        Collection of traces, as defined in TraceCollection class.
    """
    
    def __init__(
        self, dbscan_eps: float = 1.,
        dbscan_min_samples: int = 1,
        window_size: int = 10 * 10**6,
        delta: float = 3.,
        merging_proximity_length: int = 2 * 10**6,
        merging_overlap_threshold: float = 0.3,
        merging_distance_threshold: float = 1.
    ):
        # Parameters
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.window_size = window_size
        self.delta = delta
        self.merging_proximity_length = merging_proximity_length
        self.merging_overlap_threshold = merging_overlap_threshold
        self.merging_distance_threshold = merging_distance_threshold
        # Internal variables
        self._traces = TraceCollection()
    
    def fit(self, X: np.ndarray, start: np.ndarray):
        """Perform the Genomic Iterative DBSCAN algorithm.

        Args:
            X (np.ndarray of float): array of shape (n_samples, 3) containing
                the coordinates of the points.
            start (np.ndarray of int): array of shape (n_samples,) containing
                the start position of each point.
        """

        self._traces.reset()  # reset the _traces to an empty state
        
        check_input(X, start)  # check that X and start are valid
        
        windows = start // self.window_size  # compute window labels
        
        # Loop over windows to perform Genomic Iterative DBSCAN
        for w in np.unique(windows):
            
            # Take data of the current window
            X_w = X[windows == w]
            start_w = start[windows == w]
            indices_w = np.where(windows == w)[0]
            
            # Apply DBSCAN
            db = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples).fit(X_w)
            
            # Convert the output of DBSCAN to a DBSCANClusterCollection object
            clusters = DBSCANClusterCollection(X_w, start_w, indices_w, db.labels_)
            
            # Run the code that assigns these new clusters to the existing traces
            self._fill_traces(clusters)
        
        # Run the code that recursively merges traces
        self._traces.iterative_merging(
            self.merging_proximity_length,
            self.merging_overlap_threshold,
            self.merging_distance_threshold
        )
        
        # Sort the traces by number of points
        self._traces.sort()
        
        # Convert traces to a numpy array as in the output of DBSCAN and save it in the labels_ attribute
        self.labels_ = self._traces.convert_to_labels(X.shape[0])
    
    
    def _fill_traces(self, clusters: DBSCANClusterCollection):
        """ Fill the traces with the clusters data.
        
        Noise and valid clusters are treated differently:
        - noise clusters are assigned to the noise trace, regardless of the distance
        - valid clusters are assigned to the existing traces with the min-distance procedure

        Args:
            clusters (DBSCANClusterCollection): clusters obtained from DBSCAN.
        """
        
        # Add the noise cluster - if present - to the noise trace
        self._fill_noise(clusters)

        # If there are no valid traceIDs (i.e. only noise is present in _traces), create a new trace for each cluster
        if self._traces.get_n_valid_traces() == 0:
            self._initialize_valids(clusters)
            return None  # exit the function
        
        # Otherwise, append the clusters to the existing traces with the min-distance procedure
        self._assign_valids_by_min_dist(clusters)
    
    def _fill_noise(self, clusters: DBSCANClusterCollection):
        """ Includes the noise cluster to the noise trace, or create a new noise trace if needed.

        Args:
            clusters (DBSCANClusterCollection): clusters obtained from DBSCAN.
        """
        
        # Exit if there is no noise cluster
        if -1 not in clusters.get_clusterIDs():
            return None
        
        # If there is no noise trace, create a new trace for the noise cluster
        if -1 not in self._traces.get_traceIDs():
            self._traces.add_new_trace_from_cluster(clusters, clusterID=-1, traceID=-1)
            return None
        
        # Otherwise, append the noise cluster to the noise trace
        self._traces.append_to_trace_from_cluster(clusters, clusterID=-1, traceID=-1)
    
    def _initialize_valids(self, clusters: DBSCANClusterCollection):
        """ Initialize the valid traces in _traces with the clusters.

        Args:
            clusters (DBSCANClusterCollection): clusters obtained from DBSCAN.
        """
        
        for clusterID in clusters.get_clusterIDs():
            
            # Skip noise
            if clusterID == -1:
                continue
            
            # Create a new trace for each cluster
            self._traces.add_new_trace_from_cluster(clusters, clusterID=clusterID, traceID=clusterID)
    
    def _assign_valids_by_min_dist(self, clusters: DBSCANClusterCollection):
        """ Assign the clusters to the existing traces with the min-distance procedure.
        
        Iteratively, it finds the closest cluster-trace pair and appends the cluster to the trace.
        
        Clusters and traces are removed from the available pool once they are assigned: it means
        that two clusters cannot be assigned to the same trace, even if they are close enough.

        Args:
            clusters (DBSCANClusterCollection): clusters obtained from DBSCAN.
        """
        
        # get array of all clusterIDs and all traceIDs
        clusterIDs = np.array(clusters.get_clusterIDs())
        traceIDs = np.array(self._traces.get_traceIDs())
        
        # Remove noise from clusterIDs and traceIDs
        clusterIDs = clusterIDs[clusterIDs != -1]
        traceIDs = traceIDs[traceIDs != -1]
        
        # Compute the distance between each cluster in clusterIDs and each trace in traceIDs
        dists = np.zeros((len(clusterIDs), len(traceIDs))).astype(float)
        for i, c in enumerate(clusterIDs):
            for j, t in enumerate(traceIDs):
                d = self._traces.compute_trace_cluster_distance(clusters, clusterID=c, traceID=t)
                dists[i, j] = d
        
        # Find the closest cluster-trace pairs iteratively
        while len(clusterIDs) > 0 and len(traceIDs) > 0:
            
            # If the minimum distance is larger than delta, stop
            if np.min(dists) > self.delta:
                break

            # Find the best cluster-trace pair
            i, j = np.unravel_index(np.argmin(dists), dists.shape)  # position of the minimum in dists
            c_best = clusterIDs[i]
            t_best = traceIDs[j]
            
            # Append the cluster into the trace
            self._traces.append_to_trace_from_cluster(clusters, clusterID=c_best, traceID=t_best)
            
            # Remove the cluster and the trace from the available pool (dists, clusterIDs, traceIDs)
            dists = np.delete(dists, i, axis=0)
            dists = np.delete(dists, j, axis=1)
            clusterIDs = np.delete(clusterIDs, i)
            traceIDs = np.delete(traceIDs, j)
        
        # If there are clusters left, create a new trace for each of them
        for c in clusterIDs:
            self._traces.add_new_trace_from_cluster(clusters, clusterID=c, traceID=c)



def check_input(X: np.ndarray, y: np.ndarray):
    """ Check that X is a float 2D array of shape (n_samples, 3)
    and that y is an int 1D array of shape (n_samples,).
    """
    
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")
    if not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy array")
    
    if len(X.shape) != 2:
        raise ValueError("X must be a 2D array")
    if len(y.shape) != 1:
        raise ValueError("y must be a 1D array")
    
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of elements")
    if X.shape[0] == 0:
        raise ValueError("X and y must have at least one element")
    if X.shape[1] != 3:
        raise ValueError("X must have 3 columns")
    
    if X.dtype != float:
        raise ValueError("X must have dtype float")
    if y.dtype != int:
        raise ValueError("y must have dtype int")


def get_overlap_score(x1: np.ndarray, x2: np.ndarray, proximity_length: float):
    """Compute the overlap score between two sets of points with a given 1D coordinate.
    
    The score is equal to the maximum, among the two sets, of the percentage of points that are proximal to a point in the other trace.

    Args:
        x1 (np.ndarray): array of shape (n_points_1,) containing the 1D coordinate position of each point in set 1.
        x2 (np.ndarray): array of shape (n_points_2,) containing the 1D coordinate position of each point in set 2.

    Returns:
        overlap_score (float): overlap score between the two sets of points.
    """
    
    # Get a matrix of |x[i] - x[j]| for ever point i in set 1 and every point j in set 2
    dist = np.abs(x1[:, np.newaxis] - x2[np.newaxis, :])
    
    # Get the matrix that is 1 if |x[i] - x[j]| < proximity_length and 0 otherwise
    mask = dist < proximity_length
    
    # Get the number of points in set 1 that are proximal to a point in set 2 and vice versa
    nprox1 = np.sum(np.any(mask, axis=1))
    nprox2 = np.sum(np.any(mask, axis=0))
    
    # Get the overlap score, equal to the maximum, among the two sets, of the percentage of points that are proximal to a point in the other set
    overlap_score = max(nprox1 / len(x1), nprox2 / len(x2))
    
    return overlap_score
