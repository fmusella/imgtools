import os
import h5py
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score, confusion_matrix, roc_curve
from ..scf import SingleCellFeature
from ..scf import scf_utils
from .repliseq import SimulatedRepliSeqExperiment
from .. import utils


class SimulatedSingleCellRepliSeqExperiment:
    """ Class to extract single-cell replication states (analogous to single-cell Repli-seq)
    from a SingleCellFeature (SCF) object and a SimulatedRepliSeqExperiment object.
    
    This class implements a machine-learning approach:
        - First, it extracts the features required for learning.
          Required features are:
            * 'chrom' (the chromosome of each locus)
            * 'start' (the genomic start position of each locus)
            from SCF:
            * 'spotcount' (the number of detected spots per locus)
            * 'chrom' (the chromosome of each locus)
            * 'genomic_start' (the genomic start position of each locus)
            * 'z' (the z-coordinate of the loci)
            * 'zones' (the nuclear zones of the loci)
          Other features can be included as well.
          Non-categorical features are smoothed using a sliding window average, specified by the user.
        - Then, it trains an XGBoost classifier to distinguish loci from G1 and G2 cells,
          which are known to be unreplicated and replicated, respectively.
          Each chromosome is modeled separately.
        - Finally, it predicts the replication state of loci in S-phase cells.
          XGBoost returns a probability for each locus to be replicated.
    
    The pipeline also checks that the method curates genomic and spatial detection biases,
    by calculating AUC scores for different RT quantiles, z quantiles and nuclear zones.
    
    The arrays produced to create the models are stored in an HDF5 file, specified at initialization.
    This file is also used to store the results of the predictions, i.e. the single-cell replication probabilities.
    
    The trained models and validation metrics are stored in a pickle file, specified when training the models.
    
    ----------
    Attributes:
        h5_name (str): name of the HDF5 file.
        h5 (h5py.File): HDF5 file object.
    
    """
    
    def __init__(self, h5_name: str, mode: str):
        """ Initialize the SimulatedSingleCellRepliSeqExperiment object.

        Args:
            h5_name (str): name of the HDF5 file to store the data.
            mode (str): mode to open the HDF5 file.
        """
        
        # Extend the name with its absolute path
        h5_name = os.path.abspath(h5_name)
        # Check that file has a valid path
        if not os.path.exists(os.path.dirname(h5_name)):
            raise FileNotFoundError("The path of the HDF5 file does not exist.")
        # Store the name of the HDF5 file
        self.h5_name = h5_name
        # Read / create the HDF5 file
        self.h5 = h5py.File(h5_name, mode=mode)
    
    def close(self):
        """ Close the HDF5 file. """
        self.h5.close()

    
    def feature_extraction(
        self,
        scf: SingleCellFeature,
        simrep: SimulatedRepliSeqExperiment,
        scf_features: list,
        win_size: float,
        nquants: int
    ):
        """ Extract the feature data from the SCF and the SimulatedRepliSeqExperiment data.
        
        Every data is first parsed into a 3D array of shape (ncells, nloci, ncopies),
        and then flattened to a 1D array suitable for machine learning.
        
        Non-categorical features are quantized into 'nquants' quantiles (except for 'spotcount'),
        then smoothed using a sliding window average of size 'win_size' (in base pairs).
        
        The following data are extracted and stored in the HDF5 file:
            - Auxiliary data:
                * 'indices': the (cell, locus, copy) indices to convert any flattened matrix back to 3D shape.
                * 'cellIDs': the cell IDs.
                * 'states': the cell states (G1, S, G2).
                * 'ts': the pseudo-times of each cell.
                * 'chroms': the chromosome of each locus.
            - Feature data:
                * 'start': the genomic start position of each locus.
                * 'spotcount': the number of detected spots per locus.
                * 'z': the z-coordinate of each locus.
                * 'zones': the nuclear zones of each locus.
                * [other SCF features]
        
        It also separately extracts the following data for genomic and spatial bias validation:
            * 'rt': the replication timing of each locus.
            * 'z': the z-coordinate of each locus.
            * 'zones': the nuclear zones of each locus.

        Args:
            scf (SingleCellFeature)
            simrep (SimulatedRepliSeqExperiment)
            scf_features (list): list of SCF features to include. Must include 'spotcount', 'z' and 'zones'.
            win_size (float): size of the sliding window average (in base pairs).
            nquants (int, optional): Number of quantiles to use for quantization of the SCF features.
                If None, no quantization is performed.
        """
        
        # --- CHECK THAT THE REQUIRED FEATURES ARE PRESENT ---
        required_features = ['spotcount', 'z', 'zones']
        for feat in required_features:
            if feat not in scf_features:
                raise ValueError(f"The '{feat}' feature must be included in the SCF features.")
        
        
        # --- SET THE SHAPE OF THE MATRIES AND THEIR FLATTENED INDICES ---
        
        # Get the shape of the SCF matrices
        ncells, nloci, ncopies = scf.get_expected_shape()
        # Get the indices to convert any flattened matrix back to
        # the (ncells, nloci, ncopies) shape
        indices = scf_utils.get_flattened_indices(ncells, nloci, ncopies)
        
        
        # --- GET THE AUXILIARY DATA (CELL IDS, STATES, PSEUDO-TIMES, CHROMOSOMES) ---
        
        # Get the cell IDs
        cellIDs = scf.cell_labels  # (ncells,)
        cellIDs = scf_utils.tile_to_shape(cellIDs, ncells, nloci, ncopies)  # tile to SCF shape
        cellIDs = cellIDs.reshape(-1)  # flatten
        # Get the cell states
        states = scf.cell_states  # (ncells,)
        states = scf_utils.tile_to_shape(states, ncells, nloci, ncopies)  # tile to SCF shape
        states = states.reshape(-1)  # flatten
        # Get the pseudo-times
        ts = simrep.h5['cell_run']['p_c'][:]
        ts = scf_utils.tile_to_shape(ts, ncells, nloci, ncopies)
        ts = ts.reshape(-1)
        # Get the chromosome IDs
        chroms = scf.index.chromstr  # (nloci,)
        chroms = scf_utils.tile_to_shape(chroms, ncells, nloci, ncopies)
        chroms = chroms.reshape(-1)
        
        
        # --- GET THE FEATURE DATA (GENOMIC START, SCF FEATURES) ---
        
        # Convert the window from base pairs to number of loci
        win_size_bin = int(np.ceil(win_size / scf.index.resolution()))
        
        # Initialize the arrays to store the feature data and the feature names
        X = []  # to be: (ncells * nloci * ncopies, nfeatures)
        features = []  # to be: (nfeatures,)
        
        # Get the genomic start positions
        starts = scf.index.start  # (nloci,)
        starts = scf_utils.tile_to_shape(starts, ncells, nloci, ncopies)
        # Get the sliding window average
        starts = scf_utils.sliding_matrix(starts, scf.index, win_size_bin, 'mean')
        starts = starts.reshape(-1)
        # Add to the feature data
        X.append(starts)
        features.append('genomic_start')
        
        # Add the SCF features
        for feat in scf_features:
            # Skip the 'zones' feature, which is treated separately
            if feat == 'zones':
                continue
            # Get the feature matrix
            fmat = scf.get_feature(feat)  # (ncells, nloci, ncopies)
            # Quantize the feature matrix (NOT for 'spotcount')
            if nquants is not None and feat != 'spotcount':
                fmat, _ = scf_utils.quantize_matrix(fmat, nquants)
            # Get the sliding average
            fmat = scf_utils.sliding_matrix(fmat, scf.index, win_size_bin, 'mean')
            # Flatten and store
            fmat = fmat.reshape(-1)
            X.append(fmat)
            features.append(feat)
        # Add the zones feature, if present
        # We treat it differently because it's categorical
        if 'zones' in scf_features:
            zones = scf.get_feature('zones')  # (ncells, nloci, ncopies)
            # Separate the data into an array for each zone
            for zone in np.unique(zones):
                # Get the binary mask for the current zone
                zone_mask = zones == zone
                # Calculate the sliding average for the current zone
                zone_mask = scf_utils.sliding_matrix(zone_mask, scf.index, win_size_bin, 'mean')  # (ncells, nloci, ncopies)
                # Flatten and store
                zone_mask = zone_mask.reshape(-1)
                X.append(zone_mask)
                features.append(f'zones_{zone}')
        
        # Convert X and features to numpy arrays
        X = np.array(X).T  # (ncells * nloci * ncopies, nfeatures)
        features = np.array(features).astype(str)  # (nfeatures,)
        
        
        # --- GET THE DATA FOR GENOMIC AND SPATIAL BIAS VALIDATION ---
        
        # To check that the method curates genomic and spatial detection biases,
        # we also need the Replication Timing (RT), the 'z' and the 'zones' features.
        # They will only be used to calculate AUC scores in the test sets.
        
        # Get the RT
        rt = simrep.h5['locus_run']['p_i_S'][:]  # (nloci,)
        rt = utils.smooth(rt, scf.index.chromstr, k=12)  # smooth the RT values
        rt = scf_utils.tile_to_shape(rt, ncells, nloci, ncopies)  # (ncells, nloci, ncopies)
        rt, _ = scf_utils.quantize_matrix(rt, nquants=20)  # quantize the RT values
        rt = rt.reshape(-1)
        # Get the z values
        if 'z' not in scf:
            raise ValueError("The 'z' feature is required in the SCF for spatial validation.")
        z = scf.get_feature('z')  # (ncells, nloci, ncopies)
        z, _ = scf_utils.quantize_matrix(z, nquants=20)
        z = z.reshape(-1)
        # Get the zones
        if 'zones' not in scf:
            raise ValueError("The 'zones' feature is required in the SCF for spatial validation.")
        zones = scf.get_feature('zones')  # (ncells, nloci, ncopies)
        zones = zones.reshape(-1)
        
        
        # --- CLEAN UP NAN VALUES ---
        
        # Identify the samples with NaN value in any of the features
        nan_mask = np.isnan(X).any(axis=1)  # (ncells * nloci * ncopies,)
        # Remove the samples with NaN values in all arrays
        indices = indices[~nan_mask]  # (nsamples, 3)
        cellIDs = cellIDs[~nan_mask]
        states = states[~nan_mask]
        ts = ts[~nan_mask]
        chroms = chroms[~nan_mask]
        rt = rt[~nan_mask]
        z = z[~nan_mask]
        zones = zones[~nan_mask]
        X = X[~nan_mask, :]  # (nsamples, nfeatures)
        nsamples = len(indices)
        self.nsamples = nsamples
        
        
        # --- STORE THE DATA IN THE HDF5 FILE ---
        
        # Store the attributes
        self.h5.attrs['ncells'] = ncells
        self.h5.attrs['nloci'] = nloci
        self.h5.attrs['ncopies'] = ncopies
        self.h5.attrs['nfeatures'] = len(features)
        self.h5.attrs['win_size'] = win_size
        
        # Store the auxiliary data
        self.h5.create_dataset('indices', data=indices, compression='gzip')
        self.h5.create_dataset('cellIDs', data=cellIDs.astype('S'), compression='gzip')
        self.h5.create_dataset('states', data=states.astype('S'), compression='gzip')
        self.h5.create_dataset('ts', data=ts, compression='gzip')
        self.h5.create_dataset('chroms', data=chroms.astype('S'), compression='gzip')
        
        # Store the auxiliary data for genomic and spatial bias validation
        self.h5.create_dataset('rt', data=rt, compression='gzip')
        self.h5.create_dataset('z', data=z, compression='gzip')
        self.h5.create_dataset('zones', data=zones, compression='gzip')
        
        # Store the feature data
        self.h5.create_dataset('X', data=X, compression='gzip')
        # Store the feature names
        self.h5.create_dataset('features', data=features.astype('S'), compression='gzip')
        
    
    def train(self, result_pickle_name: str):
        """ Train the XGB models on each chromosome and save the results in a pickle file.
        
        The following steps are performed for each chromosome:
            - Isolate the data for the current chromosome.
            - Isolate the G1 and G2 data (remove S-phase cells).
            - Remove bad cells, i.e. G1 cells with pseudo-time > 0.05 and G2 cells with pseudo-time < 0.95.
            - Create a stratified train/test split, stratifying by cell IDs.
            - Balance the G1 and G2 labels in the training set using random undersampling.
            - Scale the data using StandardScaler.
            - Train an XGBClassifier on the training set.
            - Evaluate the model on the test set, calculating:
                * AUC score,
                * ROC curve,
                * Yield, accuracy, balanced accuracy and confusion matrix for different probability thresholds,
                * AUC scores for different RT quantiles, z quantiles and nuclear zones.
        
        In the pickle file, the following data are stored:
            - 'scalers': a dictionary with the StandardScaler for each chromosome.
            - 'models': a dictionary with the trained XGBClassifier for each chromosome.
            - 'metrics': a dictionary with the evaluation metrics for each chromosome, as described above.

        Args:
            result_pickle_name (str): The name of the pickle file to save the results.
        """
        
        # Read the data from the HDF5 file
        cellIDs = self.h5['cellIDs'][:].astype(str)
        states = self.h5['states'][:].astype(str)
        ts = self.h5['ts'][:]
        chroms = self.h5['chroms'][:].astype(str)
        rt = self.h5['rt'][:]
        z = self.h5['z'][:]
        zones = self.h5['zones'][:]
        X = self.h5['X'][:]
        
        # Initialize the scalers, models, metrics dictionaries (for each chromosome)
        scalers = {}
        models = {}
        metrics = {}
        
        
        # Model separately each chromosome
        for chrom in np.unique(chroms):
            
            
            # --- ISOLATE THE DATA FOR THE CURRENT CHROMOSOME ---
            
            # Get the data for the current chromosome
            mask_chrom = chroms == chrom
            cellIDs_chrom = cellIDs[mask_chrom]  # (nsamples_chrom,)
            states_chrom = states[mask_chrom]  # (nsamples_chrom,)
            ts_chrom = ts[mask_chrom]  # (nsamples_chrom,)
            rt_chrom = rt[mask_chrom]  # (nsamples_chrom,)
            z_chrom = z[mask_chrom]  # (nsamples_chrom,)
            zones_chrom = zones[mask_chrom]  # (nsamples_chrom,)
            X_chrom = X[mask_chrom, :]  # (nsamples_chrom, nfeatures)
            
            # Isolate the G1/G2 data
            mask_G = states_chrom != 'S'  # G1/G2 states
            cellIDs_chrom_G = cellIDs_chrom[mask_G]  # (nsamples_chrom_G1G2,)
            states_chrom_G = states_chrom[mask_G]  # (nsamples_chrom_G1G2,)
            ts_chrom_G = ts_chrom[mask_G]  # (nsamples_chrom_G1G2,)
            rt_chrom_G = rt_chrom[mask_G]  # (nsamples_chrom_G1G2,)
            z_chrom_G = z_chrom[mask_G]  # (nsamples_chrom_G1G2,)
            zones_chrom_G = zones_chrom[mask_G]  # (nsamples_chrom_G1G2,)
            X_chrom_G = X_chrom[mask_G, :]  # (nsamples_chrom_G1G2, nfeatures)
            
            # Remove bad cells, i.e. those that might be S
            # This means removing the G1 cells with t > 0.05
            # and the G2 cells with t < 0.95
            mask_bad = np.logical_or(
                np.logical_and(states_chrom_G == 'G1', ts_chrom_G > 0.05),
                np.logical_and(states_chrom_G == 'G2', ts_chrom_G < 0.95)
            )
            cellIDs_chrom_G = cellIDs_chrom_G[~mask_bad]  # (nsamples_chrom_G1G2_good,)
            states_chrom_G = states_chrom_G[~mask_bad]
            rt_chrom_G = rt_chrom_G[~mask_bad]
            z_chrom_G = z_chrom_G[~mask_bad]
            zones_chrom_G = zones_chrom_G[~mask_bad]
            X_chrom_G = X_chrom_G[~mask_bad, :]  # (nsamples_chrom_G1G2_good, nfeatures)
            
            # Create the label array: 0 for G1, 1 for G2
            y_chrom_G = np.zeros(len(states_chrom_G), dtype=int)
            y_chrom_G[states_chrom_G == 'G2'] = 1
            
            
            # --- SEPARATE THE DATA INTO TRAINING AND TESTING SETS ---
            
            # Get the unique cell IDs, separately for G1 and G2
            cells_G1 = np.unique(cellIDs_chrom_G[y_chrom_G == 0])  # (ncells_G1,)
            cells_G2 = np.unique(cellIDs_chrom_G[y_chrom_G == 1])  # (ncells_G2,)
            cells = np.concatenate([cells_G1, cells_G2])  # (ncells_G1 + ncells_G2,)
            cell_labels = np.concatenate([np.zeros(len(cells_G1), dtype=int), np.ones(len(cells_G2), dtype=int)])  # (ncells_G1 + ncells_G2,)
            
            # Create the stratified split
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            # We want to stratify by the cell labels. StratifiedShuffleSplit requires
            # a dummy array of the same length as first argument.
            dummy_arr = np.zeros(len(cells), dtype=int)
            # Since we only do one split, we just have one train/test pair.
            # It's contained in sss.split, so we just take the first element (next)
            train, test = next(sss.split(dummy_arr, cell_labels))
            # train and test are arrays going from 0 to ncells_G1 + ncells_G2 - 1,
            # which give the indices on the cells / cell_labels.
            # We can then take the actual cellIDs
            train_cells, test_cells = cells[train], cells[test]
            # Create masks for the training and testing cells
            train_mask = np.isin(cellIDs_chrom_G, train_cells)
            test_mask = np.isin(cellIDs_chrom_G, test_cells)
            
            # Get the training and testing data
            X_train = X_chrom_G[train_mask, :]
            y_train = y_chrom_G[train_mask]
            X_test = X_chrom_G[test_mask, :]
            y_test = y_chrom_G[test_mask]
            # Get the test data for the genomic and spatial bias validation data
            # (we don't use it in training)
            rt_test = rt_chrom_G[test_mask]
            z_test = z_chrom_G[test_mask]
            zones_test = zones_chrom_G[test_mask]
            
            
            # --- BALANCE THE TRAINING DATA ---
            # Balance the labels
            rus = RandomUnderSampler(random_state=42)
            X_train, y_train = rus.fit_resample(X_train, y_train)

            # Split a validation set from the training data for early stopping
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
            )


            # --- SCALING ---

            # Perform the standard scaling
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)
            # Store the scaler
            scalers[chrom] = scaler
            
            
            # --- TRAIN THE MODEL ---
            
            # Initialize the XGBoost classifier
            clf = XGBClassifier(
                    tree_method      = "hist",
                    max_depth        = 6,
                    learning_rate    = 0.05,
                    n_estimators     = 1000,
                    subsample        = 0.8,
                    colsample_bynode = 0.8,
                    reg_lambda       = 2.0,
                    reg_alpha        = 1.0,
                    random_state     = 42,
                    eval_metric      = "auc",
                    early_stopping_rounds = 10,
                    n_jobs           = -1
            )
            # Train the model with early stopping on the validation set
            clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)
            
            # Store the model
            models[chrom] = clf
            
            
            # --- EVALUATE THE MODEL ---
            
            print(f'Chromosome {chrom}: Evaluating model...')
            
            # Initialize the metrics dictionary for the current chromosome
            metrics[chrom] = {
                'auc': 0.0,  # to store the AUC score
                'roc_curve': None,  # to store the ROC curve
                'yield': {},  # to store yield for different thresholds
                'accuracy': {},  # to store accuracy for different thresholds
                'balanced_accuracy': {},  # to store balanced accuracy for different thresholds
                'confusion_matrix': {}  # to store confusion matrix for different thresholds
            }
            
            # The XGB returns the probability that each sample belongs to the positive class (G2).
            proba = clf.predict_proba(X_test)[:, 1]  # ( nsamples_test, )
            
            # We calculate the AUC score directly on the probabilities
            auc = roc_auc_score(y_test, proba)
            metrics[chrom]['auc'] = auc
            print(f'   AUC = {auc:.4f}')
            
            # Store the ROC curve
            fpr, tpr, _ = roc_curve(y_test, proba)
            metrics[chrom]['roc_curve'] = (fpr, tpr)
            
            # To calculate the accuracy, we have to threshold the probabilities.
            # We explore different thresholds and save each result.
            for thresh in np.linspace(0.1, 0.5, 9):
                print(f'   Threshold = {thresh:.2f}')
                
                # Binarize the predictions. Mixed results are set to -1.
                y_pred = np.full(y_test.shape, -1, dtype=int)
                y_pred[proba < thresh] = 0  # G1
                y_pred[proba > 1 - thresh] = 1  # G2
                
                # Calculate the yield (fraction of non-mixed predictions),
                # the accuracy, balanced accuracy and confusion matrix
                # excluding the mixed predictions.
                yield_fraction = np.mean(y_pred != -1)
                acc = accuracy_score(y_test[y_pred != -1], y_pred[y_pred != -1])
                bal_acc = balanced_accuracy_score(y_test[y_pred != -1], y_pred[y_pred != -1])
                conf_mat = confusion_matrix(y_test[y_pred != -1], y_pred[y_pred != -1])
                print(f'      Yield = {yield_fraction:.4f}, Accuracy = {acc:.4f}, Balanced Accuracy = {bal_acc:.4f}')
                print(f'      Confusion Matrix:\n{conf_mat}')
                
                # Store the metrics
                metrics[chrom]['yield'][thresh] = yield_fraction
                metrics[chrom]['accuracy'][thresh] = acc
                metrics[chrom]['balanced_accuracy'][thresh] = bal_acc
                metrics[chrom]['confusion_matrix'][thresh] = conf_mat
            
            # Calculate the AUC for each RT quantile
            print(f'   AUC for RT quantiles:')
            metrics[chrom]['auc_rt_quantiles'] = {}
            for q in np.unique(rt_test):
                mask_q = rt_test == q
                try:
                    auc_q = roc_auc_score(y_test[mask_q], proba[mask_q])
                except ValueError:
                    print(f'      RT quantile {q}: Not enough samples, skipping.')
                    metrics[chrom]['auc_rt_quantiles'][q] = np.nan
                    continue
                metrics[chrom]['auc_rt_quantiles'][q] = auc_q
                print(f'      RT quantile {q}: AUC = {auc_q:.4f}')
            # Calculate the AUC for each z quantile
            print(f'   AUC for z quantiles:')
            metrics[chrom]['auc_z_quantiles'] = {}
            for q in np.unique(z_test):
                mask_q = z_test == q
                try:
                    auc_q = roc_auc_score(y_test[mask_q], proba[mask_q])
                except ValueError:
                    print(f'      z quantile {q}: Not enough samples, skipping.')
                    metrics[chrom]['auc_z_quantiles'][q] = np.nan
                    continue
                metrics[chrom]['auc_z_quantiles'][q] = auc_q
                print(f'      z quantile {q}: AUC = {auc_q:.4f}')
            # Calculate the AUC for each zone
            print(f'   AUC for zones:')
            metrics[chrom]['auc_zones'] = {}
            for zone in np.unique(zones_test):
                mask_zone = zones_test == zone
                try:
                    auc_zone = roc_auc_score(y_test[mask_zone], proba[mask_zone])
                except ValueError:
                    print(f'      Zone {zone}: Not enough samples, skipping.')
                    metrics[chrom]['auc_zones'][zone] = np.nan
                    continue
                metrics[chrom]['auc_zones'][zone] = auc_zone
                print(f'      Zone {zone}: AUC = {auc_zone:.4f}')
        
        # Store the results (scalers, models, metrics) in a pickle file
        os.makedirs(os.path.dirname(result_pickle_name), exist_ok=True)
        with open(result_pickle_name, 'wb') as f:
            pickle.dump({
                'scalers': scalers,
                'models': models,
                'metrics': metrics
            }, f)
    
    
    def predict(self, result_pickle_name: str):
        """ Predict the replication state using the trained models and save the results in the HDF5 file.
        
        For each chromosome, the following steps are performed:
            - Isolate the data for the current chromosome.
            - Isolate the S-phase data.
            - Scale the data using the StandardScaler for the current chromosome.
            - Predict the replication state using the XGBClassifier for the current chromosome.
            - Store the predicted replication probabilities in a 3D array of shape (ncells, nloci, ncopies).

        Args:
            result_pickle_name (str): The name of the pickle file with the trained models and scalers.
        """
        
        # Read the model results from the pickle file
        with open(result_pickle_name, 'rb') as f:
            results = pickle.load(f)
        scalers = results['scalers']
        models = results['models']
        
        # Read the data from the HDF5 file
        ncells = self.h5.attrs['ncells']
        nloci = self.h5.attrs['nloci']
        ncopies = self.h5.attrs['ncopies']
        indices = self.h5['indices'][:]  # (nsamples, 3)
        states = self.h5['states'][:].astype(str)  # (nsamples,)
        chroms = self.h5['chroms'][:].astype(str)  # (nsamples,)
        X = self.h5['X'][:]  # (nsamples, nfeatures)
        
        # Initialize the replication state array
        repli_prob = np.full((ncells, nloci, ncopies), np.nan, dtype=float)
        
        # Loop over each chromosome to predict the replication state
        for chrom in models:
            
            # Get the data for the current chromosome
            mask_chrom = chroms == chrom
            indices_chrom = indices[mask_chrom, :]  # (nsamples_chrom, 3)
            states_chrom = states[mask_chrom]
            X_chrom = X[mask_chrom, :]  # (nsamples_chrom, nfeatures)
            
            # Isolate the S data
            mask_S = states_chrom == 'S'
            indices_chrom_S = indices_chrom[mask_S, :]
            X_chrom_S = X_chrom[mask_S, :]  # (nsamples_chrom_S, nfeatures)
            
            # Scale the data
            scaler = scalers[chrom]
            X_chrom_S = scaler.transform(X_chrom_S)
            
            # Predict the probability of being replicated
            clf = models[chrom]
            prob_chrom_S = clf.predict_proba(X_chrom_S)[:, 1]
            
            # Store the predictions in the repli array
            repli_prob[indices_chrom_S[:, 0], indices_chrom_S[:, 1], indices_chrom_S[:, 2]] = prob_chrom_S
            
        
        # Store the replication state in the HDF5 file
        self.h5.create_dataset('repli_prob', data=repli_prob, compression='gzip')
        