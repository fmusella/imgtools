import os
import h5py
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import StratifiedShuffleSplit
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score
from ..scf import SingleCellFeature
from ..scf import scf_utils
from .repliseq import SimulatedRepliSeqExperiment
from .. import utils

class SimulatedSingleCellRepliSeqExperiment:
    """_summary_
    """
    
    def __init__(self, h5_name: str, mode: str):
        
        # Extend the name with its absolute path
        h5_name = os.path.abspath(h5_name)
        # Check that file has a valid path
        if not os.path.exists(os.path.dirname(h5_name)):
            raise FileNotFoundError("The path of the HDF5 file does not exist.")
        # Store the name of the HDF5 file
        self.h5_name = h5_name
        # Read / create the HDF5 file
        self.h5 = h5py.File(h5_name, mode=mode)

    
    def feature_extraction(
        self,
        scf: SingleCellFeature,
        simrep: SimulatedRepliSeqExperiment,
        scf_features: list,
        win_size: float
    ):
        """ Extract the feature data from the SCF and the SimulatedRepliSeqExperiment data.

        Args:
            scf (SingleCellFeature)
            simrep (SimulatedRepliSeqExperiment)
            scf_features (list)
            win_size (float)
        """
        
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
        
        
        # --- GET THE FEATURE DATA (SPOTCOUNT, INTENSITY, GENOMIC START, RT, ADDITIONAL FEATURES) ---
        
        # Initialize the arrays to store the feature data and the feature names
        X = []  # to be: (ncells * nloci * ncopies, nfeatures)
        features = []  # to be: (nfeatures,)
        
        # Get the genomic start positions
        starts = scf.index.start  # (nloci,)
        starts = scf_utils.tile_to_shape(starts, ncells, nloci, ncopies)
        starts = starts.reshape(-1)
        # Add to the feature data
        X.append(starts)
        features.append('genomic_start')
        
        # Get the RT values
        rt = simrep.h5['locus_run']['p_i_S'][:]  # (nloci,)
        rt = utils.smooth(rt, scf.index.chromstr, k=12)
        rt = scf_utils.tile_to_shape(rt, ncells, nloci, ncopies)
        rt = rt.reshape(-1)
        X.append(rt)
        features.append('RT')
        
        # Add the SCF features
        for feat in scf_features:
            # Get the feature matrix
            fmat = scf.get_feature(feat)  # (ncells, nloci, ncopies)
            # Get the sliding average
            fmat = scf_utils.sliding_matrix(fmat, scf.index, win_size, 'mean')
            # Flatten and store
            fmat = fmat.reshape(-1)
            X.append(fmat)
            features.append(feat)
        
        # If there is the 'zones' feature in the SCF, add the nuclear zones
        if 'zones' in scf.feature_list:
            zones = scf.get_feature('zones')  # (ncells, nloci, ncopies)
            # Separate the data into an array for each zone
            for z in np.unique(zones):
                # Get the binary mask for the current zone
                zone_z = zones == z
                # Calculate the sliding average for the current zone
                zone_z = scf_utils.sliding_matrix(zone_z, scf.index, win_size, 'mean')  # (ncells, nloci, ncopies)
                # Flatten and store
                zone_z = zone_z.reshape(-1)
                X.append(zone_z)
                features.append(f'zones_{z}')
        
        # Convert X and features to numpy arrays
        X = np.array(X).T  # (ncells * nloci * ncopies, nfeatures)
        features = np.array(features).astype(str)  # (nfeatures,)
        
        
        # --- CLEAN UP NAN VALUES ---
        
        # Identify the samples with NaN value in any of the features
        nan_mask = np.isnan(X).any(axis=1)  # (ncells * nloci * ncopies,)
        # Remove the samples with NaN values in all arrays
        indices = indices[~nan_mask]  # (nsamples, 3)
        cellIDs = cellIDs[~nan_mask]
        states = states[~nan_mask]
        ts = ts[~nan_mask]
        chroms = chroms[~nan_mask]
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
        
        # Store the feature data
        self.h5.create_dataset('X', data=X, compression='gzip')
        # Store the feature names
        self.h5.create_dataset('features', data=features.astype('S'), compression='gzip')
        
    
    def train(self, result_pickle_name: str):
        """ Train the model on each chromosome and save the results in a pickle file.

        Args:
            result_pickle_name (str): The name of the pickle file to save the results.
        """
        
        # Read the data from the HDF5 file
        cellIDs = self.h5['cellIDs'][:].astype(str)
        states = self.h5['states'][:].astype(str)
        ts = self.h5['ts'][:]
        chroms = self.h5['chroms'][:].astype(str)
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
            X_chrom = X[mask_chrom, :]  # (nsamples_chrom, nfeatures)
            
            # Isolate the G1/G2 data
            mask_G = states_chrom != 'S'  # G1/G2 states
            cellIDs_chrom_G = cellIDs_chrom[mask_G]  # (nsamples_chrom_G1G2,)
            states_chrom_G = states_chrom[mask_G]  # (nsamples_chrom_G1G2,)
            ts_chrom_G = ts_chrom[mask_G]  # (nsamples_chrom_G1G2,)
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
            
            
            # --- BALANCE THE TRAINING DATA ---
            # Balance the labels
            rus = RandomUnderSampler(random_state=42)
            X_train, y_train = rus.fit_resample(X_train, y_train)
            
            
            # --- SCALING ---
            
            # Perform the standard scaling
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            # Store the scaler
            scalers[chrom] = scaler
            
            
            # --- TRAIN THE MODEL ---
            
            # Initialize the XGBoost classifier
            clf = XGBClassifier(
                    tree_method        = "hist",          # "gpu_hist" for GPU
                    max_depth          = 12,
                    learning_rate      = 0.05,
                    n_estimators       = 600,
                    subsample          = 0.8,
                    colsample_bynode   = 0.8,
                    reg_lambda         = 2.0,
                    reg_alpha          = 1.0,
                    random_state       = 42,
                    eval_metric        = "auc",
                    n_jobs             = -1
            )
            # Train the model
            clf.fit(X_train, y_train,verbose=True)
            
            # Store the model
            models[chrom] = clf
            
            
            # --- EVALUATE THE MODEL ---
            
            proba = clf.predict_proba(X_test)[:, 1]
            pred  = (proba > 0.5)
            acc = accuracy_score(y_test, pred)
            auc = roc_auc_score(y_test, proba)
            bal_acc = balanced_accuracy_score(y_test, pred)
            print(f'Chromosome {chrom}: Accuracy = {acc:.4f}, AUC = {auc:.4f}, Balanced Accuracy = {bal_acc:.4f}')
            # Store the metrics
            metrics[chrom] = {
                'accuracy': acc,
                'auc': auc,
                'balanced_accuracy': bal_acc
            }
    
    
        # Store the results (scalers, models, metrics) in a pickle file
        os.makedirs(os.path.dirname(result_pickle_name), exist_ok=True)
        with open(result_pickle_name, 'wb') as f:
            pickle.dump({
                'scalers': scalers,
                'models': models,
                'metrics': metrics
            }, f)
    
            