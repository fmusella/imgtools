import os
import numpy as np
import h5py
from ..scf import SingleCellFeature
from ..scf import scf_utils
from .repliseq import SimulatedRepliSeqExperiment
from . import sc_repliseq_utils
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
        indices = sc_repliseq_utils.get_flattened_indices(ncells, nloci, ncopies)
        
        
        # --- GET THE AUXILIARY DATA (CELL IDS, STATES, PSEUDO-TIMES, CHROMOSOMES) ---
        
        # Get the cell IDs
        cellIDs = scf.cell_labels  # (ncells,)
        cellIDs = sc_repliseq_utils.tile_to_shape(cellIDs, ncells, nloci, ncopies)  # tile to SCF shape
        cellIDs = cellIDs.reshape(-1)  # flatten
        # Get the cell states
        states = scf.cell_states  # (ncells,)
        states = sc_repliseq_utils.tile_to_shape(states, ncells, nloci, ncopies)  # tile to SCF shape
        states = states.reshape(-1)  # flatten
        # Get the pseudo-times
        ts = simrep.h5['cell_run']['p_c'][:]
        ts = sc_repliseq_utils.tile_to_shape(ts, ncells, nloci, ncopies)
        ts = ts.reshape(-1)
        # Get the chromosome IDs
        chroms = scf.index.chromstr  # (nloci,)
        chroms = sc_repliseq_utils.tile_to_shape(chroms, ncells, nloci, ncopies)
        chroms = chroms.reshape(-1)
        
        
        # --- GET THE FEATURE DATA (SPOTCOUNT, INTENSITY, GENOMIC START, RT, ADDITIONAL FEATURES) ---
        
        # Initialize the arrays to store the feature data and the feature names
        X = []  # to be: (ncells * nloci * ncopies, nfeatures)
        features = []  # to be: (nfeatures,)
        
        # Get the genomic start positions
        starts = scf.index.start  # (nloci,)
        starts = sc_repliseq_utils.tile_to_shape(starts, ncells, nloci, ncopies)
        starts = starts.reshape(-1)
        # Add to the feature data
        X.append(starts)
        features.append('genomic_start')
        
        # Get the RT values
        rt = simrep.h5['locus_run']['p_i_S'][:]  # (nloci,)
        rt = utils.smooth(rt, scf.index.chromstr, k=12)
        rt = sc_repliseq_utils.tile_to_shape(rt, ncells, nloci, ncopies)
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
        
        