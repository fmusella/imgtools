import numpy as np

def set_replication_states(self, low: float, high: float) -> None:
    """ Set the replication states based on the replication probability.
    
    It sets the replication state to 1 (non-replicating) if the replication probability is below a low threshold,
    and to 2 (replicating) if the replication probability is above a high threshold.
    The in-between values are set to 0 (not enough confidence).
    
    Stores the replication state as r_ic in the object.

    Args:
        low (float): low threshold for non-replicating regions.
        high (float): high threshold for replicating regions.
    """
    r_ic = np.zeros(self.n_ic.shape, dtype=float)
    r_ic[self.p_ic < low] = 1
    r_ic[self.p_ic > high] = 2
    self.r_ic = r_ic


# PERFORMANCE EVALUATION METHODS

def yielding(self) -> dict:
    """ Calculate the yield of the replication classification,
    i.e. the fraction of significantly-predicted replicating/non-replicating/determined loci.
    
    The yield is calculated per locus, per cell and total, and is given as absolute and relative values:
    - absolute yield: the fraction of rep/nonrep/det loci over the total number of loci
    - relative yield: the fraction of rep/nonrep/det loci over the total number of imaged loci (where n_ic > 0)
    
    The yield should be measured for the S-phase cells, but G1 and G2 yields are also calculated for validation.

    Returns:
        dict: the yield dictionaries, with the following keys:
            - abs_rep_yield_i: absolute yield of replicating loci per locus.
            - abs_nonrep_yield_i: absolute yield of non-replicating loci per locus.
            - abs_det_yield_i: absolute yield of determined loci per locus.
            - abs_rep_yield_c: absolute yield of replicating loci per cell.
            - abs_nonrep_yield_c: absolute yield of non-replicating loci per cell.
            - abs_det_yield_c: absolute yield of determined loci per cell.
            - abs_rep_yield: absolute yield of replicating loci.
            - abs_nonrep_yield: absolute yield of non-replicating loci.
            - abs_det_yield: absolute yield of determined loci.
            - rel_rep_yield_i: relative yield of replicating loci per locus.
            - rel_nonrep_yield_i: relative yield of non-replicating loci per locus.
            - rel_det_yield_i: relative yield of determined loci per locus.
            - rel_rep_yield_c: relative yield of replicating loci per cell.
            - rel_nonrep_yield_c: relative yield of non-replicating loci per cell.
            - rel_det_yield_c: relative yield of determined loci per cell.
            - rel_rep_yield: relative yield of replicating loci.
            - rel_nonrep_yield: relative yield of non-replicating loci.
            - rel_det_yield: relative yield of determined loci.
            - abs_rep_yield_G1: absolute yield of replicating loci in G1.
            - abs_nonrep_yield_G1: absolute yield of non-replicating loci in G1.
            - abs_det_yield_G1: absolute yield of determined loci in G1.
            - abs_rep_yield_G2: absolute yield of replicating loci in G2.
            - abs_nonrep_yield_G2: absolute yield of non-replicating loci in G2.
            - abs_det_yield_G2: absolute yield of determined loci in G2.
            - rel_rep_yield_G1: relative yield of replicating loci in G1.
            - rel_nonrep_yield_G1: relative yield of non-replicating loci in G1.
            - rel_det_yield_G1: relative yield of determined loci in G1.
            - rel_rep_yield_G2: relative yield of replicating loci in G2.
            - rel_nonrep_yield_G2: relative yield of non-replicating loci in G2.
            - rel_det_yield_G2: relative yield of determined loci in G2.
    """
    
    # Get a copy of the replication states
    r_ic = np.copy(self.r_ic)
    
    # Set low quality regions to NaN
    r_ic[~self.q_ic] = np.nan
    # Set 0s to NaN (non-determined regions)
    r_ic[self.r_ic == 0] = np.nan
    
    # Isolate the S-phase cells
    rS_ic = r_ic[self.states == 'S', :, :]
    nS_ic = self.n_ic[self.states == 'S', :, :]
    nScells = np.sum(self.states == 'S')
    
    # Get the total number of non-replicated, replicated and determined loci:
    # per locus,
    rep_i = np.sum(rS_ic == 2, axis=(0, 2))
    nonrep_i = np.sum(rS_ic == 1, axis=(0, 2))
    det_i = np.sum(~np.isnan(rS_ic), axis=(0, 2))
    # per cell,
    rep_c = np.sum(rS_ic == 2, axis=(1, 2))
    nonrep_c = np.sum(rS_ic == 1, axis=(1, 2))
    det_c = np.sum(~np.isnan(rS_ic), axis=(1, 2))
    # and total
    rep = np.sum(rS_ic == 2)
    nonrep = np.sum(rS_ic == 1)
    det = np.sum(~np.isnan(rS_ic))
    
    # To calculate the yield, we consider the following:
    # - absolute yield: the fraction of rep/nonrep/det loci over the total number of loci
    # - relative yield: the fraction of rep/nonrep/det loci over the total number of imaged loci (where n_ic > 0)
    abs_rep_yield_i = rep_i / (nScells * self.ncopies)
    abs_nonrep_yield_i = nonrep_i / (nScells * self.ncopies)
    abs_det_yield_i = det_i / (nScells * self.ncopies)
    abs_rep_yield_c = rep_c / (self.nloci * self.ncopies)
    abs_nonrep_yield_c = nonrep_c / (self.nloci * self.ncopies)
    abs_det_yield_c = det_c / (self.nloci * self.ncopies)
    abs_rep_yield = rep / (nScells * self.nloci * self.ncopies)
    abs_nonrep_yield = nonrep / (nScells * self.nloci * self.ncopies)
    abs_det_yield = det / (nScells * self.nloci * self.ncopies)
    rel_rep_yield_i = rep_i / np.sum(nS_ic > 0, axis=(0, 2))
    rel_nonrep_yield_i = nonrep_i / np.sum(nS_ic > 0, axis=(0, 2))
    rel_det_yield_i = det_i / np.sum(nS_ic > 0, axis=(0, 2))
    rel_rep_yield_c = rep_c / np.sum(nS_ic > 0, axis=(1, 2))
    rel_nonrep_yield_c = nonrep_c / np.sum(nS_ic > 0, axis=(1, 2))
    rel_det_yield_c = det_c / np.sum(nS_ic > 0, axis=(1, 2))
    rel_rep_yield = rep / np.sum(nS_ic > 0)
    rel_nonrep_yield = nonrep / np.sum(nS_ic > 0)
    rel_det_yield = det / np.sum(nS_ic > 0)
    
    
    # Get the absolute and relative yield in G1/G2 for validation
    rG1_ic = r_ic[self.states == 'G1', :, :]
    rG2_ic = r_ic[self.states == 'G2', :, :]
    nG1_ic = self.n_ic[self.states == 'G1', :, :]
    nG2_ic = self.n_ic[self.states == 'G2', :, :]
    nG1cells = np.sum(self.states == 'G1')
    nG2cells = np.sum(self.states == 'G2')
    repG1 = np.sum(rG1_ic == 2)
    nonrepG1 = np.sum(rG1_ic == 1)
    detG1 = np.sum(~np.isnan(rG1_ic))
    repG2 = np.sum(rG2_ic == 2)
    nonrepG2 = np.sum(rG2_ic == 1)
    detG2 = np.sum(~np.isnan(rG2_ic))
    abs_rep_yield_G1 = repG1 / (nG1cells * self.nloci * self.ncopies)
    abs_nonrep_yield_G1 = nonrepG1 / (nG1cells * self.nloci * self.ncopies)
    abs_det_yield_G1 = detG1 / (nG1cells * self.nloci * self.ncopies)
    abs_rep_yield_G2 = repG2 / (nG2cells * self.nloci * self.ncopies)
    abs_nonrep_yield_G2 = nonrepG2 / (nG2cells * self.nloci * self.ncopies)
    abs_det_yield_G2 = detG2 / (nG2cells * self.nloci * self.ncopies)
    rel_rep_yield_G1 = repG1 / np.sum(nG1_ic > 0)
    rel_nonrep_yield_G1 = nonrepG1 / np.sum(nG1_ic > 0)
    rel_det_yield_G1 = detG1 / np.sum(nG1_ic > 0)
    rel_rep_yield_G2 = repG2 / np.sum(nG2_ic > 0)
    rel_nonrep_yield_G2 = nonrepG2 / np.sum(nG2_ic > 0)
    rel_det_yield_G2 = detG2 / np.sum(nG2_ic > 0)
    
    return {
        'abs_rep_yield_i': abs_rep_yield_i,
        'abs_nonrep_yield_i': abs_nonrep_yield_i,
        'abs_det_yield_i': abs_det_yield_i,
        'abs_rep_yield_c': abs_rep_yield_c,
        'abs_nonrep_yield_c': abs_nonrep_yield_c,
        'abs_det_yield_c': abs_det_yield_c,
        'abs_rep_yield': abs_rep_yield,
        'abs_nonrep_yield': abs_nonrep_yield,
        'abs_det_yield': abs_det_yield,
        'rel_rep_yield_i': rel_rep_yield_i,
        'rel_nonrep_yield_i': rel_nonrep_yield_i,
        'rel_det_yield_i': rel_det_yield_i,
        'rel_rep_yield_c': rel_rep_yield_c,
        'rel_nonrep_yield_c': rel_nonrep_yield_c,
        'rel_det_yield_c': rel_det_yield_c,
        'rel_rep_yield': rel_rep_yield,
        'rel_nonrep_yield': rel_nonrep_yield,
        'rel_det_yield': rel_det_yield,
        'abs_rep_yield_G1': abs_rep_yield_G1,
        'abs_nonrep_yield_G1': abs_nonrep_yield_G1,
        'abs_det_yield_G1': abs_det_yield_G1,
        'abs_rep_yield_G2': abs_rep_yield_G2,
        'abs_nonrep_yield_G2': abs_nonrep_yield_G2,
        'abs_det_yield_G2': abs_det_yield_G2,
        'rel_rep_yield_G1': rel_rep_yield_G1,
        'rel_nonrep_yield_G1': rel_nonrep_yield_G1,
        'rel_det_yield_G1': rel_det_yield_G1,
        'rel_rep_yield_G2': rel_rep_yield_G2,
        'rel_nonrep_yield_G2': rel_nonrep_yield_G2,
        'rel_det_yield_G2': rel_det_yield_G2,
    }


def performance(self) -> dict:
    """ Evaluate the performance of the replication classification.
    
    This method evaluates the performance of the replication classification using two methods:
    1. Using G1/G2 as ground truth.
    2. Using early/late replicating loci in S as ground truth.

    Returns:
        dict: the performance dictionaries for the two methods.
    """
    
    if not hasattr(self, 'r_ic'):
        raise ValueError("The replication states have not been set yet.")
    
    perf_1 = self.performance_method_1(self.r_ic)
    perf_2 = self.performance_method_2(self.r_ic)
    
    return {'Method 1': perf_1, 'Method 2': perf_2}

def evaluate_thresholds(self, ranges: list) -> None:
    """ Evaluate the performance of the replication classification for different thresholds.
    
    For each pair of low and high thresholds, it calculates the replication state based on the replication probability,
    and evaluates the performance of the replication classification using two methods:
    1. Using G1/G2 as ground truth.
    2. Using early/late replicating loci in S as ground truth.

    Args:
        ranges (list): list of two ranges, one for the low threshold and one for the high threshold.
    """
    
    # To find the best thresholds, initialize the best one to 0
    best_low, best_high, best_acc = None, None, 0
    
    # Loop over the ranges of low threshold (non-replicating) and high threshold (replicating)
    for low in ranges[0]:
        for high in ranges[1]:
            
            # Calculate the replication state: 1 non-replicating, 2 replicating, 0 not enough confidence
            r_ic = np.zeros(self.n_ic.shape, dtype=float)
            r_ic[self.p_ic < low] = 1
            r_ic[self.p_ic > high] = 2
            
            # Calculate the performance of the replication classification with the two methods
            perf_1 = self.performance_method_1(r_ic)
            perf_2 = self.performance_method_2(r_ic)
            
            # Print the performance dictionaries
            print(f"Low: {low}, High: {high}")
            print("Method 1:")
            print(perf_1)
            print("Method 2:")
            print(perf_2)
            print("\n")
            
            # Calculate the average accuracy of the two methods
            acc = (perf_1['accuracy'] + perf_2['accuracy']) / 2
            
            # Update the best thresholds if the accuracy is better
            if acc > best_acc:
                best_low, best_high, best_acc = low, high, acc
    
    # Print the best thresholds
    print(f"Best thresholds: low = {best_low}, high = {best_high}, accuracy = {best_acc}")
    print("\n\n")

def performance_method_1(
    self,
    r_ic: np.ndarray
) -> dict:
    """ Evaluate the performance of the replication classification using G1/G2 as ground truth.
    
    It evaluates true positives and true negatives, respectively, as the number of regions
    classified as replicating in G2 and non-replicating in G1.

    Args:
        r_ic (np.ndarray): replication state array, shape: (ncells, nloci, ncopies).

    Returns:
        dict: the confusion matrix and the performance metrics, with the following keys:
            - TP: true positives.
            - TN: true negatives.
            - FP: false positives.
            - FN: false negatives.
            - accuracy: accuracy.
            - ppv: positive predictive value.
            - npv: negative predictive value.
            - tpr: true positive rate.
            - tnr: true negative rate.
    """
    
    # Set low quality regions to NaN
    r_ic[~self.q_ic] = np.nan
    
    # Get the replication states and volumes for G1 and G2 cells
    r_ic_G1 = r_ic[self.states == 'G1', :, :]
    r_ic_G2 = r_ic[self.states == 'G2', :, :]
    vol_G1 = self.volumes[self.states == 'G1']
    vol_G2 = self.volumes[self.states == 'G2']
    
    # Remove the top 20% of the volumes for G1 and the bottom 20% of the volumes for G2,
    # to make sure that there is no S-phase contamination
    G1_max_vol = np.percentile(vol_G1, 80)
    G2_min_vol = np.percentile(vol_G2, 20)
    r_ic_G1 = r_ic_G1[vol_G1 < G1_max_vol, :, :]
    r_ic_G2 = r_ic_G2[vol_G2 > G2_min_vol, :, :]
    
    # Remove the NaN values and flatten the arrays
    r_ic_G1 = r_ic_G1[~np.isnan(r_ic_G1)].flatten()
    r_ic_G2 = r_ic_G2[~np.isnan(r_ic_G2)].flatten()
    
    # Subsample the two arrays to have the same length,
    # so that the performance metrics are not biased by different sample sizes
    np.random.seed(0)  # for reproducibility
    min_len = min(len(r_ic_G1), len(r_ic_G2))
    r_ic_G1 = np.random.choice(r_ic_G1, min_len, replace=False)
    r_ic_G2 = np.random.choice(r_ic_G2, min_len, replace=False)
    
    # Return the confusion matrix
    return self.confusion_matrix(r_ic_G1, r_ic_G2)

def performance_method_2(
    self,
    r_ic: np.ndarray
) -> dict:
    """ Evaluate the performance of the replication classification using early/late replicating loci in S as ground truth.
    
    It evaluates true positives and true negatives, respectively, as the number of regions
    classified as replicating for the early replicating loci and non-replicating for the late replicating loci.

    Args:
        r_ic (np.ndarray): replication state array, shape: (ncells, nloci, ncopies).

    Returns:
        dict: the confusion matrix and the performance metrics, with the following keys:
            - TP: true positives.
            - TN: true negatives.
            - FP: false positives.
            - FN: false negatives.
            - accuracy: accuracy.
            - ppv: positive predictive value.
            - npv: negative predictive value.
            - tpr: true positive rate.
            - tnr: true negative rate.
    """
    
    # Set low quality regions to NaN
    r_ic[~self.q_ic] = np.nan
    
    # Get the replication states for S cells
    rS_ic = r_ic[self.states == 'S', :, :]

    # Identify the early/late replicating loci
    early = self.pS_i > np.nanpercentile(self.pS_i, 90)
    late = self.pS_i < np.nanpercentile(self.pS_i, 10)
    assert np.sum(early) == np.sum(late), "The number of early and late replicating loci should be the same"
    
    # Get the replication states in S for the early/late replicating loci
    rS_ic_early = rS_ic[:, early, :]
    rS_ic_late = rS_ic[:, late, :]
    
    # Consider only cells at the end of S phase for the early replicating loci (ensuring ground truth of r=2),
    # and at the beginning for the late replicating loci (ensuring ground truth of r=1)
    pS_c = self.p_c[self.states == 'S']
    rS_ic_early = rS_ic_early[pS_c > np.nanpercentile(pS_c, 90), :, :]
    rS_ic_late = rS_ic_late[pS_c < np.nanpercentile(pS_c, 10), :, :]
    
    # Remove the NaN values and flatten the arrays
    rS_ic_early = rS_ic_early[~np.isnan(rS_ic_early)].flatten()
    rS_ic_late = rS_ic_late[~np.isnan(rS_ic_late)].flatten()
    
    # Subsample the two arrays to have the same length,
    # so that the performance metrics are not biased by different sample sizes
    np.random.seed(0)
    min_len = min(len(rS_ic_early), len(rS_ic_late))
    rS_ic_early = np.random.choice(rS_ic_early, min_len, replace=False)
    rS_ic_late = np.random.choice(rS_ic_late, min_len, replace=False)
    
    # Return the confusion matrix
    return self.confusion_matrix(rS_ic_late, rS_ic_early)
    
@staticmethod
def confusion_matrix(y1: np.ndarray, y2: np.ndarray) -> dict:
    """ Calculate the confusion matrix of the replication classification.
    
    It calculates the classification counts as:
    - True positive: replicated regions correctly classified as replicated.
    - True negative: non-replicated regions correctly classified as non-replicated.
    - False positive: non-replicated regions incorrectly classified as replicated.
    - False negative: replicated regions incorrectly classified as non-replicated.
    
    And the performance metrics:
    - Accuracy: (TP + TN) / (TP + TN + FP + FN).
    - Positive predictive value (PPV): TP / (TP + FP).
    - Negative predictive value (NPV): TN / (TN + FN).
    - True positive rate (TPR): TP / (TP + FN).
    - True negative rate (TNR): TN / (TN + FP).

    Args:
        y1 (np.ndarray): predicted replication states, where the ground truth is 1 (non-replicated)
        y2 (np.ndarray): predicted replication states, where the ground truth is 2 (replicated)

    Returns:
        dict: the confusion matrix and the performance metrics, with the following keys:
            - TP: true positives.
            - TN: true negatives.
            - FP: false positives.
            - FN: false negatives.
            - accuracy: accuracy.
            - ppv: positive predictive value.
            - npv: negative predictive value.
            - tpr: true positive rate.
            - tnr: true negative rate.
    """
    
    # Calculate the true positives, true negatives, false positives and false negatives
    tp = np.sum(y2 == 2)  # True positives
    tn = np.sum(y1 == 1)  # True negatives
    fp = np.sum(y1 == 2)  # False positives
    fn = np.sum(y2 == 1)  # False negatives
    
    # Calculate the performance metrics
    acc = 100 * (tp + tn) / (tp + tn + fp + fn)  # Accuracy
    ppv = 100 * tp / (tp + fp)  # Positive predictive value (PPV)
    npv = 100 * tn / (tn + fn)  # Negative predictive value (NPV)
    tpr = 100 * tp / (tp + fn)  # True positive rate (TPR)
    tnr = 100 * tn / (tn + fp)  # True negative rate (TNR)

    # Return the results as a dictionary
    return {
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'accuracy': acc,
        'ppv': ppv,
        'npv': npv,
        'tpr': tpr,
        'tnr': tnr
    }