"""
ADAGES: Adaptive Aggregation with Stability for Distributed Feature Selection

This module implements the ADAGES method for aggregating multiple knockoff selection sets.
Based on the paper by Yu Gui (2020).
"""
import numpy as np


def agg_adages(shat_list, p):
    """
    Aggregate selection sets using ADAGES method.
    
    ADAGES (ADaptive AGgrEgation with Stability) aggregates multiple selection sets
    by finding an adaptive threshold that minimizes the complexity ratio while maintaining
    power. This implementation follows Gui (2020).
    
    Parameters
    ----------
    shat_list : list of array-like
        List of K selection sets, where each set contains indices of selected features.
        Each element should be an array/list of feature indices (0-based).
    p : int
        Total number of features in the original problem.
    
    Returns
    -------
    dict with keys:
        'selected' : ndarray
            Indices of features selected by ADAGES aggregation.
        'threshold' : int
            Optimal threshold c* used for aggregation.
        'n_sets' : int
            Number of selection sets aggregated (K).
    
    References
    ----------
    Gui, Y. (2020). ADAGES: Adaptive Aggregation with Stability for Distributed 
    Feature Selection. Proceedings of the 2020 ACM-IMS Foundations of Data Science 
    Conference. https://arxiv.org/pdf/2007.10776.pdf
    
    Examples
    --------
    >>> shat_list = [[0, 1, 2], [1, 2, 3], [0, 2, 4]]
    >>> result = agg_adages(shat_list, p=10)
    >>> result['selected']  # Features selected by at least c* sets
    """
    if not isinstance(shat_list, list):
        raise TypeError("shat_list must be a list")
    
    K = len(shat_list)
    
    # Step 1: Compute m_j - count how many times each feature appears
    m = np.zeros(p, dtype=int)
    for shat in shat_list:
        if len(shat) > 0:
            m[np.array(shat, dtype=int)] += 1
    
    # Step 2: Compute cardinality and complexity ratio eta for each threshold c
    s_c_card = np.zeros(K, dtype=int)
    eta = np.zeros(K - 1)
    
    for c in range(1, K + 1):
        s_c = np.where(m >= c)[0]
        s_c_card[c - 1] = len(s_c)
        if c > 1:
            # Use surrogate: (|S_{c-1}| + 1) / (|S_c| + 1)
            eta[c - 2] = (s_c_card[c - 2] + 1) / (s_c_card[c - 1] + 1)
    
    # Step 3: Compute s_bar - mean cardinality of input sets
    s_card = np.array([len(shat) for shat in shat_list])
    s_bar = np.mean(s_card)
    
    # Step 4: Find candidate thresholds where |S_c| >= s_bar
    c_seq = np.where(s_c_card >= s_bar)[0] + 1  # +1 because c is 1-indexed
    
    # Step 5: Find optimal threshold c*
    if K <= 2:
        c_opt = 2 if K == 2 else 1
    else:
        ll = min(K - 1, len(c_seq))
        if ll == 0:
            c_opt = 1
        else:
            # Among candidate thresholds, find the one with minimum eta
            valid_eta_indices = c_seq[:ll] - 1  # Convert to 0-indexed for eta
            valid_eta_indices = valid_eta_indices[valid_eta_indices < len(eta)]
            if len(valid_eta_indices) == 0:
                c_opt = c_seq[0] if len(c_seq) > 0 else 1
            else:
                min_eta = np.min(eta[valid_eta_indices])
                # Take the maximum c among those with minimum eta
                candidates = c_seq[:ll][eta[c_seq[:ll] - 1] == min_eta]
                c_opt = np.max(candidates)
    
    # Step 6: Aggregate selection set with optimal threshold
    selected = np.where(m >= c_opt)[0]
    
    return {
        'selected': selected,
        'threshold': int(c_opt),
        'n_sets': K
    }


def agg_adages_mod(shat_list, p):
    """
    Aggregate selection sets using modified ADAGES method.
    
    Modified ADAGES minimizes the trade-off between threshold and model complexity
    c * |S_c| instead of the complexity ratio used in standard ADAGES.
    
    Parameters
    ----------
    shat_list : list of array-like
        List of K selection sets, where each set contains indices of selected features.
    p : int
        Total number of features in the original problem.
    
    Returns
    -------
    dict with keys:
        'selected' : ndarray
            Indices of features selected by modified ADAGES.
        'threshold' : int
            Optimal threshold c* used for aggregation.
        'n_sets' : int
            Number of selection sets aggregated (K).
    
    References
    ----------
    Gui, Y. (2020). ADAGES: Adaptive Aggregation with Stability for Distributed 
    Feature Selection. Proceedings of the 2020 ACM-IMS Foundations of Data Science 
    Conference. https://arxiv.org/pdf/2007.10776.pdf
    """
    if not isinstance(shat_list, list):
        raise TypeError("shat_list must be a list")
    
    K = len(shat_list)
    
    # Step 1: Compute m_j
    m = np.zeros(p, dtype=int)
    for shat in shat_list:
        if len(shat) > 0:
            m[np.array(shat, dtype=int)] += 1
    
    # Step 2: Compute cardinalities for all thresholds
    s_c_card = np.zeros(K, dtype=int)
    for c in range(1, K + 1):
        s_c = np.where(m >= c)[0]
        s_c_card[c - 1] = len(s_c)
    
    # Step 3: Compute c0 - upper bound based on s_bar
    s_card = np.array([len(shat) for shat in shat_list])
    s_bar = np.mean(s_card)
    c_seq = np.where(s_c_card >= s_bar)[0] + 1
    
    if len(c_seq) == 0:
        c0 = 1
    else:
        c0 = int(np.max(c_seq))
    
    # Step 4: Minimize c * |S_c| for c in [1, c0]
    obj = s_c_card[:c0] * np.arange(1, c0 + 1)
    c_opt = int(np.argmin(obj) + 1)  # +1 because we're 0-indexed
    
    # Step 5: Get aggregated selection set
    selected = np.where(m >= c_opt)[0]
    
    return {
        'selected': selected,
        'threshold': c_opt,
        'n_sets': K
    }


def agg_union(shat_list):
    """
    Aggregate selection sets by taking their union.
    
    Parameters
    ----------
    shat_list : list of array-like
        List of selection sets to aggregate.
    
    Returns
    -------
    ndarray
        Union of all selection sets (sorted).
    """
    if not isinstance(shat_list, list):
        raise TypeError("shat_list must be a list")
    
    if len(shat_list) == 1:
        return np.sort(np.array(shat_list[0], dtype=int))
    
    # Take union of all sets
    union_set = set()
    for shat in shat_list:
        union_set.update(shat)
    
    return np.sort(np.array(list(union_set), dtype=int))
