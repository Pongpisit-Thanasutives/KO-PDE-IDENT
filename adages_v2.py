"""
ADAGES aggregation methods - Python implementation matching R code exactly
"""

import numpy as np
from typing import List, Tuple, Dict


def agg_ADAGES(Shat_list: List[List[int]], p: int) -> Dict:
    """
    Aggregation by ADAGES
    
    This function aggregates a list of selection sets by ADAGES, matching the R implementation.
    
    Parameters:
    -----------
    Shat_list : list of lists
        List of K elements containing the selection sets (1-indexed variable indices)
    p : int
        Number of original variables in the model
        
    Returns:
    --------
    dict with keys:
        'Shat': list of selected variables (1-indexed)
        'c': optimal threshold
        'K': number of aggregated sets
        
    References:
    -----------
    Gui (2020). ADAGES: adaptive aggregation with stability for distributed feature selection.
    Proceedings of the 2020 ACM-IMS on Foundations of Data Science Conference.
    """
    
    # Input check
    if not isinstance(Shat_list, list):
        raise ValueError('Input Shat_list must be a list')
    
    K = len(Shat_list)
    
    # Step 1: Compute m_j (count occurrences of each variable across all sets)
    # In R: m <- sapply(seq_len(p), function(x) { sum(x == unlist(Shat.list)) })
    m = np.zeros(p, dtype=int)
    all_vars = []
    for s in Shat_list:
        all_vars.extend(s)
    
    for j in range(1, p + 1):  # j from 1 to p (1-indexed)
        m[j - 1] = all_vars.count(j)
    
    # Step 2: Compute complexity ratio eta
    S_c_card = np.zeros(K, dtype=int)
    eta = np.zeros(K - 1)
    
    for c in range(1, K + 1):  # c from 1 to K
        # S_c <- which(m >= c)
        S_c = [j for j in range(1, p + 1) if m[j - 1] >= c]
        S_c_card[c - 1] = len(S_c)
        
        if c > 1:
            # eta[c-1] <- (S_c_card[c-1]+1)/(S_c_card[c]+1)
            eta[c - 2] = (S_c_card[c - 2] + 1) / (S_c_card[c - 1] + 1)
    
    # Step 3: Compute s_bar
    S_card = [len(s) for s in Shat_list]  # Cardinality of each set
    s_bar = np.mean(S_card)
    
    # Step 4: Compute sequence of c: |S_c| >= s_bar
    # c_seq <- which(S_c_card >= s_bar)
    c_seq = [c for c in range(1, K + 1) if S_c_card[c - 1] >= s_bar]
    
    # Step 5: Find c_op
    ll = min(K - 1, len(c_seq))
    
    if K <= 2:
        c_op = 2
    else:
        # c_op <- max(c_seq[eta[1:ll] == min(eta[1:ll])])
        # In R: eta[1:ll] gets eta values at indices 1 to ll (R is 1-indexed)
        # In Python: eta is 0-indexed, so eta[0:ll] gets first ll values
        # c_seq is 1-indexed values, we need first ll elements
        
        # Get first ll elements of c_seq
        c_seq_subset = c_seq[:ll]
        
        # For each c in c_seq_subset, get corresponding eta value
        # eta[c-1] corresponds to threshold c (since eta[0] is for c=2, eta[1] for c=3, etc.)
        eta_values = [eta[c - 2] for c in c_seq_subset]
        min_eta = min(eta_values)
        
        # Find all c in c_seq_subset where eta equals min_eta
        candidates = [c for c, eta_val in zip(c_seq_subset, eta_values) if eta_val == min_eta]
        c_op = max(candidates)
    
    # Step 6: Aggregated selection set
    # Shat <- which(m >= c_op)
    Shat = [j for j in range(1, p + 1) if m[j - 1] >= c_op]
    
    return {
        'Shat': Shat,
        'c': c_op,
        'K': K
    }


def agg_ADAGES_mod(Shat_list: List[List[int]], p: int) -> Dict:
    """
    Aggregation by modified ADAGES
    
    This function aggregates a list of selection sets by modified ADAGES, matching the R implementation.
    
    Parameters:
    -----------
    Shat_list : list of lists
        List of K elements containing the selection sets (1-indexed variable indices)
    p : int
        Number of original variables in the model
        
    Returns:
    --------
    dict with keys:
        'Shat': list of selected variables (1-indexed)
        'c': optimal threshold
        'K': number of aggregated sets
        
    References:
    -----------
    Gui (2020). ADAGES: adaptive aggregation with stability for distributed feature selection.
    Proceedings of the 2020 ACM-IMS on Foundations of Data Science Conference.
    """
    
    # Input check
    if not isinstance(Shat_list, list):
        raise ValueError('Input Shat_list must be a list')
    
    K = len(Shat_list)
    
    # Step 1: Compute m_j (count occurrences of each variable across all sets)
    m = np.zeros(p, dtype=int)
    all_vars = []
    for s in Shat_list:
        all_vars.extend(s)
    
    for j in range(1, p + 1):  # j from 1 to p (1-indexed)
        m[j - 1] = all_vars.count(j)
    
    # Compute Cardinalities
    S_c_card = np.zeros(K, dtype=int)
    for c in range(1, K + 1):
        S_c = [j for j in range(1, p + 1) if m[j - 1] >= c]
        S_c_card[c - 1] = len(S_c)
    
    # Compute c0
    S_card = [len(s) for s in Shat_list]
    s_bar = np.mean(S_card)
    
    # Compute sequence of c: |S_c| >= s_bar
    c_seq = [c for c in range(1, K + 1) if S_c_card[c - 1] >= s_bar]
    c0 = max(c_seq)
    
    # Compute criterion and selection set
    # obj <- S_c_card[1:c0] * (1:c0)
    obj = S_c_card[:c0] * np.arange(1, c0 + 1)
    
    # c_op <- (1:c0)[obj == min(obj)]
    min_obj = np.min(obj)
    c_op_candidates = [c for c in range(1, c0 + 1) if obj[c - 1] == min_obj]
    c_op = c_op_candidates[0]  # In R, if multiple minima, takes the first
    
    # Shat <- which(m >= c_op)
    Shat = [j for j in range(1, p + 1) if m[j - 1] >= c_op]
    
    return {
        'Shat': Shat,
        'c': c_op,
        'K': K
    }


def agg_union(Shat_list: List[List[int]]) -> List[int]:
    """
    Union of selection sets
    
    This function aggregates a list of selection sets by their union.
    
    Parameters:
    -----------
    Shat_list : list of lists
        List of K elements containing the selection sets
        
    Returns:
    --------
    list
        Aggregated union set of selected variables (sorted)
    """
    
    if not isinstance(Shat_list, list):
        raise ValueError('Input Shat_list must be a list')
    
    K = len(Shat_list)
    
    if K == 1:
        S_hat_final = Shat_list[0]
    else:
        # Take union and sort
        S_hat_final = sorted(set().union(*[set(s) for s in Shat_list]))
    
    return S_hat_final


# Test with the example from R documentation
if __name__ == "__main__":
    # General example (selection sets with indices between 1 and 30)
    Shat_list = [
        [2, 4, 3, 1, 20, 30],  # s1
        [3, 30, 23, 1, 4, 8],  # s2
        [3, 4, 5, 13, 15, 12] + list(range(23, 30)) + [30],  # s3
        list(range(1, 11)) + list(range(13, 16)) + [17],  # s4
        list(range(15, 21)) + [23]  # s5
    ]
    
    print("Testing agg_ADAGES:")
    result_adages = agg_ADAGES(Shat_list, p=30)
    print(f"Shat: {result_adages['Shat']}")
    print(f"c: {result_adages['c']}")
    print(f"K: {result_adages['K']}")
    
    print("\nTesting agg_ADAGES_mod:")
    result_mod = agg_ADAGES_mod(Shat_list, p=30)
    print(f"Shat: {result_mod['Shat']}")
    print(f"c: {result_mod['c']}")
    print(f"K: {result_mod['K']}")
    
    print("\nTesting agg_union:")
    result_union = agg_union(Shat_list)
    print(f"Union: {result_union}")
