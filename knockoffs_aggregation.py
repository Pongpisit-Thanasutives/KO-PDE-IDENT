# See https://github.com/mind-inria/hidimstat for any updates
import numpy as np


def quantile_aggregation(pvals, gamma=0.5, adaptive=False):
    """
    Implements the quantile aggregation method for p-values.

    This method is based on :footcite:t:meinshausen2009pvalues.

    The function aggregates multiple p-values into a single p-value while controlling
    the family-wise error rate. It supports both fixed and adaptive quantile aggregation.

    Parameters
    ----------
    pvals : ndarray of shape (n_sampling*2, n_test)
        Matrix of p-values to aggregate. Each row represents a sampling instance
        and each column a hypothesis test.
    gamma : float, default=0.5
        Quantile level for aggregation. Must be in range (0,1].
    adaptive : bool, default=False
        If True, uses adaptive quantile aggregation which optimizes over multiple gamma values.
        If False, uses fixed quantile aggregation with the provided gamma value.

    Returns
    -------
    ndarray of shape (n_test,)
        Vector of aggregated p-values, one for each hypothesis test.

    References
    ----------
    .. footbibliography::

    Notes
    -----
    The aggregated p-values are guaranteed to be valid p-values in [0,1].
    When adaptive=True, gamma is treated as the minimum gamma value to consider.
    """
    # if pvalues are one-dimensional, do nothing
    if pvals.shape[0] == 1:
        return pvals[0]
    if adaptive:
        return _adaptive_quantile_aggregation(pvals, gamma)
    else:
        return _fixed_quantile_aggregation(pvals, gamma)


def _fixed_quantile_aggregation(pvals, gamma=0.5):
    """
    Quantile aggregation function

    For more details, see footcite:t:meinshausen2009pvalues

    Parameters
    ----------
    pvals : 2D ndarray (n_sampling*2, n_test)
        p-value

    gamma : float
        Percentile value used for aggregation.

    Returns
    -------
    pvalue aggregate: 1D ndarray (n_tests, )
        Vector of aggregated p-values

    References
    ----------
    .. footbibliography::
    """
    assert gamma > 0 and gamma <= 1, "gamma should be between 0 and 1"
    # equation 2.2 of meinshausen2009pvalues
    converted_score = np.quantile(pvals, q=gamma, axis=0) / gamma
    return np.minimum(1, converted_score)


def _adaptive_quantile_aggregation(pvals, gamma_min=0.05):
    """
    Adaptive version of quantile aggregation method

    For more details, see footcite:t:meinshausen2009pvalues

    Parameters
    ----------
    pvals : 2D ndarray (n_sampling*2, n_test)
        p-value
    gamma_min : float, default=0.05
        Minimum percentile value for adaptive aggregation

    Returns
    -------
    pvalue aggregate: 1D ndarray (n_tests, )
        Vector of aggregated p-values

    References
    ----------
    .. footbibliography::
    """
    assert gamma_min > 0 and gamma_min <= 1, "gamma min should between 0 and 1"

    n_iter, n_features = pvals.shape

    n_min = int(np.floor(gamma_min * n_iter))
    ordered_pval = np.sort(pvals, axis=0)[n_min:]
    # calculation of the pvalue / quantile (=j/m)
    # see equation 2.2 of `meinshausen2009p`
    P = (
        np.min(ordered_pval / np.arange(n_min, n_iter, 1).reshape(-1, 1), axis=0)
        * n_iter
    )
    # see equation 2.3 of `meinshausen2009p`
    pval_aggregate = np.minimum(1, (1 - np.log(gamma_min)) * P)
    return pval_aggregate


def fdp_power(selected, ground_truth):
    """
    Calculate False Discovery Proportion and statistical power

    Parameters
    ----------
    selected : ndarray (n_features,)
        Array of selected variable. 0 for non-selected, non-zero for selected
    ground_truth : ndarray (n_features,)
        Array of true relevant variables. 0 for null, 1 for non-null

    Returns
    -------
    fdp : float
        False Discovery Proportion (number of false discoveries / total discoveries)
    power : float
        Statistical power (number of true discoveries / number of non-null variables)
    """

    # Make sure arrays are binary
    selected_binary = selected != 0
    ground_truth_binary = ground_truth != 0

    true_positive = np.sum(selected_binary & ground_truth_binary)
    false_positive = np.sum(selected_binary & ~ground_truth_binary)

    fdp = false_positive / max(1, np.sum(selected_binary))
    power = true_positive / max(1, np.sum(ground_truth_binary))

    return fdp, power


def fdr_threshold(pvals, fdr=0.1, method="bhq", reshaping_function=None):
    """
    Calculate threshold for False Discovery Rate control methods.

    Parameters
    ----------
    pvals : 1D ndarray
        Set of p-values to threshold
    fdr : float, default=0.1
        Target False Discovery Rate level
    method : {'bhq', 'bhy', 'ebh'}, default='bhq'
        Method for FDR control:
        * 'bhq': Standard Benjamini-Hochberg procedure
        * 'bhy': Benjamini-Hochberg-Yekutieli procedure
        * 'ebh': e-Benjamini-Hochberg procedure
    reshaping_function : callable
        Reshaping function for BHY method, default uses sum of reciprocals

    Returns
    -------
    threshold : float
        Threshold value for p-values. P-values below this threshold are rejected.

    References
    ----------
    .. footbibliography::
    """
    if method == "bhq":
        threshold = _bhq_threshold(pvals, fdr=fdr)
    elif method == "bhy":
        threshold = _bhy_threshold(
            pvals, fdr=fdr, reshaping_function=reshaping_function
        )
    elif method == "ebh":
        threshold = _ebh_threshold(pvals, fdr=fdr)
    else:
        raise ValueError("{} is not support FDR control method".format(method))
    return threshold


def _bhq_threshold(pvals, fdr=0.1):
    """
    Standard Benjamini-Hochberg
    for controlling False discovery rate

    Calculate threshold for standard Benjamini-Hochberg procedure
    :footcite:`benjamini1995controlling,bhy_2001` for False Discovery Rate (FDR)
    control.

    Parameters
    ----------
    pvals : 1D ndarray
        Array of p-values to threshold
    fdr : float, default=0.1
        Target False Discovery Rate level

    Returns
    -------
    threshold : float
        Threshold value for p-values. P-values below this threshold are rejected.

    References
    ----------
    .. footbibliography::
    """
    n_features = len(pvals)
    pvals_sorted = np.sort(pvals)
    selected_index = 2 * n_features
    for i in range(n_features - 1, -1, -1):
        if pvals_sorted[i] <= fdr * (i + 1) / n_features:
            selected_index = i
            break
    if selected_index <= n_features:
        threshold = pvals_sorted[selected_index]
    else:
        threshold = -1.0
    return threshold


def _ebh_threshold(evals, fdr=0.1):
    """
    e-BH procedure for FDR control described in equation 5 :footcite:`wang2022false`

    Parameters
    ----------
    evals : 1D ndarray
        Array of e-values to threshold
    fdr : float, default=0.1
        Target False Discovery Rate level

    Returns
    -------
    threshold : float
        Threshold value for e-values. E-values above this threshold are rejected.

    References
    ----------
    .. footbibliography::
    """
    n_features = len(evals)
    evals_sorted = np.sort(evals)[::-1]  # sort in descending order
    k_star = 1
    # The for loop over all e-values could be optimized by considering a descending list
    # and stopping when the condition is not satisfied anymore.
    # The condition k * e_k >= n_features / fdr, is defined for k = 1, ..., n_features
    # the for loop therefore starts at k=1
    for k, e_k in enumerate(evals_sorted, start=1):
        if k * e_k >= n_features / fdr:
            k_star = k
    if k_star <= n_features:
        threshold = evals_sorted[k_star - 1]
    else:
        threshold = np.inf
    return threshold


def _bhy_threshold(pvals, reshaping_function=None, fdr=0.1):
    """
        Benjamini-Hochberg-Yekutieli  procedure for
    controlling FDR

    Calculate threshold for Benjamini-Hochberg-Yekutieli procedure
    :footcite:p:`bhy_2001` for False Discovery Rate control,
    with input shape function :footcite:p:`ramdas2017online`.

    Parameters
    ----------
    pvals : 1D ndarray
        Array of p-values to threshold
    reshaping_function : callable, default=None
        Function to reshape FDR threshold. If None, uses sum of reciprocals.
    fdr : float, default=0.1
        Target False Discovery Rate level

    Returns
    -------
    threshold : float
        Threshold value for p-values. P-values below this threshold are rejected.

    References
    ----------
    .. footbibliography::
    """
    n_features = len(pvals)
    pvals_sorted = np.sort(pvals)
    selected_index = 2 * n_features
    # Default value for reshaping function -- defined in
    # Benjamini & Yekutieli (2001)
    if reshaping_function is None:
        temp = np.arange(n_features)
        sum_inverse = np.sum(1 / (temp + 1))
        threshold = _bhq_threshold(pvals, fdr / sum_inverse)
    else:
        for i in range(n_features - 1, -1, -1):
            if pvals_sorted[i] <= fdr * reshaping_function(i + 1) / n_features:
                selected_index = i
                break
        if selected_index <= n_features:
            threshold = pvals_sorted[selected_index]
        else:
            threshold = -1.0
    return threshold


def empirical_knockoff_pval(test_score):
    """
    Compute the empirical p-values from the knockoff+ test.

    Parameters
    ----------
    test_score : 1D ndarray, shape (n_features, )
        Vector of test statistics.

    Returns
    -------
    pvals : 1D ndarray, shape (n_features, )
        Vector of empirical p-values.
    """
    pvals = []
    n_features = test_score.size

    offset = 1  # Offset equals 1 is the knockoff+ procedure.

    test_score_inv = -test_score
    for i in range(n_features):
        if test_score[i] <= 0:
            pvals.append(1)
        else:
            pvals.append(
                (offset + np.sum(test_score_inv >= test_score[i])) / n_features
            )

    return np.array(pvals)


def empirical_knockoff_eval(test_score, ko_threshold):
    """
    Compute the empirical e-values from the knockoff test.

    Parameters
    ----------
    test_score : 1D ndarray, shape (n_features, )
        Vector of test statistics.

    ko_threshold : float
        Threshold level.

    Returns
    -------
    evals : 1D ndarray, shape (n_features, )
        Vector of empirical e-values.
    """
    evals = []
    n_features = test_score.size

    offset = 1  # Offset equals 1 is the knockoff+ procedure.

    for i in range(n_features):
        if test_score[i] < ko_threshold:
            evals.append(0)
        else:
            evals.append(n_features / (offset + np.sum(test_score <= -ko_threshold)))

    return np.array(evals)


def knockoff_threshold(test_score, fdr=0.1):
    """
    Calculate the knockoff threshold based on the procedure stated in the article.

    Original code:
    https://github.com/msesia/knockoff-filter/blob/master/R/knockoff/R/knockoff_filter.R

    Parameters
    ----------
    test_score : 1D ndarray, shape (n_features, )
        Vector of test statistic.

    fdr : float
        Desired controlled FDR (false discovery rate) level.

    Returns
    -------
    threshold : float or np.inf
        Threshold level.
    """
    offset = 1  # Offset equals 1 is the knockoff+ procedure.

    threshold_mesh = np.sort(np.abs(test_score[test_score != 0]))
    np.concatenate(
        [[0], threshold_mesh, [np.inf]]
    )  # if there is no solution, the threshold is inf
    # find the right value of t for getting a good fdr
    # Equation 1.8 of barber2015controlling and 3.10 in Candès 2018
    threshold = 0.0
    for threshold in threshold_mesh:
        false_pos = np.sum(test_score <= -threshold)
        selected = np.sum(test_score >= threshold)
        if (offset + false_pos) / np.maximum(selected, 1) <= fdr:
            break
    return threshold


def fdr_selection(
    test_scores,
    fdr,
    fdr_control="bhq",
    evalues=False,
    reshaping_function=None,
    adaptive_aggregation=False,
    gamma=0.5,
):
    if test_scores.shape[0] == 1:
        threshold_fdr_ = knockoff_threshold(test_scores, fdr=fdr)
        selected = test_scores[0] >= threshold_fdr_
    elif not evalues:
        assert fdr_control != "ebh", "for p-values, the fdr control can't be 'ebh'"
        pvalues = np.array(
            [empirical_knockoff_pval(test_score) for test_score in test_scores]
        )
        aggregated_pval_ = quantile_aggregation(
            pvalues, gamma=gamma, adaptive=adaptive_aggregation
        )
        threshold_fdr_ = fdr_threshold(
            aggregated_pval_,
            fdr=fdr,
            method=fdr_control,
            reshaping_function=reshaping_function,
        )
        selected = aggregated_pval_ <= threshold_fdr_
    else:
        assert fdr_control == "ebh", "for e-value, the fdr control need to be 'ebh'"
        evalues = []
        for test_score in test_scores:
            ko_threshold = knockoff_threshold(test_score, fdr=fdr)
            evalues.append(empirical_knockoff_eval(test_score, ko_threshold))
        aggregated_eval_ = np.mean(evalues, axis=0)
        threshold_fdr_ = fdr_threshold(
            aggregated_eval_,
            fdr=fdr,
            method=fdr_control,
            reshaping_function=reshaping_function,
        )
        selected = aggregated_eval_ >= threshold_fdr_
    return selected
