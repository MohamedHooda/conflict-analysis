import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, f_oneway, ttest_ind, fisher_exact, wilcoxon, mannwhitneyu


def one_way_anova(*groups, alpha):
    # Perform one-way ANOVA
    f_statistic, p_value = f_oneway(*groups)

    # Print the results
    print("F-statistic:", f_statistic)
    print("P-value:", p_value)

    # Interpret the results
    if p_value < alpha:
        print("Reject the null hypothesis. There is a significant difference between at least two groups.")
    else:
        print("Fail to reject the null hypothesis. There is no significant difference between the groups.")


def pairwise_t_tests(data, alpha):
    """
    Perform pairwise t-tests for all combinations of groups in a DataFrame.

    Parameters:
    - dataframe (pd.DataFrame): DataFrame with columns representing different groups.

    Returns:
    - results (list of dictionaries): List containing dictionaries with t-test results.
    """
    results = []
    groups = list(data.keys())
    groups_with_differences = []
    groups_without_differences = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            group1 = data[groups[i]]
            group2 = data[groups[j]]

            # Perform welch paired t-test
            t_stat, p_value = ttest_ind(group1, group2, equal_var=False)

            # Interpret the results
            if p_value < alpha:
                groups_with_differences.append(
                    str(groups[i])+" and " + str(groups[j]))
            else:
                groups_without_differences.append(
                    str(groups[i])+" and " + str(groups[j]))

            # Store results in a dictionary
            result_dict = {
                'Group1': groups[i],
                'Group2': groups[j],
                'T-statistic': t_stat,
                'P-value': p_value
            }

            results.append(result_dict)

    print("groups with significant differences: ", groups_with_differences)
    print("group wihout signifcant difference: ", groups_without_differences)
    return results


def chi_square_test(group1_labels, group2_labels, alpha):
    """
    Perform a chi-square test to determine if there is a significant difference
    in the distribution of labels between two groups.

    Parameters:
    - group1_labels: List of labels for group 1 (e.g., ["hate speech", "not hate speech", ...]).
    - group2_labels: List of labels for group 2 (e.g., ["hate speech", "not hate speech", ...]).

    Returns:
    - result: String indicating the result of the test ("significant" or "not significant").
    """
    # Create a contingency table
    contingency_table = pd.crosstab(pd.Series(group1_labels, name='Group1'),
                                    pd.Series(group2_labels, name='Group2'))

    # Perform chi-square test
    chi2, p, _, _ = chi2_contingency(contingency_table)

    print("p-value:", p)
    # Check p-value to determine significance
    if p < alpha:
        print("There is a significant difference between the groups.")
    else:
        print("There is no significant difference between the groups.")


def fishers_exact_test(group1_labels, group2_labels, alpha):
    """
    Perform Fisher's exact test to determine if there is a significant difference
    in the distribution of labels between two groups.

    Parameters:
    - group1_labels: List of labels for group 1 (e.g., ["hate speech", "not hate speech", ...]).
    - group2_labels: List of labels for group 2 (e.g., ["hate speech", "not hate speech", ...]).

    Returns:
    - result: String indicating the result of the test ("significant" or "not significant").
    """
    # Create a contingency table
    contingency_table = pd.crosstab(pd.Series(group1_labels, name='Group1'),
                                    pd.Series(group2_labels, name='Group2'))

    # Perform Fisher's exact test
    odds_ratio, p = fisher_exact(contingency_table)

    # Check p-value to determine significance
    print("p-value:", p)
    # Check p-value to determine significance
    if p < alpha:
        print("There is a significant difference between the groups.")
    else:
        print("There is no significant difference between the groups.")


def paired_mannwhitneyu_rank_test(sample1, sample2, alpha):

    # Perform Wilcoxon signed-rank test
    statistic, p_value = mannwhitneyu(sample1, sample2)

    # Print the results
    print("Mann-Whitney U statistic:", statistic)
    print("p-value:", p_value)

    # Interpret the results
    if p_value < alpha:
        print("Reject the null hypothesis. There is a significant difference between the paired samples.")
    else:
        print("Fail to reject the null hypothesis. There is no significant difference between the paired samples.")
