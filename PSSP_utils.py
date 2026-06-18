# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 09:30:34 2026

@author: E.Mousavi
"""

from scipy import stats

import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.datasets import make_blobs, make_moons
from sklearn.ensemble import IsolationForest
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDOneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline

import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import NMF
from matplotlib.pyplot import xticks

import pyreadstat
from scipy.stats import kruskal
import scikit_posthocs as sp
import scipy.stats as stats
from itertools import combinations

import requests
from bs4 import BeautifulSoup

import numpy as np
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment as hungarian

def clustering_accuracy(true_labels, predicted_labels):
    # Create a cost matrix based on label matching
    D = max(predicted_labels.max(), true_labels.max()) + 1
    cost_matrix = np.zeros((D, D), dtype=np.int64)
    
    for i in range(predicted_labels.size):
        cost_matrix[predicted_labels[i], true_labels[i]] += 1
    
    # Use the Hungarian algorithm to solve the assignment problem
    row_ind, col_ind = hungarian(cost_matrix.max() - cost_matrix)
    
    # Calculate the accuracy based on the optimal assignment
    return accuracy_score(true_labels, [col_ind[i] for i in predicted_labels])


# Function to calculate Cramér's V for effect size
def cramers_v(confusion_matrix):
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    return np.sqrt(phi2 / min(r-1, k-1))


# Function to calculate epsilon squared (ε²) for effect size
def epsilon_squared(H, N, k):
    return (H - (k - 1)) / (N - k)

def comparison_cont_var_across_clusters(df_total,index_var):
    # Initialize results DataFrame dynamically
    sorted_clusters = sorted(df_total['group'].unique())

    results_cont = pd.DataFrame(columns=['Variable', 'Kruskal_p', 'Effect_Size'] + 
                                [f'C{i+1} (Mean±SD)' for i in range(len(sorted_clusters))] +
                                ['Pairwise_Comparison', 'Pairwise_p', 'Adjusted_Pairwise_p'])

    # Step 1: Descriptive Statistics and Kruskal-Wallis Test
    for var in index_var:  # continuous_vars:
        df = df_total[[var, 'group']].dropna()
        clusters = sorted(df_total['group'].unique())
        cluster_means = []
        cluster_sds = []
        cluster_data = []
        
        for cluster in clusters:
            cluster_values = df[df['group'] == cluster][var]
            cluster_means.append(cluster_values.mean())
            cluster_sds.append(cluster_values.std())
            cluster_data.append(cluster_values)
        
        # Kruskal-Wallis Test
        H, p = stats.kruskal(*cluster_data)
        effect_size = epsilon_squared(H, len(df), len(cluster_data))
        
        result_row = {
            'Variable': var,
            'Kruskal_p': np.round(p , 3),
            'Effect_Size': np.round(effect_size , 3),
            'Pairwise_Comparison': None,
            'Pairwise_p': None,
            'Adjusted_Pairwise_p': None
        }
        
        for i, cluster in enumerate(clusters):
            result_row[f'C{i+1} (Mean±SD)'] = f"{cluster_means[i]:.2f}±{cluster_sds[i]:.2f}"
        
        results_cont = pd.concat([results_cont, pd.DataFrame([result_row])], ignore_index=True)
        
        # Step 2: Pairwise Comparisons (Post-Hoc Dunn's Test with Bonferroni Correction)
        posthoc_results = sp.posthoc_dunn(df, val_col=var, group_col='group', p_adjust='bonferroni')
        pairs = list(combinations(clusters, 2))
        
        for pair in pairs:
            p_pair = posthoc_results.loc[pair[0], pair[1]]
            adjusted_p = p_pair  # Already adjusted by 'bonferroni' in posthoc_dunn
            
            pairwise_row = {
                'Variable': var,
                'Kruskal_p': None,
                'Effect_Size': None,
                'Pairwise_Comparison': f"{pair[0]} vs {pair[1]}",
                'Pairwise_p': p_pair,
                'Adjusted_Pairwise_p': np.round(adjusted_p,3)
            }
            
            for i in range(len(clusters)):
                pairwise_row[f'C{i+1} (Mean±SD)'] = None
            
            results_cont = pd.concat([results_cont, pd.DataFrame([pairwise_row])], ignore_index=True)

    return results_cont

def comparison_cat_var_across_clusters(df_total,categorical_vars):

    cluster_labels = sorted(df_total['group'].unique())

    #cluster_labels = df_total['group'].unique().tolist()
    
    # Initialize results DataFrame with dynamic cluster columns
    cluster_columns = [f'{cluster} (Count,%)' for cluster in cluster_labels]
    results_cat = pd.DataFrame(columns=['Variable', 'Chi2_p', 'Effect_Size'] + 
                               cluster_columns + 
                               ['Pairwise_Comparison', 'Pairwise_p', 
                                'Adjusted_Pairwise_p'])
    
    # Step 1: Descriptive Statistics and Chi-Square Test
    for var in categorical_vars:
        df = df_total[[var, 'group']].dropna()
        cluster_counts = []
        
        for cluster in cluster_labels:
            cluster_data = df[df['group'] == cluster]
            count_1 = cluster_data[var].sum()
            if var == "DepressionGroups" or var == "AnxietyGroups":
                count_1 = np.sum(cluster_data[var] == 3)
            percentage_1 = (count_1 / len(cluster_data)) * 100 if len(cluster_data) > 0 else 0
            cluster_counts.append(f"{count_1} ({percentage_1:.2f}%)")
        
        # Overall Chi-Square Test
        contingency_table = pd.crosstab(df['group'], df[var])
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
        effect_size = cramers_v(contingency_table)
        expected_min = np.min(expected)
        if(expected_min<5):                
            print(f"for {var} you need to perform onother test")

        results_cat = pd.concat([results_cat, pd.DataFrame({
            'Variable': [var],
            'Chi2_p': [np.round(p , 3)],
            'Effect_Size': [np.round(effect_size,2)],
            **{col: cluster_counts[i] for i, col in enumerate(cluster_columns)},
            'Pairwise_Comparison': [None],
            'Pairwise_p': [None],
            'Adjusted_Pairwise_p': [None]
        })], ignore_index=True)
        
        # Step 2: Pairwise Comparisons (Post-Hoc Analysis with Bonferroni Correction)
        pairs = list(combinations(cluster_labels, 2))
        alpha = 0.05 / len(pairs)  # Bonferroni adjustment
        
        for pair in pairs:
            subset = df[df['group'].isin(pair)]
            pair_table = pd.crosstab(subset['group'], subset[var])
            
            chi2_pair, p_pair, dof_pair, expected_pair = stats.chi2_contingency(pair_table)
            adjusted_p = min(p_pair * len(pairs), 1.0)  # Bonferroni correction
            
            results_cat = pd.concat([results_cat, pd.DataFrame({
                'Variable': [var],
                'Chi2_p': [None],
                'Effect_Size': [None],
                **{col: None for col in cluster_columns},
                'Pairwise_Comparison': [f"{pair[0]} vs {pair[1]}"],
                'Pairwise_p': [p_pair],
                'Adjusted_Pairwise_p': [np.round(adjusted_p,3)]
            })], ignore_index=True)
    
    return results_cat


def extract_rules(tree, feature_names):
    from sklearn.tree import _tree
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    def recurse(node, depth):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print(f"{'|   ' * depth}{name} <= {threshold:.2f}")
            recurse(tree_.children_left[node], depth + 1)
            print(f"{'|   ' * depth}{name} > {threshold:.2f}")
            recurse(tree_.children_right[node], depth + 1)
        else:
            print(f"{'|   ' * depth}Class: {tree_.value[node]}")

    recurse(0, 0)
