# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 10:20:03 2026

@author: E.Mousavi
"""

import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score

def pca_plots(explained_variance_ratio):
    explained_variance = np.cumsum(explained_variance_ratio)
    plt.plot(explained_variance)
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Explained Variance')
    plt.grid()
    plt.show()

    plt.plot(explained_variance_ratio)
    plt.xlabel('Number of Components')
    plt.ylabel('Explained Variance')
    plt.title('PCA Explained Variance')
    plt.grid()
    plt.show()

    from kneed import KneeLocator  # Install using `pip install kneed`
    cumulative_variance =explained_variance 
    knee_locator = KneeLocator(range(1, len(explained_variance)+1), cumulative_variance, curve="convex", direction="decreasing")
    elbow_point = knee_locator.knee
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(explained_variance) + 1),
    cumulative_variance, marker='o', label="Cumulative Explained Variance")
    plt.axvline(elbow_point, color='r', linestyle='--', label=f"Elbow Point ({elbow_point})")
    plt.title("Elbow Point in PCA Explained Variance")
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.legend()
    plt.grid()
    plt.show()


def print_plot_results(data_in , labels, feature_name = None):
    import numpy as np
    import matplotlib.pyplot as plt

    if labels.min()==0:
        labels+=1

    in_var = data_in.keys()
    if feature_name:
        fea_name = feature_name
    else:
        fea_name = in_var
    No_clusters = max(labels)
    n_vars = len(in_var)
    n_rows, n_cols = 5, 4  # 2 rows, 3 columns for the 6 variables
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 20),constrained_layout=True)
    axes = axes.flatten()  # Flatten axes to iterate easily

    axes = axes.flatten()

    for i, fea in enumerate(in_var):
        data = [data_in[fea][labels == j + 1] for j in range(No_clusters)]

        # Plot median line
        median_values = [np.median(data_in[fea]) for _ in range(No_clusters + 2)]
        axes[i].plot(range(0, No_clusters + 2), median_values, color='gray', linestyle='-.')

        mean_values = [np.mean(data_in[fea]) for _ in range(No_clusters + 2)]
        axes[i].plot(range(0, No_clusters + 2), mean_values, color='black', linestyle='--')

        axes[i].legend(loc="upper right", labels=['Total Median' ,'Total Mean'])#,fontsize=8)

        axes[i].set_title(fea_name[i])
        # axes[i].boxplot(data)
        axes[i].boxplot(data,
                medianprops=dict(color='black'))

        x_axis = range(1, No_clusters + 1)
        x_tic = ['C' + str(k + 1) for k in range(No_clusters)]
        axes[i].set_xticks(x_axis)
        axes[i].set_xticklabels(x_tic, fontsize=12)

    plt.tight_layout()
    plt.show()

    unique, counts = np.unique(labels, return_counts=True)
    print(dict(zip(unique, counts)))


def evaluate_clustering(embeddings, labels):
    #silhouette_avg = silhouette_score(embeddings, labels)
    davies_bouldin = davies_bouldin_score(embeddings, labels)
    calinski_harabasz = calinski_harabasz_score(embeddings, labels)

    return davies_bouldin,calinski_harabasz


def print_plot_categorical(df_binary, clusters , feature_name):
    
    import matplotlib.pyplot as plt

    df_binary = pd.DataFrame(df_binary.copy())
    group_counts = df_binary['Group'].value_counts()
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(group_counts, labels=group_counts.index, 
           autopct='%1.1f%%', startangle=90, 
           colors=plt.cm.Set2.colors)
    plt.title('Pie Plot of Percent of Samples in Each Group')
    # plt.legend(labels=group_counts.index, loc="upper left")
    plt.show()

    if 'Sex' in df_binary.keys():
        df_binary = pd.DataFrame(df_binary.copy())
        group_counts = df_binary['Sex'].value_counts()
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.pie(group_counts, #labels=group_counts.index, 
               autopct='%1.1f%%', startangle=90, 
               colors=['#f72585','#4361ee'])
        plt.title('Pie Plot of Number of Samples in Each Group')
        plt.legend(title= 'Sex',labels=['Female', 'Male'], loc="upper right")
        plt.show()
    
    
        grouped_counts = df_binary.groupby(["Group" , 'Sex']).size().unstack(fill_value=0)
         # Optional: Rename index if Group contains numeric categories like 1, 2, 3, 4
        custom_labels = {1: "C1", 2: "C2", 3: "C3", 4: "C4"}  # Adjust this mapping to match your actual Group values
        grouped_counts.rename(index=custom_labels, inplace=True)
        
        # Create percentage DataFrame
        percentages = grouped_counts.div(grouped_counts.sum(axis=1), axis=0) * 100
        
        # Plotting
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
        
        # Plot 1: Count
        grouped_counts.plot(kind="bar", stacked=True, ax=axes[0], 
                            # color=['#4361ee','#f72585'])  # Male, Female
                            color = ['#a9a9a9', '#d3d3d3'] , # apply to Male and Female bars
                            edgecolor='black')        # Black outlines around bars
        
        axes[0].set_title("Count of Sex by Group", fontsize=14)
        axes[0].set_xlabel("Groups", fontsize=12)
        axes[0].set_ylabel("Count", fontsize=12)
        axes[0].legend(title='Sex', labels=['Male', 'Female'], loc="upper right")
        axes[0].grid(axis="y", linestyle="--", alpha=0.7)
        axes[0].tick_params(axis='x', rotation=0)
        
        # Plot 2: Percentage
        percentages.plot(kind="bar", stacked=True, ax=axes[1], 
                         color = ['#a9a9a9', '#d3d3d3'] , # apply to Male and Female bars
                         # color=['#4361ee','#f72585'])  # Male, Female
                         edgecolor='black')        # Black outlines around bars
        
        axes[1].set_title("Percentage of Sex by Group", fontsize=14)
        # axes[1].set_xlabel("Groups", fontsize=12)
        axes[1].set_ylabel("Percentage (%)", fontsize=12)
        axes[1].legend(title='Sex', labels=['Male', 'Female'], loc="upper right")
        axes[1].grid(axis="y", linestyle="--", alpha=0.7)
        axes[1].tick_params(axis='x', rotation=0)
        
        plt.tight_layout()
        plt.show()

    features = df_binary.keys()[:-1]
    # n_cols = 4  # Number of columns in subplot grid
    # n_rows = (n_features + n_cols - 1) // n_cols  # Calculate rows dynamically

    # fig, axes = plt.subplots(n_rows, n_cols, figsize=(8, 4), constrained_layout=True)
    n_rows, n_cols = 5, 4  # 2 rows, 3 columns for the 6 variables
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 20),constrained_layout=True)
    axes = axes.flatten()  # Flatten axes to iterate easily

    # Plot each feature
    for i, fea in enumerate(features):
        # Group and calculate percentages
        grouped_counts = df_binary.groupby(["Group", fea]).size().unstack(fill_value=0)
        custom_labels = {0: "C1", 1: "C2", 2: "C3", 3: "C4"}  # Adjust this mapping to match your actual Group values
        grouped_counts.rename(index=custom_labels, inplace=True)        
        percentages = grouped_counts.div(grouped_counts.sum(axis=1), axis=0) * 100  # Convert to percentages

        # Plot the stacked bar chart
        percentages.plot(kind="bar", stacked=True, ax=axes[i], 
                            color = ['#a9a9a9', '#d3d3d3'] , # apply to Male and Female bars
                            edgecolor='black')        # Black outlines around bars
                         # color = ["#57cc99", "#e5383b"])
                        # color=["green", "lightcoral", "orange", "green", "purple"][:len(grouped_counts.columns)])
        
        # Customize the plot
        axes[i].set_title(feature_name[i])#, fontsize=8)
        # axes[i].set_xlabel("Groups", fontsize=10)
        axes[i].set_ylabel("Percentage (%)")#, fontsize=7)
        axes[i].tick_params(axis='x', rotation=0)
        if(fea == "physicalActivity"):
            fea = "PA"
        axes[i].legend(title=fea, loc="upper right")#, fontsize=5)
        axes[i].grid(axis="y", linestyle="--", alpha=0.7)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Show the final plot
    plt.show()