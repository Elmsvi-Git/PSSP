# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 09:24:29 2026

@author: E. Mousavi

"""
# print("scikit-learn:", sklearn.__version__)
# print("numpy:", np.__version__)
# print("scipy:", scipy.__version__)
# print("pandas:", pd.__version__)
# print("xgboost:", xgboost.__version__)

#------------------------------------------------------------------------------
import xgboost
from sklearn.tree import plot_tree

import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, accuracy_score

import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import shap
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score

from sklearn.tree import DecisionTreeClassifier



from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from utils_plots import evaluate_clustering , pca_plots , print_plot_categorical , print_plot_results
from utils_plots import print_plot_categorical

from sklearn.manifold import SpectralEmbedding
from GUDMM import caculate_affinity, ditance_dependency_mixed_matrix
from sklearn.cluster import SpectralClustering
# from silhouette_score , calinski_harabasz_score,davies_bouldin_score
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score

#------------------------------------------------------------------------------
questions_file = 'questions.txt'
with open(questions_file, 'r') as file:
    questions = [line.strip() for line in file.readlines()]

survey_data_file = 'all_data_psycho_new6.csv'
df = pd.read_csv(survey_data_file)

df.keys()
df.shape

#-----------------------------------------------------------------------
# Seperation of ordinal and nominal varibles
#-----------------------------------------------------------------------
ouput_var = ['Neuroticism','Extraversion',
             'Openness', 'Agreeableness', 
             'Conscientiousness', 
             'DepressionScore','AxietyScore',
             'healthy_seeking' ,
             'IBS','FD',
             'Lifestyle_score',
             'GERD' , 'FCP', 'FB' , 'FC' , 'FunctionalDiarrhea',
             ]


input_var = ['Seeking_social_support','Problem_engagement',
          'Positive_reinterpretation',
          'Avoidance','Acceptance',
          'GHQscore' , 'AHEI','BMI' , 
          'physicalActivity' , 'Smoking',
          'Age' , 'Sex']
#-----------------------------------------------------------------------
df['physicalActivity']=np.where(df['physicalActivity']<3, 0, 1)
df['Smoking']=np.where(df['Smoking']<2, 0, 1)
df['Sex']=np.where(df['Sex']==1, 0, 1)

data_in = df[input_var].copy()
data_out = df[ouput_var].copy()

#------------------------------------------------------------------------------
#                   Load the trained embedding 
#------------------------------------------------------------------------------
save_path = 'combined_embeddings_5fa_distilbert-base-uncased_28_1.npy'
combined_embeddings = np.load(save_path)
print("Array loaded successfully!")
combined_embeddings.shape

#------------------------------------------------------------------------------
#                   PCA on learned embedding
#------------------------------------------------------------------------------
# Standardize the embeddings before PCA
scaler = StandardScaler()
scaled_embeddings = scaler.fit_transform(combined_embeddings)
scaled_embeddings.shape

# Apply PCA for dimensionality reduction
no_pc = 8 #14#30 for 90
pca = PCA(n_components=no_pc)  # Experiment with different values
reduced_embeddings = pca.fit_transform(scaled_embeddings)
explained_variance_ratio = pca.explained_variance_ratio_

#---------------------------------------------------------------------
# Standardize all the embeddings After PCA
# combined_embeddings_scaled: the scaled embeding for pca reduced embeddings
scaler = StandardScaler()
combined_embeddings_scaled = scaler.fit_transform(reduced_embeddings)

#------------------------------------------------------------------------------
categorical_feature = ['Sex','Smoking', 'physicalActivity',]#
categorical_data = df[categorical_feature]

for fea in categorical_feature:
    if np.min(categorical_data[fea])==0:
        categorical_data[fea] = (categorical_data[fea]+1).astype(int)
        print(np.min(categorical_data[fea]))
        
# Normalize numerical features
scaler = StandardScaler()
numerical_feature = ['Seeking_social_support','Problem_engagement',
                     'Positive_reinterpretation','Acceptance',
                     'Avoidance',
                     'GHQscore' , 
                     'AHEI', 'Age' , 'BMI',
                     ]#9
numerical_scaled = scaler.fit_transform(df[numerical_feature])

# Combine encoded categorical and normalized numerical features
non_text_features = np.hstack((numerical_scaled, categorical_data)) #numerical_scaled#

# Combine all features
scaled_embeddings_all = np.hstack((combined_embeddings_scaled, non_text_features))


#------------------------------------------------------------------------------
reduced_embeddings_df = pd.DataFrame(data = reduced_embeddings, 
                                     columns = ['pc'+str(i) for i in range(reduced_embeddings.shape[1])])

data_total = pd.concat((reduced_embeddings_df, data_out),axis = 1)
cor_matrix = np.abs(data_total.corr())
plt.figure(figsize=(20, 20))
sns.heatmap(cor_matrix, annot=True, cmap="coolwarm", fmt=".1f", 
            linewidths=0.5,annot_kws={"size": 8} )
plt.title("Correlation Heatmap")
plt.show()

data_total = pd.concat((reduced_embeddings_df, data_in),axis = 1)
cor_matrix = np.abs(data_total.corr())
plt.figure(figsize=(20, 20))
sns.heatmap(cor_matrix, annot=True, cmap="coolwarm", fmt=".1f", 
            linewidths=0.5,annot_kws={"size": 8} )
plt.title("Correlation Heatmap")
plt.show()

#------------------------------------------------------------------------------
# Distance Claculation and Clustering
#------------------------------------------------------------------------------

distance_matrix = ditance_dependency_mixed_matrix(scaled_embeddings_all ,
                                                  17 , 0, 'DM5')#no_cot, no ord: 0,no_nom:3
plt.figure()
plt.imshow(distance_matrix)

affinity_matrix = caculate_affinity(distance_matrix, type_affinity='global' ,
                                    p = 3, S = 11) #5,16
plt.figure()
plt.title("Affinity Distribution p: 5, s:16")
plt.imshow(affinity_matrix)

clustering = SpectralClustering(n_clusters=4, affinity="precomputed",
                                assign_labels='kmeans',random_state=42)

clusters = clustering.fit_predict(affinity_matrix)
