# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 09:23:19 2026

@author: E.Mousavi
"""
from sklearn.tree import plot_tree

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import shap
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier

from PSSP_plot_utils import evaluate_clustering , pca_plots , print_plot_categorical , print_plot_results
from PSSP_utils import extract_rules

#---------------------------------------------------------------------
# Load data and cluster labels calculated earlier
#---------------------------------------------------------------------
df = pd.read_csv('PSSP_data.csv')
clusters = df['cluster'].values.copy()

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

categorical_feature = ['Sex','Smoking', 'physicalActivity',]#

categorical_data = df[categorical_feature]

#------------------------------------------------------------------------------------------------------
plot_feature = ['Seeking_social_support','Problem_engagement',
                     'Positive_reinterpretation','Acceptance',
                     'Avoidance',
                     'Neuroticism','Extraversion',
                     'Openness', 'Agreeableness', 
                     'Conscientiousness', 
                     'GHQscore','AHEI','Age' , 'BMI' ,
                     'healthy_seeking' ,#'Lifestyle_score',
                     #'Sex','physicalActivity',#'Smoking',
                     #'DepressionScore','AxietyScore',
                     #'IBS','FD'
                     ]


data_plot = df[plot_feature]
df_binary = {}
df_binary['Group'] = clusters
df_binary['Sex'] = categorical_data['Sex']
df_binary['physicalActivity'] = categorical_data['physicalActivity']
df_binary['Smoking'] = categorical_data['Smoking']
df_binary['IBS'] = df['IBS']
df_binary['FD'] = df['FD']
df_binary['GERD'] = df['GERD']
df_binary['FCP'] = df['FCP']
df_binary['FB'] = df['FB']
df_binary['FC'] = df['FC']
#df_binary['FunctionalDiarrhea'] = df['FunctionalDiarrhea']
for fea in df_binary.keys():
    if(np.min(df_binary[fea])==1):
        df_binary[fea] = df_binary[fea]-1

data_name_in = ['Seeking Social Support', 'Problem Engagement',
       'Positive Reinterpretation', 'Acceptance', 'Avoidance', 'Neuroticism',
       'Extraversion', 'Openness', 'Agreeableness', 'Conscientiousness',
       'General Health Questionnaire Score', 'Alternative Healthy Eating Index' , 'Age', 'Body Mass Index']

data_name_bin_in = ['Sex' , 'Physical Activity (PA)' , 'Smoking Status' , 'FC']
data_name_out= ['Depression Score','Anxiety Score' , 'Healthy Seeking Behavior Score']
data_name_bin_out = ['Irritable Bowel Syndrome (IBS)' , 'Functional Dyspepsia (FD)' ,  'GERD','FCP']


df_binary = pd.DataFrame(df_binary)
df_binary1 = df_binary[['Sex' ,'physicalActivity' , 'Smoking', 'FC' , 'Group']]
df_binary2 = df_binary[['IBS', 'FD', 'GERD','FCP', 'Group']]


print_plot_results(data_plot.iloc[:,:14], clusters  , data_name_in)
print_plot_results(df[['DepressionScore','AxietyScore', 'healthy_seeking']] , clusters ,data_name_out)

print_plot_categorical(df_binary1 , clusters , data_name_bin_in )
print_plot_categorical(df_binary2, clusters  , data_name_bin_out)

#-----------------------------------------------------------------------------------------------
#                       Statistical Test
#-----------------------------------------------------------------------------------------------
cluster_mapping = {1: 'C1', 2: 'C2', 3: 'C3', 4: 'C4'}

test_feature = ['Seeking_social_support','Problem_engagement',
                     'Positive_reinterpretation','Acceptance',
                     'Avoidance',
                     'Neuroticism','Extraversion',
                     'Openness', 'Agreeableness', 
                     'Conscientiousness', 
                     'DepressionScore','AxietyScore',
                     'GHQscore' , 'AHEI', 'Age' , 'BMI' ,
                     'healthy_seeking' ,'Lifestyle_score',
                     'Sex','physicalActivity','Smoking',
                     'IBS','FD','GERD','FCP','FC','FB',
                     'DepressionGroups','AnxietyGroups',
                     ]

data_test = df[test_feature].copy()
data_test['group'] = clusters

data_test['group'] = data_test['group'].replace(cluster_mapping)

groups = data_test['group'].unique()
#-------------------------------------------------------------------------
#                       Varible Examination
#-------------------------------------------------------------------------
from utils import comparison_cont_var_across_clusters

continuous_vars = ['Seeking_social_support','Problem_engagement',
                     'Positive_reinterpretation','Acceptance',
                     'Avoidance',
                     'Neuroticism','Extraversion',
                     'Openness', 'Agreeableness', 
                     'Conscientiousness', 
                     'DepressionScore','AxietyScore',
                     'AHEI', 'healthy_seeking','Lifestyle_score',
                     'GHQscore' ,'Age' , 'BMI' ,]

results_continuous_vars = comparison_cont_var_across_clusters(data_test,continuous_vars)
filtered_results_cont = results_continuous_vars[np.logical_or(
    (pd.to_numeric(results_continuous_vars['Adjusted_Pairwise_p'], errors='coerce') < 0.05) ,
    (results_continuous_vars['Pairwise_p'].isna()))]

#------------------------------------------------------------------------------
# 4 to 3 class conversion 
#----------------------------------------------------------------------------
cluster_1_4 = clusters#np.where(clusters !=3)[0]
X_14 = df.copy()

all_features = []
for i in range(1,61):    
    all_features.append("question_" + str(i))

for i in range(1,24):    
    all_features.append("Q" + str(i))
   
X = X_14[all_features]
X = X.reset_index(drop=True)

y = clusters
y[y==4]=3

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
#----------------------------------------------------------------------------
#Three-class classification and feature imprtance caculation
#----------------------------------------------------------------------------
weights = { 0: 2326/646, 1: 2326/235, 2:1}

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
iii = 0

outer_scores = []
shap_feature = []
xgb_features = []
for train_idx, test_idx in outer_cv.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    xgb_model = xgb.XGBClassifier(eval_metric='logloss')
    param_grid = {
        'n_estimators': [50, 100, 200],         # Number of boosting rounds
        'max_depth': [3, 5, 7],                # Maximum tree depth
        'learning_rate': [0.01, 0.1, 0.2],     # Step size shrinkage
        'subsample': [0.6, 0.8, 1.0],          # Fraction of samples used per tree
        'colsample_bytree': [0.6, 0.8, 1.0],   # Fraction of features used per tree
        'gamma': [0, 1, 3, 5],                 # Minimum loss reduction to make a split
    }

    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        #scoring='f1',                    # You can change this to other metrics like 'roc_auc'
        cv=5,                                  # 3-fold cross-validation
        verbose=1,                             # Verbosity level for progress updates
        n_jobs=-1                              # Use all available CPU cores
    )
    
    sample_weights = [weights[int(label)] for label in y_train]
    grid_search.fit(X_train, y_train,sample_weight=sample_weights)
    
    print("Best Parameters:", grid_search.best_params_)
    print("Best Accuracy:", grid_search.best_score_)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    print("Test Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    accuracy = accuracy_score(y_test, y_pred)
    outer_scores.append(accuracy)

    #------------------------------
    #XGBoost Feature Importance
    #-----------------------------        
    importances = best_model.feature_importances_
    importance_df = pd.DataFrame({'Feature': all_features,'Importance': importances}).sort_values(by='Importance', ascending=False)
    print("\nFeature Importances:")
    xgb_features.append(importance_df['Feature'][importance_df['Importance']>0.03])
    
    plt.figure(figsize=(8, 6))
    plt.barh(importance_df['Feature'][:20], importance_df['Importance'][:20], color='blue')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importances (XGBClassifier)')
    plt.gca().invert_yaxis()
    plt.show()
    
    #----------------------------
    #SHAP feature selection
    #---------------------------
    explainer = shap.TreeExplainer(best_model, feature_perturbation="tree_path_dependent")
    shap_values = explainer(X_train)
    shap.summary_plot(shap_values, X_train)
    
    # absolute SHAP values
    shap_abs = np.abs(shap_values.values)  # shape: (samples, features, classes)
    
    # average over samples AND classes
    mean_shap = shap_abs.mean(axis=(0, 2))
    
    mean_shap_values = pd.DataFrame({
        "Feature": X_train.columns,
        "Mean |SHAP Value|": mean_shap
    }).sort_values(by="Mean |SHAP Value|", ascending=False)
    
    shap_feature.append(mean_shap_values['Feature'][mean_shap_values['Mean |SHAP Value|']>.2])
  
    
all_shape_fea = pd.concat(shap_feature).values
value_shap_counts = pd.Series(all_shape_fea).value_counts()
result_shap = value_shap_counts.reset_index()
result_shap.columns = ['Value', 'Repetitions']

all_xgb_values = pd.concat(xgb_features).values
value_xgb_counts = pd.Series(all_xgb_values).value_counts()
result_xgb = value_xgb_counts.reset_index()
result_xgb.columns = ['Value', 'Repetitions']


#----------------------------------------------------------------------------------------------
#                             After feature selection
#----------------------------------------------------------------------------------------------
selected_features = ['question_42','Q5','Q7','question_51',
                     'question_26','Q4','Q18','question_21']

#-----------------------------------------------------------------------------
X_selected = X[selected_features]

sns.heatmap(np.abs(X_selected.corr()))
outer_scores = []

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, test_idx in outer_cv.split(X_selected, y):
    X_train, X_test = X_selected.iloc[train_idx], X_selected.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    sample_weights = [weights[int(label)] for label in y_train]

    xgb_model = xgb.XGBClassifier(
        eval_metric='logloss',  # Evaluation metric for multi-class classification
        #use_label_encoder=False
    )
    param_grid = {
        'n_estimators': [50, 100, 200],        # Number of boosting rounds
        'max_depth': [3, 5, 7],                # Maximum tree depth
        'learning_rate': [0.01, 0.1, 0.2],     # Step size shrinkage
        'subsample': [0.6, 0.8, 1.0],          # Fraction of samples used per tree
        'colsample_bytree': [0.6, 0.8, 1.0],   # Fraction of features used per tree
        'gamma': [0, 1, 5],                    # Minimum loss reduction to make a split
    }
    
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        # scoring='f1',                    # You can change this to other metrics like 'roc_auc'
        cv=5,                                  # 5-fold cross-validation
        verbose=1,                             # Verbosity level for progress updates
        n_jobs=-1                              # Use all available CPU cores
    )
    
    grid_search.fit(X_train, y_train,sample_weight=sample_weights)
    
    print("Best Parameters:", grid_search.best_params_)
    print("Best Accuracy:", grid_search.best_score_)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_train_pred = best_model.predict(X_train)

    print("Test Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report XGB:\n", classification_report(y_test, y_pred))

    
    surrogate_tree = DecisionTreeClassifier(max_depth=3)  # Limit depth for interpretability
    param_grid_dt = {
        'max_depth': [3],#[2, 3, 4, 5, None],                # Tree depth
        'min_samples_split': [2, 3, 4],                # Minimum samples to split a node
        'min_samples_leaf': [1, 2, 3],                 # Minimum samples in a leaf node
        'criterion': ['gini', 'entropy']
        }
    
    grid_search_dt = GridSearchCV( 
        estimator=surrogate_tree,
        param_grid= param_grid_dt,
        # scoring='f1',                    
        cv=5,verbose=1,n_jobs=-1 
        )

    grid_search_dt.fit(X_train, y_train_pred,sample_weight=sample_weights)

    best_model_dt = grid_search_dt.best_estimator_
    y_pred_dt = best_model_dt.predict(X_test)
    
    plt.figure(figsize=(30, 15)) 
    plot_tree(best_model_dt, feature_names=selected_features,
              class_names=['C1', 'C2', 'C3', 'C4'], filled=True)
    plt.show()
    
    print("\nClassification Report DT:\n", 
          classification_report(y_test, y_pred_dt))


report = classification_report(y_test, y_pred_dt, output_dict=True)
report_df = pd.DataFrame(report).transpose()

extract_rules(best_model_dt, selected_features)

