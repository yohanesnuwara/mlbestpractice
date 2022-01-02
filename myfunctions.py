import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate, train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import plot_confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

def OrdinalEncoder(df, ordinal_col, encode_dict):
    encode_dict = {ordinal_col: encode_dict}
    return df.replace(encode_dict)

def NominalEncoder(df, nominal_cols):
    df_dummy = pd.get_dummies(df[nominal_cols])
    return pd.concat((df, df_dummy), axis=1).drop(columns=nominal_cols)

def compare_classifiers(X_train, y_train, pipeline, cv, seed=42):
    # Ignore warnings
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)    
    
    # List of classifiers
    classifiers = [LogisticRegression(), GaussianNB(), 
                   KNeighborsClassifier(), SVC(kernel='linear'), 
                   SVC(kernel='rbf'), DecisionTreeClassifier(), 
                   RandomForestClassifier(), CatBoostClassifier(verbose=False),
                   AdaBoostClassifier(), LGBMClassifier(), XGBClassifier()] 
    
    # Define multiple scoring metrics
    scoring = {
        'acc': 'accuracy',
        'prec_macro': 'precision_macro',
        'rec_macro': 'recall_macro',
        'f1_macro': 'f1_macro',
        'roc_auc': 'roc_auc'
    }    
 
    # Make dataframe
    compare_model = pd.DataFrame(columns=['Classifier', 'Accuracy', 'Macro Precision', 
                                          'Macro Recall', 'Macro F1', 'ROC'])
    compare_model['Classifier'] = classifiers

    for i in range(len(classifiers)):
        if i>0:
            # When classifier move to the second, remove the previous classifier by pop
            pipeline.steps.pop(-1)
        # Update classifier to existing pipeline
        pipeline.steps.append(('classifier', classifiers[i]))

        # Cross-validation
        cv_scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring)

        # Print scoring results from dictionary
        cv_means = []
        for metric_name, metric_value in cv_scores.items():
            mean = np.mean(metric_value)
            mean = np.round(mean, 3) # rounding
            cv_means.append(mean)

        # Put in dataframe
        compare_model.loc[i,1:] = cv_means[2:]
    
    # Change sklearn object in Classifier column to names
    names = ['LogReg', 'Bayes', 'KNN', 'Linear SVC', 'RBF SVC', 'DT', 
             'RF', 'CatBoost', 'AdaBoost', 'LightGBM', 'XGBoost']
    compare_model['Classifier'] = names
    
    # Sort model by recall, precision, F1, and ROC
    sort_rec = compare_model.sort_values(by='Macro Recall', ascending=False)
    sort_prec = compare_model.sort_values(by='Macro Precision', ascending=False)
    sort_f1 = compare_model.sort_values(by='Macro F1', ascending=False)
    sort_roc = compare_model.sort_values(by='ROC', ascending=False)

    # Bar plot
    f, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2,2, figsize=(7,6))

    sns.lineplot(data=sort_roc[:5], x='Classifier', y='ROC', marker='o', color='red', ax=ax1)
    ax1.tick_params(axis='x', rotation=90)
    ax1.set_title('ROC')

    sns.lineplot(data=sort_f1[:5], x='Classifier', y='Macro F1', marker='o', color='green', ax=ax2)
    ax2.tick_params(axis='x', rotation=90)
    ax2.set_title('Macro F1')

    sns.lineplot(data=sort_rec[:5], x='Classifier', y='Macro Recall', color='blue', marker='o', ax=ax3)
    ax3.tick_params(axis='x', rotation=90)
    ax3.set_title('Macro Recall')

    sns.lineplot(data=sort_prec[:5], x='Classifier', y='Macro Precision', color='purple', marker='o', ax=ax4)
    ax4.tick_params(axis='x', rotation=90)
    ax4.set_title('Macro Precision')

    plt.tight_layout()
        
    return compare_model

def cv_classification_multiple_score(X_train, y_train, pipeline, cv, scoring):
    # Cross-validation
    cv_scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring)

    # Print scoring results from dictionary
    cv_means = []
    for metric_name, metric_value in cv_scores.items():
        mean = np.mean(metric_value)
        mean = np.round(mean, 3) # rounding
        print(f'{metric_name}: {np.round(metric_value, 3)}, Mean: {np.round(mean, 3)}')   
