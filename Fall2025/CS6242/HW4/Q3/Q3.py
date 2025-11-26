#!/usr/bin/env python
# coding: utf-8

# # Q3 Using Scikit-Learn

# # Imports
# Do not modify

# In[1]:


#export
import pkg_resources
from pkg_resources import DistributionNotFound, VersionConflict
from platform import python_version
import numpy as np
import pandas as pd
import time
import gc
import random
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer


# In[2]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import tests as tests


# # Verify your Python version and setup

# In[3]:


#function to check setup
def check_env_setup():
    dependencies = open("requirements.txt").readlines()
    try:
        pkg_resources.require(dependencies)
        print("✅ ALL GOOD")
    except DistributionNotFound as e:
        print("⚠️ Library is missing")
        print(e)
    except VersionConflict as e:
        print("⚠️ Library version conflict")
        print(e)
    except Exception as e:
        print("⚠️ Something went wrong")
        print(e)


# In[4]:


# verify the environment setup
check_env_setup()


# # Add your Georgia Tech Username

# In[5]:


#export
class GaTech():
    # Change to your GA Tech Username
    # NOT your 9-Digit GTId
    def GTusername(self):
        gt_username = "urafi3"
        return gt_username


# # Q3.1 Data Import
# Now for the fun stuff. Let’s import some data!

# In[6]:


#export
class Data():
    # ------------- dataAllocation() -------------
    def dataAllocation(self, path):
        df = pd.read_csv(path)

        # Detect label column
        target_candidates = ['Class', 'class', 'diagnosis', 'Diagnosis', 'target', 'Target', 'label', 'Label', 'outcome', 'Outcome']
        target_col = None
        for c in target_candidates:
            if c in df.columns:
                target_col = c
                break
        if target_col is None:
            target_col = df.columns[-1]

        # Drop ID columns if present
        id_candidates = ['id', 'ID', 'Id', 'patient_id', 'PatientID', 'Patient Id']
        for idc in id_candidates:
            if idc in df.columns and idc != target_col:
                df = df.drop(columns=[idc])
                break

        # Separate features and labels
        y_data = df[target_col].copy()
        x_data = df.drop(columns=[target_col]).copy()

        # Drop non-numeric feature columns
        non_numeric = [c for c in x_data.columns if not np.issubdtype(x_data[c].dtype, np.number)]
        if len(non_numeric) > 0:
            x_data = x_data.drop(columns=non_numeric)

        # Impute missing values
        if x_data.isna().any().any():
            imputer = SimpleImputer(strategy='mean')
            x_data[:] = imputer.fit_transform(x_data)

        return x_data, y_data

    # ------------- trainSets() -------------
    def trainSets(self, x_data, y_data):
        x_train, x_test, y_train, y_test = train_test_split(
            x_data, y_data, test_size=0.3, shuffle=True, random_state=614
        )
        return x_train, x_test, y_train, y_test

##################################################
##### Do not add anything below this line ########
tests.dataTest(Data)
##################################################


# # Q3.2 Linear Regression 

# In[7]:


#export
class LinearRegressionModel():
    # ------------- linearClassifier() -------------
    def linearClassifier(self, x_train, x_test, y_train):
        lr = LinearRegression()
        lr.fit(x_train, y_train)
        y_predict_train = lr.predict(x_train)
        y_predict_test = lr.predict(x_test)
        return y_predict_train, y_predict_test

    # ------------- lgTrainAccuracy() -------------
    def lgTrainAccuracy(self, y_train, y_predict_train):
        rounded = np.where(np.array(y_predict_train) >= 0.5, 1, 0)
        train_accuracy = accuracy_score(np.array(y_train), rounded)
        return train_accuracy

    # ------------- lgTestAccuracy() -------------
    def lgTestAccuracy(self, y_test, y_predict_test):
        rounded = np.where(np.array(y_predict_test) >= 0.5, 1, 0)
        test_accuracy = accuracy_score(np.array(y_test), rounded)
        return test_accuracy
    
##################################################
##### Do not add anything below this line ########
tests.linearTest(Data,LinearRegressionModel)
##################################################


# # Q3.3 Random Forest Classifier

# In[8]:


#export
class RFClassifier():

    # ------------- randomForestClassifier() -------------
    def randomForestClassifier(self, x_train, x_test, y_train):
        rf_clf = RandomForestClassifier(random_state=614)
        rf_clf.fit(x_train, y_train)
        y_predict_train = rf_clf.predict(x_train)
        y_predict_test = rf_clf.predict(x_test)
        return rf_clf, y_predict_train, y_predict_test

    # ------------- rfTrainAccuracy() -------------
    def rfTrainAccuracy(self, y_train, y_predict_train):
        return accuracy_score(np.array(y_train), np.array(y_predict_train))

    # ------------- rfTestAccuracy() -------------
    def rfTestAccuracy(self, y_test, y_predict_test):
        return accuracy_score(np.array(y_test), np.array(y_predict_test))

    # ------------- rfFeatureImportance() -------------
    def rfFeatureImportance(self, rf_clf):
        return rf_clf.feature_importances_

    # ------------- sortedRFFeatureImportanceIndicies() -------------
    def sortedRFFeatureImportanceIndicies(self, rf_clf):
        return np.argsort(rf_clf.feature_importances_)[::-1]

    # ------------- hyperParameterTuning() -------------
    def hyperParameterTuning(self, rf_clf, x_train, y_train):
        param_grid = {'n_estimators': [4, 16, 256], 'max_depth': [2, 8, 16]}
        gscv_rfc = GridSearchCV(rf_clf, param_grid, cv=3, n_jobs=-1)
        gscv_rfc.fit(x_train, y_train)
        return gscv_rfc


    # ------------- bestParams() -------------
    def bestParams(self, gscv_rfc):
        return gscv_rfc.best_params_

    # ------------- bestScore() -------------
    def bestScore(self, gscv_rfc):
        return gscv_rfc.best_score_
    
##################################################
##### Do not add anything below this line ########
tests.RandomForestTest(Data,RFClassifier)
##################################################


# # Q3.4 Support Vector Machine

# In[9]:


#export
class SupportVectorMachine():

    # ------------- dataPreProcess() -------------
    def dataPreProcess(self, x_train, x_test):
        scaler = StandardScaler()
        scaled_x_train = scaler.fit_transform(x_train)
        scaled_x_test = scaler.transform(x_test)
        return scaled_x_train, scaled_x_test

    # ------------- SVCClassifier() -------------
    def SVCClassifier(self, scaled_x_train, scaled_x_test, y_train):
        svc = SVC(gamma='auto')
        svc.fit(scaled_x_train, y_train)
        y_predict_train = svc.predict(scaled_x_train)
        y_predict_test = svc.predict(scaled_x_test)
        return y_predict_train, y_predict_test

    # ------------- SVCTrainAccuracy() -------------
    def SVCTrainAccuracy(self, y_train, y_predict_train):
        return accuracy_score(np.array(y_train), np.array(y_predict_train))

    # ------------- SVCTestAccuracy() -------------
    def SVCTestAccuracy(self, y_test, y_predict_test):
        return accuracy_score(np.array(y_test), np.array(y_predict_test))

    # ------------- SVMBestScore() -------------
    def SVMBestScore(self, scaled_x_train, y_train):
        svm_parameters = {'kernel': ('linear', 'rbf'), 'C': [0.01, 0.1, 1.0]}
        svc = SVC(gamma='auto')
        svm_cv = GridSearchCV(svc, svm_parameters, n_jobs=-1, return_train_score=True)
        svm_cv.fit(scaled_x_train, y_train)
        best_score = svm_cv.best_score_
        return svm_cv, best_score

    # ------------- SVCClassifierParam() -------------
    def SVCClassifierParam(self, svm_cv, scaled_x_train, scaled_x_test, y_train):
        best_est = svm_cv.best_estimator_
        y_predict_train = best_est.predict(scaled_x_train)
        y_predict_test = best_est.predict(scaled_x_test)
        return y_predict_train, y_predict_test

    # ------------- svcTrainAccuracy() -------------
    def svcTrainAccuracy(self, y_train, y_predict_train):
        return accuracy_score(np.array(y_train), np.array(y_predict_train))

    # ------------- svcTestAccuracy() -------------
    def svcTestAccuracy(self, y_test, y_predict_test):
        return accuracy_score(np.array(y_test), np.array(y_predict_test))

    # ------------- SVMRankTestScore() -------------
    def SVMRankTestScore(self, svm_cv):
        return np.array(svm_cv.cv_results_['rank_test_score'])

    # ------------- SVMMeanTestScore() -------------
    def SVMMeanTestScore(self, svm_cv):
        return np.array(svm_cv.cv_results_['mean_test_score'])
        
##################################################
##### Do not add anything below this line ########
tests.SupportVectorMachineTest(Data,SupportVectorMachine)
##################################################


# # Q3.5 PCA
# 
# 1. Perform dimensionality reduction of the data using PCA.  
# 2. Return Explained Variance Ratios. 
# 3. Report Singular Values. 
# 4. Report how many principal components (PCs) are needed to reach ≥ 90% cumulative explained variance.  
# 5. Report the top 5 feature names that contribute to PC1 (by absolute coefficient magnitude).  

# In[10]:


#export
class PCAClassifier():

    def __init__(self, random_state: int = 0):
        self.random_state = random_state
        self.scaler_ = None
        self.pca_ = None
        self.feature_names_ = None
        self.X_scaled_ = None


    # ------------- pcaClassifier() -------------
    def pcaClassifier(self, x_data):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(x_data.values)
        pca = PCA(n_components=8, svd_solver='full', random_state=self.random_state)
        pca.fit(X_scaled)
        self.pca_ = pca
        self.feature_names_ = list(x_data.columns)
        self.X_scaled_ = pd.DataFrame(X_scaled, columns=self.feature_names_)
        return pca

    # ------------- pcaExplainedVarianceRatio() -------------
    def pcaExplainedVarianceRatio(self, pca):
        return pca.explained_variance_ratio_

    # ------------- pcaSingularValues() -------------
    def pcaSingularValues(self, pca):
        return pca.singular_values_

    # ------------- n_components_for_variance() -------------
    def n_components_for_variance(self, pca, threshold=0.90):
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        idx = np.searchsorted(cumvar, threshold, side='left')
        return int(idx + 1) if idx < len(cumvar) else len(cumvar)

    # ------------- top_pc1_contributors() -------------
    def top_pc1_contributors(self, pca, top_n: int = 5):
        names = getattr(pca, "feature_names_in_", None)
        if names is None:
            names = self.feature_names_ or []

        pc1 = pca.components_[0]
        abs_vals = np.abs(pc1)
        tuples = [(abs_vals[i], names[i]) for i in range(len(names))]
        tuples_sorted = sorted(tuples, key=lambda t: (-t[0], t[1]))
        top_features = [t[1] for t in tuples_sorted[:top_n]]
        return top_features
    
##################################################
##### Do not add anything below this line ########
tests.PCATest(Data,PCAClassifier)
##################################################

