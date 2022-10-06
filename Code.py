#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


pd.set_option('display.max_row', 111)
pd.set_option('display.max_column', 111)


# In[3]:


data = pd.read_csv(r'Z:\Repertoires Nominatifs\AntoineChen\Memoire\maj_csv.csv', sep=';')


#  data = pd.read_csv("Z:\Repertoires Nominatifs\AntoineChen\Memoire\Extracts_csv\contrats_maj_csv.csv", sep=';')

# In[18]:


data.head()


# In[4]:


data2020 = data[data['encours2020'] == 1]
data2021 = data[data['encours2021'] == 1]


# In[5]:


data2020 = data2020.drop(columns=['NUMCNT', 'DTNAIS', 'DTSORTIE', 'DTEFFAN', 'ORGDATA', 'resil', 'annee_resil', 'tranche', 'DTEFFRES', 'DTEMIRES', 'code_commune', 'mois_resil', 'motif', 'totRC2019', 'cotacq2019',  'encours2020', 'encours2021', 'resil2021','code_region', 'compteur'])
data2021 = data2021.drop(columns=['NUMCNT','DTNAIS', 'DTSORTIE', 'DTEFFAN', 'ORGDATA', 'resil', 'annee_resil', 'tranche', 'DTEFFRES', 'DTEMIRES', 'code_commune', 'mois_resil', 'motif', 'totRC2019', 'cotacq2019',  'encours2020', 'encours2021', 'resil2020', 'code_region', 'compteur'])


# In[89]:


data2020.head()


# # 1. Exploratory Data Analysis
# 
# ## Objectif :
# - Comprendre du mieux possible nos données
# - Développer une premiere stratégie de modélisation
# 
# ## Checklist de base
# On a d'abord séparé le dataset en 2, avec les contrats en cours début 2020, et les contrats en cours début 2021
# #### Analyse de Forme :
# - **variable target** : resil2020 ou resil2021
# - **lignes et colonnes** : 400 k colonnes et 17 variables
# - **types de variables** : qualitatives : 11, quantitatives : 6
# - **Analyse des valeurs manquantes** :
#     - NBENFCH, ORGDATA et CDSITFAC avec ~30% de NaN
#     - le reste presque complet
# 
# 
# #### Analyse de Fond :
# - **Visualisation de la target** : données non équilibrées, à faire attention pour la suite
# - **Corellations**: coréllation moyenne entre le nombre d'enfants à charge et le nombre d'assurés au contrat
# - **Relation Age/Zone**: Aucun lien entre l'âge du souscripteur et sa zone d'habitation
# - **Relation Target/Sexe**: Taux de résiliation légèrement plus élevé chez les femmes

# ### Analyse de la forme des données

# In[6]:


df2020 = data2020.copy()
df2021 = data2021.copy()


# In[19]:


df2020.shape


# In[20]:


df2021.shape


# In[21]:


df2020.dtypes.value_counts().plot.pie()


# In[22]:


df2020.dtypes.value_counts()
df2020.info()

# In[91]:


df2020.dtypes


# In[24]:


plt.figure(figsize=(20, 10))
sns.heatmap(df2020.isna(), cbar=False)


# In[46]:


(df2020.isna().sum()/df2020.shape[0]).sort_values(ascending=True)


# In[26]:


plt.figure(figsize=(20, 10))
sns.heatmap(df2021.isna(), cbar=False)


# In[51]:


(df2021.isna().sum()/df2021.shape[0]).sort_values(ascending=True)


# ## Analyse de fond

# ### Analyse de la Target

# In[27]:


df2020['resil2020'].value_counts(normalize=True)


# In[28]:


df2021['resil2021'].value_counts(normalize=True)


# In[8]:


resilied2020_df = df2020[df2020['resil2020'] == 1]
noresilied2020_df = df2020[df2020['resil2020'] == 0]
resilied2021_df = df2021[df2021['resil2021'] == 1]
noresilied2021_df = df2021[df2021['resil2021'] == 0]

urbaine2020_df = df2020[df2020['zone2'] == 'Urbaine']
rurale2020_df =  df2020[df2020['zone2'] == 'Rurale']
urbaine2021_df = df2021[df2021['zone2'] == 'Urbaine']
rurale2021_df =  df2021[df2021['zone2'] == 'Rurale']

idf2020_df = df2020[df2020['code_region'] == 11]
hdf2020_df = df2020[df2020['code_region'] == 32]
paca2020_df = df2020[df2020['code_region'] == 94]
cvl2020_df = df2020[df2020['code_region'] == 24]
pdl2020_df = df2020[df2020['code_region'] == 52]
bfc2020_df = df2020[df2020['code_region'] == 27]

plt.figure()
sns.distplot(idf2020_df['age'], label='Ile de France')
sns.distplot(hdf2020_df['age'], label='Hauts de France')
sns.distplot(paca2020_df['age'], label='PACA')
sns.distplot(cvl2020_df['age'], label='Centre Val de Loire')
sns.distplot(pdl2020_df['age'], label='Pays de la Loire')
plt.legend()
# ### Variable âge

# In[11]:


plt.figure()
sns.distplot(resilied2020_df['age'], label='résiliés')
sns.distplot(noresilied2020_df['age'], label='non-résiliés')
plt.legend()

plt.figure()
sns.distplot(df2020['age'])
plt.legend()

plt.figure()
sns.distplot(df2021['age'])
plt.legend()



# In[59]:


plt.figure()
sns.distplot(resilied2021_df['age'], label='résiliés')
sns.distplot(noresilied2021_df['age'], label='non-résiliés')
plt.legend()

from numpy.random import seed
from numpy.random import randn
from numpy.random import lognormal

seed(0)

from scipy.stats import ks_2samp

#perform Kolmogorov-Smirnov test
ks_2samp(resilied2020_df['age'], noresilied2020_df['age'])
ks_2samp(noresilied2020_df['age'], noresilied2021_df['age'])
# ### Variables discrètes

# In[11]:


#for discrete variables
for col in df2020.select_dtypes('int'):
  if col != 'DEPT' and col != 'resil2020':
    plt.figure()
    sns.countplot(x=col, hue='resil2020', data=df2020)
    plt.plot()


# In[12]:


sns.countplot(x='NBASSUR', hue='resil2020', data=df2020)


# ### Variables qualitatives

# In[50]:


#for categorical variables
for col in df2020.select_dtypes('object'):
  if col != 'CUMCOTIS' and col !='ORGDATA' and col !='SsurC19':
    plt.figure(figsize= [11, 4.8])
    sns.countplot(x=col, hue='resil2020', data=df2020)
    plt.plot()


# In[1]:


#sns.countplot(x='SsurC19', hue='resil2020', data=df2020)


# In[56]:


#for categorical variables
for col in df2021.select_dtypes('object'):
  if col != 'CUMCOTIS' and col !='ORGDATA' and col !='SsurC19' and col !='region':
    plt.figure(figsize= [11, 4.8])
    sns.countplot(x=col, hue='resil2021', data=df2021)
    plt.plot()


# ### Corellations

# In[13]:


sns.clustermap(df2020.corr())


# In[42]:


sns.clustermap(df2021.corr())


# ~0.4 de corellation entre le nb d'assurés et le nb d'enfants à charge

# In[43]:


plt.figure()
sns.distplot(urbaine2020_df['age'], label='urbaine')
sns.distplot(rurale2020_df['age'], label='rurale')
plt.legend()


# In[34]:


plt.figure()
sns.distplot(urbaine2021_df['age'], label='urbaine')
sns.distplot(rurale2021_df['age'], label='rurale')
plt.legend()


# In[92]:


df2020["CUMCOTIS"]=df2020["CUMCOTIS"].str.replace(',','.')
df2021["CUMCOTIS"]=df2020["CUMCOTIS"].str.replace(',','.')


# In[97]:


df2020["CUMCOTIS"]=df2020["CUMCOTIS"].astype(float)
df2021["CUMCOTIS"] = df2021["CUMCOTIS"].astype(float)

df2020.head()
# In[96]:


df2020.dtypes


# In[7]:
    
# Installation et importation des librairies et packages
from imblearn.over_sampling import SMOTE, SMOTENC, RandomOverSampler
from sklearn.model_selection import train_test_split,  RandomizedSearchCV, learning_curve
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import f1_score, confusion_matrix, classification_report, precision_recall_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel, RFECV
from xgboost import XGBClassifier
from sklearn.compose import make_column_selector, make_column_transformer, ColumnTransformer
from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import missingno as msno
from sklearn import metrics
import shap

# In[8]:
df2020 = df2020.drop(['ind_OD', 'ind_hospi', 'ind_confort', 'regime2', 'gamme'], axis=1)
df2021 = df2021.drop(['ind_OD', 'ind_hospi', 'ind_confort', 'regime2', 'gamme'], axis=1)
df2020['amour'] = df2020['amour'].fillna('na')
df2021['amour'] = df2021['amour'].fillna('na')
df2020.loc[df2020['NBASSUR'] == 4, ['NBASSUR']] = 3
df2020.loc[df2020['NBASSUR'] == 5, ['NBASSUR']] = 3
df2020.loc[df2020['NBASSUR'] == 6, ['NBASSUR']] = 3
df2021.loc[df2021['NBASSUR'] == 4, ['NBASSUR']] = 3
df2021.loc[df2021['NBASSUR'] == 5, ['NBASSUR']] = 3
df2021.loc[df2021['NBASSUR'] == 6, ['NBASSUR']] = 3

msno.bar(df2020)
dft = df2020.drop(['SEXE', 'DEPT', 'region', 'ORIGINE', 'resil2020'], axis=1)
msno.bar(dft)

dfshow20 = df2020.drop([ 'DEPT', 'region', 'ind_OD', 'ind_hospi', 'ind_confort', 'gamme', 'regime2' ], axis=1)
msno.bar(dfshow20)
dfshow20.info()

dfshow21 = df2021.drop([ 'DEPT', 'region', ], axis=1)
dfshow21.info()

trainset, testset = train_test_split(df2020, test_size=0.3, random_state=0)


# In[9]:
def preproc_cor(df):
    df.loc[df['SEXE'] == 1, ['sex']] = 'homme'
    df.loc[df['SEXE'] == 2, ['sex']] = 'femme'
    df.loc[df['SEXE'] == 9, ['sex']] = np.nan
    
    df.loc[df['ORIGINE'] == 1, ['ORG']] = '1er assu'
    df.loc[df['ORIGINE'] == 2, ['ORG']] = 'concurrence'
    df.loc[df['ORIGINE'] == 3, ['ORG']] = 'Axa'
    df = df.drop(['SEXE', 'DEPT', 'region', 'ORIGINE'], axis=1)

    
    #Scaling before smote
    categorical_cols = ['regime2', 'niveau', 'anciennete1', 'SsurC2019', 'cible', 'amour', 'zone2', 'sex', 'ORG']
    numerical_cols = ['CUMCOTIS', 'NBASSUR', 'Nb_renfort', 'rabais', 'age', 'multi_detention']
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    

    
    #preprocessing avant intégration modèle
    #df = feature_engineering(df)
    #df = imputation(df)
    
    df = df.drop('resil2020', axis=1)
    
    df=pd.get_dummies(df)
    
    
    return df

reg = preproc_cor(df2020)
reg.info()
sns.clustermap(reg.corr())

fig = plt.figure(figsize=(36,36), dpi = 480)
sns.heatmap(reg.corr(method='pearson'), annot = True, fmt = '.2f')

reg.info()

fig = plt.figure(figsize=(36,36), dpi = 480)
sns.heatmap(reg.corr(method='spearman'), annot = False, fmt = '.2f')


#Pré-processing des données train et test
def preprocessingtrain(df): 
          
    df.loc[df['SEXE'] == 1, ['sex']] = 0 #homme
    df.loc[df['SEXE'] == 2, ['sex']] = 1 #femme
    df.loc[df['SEXE'] == 9, ['sex']] = np.nan
    
    df.loc[df['zone2'] == 'Urbaine', ['zone']] = 1 
    df.loc[df['zone2'] == 'Rurale', ['zone']] = 0 
    
    
    df.loc[df['ORIGINE'] == 1, ['ORG']] = '1er assu'
    df.loc[df['ORIGINE'] == 2, ['ORG']] = 'concurrence'
    df.loc[df['ORIGINE'] == 3, ['ORG']] = 'Axa'
    df = df.drop(['SEXE', 'DEPT', 'region', 'ORIGINE', 'zone2', 'reseau2'], axis=1)

    
    #Scaling before smote
    categorical_cols = ['niveau', 'anciennete1', 'SsurC2019', 'cible', 'amour', 'ORG']
    numerical_cols = ['CUMCOTIS', 'NBASSUR', 'Nb_renfort', 'rabais', 'age', 'multi_detention','sex', 'zone']
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    X = df.drop('resil2020', axis=1)
    y = df['resil2020']
    
    #missing values
    imp = SimpleImputer(missing_values=np.nan)
    impcat = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    X[numerical_cols] = imp.fit_transform(X[numerical_cols])
    X[categorical_cols] = impcat.fit_transform(X[categorical_cols])
    print(X.dtypes)
    #smotenc
    a = [1, 2, 4, 5, 6, 9, 11, 12, 13, 15, 16]
    smote_nc = SMOTENC(categorical_features=a, random_state=0)
    X_res, y_res = smote_nc.fit_resample(X,y)
    
    #Random Over Sampling
    #ros = RandomOverSampler(random_state=0)
    #X_res, y_res = ros.fit_resample(X, y)
      
    return X_res, y_res, scaler


def preprocessing(df, scaler): 
    
    df.loc[df['SEXE'] == 1, ['sex']] = 0 #homme
    df.loc[df['SEXE'] == 2, ['sex']] = 1 #femme
    df.loc[df['SEXE'] == 9, ['sex']] = np.nan
    
    df.loc[df['zone2'] == 'Urbaine', ['zone']] = 1 
    df.loc[df['zone2'] == 'Rurale', ['zone']] = 0 
    
    df.loc[df['ORIGINE'] == 1, ['ORG']] = '1er assu'
    df.loc[df['ORIGINE'] == 2, ['ORG']] = 'concurrence'
    df.loc[df['ORIGINE'] == 3, ['ORG']] = 'Axa'
    df = df.drop(['SEXE', 'DEPT', 'region', 'ORIGINE', 'zone2', 'reseau2'], axis=1)
 
    categorical_cols = [ 'niveau', 'anciennete1', 'SsurC2019', 'cible', 'amour',  'ORG']
    numerical_cols = ['CUMCOTIS', 'NBASSUR', 'Nb_renfort', 'rabais', 'age', 'multi_detention', 'sex', 'zone']
    df[numerical_cols] = scaler.transform(df[numerical_cols])
       
    #preprocessing avant intégration modèle
    #df = feature_engineering(df)
    #df = imputation(df)
    
    X = df.drop('resil2020', axis=1)
    y = df['resil2020']
    
    imp = SimpleImputer(missing_values=np.nan)
    impcat = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    X[numerical_cols] = imp.fit_transform(X[numerical_cols])
    X[categorical_cols] = impcat.fit_transform(X[categorical_cols])
    
    return X, y

# In[10]:


X_train, y_train, scaler = preprocessingsmote(trainset) 

X_test, y_test = preprocessing(testset, scaler)  

X_test, y_test = preprocessing(df2021, scaler)
#encodage des features catégorielles pour la modéelisation
X_train=pd.get_dummies(X_train)
X_test=pd.get_dummies(X_test)
X_test.head()



sns.clustermap(X_train.corr())

# In[13]:
# Confusion matrix 
mat_con = (confusion_matrix(y_test, ypred, labels=[0, 1]))
fig, px = plt.subplots(figsize=(7.5, 7.5))
px.matshow(mat_con, cmap=plt.cm.YlOrRd, alpha=0.5)
for m in range(mat_con.shape[0]):
    for n in range(mat_con.shape[1]):
        px.text(x=m,y=n,s=mat_con[m, n], va='center', ha='center', size='xx-large')
plt.xlabel('Actuals', fontsize=16)
plt.ylabel('Predictions', fontsize=16)
plt.title('Confusion Matrix', fontsize=15)
plt.show()

#fonction d'évaluation avec learning curve
def evaluation(model, name):
    target = 1
    scorer = metrics.make_scorer(lambda y_true, y_pred: f1_score(
    y_true, y_pred, 
    labels=None, 
    pos_label=1, 
    average='binary', 
    sample_weight=None)) 
    
    model.fit(X_train, y_train)
    ypred = model.predict(X_test)
    
    print(confusion_matrix(y_test, ypred))
    print(classification_report(y_test, ypred))
    
    N, train_score, val_score = learning_curve(model, X_train, y_train,
                                              cv=3, 
                                              scoring='accuracy',
                                               train_sizes=np.linspace(0.1, 1, 10))
    
    
    plt.figure(figsize=(12, 8))
    plt.plot(N, train_score.mean(axis=1), label='train score')
    plt.plot(N, val_score.mean(axis=1), label='validation score')
    plt.title(name)
    plt.legend()

tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
ypred = tree.predict(X_test)
print(confusion_matrix(y_test, ypred))
print(classification_report(y_test, ypred))
print(pd.crosstab(y_test, ypred))


import re
regex = re.compile(r"\[|\]|<", re.IGNORECASE)

X_train.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in X_train.columns.values]
X_train.head()
X_test.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in X_test.columns.values]

# Load Package
from sklearn.dummy import DummyClassifier

# Initialize Estimator
dummy_clf = DummyClassifier(strategy='constant', random_state=0, constant=0)
dummy_clf.fit(X_train,y_train)
ypred = dummy_clf.predict(X_test)
print(confusion_matrix(y_test, ypred))
print(classification_report(y_test, ypred))
y_pred_proba = dummy_clf.predict_proba(X_test)[::,1]
auc = metrics.roc_auc_score(y_test, y_pred_proba)
print(auc)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
ypred = knn.predict(X_test)
print(confusion_matrix(y_test, ypred))
print(classification_report(y_test, ypred))
y_pred_proba = knn.predict_proba(X_test)[::,1]
auc = metrics.roc_auc_score(y_test, y_pred_proba)
print(auc)

svm = LinearSVC(random_state=0, probabilities=True)
svm.fit(X_train, y_train)
ypred = svm.predict(X_test)
print(confusion_matrix(y_test, ypred))
print(classification_report(y_test, ypred))
y_pred_proba = svm.predict_proba(X_test)[::,1]
auc = metrics.roc_auc_score(y_test, y_pred_proba)
print(auc)


XGBoost =  XGBClassifier(random_state=0, use_label_encoder=False)
XGBoost.fit(X_train, y_train)
ypred = XGBoost.predict(X_train)
print(confusion_matrix(y_test, ypred))
print(classification_report(y_test, ypred))
y_pred_proba = XGBoost.predict_proba(X_test)[::,1]
auc = metrics.roc_auc_score(y_test, y_pred_proba)
print(auc)

lr = LogisticRegression()
lr.fit(X_train, y_train)
ypred = lr.predict(X_test)
print(confusion_matrix(y_test, ypred))
print(classification_report(y_test, ypred))
y_pred_proba = lr.predict_proba(X_test)[::,1]
auc = metrics.roc_auc_score(y_test, y_pred_proba)
print(auc)
 #good learning curve

rf = RandomForestClassifier(random_state=0)
rf.fit(X_train, y_train)
ypred = rf.predict(X_test)
print(confusion_matrix(y_test, ypred))
print(classification_report(y_test, ypred))
y_pred_proba = XGBoost.predict_proba(X_test)[::,1]
auc = metrics.roc_auc_score(y_test, y_pred_proba)
print(auc)

importances = rf.feature_importances_
sorted_indices = np.argsort(importances)[::-1]

plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]), importances[sorted_indices], align='center')
plt.xticks(range(X_train.shape[1]), X_train.columns[sorted_indices], rotation=90, fontsize = 6)
plt.tight_layout()
plt.show()

#Hyperparameters Optimisation
scorer = metrics.make_scorer(lambda y_true, y_pred: f1_score(
y_true, y_pred, 
labels=None, 
pos_label=1, 
average='binary', 
sample_weight=None)) 
params = { 'max_depth': [4, 6, 10, 15],
           'learning_rate': [0.01, 0.1, 0.2, 0.3],
           'subsample': np.arange(0.5, 1.0, 0.1),
           'gamma':[0, 0.1, 0.2],
           'n_estimators': [100, 250, 400]}

xgbr = XGBClassifier(random_state = 0)
clf = RandomizedSearchCV(estimator=xgbr,
                         param_distributions=params,
                         scoring=scorer,
                         n_iter=25,
                         verbose=1)

clf.fit(X_train, y_train)
print("Best parameters:", clf.best_params_)

xgbtuned = XGBClassifier(n_estimators=200, reg_alpha=50, reg_lambda=50, learning_rate = 0.3)
xgbtuned.fit(X_train, y_train)
ypredtrain = xgbtuned.predict(X_train)
ypredtest = xgbtuned.predict(X_test)

print(confusion_matrix(y_test, ypredtest))
print(classification_report(y_train, ypredtrain))
print(classification_report(y_test, ypredtest))
evaluation(xgbtuned, xgbtuned)
y_pred_proba = xgbtuned.predict_proba(X_test)[::,1]
auc = metrics.roc_auc_score(y_test, y_pred_proba)
print(auc)

# Importance des variables
importances = xgbtuned.feature_importances_
sorted_indices = np.argsort(importances)[::-1]

plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]), importances[sorted_indices], align='center')
plt.xticks(range(X_train.shape[1]), X_train.columns[sorted_indices], rotation=90, fontsize = 6)
plt.tight_layout()
plt.show()

# load JS visualization code to notebook
shap.initjs()

# explain the model's predictions using SHAP (code takes around 10 minutes)
explainer = shap.TreeExplainer(xgbtuned)
shap_values = explainer.shap_values(X_test)

# shap.force_plot
index = 88883
arr=X_test.iloc[index,:].values
pred = xgbtuned.predict(arr.reshape(1,-1))[0]
true_label = y_test.iloc[index]
if true_label == pred:
        accurate = 'Correct!'
else:
        accurate = 'Incorrect'
    
print('***'*12)
# Print ground truth label for row at index
print(f'Ground Truth Label: {true_label}')
print()
# Print model prediction for row at index
print(f'Model Prediction:  {pred} -- {accurate}')
print('***'*12)
print()
    
shap.force_plot(explainer.expected_value, shap_values[index,:], X_test.iloc[index,:], matplotlib=True)

y_pred_proba = xgbtuned.predict_proba(X_test)[88888,1]

#shap.summary_plot
shap.summary_plot(shap_values, X_test, max_display=40)
#CUSTOMIZED NB OF FEATURES
features = X_train.columns
indices = np.argsort(importances)

# customized number 
num_features = 15

plt.figure(figsize=(10,100))
plt.title('Feature Importances')

# only plot the customized number of features
plt.barh(range(num_features), importances[indices[:num_features]], color='b', align='center')
plt.yticks(range(num_features), [features[i] for i in indices[:num_features]], fontsize=7)
plt.xlabel('Feature Importance')
plt.show()

print(confusion_matrix(y_test, ypred))
print(classification_report(y_test, ypred))

min_features_to_select = 7
rfecv = RFECV(estimator=rf, cv=4, scoring="f1", min_features_to_select= 7)
rfecv.fit(X_train, y_train)

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (accuracy)")
plt.plot(
    range(7, len(rfecv.grid_scores_) + 7),
    rfecv.grid_scores_,
)
plt.show()

#evaluation
evaluation(tree, tree)
evaluation(XGBoost, XGBoost)
evaluation(lr, lr)
evaluation(rf, rf)

from sklearn.metrics import precision_recall_curve
precision, recall, threshold = precision_recall_curve(y_test, xgbtuned.decision_function(X_test))
plt.plot(threshold, precision[:-1], label='precision')
plt.plot(threshold, recall[:-1], label='recall')
plt.legend()

def model_final(model, X, threshold=0):
    return model.decision_function(X) > threshold

y_pred = model_final(lr, X_test, threshold=0)


f1_score(y_test, y_pred)

from sklearn.metrics import recall_score
recall_score(y_test, y_pred)

#Feature Importance

features = np.array(X_train.columns)
importances = lr.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(10,20))
plt.title('Feature Importances')
nf = 50
plt.barh(range(len(indices))[-nf:-1], importances[indices][-nf:-1], color='b', align='center')
plt.yticks(range(len(indices))[-nf:-1], features[indices][-nf:-1])
plt.xlabel('Relative Importance')
# In[14]:


# numerical_features = make_column_selector(dtype_include=np.number)
# categorical_features = make_column_selector(dtype_exclude=np.number)


# # In[15]:


# numerical_pipeline = make_pipeline ( SimpleImputer(),
#                                     StandardScaler())
# categorical_pipeline = make_pipeline ( SimpleImputer(strategy='most_frequent'),
#                                       OneHotEncoder())


# # In[17]:


# preprocessor = make_column_transformer ((numerical_pipeline, numerical_features),
#                                         (categorical_pipeline, categorical_features))


# # In[18]:


# RandomForest = make_pipeline (preprocessor, RandomForestClassifier(random_state=0))
# tree = make_pipeline (preprocessor, DecisionTreeClassifier(random_state=0))
# KNN = make_pipeline (preprocessor, KNeighborsClassifier())
# XGBoost = make_pipeline(preprocessor, XGBClassifier(random_state=0))


# In[19]:


dict_of_models = {'tree': tree,
                  'RandomForest': RandomForest,
                  'XGBoost' : XGBoost,
                  'KNN': KNN
                 }


# In[20]:


for name, model in dict_of_models.items():
    print(name)
    evaluation(model, name)


# In[ ]:





# In[ ]:




