##
import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pickle
import ml_utils as mlu
import config as cfg
##

##
df = pd.read_csv(cfg.DATA_FILE)
##

##
# Cleaning columns with no intereset
df.drop(['ID', 'Source'],
        axis=1,
        inplace=True)

#mlu.missing_values_heatmap(df)

df['TMC'].count_values(dropna=False)
df['End_Lat'].count_values(dropna=False)
df['End_Lng'].count_values(dropna=False)
df['Number'].count_values(dropna=False)
df['Wind_Chill(F)'].count_values(dropna=False)
df['Precipitation(in)'].count_values(dropna=False)

df.drop(['End_Lat', 'End_Lng', 'Number'], 
        axis=1, 
        inplace=True)

##

##
# Fill missing values
df['Wind_Speed(mph)'].fillna(df['Wind Chill(F)'].mean(), inplace=True)
df['Wind_Chill(F)'].fillna(df['Wind Chill(F)'].mean(), inplace=True)
df['Precipitation'].fillna(df['Wind Chill(F)'].mean(), inplace=True)
##

##
# Boolean columns

##

##
# Categorical data preperation
##

##
# Date preparation
##

##
# save pickle
with open(cfg.OUTPUT_PATH+'df.pickle','wb') as f:
    pickle.dump(df,f)
##

##
# working with a subset of the data
df_s = df.loc[df['State'] == 'CT' and df['County'] == 'Fairfield'].copy
mlu.cor_matrix(df)
##

##
# Feature Selection

features = []
##

##
ax = cell_df[cell_df['Class'] == 4][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='DarkBlue', label='malignant');
cell_df[cell_df['Class'] == 2][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='Yellow', label='benign', ax=ax);
plt.savefig("fig_1.png")
##

##
cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()]
cell_df['BareNuc'] = cell_df['BareNuc'].astype('int')

feature_df = cell_df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
X = np.asarray(feature_df)

y = np.array(cell_df['Class'].astype('int'))

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)
##

##
from sklearn import svm
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train) 

y_hat = clf.predict(X_test)
##

from sklearn.metrics import classification_report, confusion_matrix, f1_score, jaccard_similarity_score
import itertools

f1_score(y_test, y_hat, average='weighted')
jaccard_similarity_score(y_test, y_hat)

##
