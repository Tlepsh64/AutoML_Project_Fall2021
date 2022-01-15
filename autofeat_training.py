import pandas as pd
import numpy as np

# Models to use
#import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import RandomForestClassifier
# Importing the metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import plot_confusion_matrix

# For measuring the training time taken during the fit process
from sklearn.model_selection import cross_val_score
import time

import pandas as pd
import numpy as np 

df = pd.read_csv('higgs_cleaned.csv')

object_columns = ['jet4phi','jet4b-tag','m_jj','m_jjj','m_lv','m_jlv','m_bb','m_wbb','m_wwbb']

X, y = df.drop('class', axis=1), df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1864)

clf = cb.CatBoostClassifier(
    bagging_temperature = 1.4426651823376004,
    border_count = 128,
    depth = 7,
    eval_metric = 'Accuracy',
    l2_leaf_reg = 10,
    learning_rate = 0.05212984419824801,
    loss_function = 'Logloss',
    random_seed = 1864,
    random_strength = 62.078606789809335
)

clf.fit(X_train, y_train, verbose=False)
preds = clf.predict(X_test)
print(f'Test accuracy of the current optimal catboost model: {accuracy_score(y_test, preds)}')

from autofeat import AutoFeatClassifier
feateng_columns = ('lepton_pT','missing_energy_magnitude','jet1pt')
model = AutoFeatClassifier(verbose=1, feateng_cols = feateng_columns, feateng_steps=3, n_jobs=-1, featsel_runs=3, transformations = ("sin", "exp", 'sqrt', 'log'))

X_train_feature_creation = model.fit_transform(X_train,y_train)

X_train_feature_creation.to_csv('generated_df.csv')
#preds = model.predict(X_test)
#print(f'Test accuracy of the current Autofeat model: {accuracy_score(y_test, preds)}')

X_train_feature_creation = model.transform(X_train)
X_test_feature_creation = model.transform(X_test)

clf1 = cb.CatBoostClassifier(
    bagging_temperature = 1.4426651823376004,
    border_count = 128,
    depth = 7,
    eval_metric = 'Accuracy',
    l2_leaf_reg = 10,
    learning_rate = 0.05212984419824801,
    loss_function = 'Logloss',
    random_strength = 62.078606789809335
)

clf1.fit(X_train_feature_creation, y_train, verbose=False)
preds = clf1.predict(X_test_feature_creation)
print(f'Test accuracy of the current optimal Autofeat boosted catboost model: {accuracy_score(y_test, preds)}')


rf = RandomForestClassifier(random_state=0)
rf.fit(X_train_feature_creation, y_train)
preds = rf.predict(X_test_feature_creation)

print(f'Test accuracy of the current optimal Autofeat boosted Random Forest model: {accuracy_score(y_test, preds)}')



