import pickle
import sklearn
from sklearn import svm
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)
import df_utils

feature_dict = df_utils.read_feats()
feature_list = list(feature_dict)

# Load predetermined features from dataset
df = pd.read_pickle(os.path.join(ROOT_DIR, 'FINAL_DF_light'))
df.dropna(inplace=True)

# Extract features
x = df[feature_list].to_numpy()
y = df.apply(lambda a: int(a['cell'] == 'hl60'), axis=1).to_numpy()

# Normalize and standardize first
scalar = sklearn.preprocessing.StandardScaler()
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.3, random_state=123)

# Fit scalar on training, apply transformation to val/test
scalar.fit(x_train)
x_train = scalar.transform(x_train)
x_val = scalar.transform(x_val)

x_val, x_test, y_val, y_test = train_test_split(x_val, y_val, test_size=0.5, random_state=123)

score = {}
# Grid search all hyperparameters
c_list = np.linspace(0.1, 1.5, 10)
degree = np.arange(1, 10)
k_list = ['linear', 'poly', 'rbf', 'sigmoid']
for c in c_list:
    for k in k_list:
        if k == 'poly':
            for d in degree:
                svm_clf = svm.SVC(C=c, kernel=k, degree=d, random_state=123, probability=True)
                svm_clf.fit(x_train, y_train.ravel())
                score[(str(c), k, str(d))] = svm_clf.score(x_val, y_val)
        svm_clf = svm.SVC(C=c, kernel=k, random_state=123, probability=True)
        svm_clf.fit(x_train, y_train.ravel())
        score[(str(c), k, 'N/A')] = svm_clf.score(x_val, y_val)

# Select best weights
c, k, d = max(score, key=lambda key: score[key])
c = float(c)
if d == 'N/A':
    d = 0
print('Optimized hyper params:')
print('C: ', c, '\n', 'Kernel: ', k, '\n', 'D: ', d, '\n')
svm_clf = svm.SVC(C=c, kernel=k, degree=d, random_state=123, probability=True)
svm_clf.fit(x_train, y_train.ravel())
print(svm_clf.score(x_test, y_test))

sklearn.metrics.plot_roc_curve(svm_clf, x_test, y_test.ravel())
plt.title('SVM' + ' C: ' + str(round(c, 2)) + ' Kernel: ' + k + ' D: ' + str(d))
plt.savefig('svm_roc.png', dpi=300)

pickle.dump(svm_clf, open('svm.pkl', 'wb'))
