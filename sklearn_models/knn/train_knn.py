import pickle
import sklearn
from sklearn.neighbors import KNeighborsClassifier
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

# Grid search all hyperparameters
leaf_size = list(range(1, 200, 5))
neighbors = np.arange(1, 25)
score = {}
for leaf in leaf_size:
    for n in neighbors:
        clf = KNeighborsClassifier(n_neighbors=n, leaf_size=leaf)
        clf.fit(x_train, y_train.ravel())
        score[(str(n), str(leaf))] = clf.score(x_val, y_val)

# Select best weights
n, leaf = max(score, key=lambda key: score[key])

print('Optimized hyper params:')
print('N: ', n, '\n', 'Leaf size: ', leaf)
clf = KNeighborsClassifier(n_neighbors=int(n), leaf_size=int(leaf))
clf.fit(x_train, y_train.ravel())
print(clf.score(x_test, y_test))

sklearn.metrics.plot_roc_curve(clf, x_test, y_test)
plt.title('k Nearest Neighbors' + ' K: ' + str(n) + ' Leaf Size: ' + str(leaf))
plt.savefig('knn_roc.png', dpi=300)

pickle.dump(clf, open('knn.pkl', 'wb'))
