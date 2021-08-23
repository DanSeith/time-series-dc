import pickle
import sklearn
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import sys
from matplotlib import rcParams
from matplotlib.pyplot import figure

rcParams['font.family'] = 'arial'

ROOT_DIR = os.path.abspath("../")
current_dir = os.getcwd()
sys.path.append(ROOT_DIR)
import df_utils

feature_dict = df_utils.read_feats()
feature_list = list(feature_dict)
feature_list_cleaned = [feature_dict[i] for i in list(feature_dict.keys())]

# Load dataset
df = pd.read_pickle(os.path.join(ROOT_DIR, 'FINAL_DF_light'))
df = df_utils.filter_df(df, ymax=5, max_ar=1.1, radius_std=3)
df = df[(df.cell == 'hl60') | (df.cell == 'hl60d')]
df = df[np.logical_not((df.cell == 'hl60') & (df.date == '11-3-20') & (df.run == '0'))]
df = df[np.logical_not((df.cell == 'hl60') & (df.date == '11-5-20') & (df.run == '3'))]
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

# Iterate over models and plot roc
model_dict = {}

# figure(figsize=(5, 2.5), dpi=300)
for rel_dir in ['knn', 'logistic_regression', 'random_forest', 'svm']:
    model_dir = os.path.join(current_dir, rel_dir, rel_dir+'.pkl')
    # Load model
    with open(model_dir, 'rb') as pkl_file:
        model = pickle.load(pkl_file)

    # Obtain fpr, tpr
    y_pred = model.predict_proba(x_test)[:, 1]
    fpr, tpr, _ = sklearn.metrics.roc_curve(y_test, y_pred)
    auc = sklearn.metrics.roc_auc_score(y_test, y_pred)
    plt.plot(fpr, tpr, label=rel_dir + ' (AUC: ' + str(round(auc, 2)) + ')')
    model_dict[rel_dir] = sklearn.metrics.accuracy_score(y_test, y_pred.round())*100
    print(sklearn.metrics.accuracy_score(y_test, y_pred.round()))
    # Save SVM performance for deep learning comparison plot
    if rel_dir == 'svm':
        os.chdir(os.path.join(current_dir, rel_dir))
        np.save('svm_pred', y_pred)
        np.save('svm_ground_truth', y_test)
        np.savetxt('svm_fpr.csv', fpr)
        np.savetxt('svm_tpr.csv', tpr)
        np.savetxt('svm_auc.csv', np.ones(1)*auc)
        os.chdir(current_dir)

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False positive rate', fontsize=18)
plt.ylabel('True positive rate', fontsize=16)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.title('ROC Curve', fontsize=24)
plt.legend(loc='best', prop={'size': 12})
plt.tight_layout()

plt.show()
plt.savefig('sklearn_roc.png', format='png')

plt.close()
fig = plt.figure(figsize=(5, 2.5))

plt.bar(range(len(model_dict.keys())), [model_dict[i] for i in list(model_dict.keys())], tick_label=list(model_dict.keys()))
plt.ylabel('Accuracy (Percent)', fontsize=9)

plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

plt.title('Traditional Machine Learning Model Performance', fontsize=10)
plt.tight_layout()
plt.savefig('sklearn_bar.eps', format='eps')
plt.savefig('sklearn_bar.png', format='png')
plt.show()
