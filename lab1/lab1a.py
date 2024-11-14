import pandas as pd
import numpy as np
import time
import os
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import RepeatedKFold, cross_val_score

def load_dataset(file_path):
    data = pd.read_csv(file_path, header=None, delimiter=',')
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    return X, y

def run_comparision():
    for idx, filename in enumerate(datasets_files):
        dir = os.path.join(datasets_path, filename)
        X, y = load_dataset(dir)

        for jdx, (name, clf) in enumerate(classifiers):
            scores = cross_val_score(clf, X, y, cv=rkf, scoring='accuracy')
            scores_fixed = np.mean(scores)

            result_matxir[idx, jdx] = scores_fixed  

datasets_path = 'datasets/'
datasets_files = [file for file in os.listdir(datasets_path)]

classifiers = [
    ("Random Forest", RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, random_state=42)),
    ("AdaBoost", AdaBoostClassifier(algorithm="SAMME", random_state=42))
]

rkf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)
result_matxir = np.zeros((len(datasets_files), len(classifiers)))

start_time = time.time()
run_comparision()
print(f'\nExecution time: {time.time() - start_time}')
print(f'\nData: \n{result_matxir}')
 