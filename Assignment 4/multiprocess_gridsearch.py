import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
import time
from multiprocessing import Process
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier

ziptrain = np.loadtxt(r"C:\Users\olive\Documents\GitHub\Computational-Applied-Statistics\Assignment 4\ziptrain.csv")
ziptest = np.loadtxt(r"C:\Users\olive\Documents\GitHub\Computational-Applied-Statistics\Assignment 4\ziptest.csv")

X_train, X_test = ziptrain[:, 1:], ziptest[:, 1:]
y_train, y_test = ziptrain[:, 0], ziptest[:, 0]

y_train = label_binarize(y_train, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
y_test = label_binarize(y_train, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

Grid = namedtuple("Grid", ['model', 'param_grid'])

grids = [
    Grid(BaggingClassifier,
        {'estimator__n_estimators': [200, 400, 600, 800, 1000],
        'estimator__max_samples': [100, 250, 500, 750, 1000]}),
    Grid(RandomForestClassifier,
        {'estimator__max_depth': [70, 80, 90, 100, None],
         'estimator__max_features': ['auto', 'sqrt'],
         'estimator__n_estimators': [200, 400, 600, 800, 1000]}),
    Grid(GradientBoostingClassifier,
         {'max_depth':[3, 4, 5, 6], 
          'estimator__n_estimators':[100,250,500,750,1000]}),
    Grid(SVC, 
    {"estimator__C": [4, 8, 10, 12], 
     "estimator__degree":[3, 4, 5, 6]})
]

def perform_gridsearch(grid):

    print("Starting grid {}".format(str(grid)))
    model = OneVsRestClassifier(grid.model())
    gs = GridSearchCV(model, param_grid=grid.param_grid, cv=3, scoring='accuracy', n_jobs=1, verbose=1)
    gs.fit(X_train, y_train)
    res = {"Model" : grid.model, "Best Parameters" : gs.best_params_, "Accuracy" : gs.best_score_}
    print("Results from grid {}:".format(str(grid)))
    print(res)
    
p1 = Process(target=perform_gridsearch, args=(grids[0],))
p2 = Process(target=perform_gridsearch, args=(grids[1],))
p3 = Process(target=perform_gridsearch, args=(grids[2],))
p4 = Process(target=perform_gridsearch, args=(grids[3],))

p1.start()
p2.start()
p3.start()
p4.start()

p1.join()
p2.join()
p3.join()
p4.join()

print("All Processes Completed Successfully.")