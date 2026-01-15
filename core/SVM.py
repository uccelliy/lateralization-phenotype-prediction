from core.util import n_iter, kfold, random_state
import pandas as pd
import core.util as util
from time import perf_counter
import shap
from datetime import timedelta
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC, SVR
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import core.FeatureImportance as FI
import core.DrawPic as DrawPic
## Support vector machines

def run_svm( X_new, X_test_new, Y_train, Y_test,Y_name,groups,model_type):
    print("Running SVM regression")
    model_name = "SVM"
    grid_svm = {'C': [0.01, 0.1, 1, 10],
                'kernel': ["linear", "poly", "rbf", "sigmoid"],
                'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10, "scale", "auto"]}

    grid_svm_debug={'C': [0.01, 0.1, 1, 10], 'kernel': ["linear"]}

    # Set up model
    if(model_type == "regr"):
        model = SVR()
        scoring = 'neg_mean_squared_error'  
    elif(model_type == "class"):
        grid_svm.setdefault('degree', [2, 3, 4])          
        grid_svm.setdefault('class_weight', [None, 'balanced'])
        model = SVC(probability=True,class_weight='balanced')
        scoring = 'balanced_accuracy'  
    else:
        raise ValueError("model_type must be 'regr' or 'class'")
    
    svm = RandomizedSearchCV(estimator = model, param_distributions = grid_svm, scoring = scoring, 
                                  n_iter = n_iter, cv = util.PseudoGroupCV(kfold,groups), verbose = 0, 
                                  random_state = random_state, n_jobs = -1)

    start = perf_counter()
    print("Fitting SVM model")
    svm.fit(X_new, Y_train.values.ravel())
    stop = perf_counter()
    print("Time: ", timedelta(seconds = stop -start))
    svm_best = svm.best_estimator_
    util.save_results_cv_pipe(svm_best, model_name, model_type, scoring, Y_name)
    stop = perf_counter()
    print("Time: ", timedelta(seconds = stop-start))
    joblib.dump(svm_best, f'svm_{Y_name}.pkl')
    y_pred_test = svm_best.predict(X_test_new)
    if model_type == "class":
        performance = util.calc_performance(Y_test, y_pred_test, model_name, Y_name,X_test_new,model_type,X_new=X_new,Y_train=Y_train.values.ravel(),model=svm_best)
    else:
        performance = util.calc_performance(Y_test, y_pred_test, model_name, Y_name,X_test_new,model_type)
    print(performance)

    ### Calculate feature_importances
    FI.calc_permutation_feature_importance(svm_best,X_test_new,Y_test,model_name,model_type,Y_name)
    shap_value=FI.calc_shap_feature_importances(svm_best,X_test_new,X_new,Y_name,model_type,model_name)
    DrawPic.draw_pic(shap_value,X_test_new,Y_name)
