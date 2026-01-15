from core.util import n_iter, kfold, random_state
import core.util as util
import pandas as pd
from time import perf_counter
import shap
from datetime import timedelta
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, KFold, train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import core.FeatureImportance as FI
import core.DrawPic as DrawPic

### Random forest regression
def run_rf(X_new, X_test_new, Y_train, Y_test,Y_name,groups,model_type):
    print("Running RF regression")
    model_name = "RF"
    grid_rf = {'n_estimators': list(range(100, 1100, 100)), # Nr of trees
              'max_features': list(range(4, 32)), # Number of features to consider at every split
              'max_depth': list(range(2, 15)), # Max depth of tree
              'min_samples_split': list(range(2 ,11)), # Minimum number of samples required to split a node
              'min_samples_leaf': list(range(1 ,11))}
    grid_rf_debug={'n_estimators': list(range(100, 1100, 100)),'max_depth': list(range(2, 15))}
    
    if(model_type == "regr"):
        model = RandomForestRegressor(random_state = random_state)
        scoring = 'neg_mean_squared_error'  # Default scoring for regression
    elif(model_type == "class"):
        model = RandomForestClassifier(random_state = random_state)
        scoring = 'balanced_accuracy'  # Default scoring for classification  试了三种accuracy f1 balanced_accuracy最好的时balanced_accuracy 这里都可以写什么问一下chat
        grid_rf.setdefault("class_weight", [None, 'balanced'])
    else:
        raise ValueError("model_type must be 'regr' or 'class'")
    
    rf = RandomizedSearchCV(estimator = model, param_distributions = grid_rf, scoring = scoring, 
                                           n_iter = n_iter, cv = util.PseudoGroupCV(kfold,groups), verbose = 0, 
                                           random_state = random_state, n_jobs = -1)
    
    start = perf_counter()
    print("Fitting RF model")
    rf.fit(X_new, Y_train)
    stop = perf_counter()
    print("Time: ", timedelta(seconds = stop -start))
    rf_best = rf.best_estimator_
    util.save_results_cv_pipe(rf_best, model_name, model_type, scoring,Y_name)
    joblib.dump(rf_best, f'rf_{Y_name}.pkl')
    y_pred_test = rf_best.predict(X_test_new)
    if model_type == "class":
        performance = util.calc_performance(Y_test, y_pred_test, model_name,Y_name,X_test_new,model_type,X_new=X_new,Y_train=Y_train.values.ravel(),model=rf_best)
    else:
        performance = util.calc_performance(Y_test, y_pred_test, model_name,Y_name,X_test_new,model_type)
    print(performance)

    ### Calculate feature_importances
    FI.calc_permutation_feature_importance(rf_best,X_test_new,Y_test,model_name,model_type,Y_name)
    FI.calc_tree_gini_feature_importance(rf_best,model_name,model_type,Y_name)
    shap_value=FI.calc_shap_feature_importances(rf_best,X_test_new,X_new,Y_name,model_type,model_name)
    DrawPic.draw_pic(shap_value,X_test_new,Y_name)
