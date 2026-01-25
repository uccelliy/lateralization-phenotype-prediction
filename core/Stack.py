from sklearn.utils import compute_sample_weight
import core.util as util
from core.util import n_iter, kfold, random_state
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from mlxtend.regressor import StackingCVRegressor
from mlxtend.classifier import StackingCVClassifier
from time import perf_counter
from datetime import timedelta
import joblib
from sklearn.preprocessing import LabelEncoder as LE
## Stacked model
def run_stack(X_new, X_test_new, Y_train, Y_test,Y_name,groups,model_type):
    print("Running Stacked model")
    sample_weight = compute_sample_weight(class_weight='balanced', y=Y_train)
# Set up model
    model_name = "Stack"
    rf=joblib.load(f'rf_{Y_name}.pkl')
    svm=joblib.load(f'svm_{Y_name}.pkl')
    xgb_mod=joblib.load(f'xgb_{Y_name}.pkl')

# Define parameter grid
    grid_stack = {'n_estimators': list(range(100, 1100, 100)),
                'learning_rate': [0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.3],
                'gamma': [i/10 for i in range(0,6)],
                'max_depth': list(range(2, 16)),
                'min_child_weight': list(range(1,11)),
                'subsample': [x/10 for x in range(2, 11)],
                'colsample_bytree': [x/10 for x in range(2, 11)],
                'reg_lambda': [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5],
                'reg_alpha': [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5]}
    grid_pipe_debug={'learning_rate': [ 0.05, 0.075, 0.1], 'max_depth': [2,4,6]}
    
    if (model_type == "regr"):
        scoring = 'neg_mean_squared_error'
        f_clf = xgb.XGBRegressor(random_state=random_state)
    elif (model_type == "class"):
        scoring = 'balanced_accuracy'
        f_clf = xgb.XGBClassifier(random_state=random_state)
        le = LE()
        Y_train = pd.Series(le.fit_transform(Y_train), index=Y_train.index, name="target")
        Y_test = pd.Series(le.transform(Y_test), index=Y_test.index, name="target")
        
    grid_search_meta = RandomizedSearchCV(estimator = f_clf,
                                param_distributions  = grid_stack,
                                scoring = scoring,
                                cv = util.PseudoGroupCV(kfold,groups),
                                verbose = 0, random_state = random_state,
                                n_jobs = -1, n_iter=n_iter)

    grid_search_meta.fit(X_new, Y_train,sample_weight=sample_weight)
    if model_type == "regr":
        stack_pipeline = StackingCVRegressor(regressors = (svm, rf, xgb_mod), meta_regressor = grid_search_meta.best_estimator_,
                                            refit =False,verbose=0, n_jobs = 1)
    elif model_type == "class":
        stack_pipeline = StackingCVClassifier(classifiers = (svm, rf, xgb_mod), meta_classifier = grid_search_meta.best_estimator_,
                                            use_probas=True, verbose=0, n_jobs = 1)

    start = perf_counter()
    print("Fitting Stacked model")
    stack_pipeline.fit(X_new, Y_train)
    stop = perf_counter()
    print("Time: ", timedelta(seconds = stop -start))
    util.save_results_cv_pipe(grid_search_meta, model_name, model_type, scoring,Y_name)

    y_pred_test = stack_pipeline.predict(X_test_new)
    if model_type == "class":
        performance = util.calc_performance(Y_test, y_pred_test, model_name,Y_name,X_test_new,model_type,X_new=X_new,Y_train=Y_train.values.ravel(),model=stack_pipeline)
    else:
        performance = util.calc_performance(Y_test, y_pred_test, model_name,Y_name,X_test_new,model_type)

