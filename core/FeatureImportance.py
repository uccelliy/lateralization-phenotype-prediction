import pandas as pd
import shap
from sklearn.inspection import permutation_importance
import numpy as np

def calc_tree_gini_feature_importance(estimator,model_name,model_type,Y_name):

    if model_name.startswith("RF") or model_name.startswith("XGB"):

        gini_imp = estimator.feature_importances_
        gini_imp_scaled = (gini_imp - np.min(gini_imp)) / (np.max(gini_imp) - np.min(gini_imp))
        df_featimp = pd.read_csv(f"../results/tree_feature_importances_{model_type}_{Y_name}.csv", index_col=0)
        feature_importances = pd.DataFrame([gini_imp, gini_imp_scaled], columns=df_featimp.columns.to_list(),
                                           index=[f"{model_name}_{model_type}_gini",
                                                  f"{model_name}_{model_type}_gini_scaled"])
        df_featimp = pd.concat([df_featimp, feature_importances])
        df_featimp.to_csv(f"../results/tree_feature_importances_{model_type}_{Y_name}.csv")
        return df_featimp
    else:
        return None


def calc_permutation_feature_importance(estimator, X, y, model_name, model_type,Y_name):
    if model_type == "class":
        scoring = 'balanced_accuracy'
    elif model_type == "regr":
        scoring = "neg_root_mean_squared_error"
    else:
        scoring = None

    feature_imp = permutation_importance(estimator, X, y, scoring=scoring, n_jobs = -1, random_state=42).importances_mean
    feature_imp_scaled = (feature_imp - np.min(feature_imp)) / (np.max(feature_imp) - np.min(feature_imp))

    # Save feature importances
    df_perm_featimp = pd.read_csv(f"../results/perm_feature_importances_{model_type}_{Y_name}.csv", index_col=0)
    feature_importances = pd.DataFrame([feature_imp, feature_imp_scaled], columns=df_perm_featimp.columns.tolist(),
                                       index=[f"{model_name}_{model_type}", f"{model_name}_{model_type}_scaled"])
    df_perm_featimp = pd.concat([df_perm_featimp, feature_importances])
    df_perm_featimp.to_csv(f"../results/perm_feature_importances_{model_type}_{Y_name}.csv")
    return feature_importances


def calc_shap_feature_importances(estimator,X_test_new,X_new,Y_name,model_type,model_name):
    if model_name in ["XGB", "RF"]:
        explainer = shap.TreeExplainer(estimator, X_test_new)
        shap_values = explainer(X_test_new)
    elif model_name=="SVM":
        num_features = X_test_new.shape[1]
        max_evals = max(2 * num_features + 1, 1500) 
        background = shap.sample(X_new, 100)  
        explainer = shap.KernelExplainer(estimator.predict, background)
        shap_values = explainer.shap_values(X_test_new, nsamples=max_evals)
    
    
    # Save SHAP values per person
    shap_pp_df_rf = pd.DataFrame(shap_values.values, columns = X_test_new.columns)
    shap_pp_df_rf.to_csv(f"../results/shap_rf_pp_{Y_name}.csv")

    # Average over all participants
    importances = []
    for i in range(shap_values.values.shape[1]):
        importances.append(np.mean(np.abs(shap_values.values[:, i])))

    
    df_shap_featimp=pd.read_csv(f"../results/shap_feature_importances_{model_type}_{Y_name}.csv", index_col=0)
    feature_importances = pd.DataFrame([importances],columns=df_shap_featimp.columns.tolist(),index=[f"{Y_name}_shap"])
    df_shap_featimp=pd.concat([df_shap_featimp,feature_importances])
    df_shap_featimp.to_csv(f"../results/shap_feature_importances_{model_type}_{Y_name}.csv")

    return shap_values


