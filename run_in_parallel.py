import pandas as pd
from joblib import Parallel, delayed
import os
from tqdm import tqdm  # 可选：进度条
import core.util as util
import core.RF as RF
import core.SVM as SVM
import core.XGB as XGB
import core.Stack as Stack
import core.FeatureSelection as FS
def run_models_for_behavior(behav_name, X, Y_all,model_type,transform_method="None"):
    
    Y = Y_all
    X_train, X_test, Y_train, Y_test, groups = util.prepare_data(X, Y, behav_name,model_type)
    X_train_new, transform = FS.feature_selection(X_train, Y_train, groups, model_type=model_type, method=transform_method)
    if transform is None:
        X_test_new = X_test 
    else:
        X_test_new = transform.transform(X_test)
        
    util.result_file_init_best(behav_name)
    util.result_file_init_performance(behav_name, model_type)
    
    Parallel(n_jobs=3)(
        delayed(model_func)(X_train_new, X_test_new, Y_train, Y_test, behav_name, groups,model_type)
        for model_func in [RF.run_rf, SVM.run_svm, XGB.run_xgb]
    )

    Stack.run_stack(X_train_new, X_test_new, Y_train, Y_test, behav_name, groups,model_type)


def main():
    # Read in the data 可以修改路径来读目标文件
    model_type = "regr"
    os.chdir(os.path.dirname(__file__))
    input_file_X = 'C:/Users/77260/Desktop/111/dti_5e6.txt'
    X = pd.read_csv(input_file_X, sep="\t", decimal='.', encoding='cp1252')
    print(X.shape)

    input_file_Y = 'C:/Users/77260/Desktop/111/regr_test.txt'
    Y_all = pd.read_csv(input_file_Y, sep="\t", decimal='.', encoding='cp1252')
    print(Y_all.shape)
    behaviors = Y_all.columns[2:]
    
    # 并行运行所有 behaviors
    Parallel(n_jobs=os.cpu_count() - 1)(
        [delayed(run_models_for_behavior)(behav_name, X, Y_all[["IID","FID",behav_name]], model_type) for behav_name in behaviors]
    )

if __name__ == "__main__":
    main()
