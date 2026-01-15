
from joblib import Parallel,delayed
import pandas as pd

import core.util as util
import core.RF as RF
import core.SVM as SVM
import core.XGB as XGB
import core.Stack as Stack
import os
import core.FeatureSelection as FS


# Read in the data
input_file = 'C:/Users/77260/Desktop/111/dti_5e6.txt'
input_file4 = 'C:/Users/77260/Desktop/111/class_test.txt'
X = pd.read_csv(input_file, sep="\t", decimal='.', encoding='cp1252')
print(X.shape)
Y_all = pd.read_csv(input_file4, sep="\t", decimal='.', encoding='cp1252')
print(Y_all.shape)
behaviors = Y_all.columns[2:]  # 跳过前两列（假设前两列非目标变量），保证前两是被试编号和家庭编号，如果有bug确保没多删或者少删
subjects_num_X = X.shape[0]-1
subjects_num_Y = Y_all.shape[0]-1
summary_df = pd.read_csv("behavior_summary.csv", index_col=0) if os.path.exists("behavior_summary.csv") else pd.DataFrame()
model_type = "regr" #选择模型类型 class/regr 唯一需要修改的地方
for behav_name in behaviors:

    print("Running model for behavior: ", behav_name)
    
    Y=Y_all[["IID","FID",behav_name]]
    X_train, X_test, Y_train, Y_test, groups = util.prepare_data(X, Y, behav_name,model_type)
    #保存当前模型的基本统计信息包括 1.输入X有多少被试2.输入的Y有多少被试3.处理后训练集有多少被试4.处理后的测试集有多少被试5.如果是回归Y的平均值是多少如果是分类每一类占比有多少
    subject_co_train= X_train.shape[0] - 1
    subject_co_test= X_test.shape[0] - 1
    row_data = {"behavior": behav_name,
                "train_subjects": subject_co_train,
                "test_subjects": subject_co_test,
                "all_subjects_X": subjects_num_X,"all_Y_subjects": subjects_num_Y}
    if model_type == "class":
        type_percent = []
        test_type_percent = []
        types = sorted(Y_train[behav_name].unique())
        train_counts = Y_train[behav_name].value_counts(normalize=True)
        test_counts = Y_test[behav_name].value_counts(normalize=True)
        for c in types:
            row_data[f"class_{c}_train_ratio"] = train_counts.get(c, 0)
            row_data[f"class_{c}_test_ratio"] = test_counts.get(c, 0)
    elif model_type == "regr":
        mean_value = Y_train[behav_name].mean()
        mean_value_test = Y_test[behav_name].mean()
        row_data["mean_train"] = mean_value
        row_data["mean_test"] = mean_value_test
    summary_df = pd.concat([summary_df, pd.DataFrame([row_data])], ignore_index=True)
    summary_df.to_csv("behavior_summary.csv", index=False)
    
    #进行特征筛选
    X_train_new, transform = FS.feature_selection(X_train, Y_train, groups, model_type=model_type,method="None")
    util.featimp_file_init(X_train_new,model_type,behav_name)
    if transform is None:
        print(f"No features selected for {behav_name}. Skipping...")
        X_test_new = X_test 
    else:
        X_test_new = transform.transform(X_test)
    
    util.result_file_init_best(behav_name)
    util.result_file_init_performance(behav_name, model_type)


    RF.run_rf(X_train_new, X_test_new, Y_train, Y_test, behav_name, groups,model_type)
    SVM.run_svm(X_train_new, X_test_new, Y_train, Y_test, behav_name, groups,model_type)
    XGB.run_xgb(X_train_new, X_test_new, Y_train, Y_test, behav_name, groups,model_type)
 
    # Parallel(n_jobs=3)(
    #     delayed(model_func)(X_train_new, X_test_new, Y_train, Y_test, behav_name, groups,model_type)
    #     for model_func in [RF.run_rf, SVM.run_svm, XGB.run_xgb]
    #     )
    
    # Stack.run_stack(X_train_new, X_test_new, Y_train, Y_test, behav_name, groups,model_type)
    
   


