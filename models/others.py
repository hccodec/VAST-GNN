from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from prophet import Prophet

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from statsmodels.tsa.arima.model import ARIMA

def gaussian_reg_time(start_exp,n_samples,labels,i_ahead,rand_seed=0):
    y_pred_mat = np.zeros((0))
    y_true_mat = np.zeros((0))

    for test_sample in range(start_exp,n_samples-i_ahead):#
        print(test_sample)
        y_pred_arr = np.zeros((0))
        y_true_arr = np.zeros((0))
        for j in range(labels.shape[0]):
            ds = labels.iloc[j,:test_sample-1].reset_index()

            X_train = np.empty((0, 1))
            y_train = np.empty((0))
            for k in range(ds.shape[0]):
                X_train = np.append(X_train, [[k]], axis=0)
                y_train = np.append(y_train, ds.iloc[k, 1])

            if(sum(ds.iloc[:,1])==0):
                yhat = np.array([0])
            elif(X_train.shape[0]==0):
                continue
            else:
                reg = GaussianProcessRegressor(kernel=RBF()).fit(X_train, y_train)
                yhat = reg.predict([[test_sample+i_ahead-1]])
            y_me = labels.iloc[j,test_sample+i_ahead-1]
            y_pred_arr = np.append(y_pred_arr, yhat)
            y_true_arr = np.append(y_true_arr, y_me)
        y_pred_mat = np.append(y_pred_mat, y_pred_arr)
        y_true_mat = np.append(y_true_mat, y_true_arr)

    return y_pred_mat, y_true_mat

def lin_reg_time(start_exp,n_samples,labels,i_ahead):
    y_pred_mat = np.zeros((0))
    y_true_mat = np.zeros((0))

    for test_sample in range(start_exp,n_samples-i_ahead):#
        print(test_sample)
        y_pred_arr = np.zeros((0))
        y_true_arr = np.zeros((0))
        for j in range(labels.shape[0]):
            ds = labels.iloc[j,:test_sample-1].reset_index()

            X_train = np.empty((0, 1))
            y_train = np.empty((0))
            for k in range(ds.shape[0]):
                X_train = np.append(X_train, [[k]], axis=0)
                y_train = np.append(y_train, ds.iloc[k, 1])

            if(sum(ds.iloc[:,1])==0):
                yhat = np.array([0])
            elif(X_train.shape[0]==0):
                continue
            else:
                reg = LinearRegression().fit(X_train, y_train)
                yhat = reg.predict([[test_sample+i_ahead-1]])
            y_me = labels.iloc[j,test_sample+i_ahead-1]
            y_pred_arr = np.append(y_pred_arr, yhat)
            y_true_arr = np.append(y_true_arr, y_me)
        y_pred_mat = np.append(y_pred_mat, y_pred_arr)
        y_true_mat = np.append(y_true_mat, y_true_arr)

    return y_pred_mat, y_true_mat

def rand_forest_time(start_exp,n_samples,labels,i_ahead,rand_seed=0):
    y_pred_mat = np.zeros((0))
    y_true_mat = np.zeros((0))

    for test_sample in range(start_exp,n_samples-i_ahead):#
        print(test_sample)
        y_pred_arr = np.zeros((0))
        y_true_arr = np.zeros((0))
        for j in range(labels.shape[0]):
            ds = labels.iloc[j,:test_sample-1].reset_index()

            X_train = np.empty((0, 1))
            y_train = np.empty((0))
            for k in range(ds.shape[0]):
                X_train = np.append(X_train, [[k]], axis=0)
                y_train = np.append(y_train, ds.iloc[k, 1])

            if(sum(ds.iloc[:,1])==0):
                yhat = np.array([0])
            elif(X_train.shape[0]==0):
                continue
            else:
                reg = RandomForestRegressor().fit(X_train, y_train)
                yhat = reg.predict([[test_sample+i_ahead-1]])
            y_me = labels.iloc[j,test_sample+i_ahead-1]
            y_pred_arr = np.append(y_pred_arr, yhat)
            y_true_arr = np.append(y_true_arr, y_me)
        y_pred_mat = np.append(y_pred_mat, y_pred_arr)
        y_true_mat = np.append(y_true_mat, y_true_arr)

    return y_pred_mat, y_true_mat

def xgboost(start_exp,n_samples,labels,i_ahead,rand_seed=0):
    y_pred_mat = np.zeros((0))
    y_true_mat = np.zeros((0))

    for test_sample in range(start_exp,n_samples-i_ahead):#
        print(test_sample)
        y_pred_arr = np.zeros((0))
        y_true_arr = np.zeros((0))
        for j in range(labels.shape[0]):
            ds = labels.iloc[j,:test_sample-1].reset_index()

            X_train = np.empty((0, 1))
            y_train = np.empty((0))
            for k in range(ds.shape[0]):
                X_train = np.append(X_train, [[k]], axis=0)
                y_train = np.append(y_train, ds.iloc[k, 1])

            if(sum(ds.iloc[:,1])==0):
                yhat = np.array([0])
            elif(X_train.shape[0]==0):
                continue
            else:
                reg = XGBRegressor(n_estimators=1000).fit(X_train, y_train)
                yhat = reg.predict([[test_sample+i_ahead-1]])
            y_me = labels.iloc[j,test_sample+i_ahead-1]
            y_pred_arr = np.append(y_pred_arr, yhat)
            y_true_arr = np.append(y_true_arr, y_me)
        y_pred_mat = np.append(y_pred_mat, y_pred_arr)
        y_true_mat = np.append(y_true_mat, y_true_arr)

    return y_pred_mat, y_true_mat

def prophet(ahead, start_exp, n_samples, labels):
    var = []
    y_pred = []
    y_true = []
    for idx in range(ahead):
        var.append([])

    error= np.zeros(ahead)
    count = 0
    for test_sample in range(start_exp,n_samples-ahead):#
        print(test_sample)
        count+=1
        err = 0
        for j in range(labels.shape[0]):
            ds = labels.iloc[j,:test_sample].reset_index()
            ds.columns = ["ds","y"]
            #with suppress_stdout_stderr():
            m = Prophet(interval_width=0.95)
            m.fit(ds)
            future = m.predict(m.make_future_dataframe(periods=ahead))
            yhat = future["yhat"].tail(ahead)
            y_me = labels.iloc[j,test_sample:test_sample+ahead]
            e =  abs(yhat-y_me.values).values
            err += e
            error += e
            y_pred.append(yhat)
            y_true.append(y_me.values)
        for idx in range(ahead):
            var[idx].append(err[idx])
            
    return error, var, y_pred, y_true

def arima(ahead,start_exp,n_samples,labels):
    var = []
    y_pred = []
    y_true = []
    for idx in range(ahead):
        var.append([])

    error= np.zeros(ahead)
    count = 0
    for test_sample in range(start_exp,n_samples-ahead):#
        print(test_sample)
        count+=1
        err = 0
        for j in range(labels.shape[0]):
            ds = labels.iloc[j,:test_sample-1].reset_index()

            if(sum(ds.iloc[:,1])==0):
                yhat = [0]*(ahead)
            else:
                try:
                    fit2 = ARIMA(ds.iloc[:,1].values, order=(2, 0, 2)).fit()
                except:
                    fit2 = ARIMA(ds.iloc[:,1].values, order=(1, 0, 0)).fit()
                yhat = abs(fit2.predict(start = test_sample , end = (test_sample+ahead-1) ))
            y_me = labels.iloc[j,test_sample:test_sample+ahead]
            e =  abs(yhat - y_me.values)
            err += e
            error += e
            y_pred.append(yhat)
            y_true.append(y_me.values)

        for idx in range(ahead):
            var[idx].append(err[idx])
    return error, var, y_pred, y_true

# def write_results(model_name, idx, error, var, y_true, y_pred, count, n_nodes):
#     # 写入结果到文件
#     with open(f"../results/results_{country}_baseline.csv", "a") as fw:
#         fw.write(f"{model_name},{idx},{:.5f}".format(error/(count * n_nodes)) +
#                  f",{:.5f}".format(np.std(var[idx])) +
#                  f",{:.5f}".format(mean_absolute_error(y_true, y_pred)) +
#                  f",{:.5f}".format(mean_squared_error(y_true, y_pred)) +
#                  f",{:.5f}".format(mean_squared_error(y_true, y_pred, squared=False)) +
#                  f",{:.5f}".format(r2_score(y_true, y_pred)) + "\n")


def process_model(model_name, ahead, start_exp, n_samples, labels, rand_seed=None):
    count = len(range(start_exp, n_samples - ahead))
    
    if model_name == "PROPHET":
        error, var, y_pred, y_true = prophet(ahead, start_exp, n_samples, labels)
        # for idx, e in enumerate(error):
        #     write_results("PROPHET", idx, e, var, y_true, y_pred, count, n_nodes)
    
    elif model_name == "ARIMA":
        error, var, y_pred, y_true = arima(ahead, start_exp, n_samples, labels)
        # for idx, e in enumerate(error):
        #     write_results("ARIMA", idx, e, var, y_true, y_pred, count, n_nodes)
    
    else:
        model_functions = {
            "LIN_REG": lin_reg_time,
            "RAND_FOREST": rand_forest_time,
            "GAUSSIAN_REG": gaussian_reg_time,
            "XGBOOST": xgboost
        }
        
        if model_name in model_functions:
            model_func = model_functions[model_name]
            for shift_time in range(ahead):
                y_pred, y_true = model_func(start_exp, n_samples, labels, shift_time, rand_seed=rand_seed)
                # write_results(model_name, shift_time, 0, None, y_true, y_pred, count, n_nodes)

# # 调用主处理函数
# process_model(args.model, args.ahead, args.start_exp, n_samples, labels, rand_seed=args.rand_seed)
