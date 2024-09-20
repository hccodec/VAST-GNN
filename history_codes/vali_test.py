import torch, numpy as np


#5.1 RMSE, MAPE, MAE, RMSLE
def RMSELoss(yhat,y):
    return float(torch.sqrt(torch.mean((yhat-y)**2)))

def MAPELoss(yhat,y):
    return float(torch.mean(torch.div(torch.abs(yhat-y), y)))

def MAELoss(yhat,y):
    return float(torch.mean(torch.div(torch.abs(yhat-y), 1)))

def RMSLELoss(yhat,y):
    log_yhat = torch.log(yhat+1)
    log_y = torch.log(y+1)
    return float(torch.sqrt(torch.mean((log_yhat-log_y)**2)))    

def compute(
        validate_x_y, validate_hat, validate_real,
        test_x_y, test_hat, test_real,
        infection_normalize_ratio
):
    #compute RMSE
    rmse_validate = list()
    rmse_test = list()
    for i in range(len(validate_x_y)):
        rmse_validate.append(float(RMSELoss(validate_hat[i],validate_real[i])))
    for i in range(len(test_x_y)):
        rmse_test.append(float(RMSELoss(test_hat[i],test_real[i])))
    # print ("rmse_validate mean", np.mean(rmse_validate))
    # print ("rmse_test mean", np.mean(rmse_test))

    #compute MAE
    mae_validate = list()
    mae_test = list()
    for i in range(len(validate_x_y)):
        mae_validate.append(float(MAELoss(validate_hat[i],validate_real[i])))
    for i in range(len(test_x_y)):
        mae_test.append(float(MAELoss(test_hat[i],test_real[i])))

    # print('---- COMPUTE START ----')
        
    # print ("mae_validate mean", np.mean(mae_validate))
    # print ("mae_test mean", np.mean(mae_test))

    #show RMSE and MAE together
    mae_validate, rmse_validate, mae_test, rmse_test =np.array(mae_validate)*infection_normalize_ratio, np.array(rmse_validate)*infection_normalize_ratio,np.array(mae_test)*infection_normalize_ratio, np.array(rmse_test)*infection_normalize_ratio
    # print ("-----------------------------------------")
    print(f"\ntest(mae,rmse) {round(np.mean(mae_test),3)}/{round(np.mean(rmse_test),3)}",
          f"validate(mae,rmse) {round(np.mean(mae_validate),3)}/{round(np.mean(rmse_validate),3)}")
    # print ("\nmae_validate mean", round(np.mean(mae_validate),3), "     rmse_validate mean", round(np.mean(rmse_validate),3))
    # print ("mae_test mean", round(np.mean(mae_test),3), "         rmse_test mean", round(np.mean(rmse_test),3))
    # print ("-----------------------------------------")
    # print('---- COMPUTE END ----')

    return rmse_validate, rmse_test, mae_validate, mae_test