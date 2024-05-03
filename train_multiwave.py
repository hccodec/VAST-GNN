
# # Step 4. Training and testing

# In[5]:

from preprocess_multiwave import *

#4.1
#read the data
train_x_y, validate_x_y, test_x_y, all_mobility, all_infection, train_original, validate_original, test_original, train_list, validation_list =read_data()
#train_x_y, validate_x_y, test_x_y = normalize(train_x_y, validate_x_y, test_x_y)

#train_x_y = train_x_y[0:30]
print (len(train_x_y))
print ("---------------------------------finish data preparation------------------------------------")


# In[6]:

#4.2
#train the model
e_losses, trained_model = model_train(train_x_y, validate_x_y, test_x_y, device)
print ("---------------------------finish model training-------------------------")


# In[7]:


#4.3 
print (len(train_x_y))
print (len(validate_x_y))
print (len(test_x_y))
#4.3.1 model validation
validation_result, validate_hat, validate_real = validate_test_process(trained_model, validate_x_y)
print ("---------------------------------finish model validation------------------------------------")
print (len(validate_hat))
print (len(validate_real))
#4.3.2 model testing
#4.4. model test
test_result, test_hat, test_real = validate_test_process(trained_model, test_x_y)
print ("---------------------------------finish model testing------------------------------------")
print (len(test_real))
print (len(test_hat))


# # Step 5: Evaluation

# In[8]:


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

#compute RMSE
rmse_validate = list()
rmse_test = list()
for i in range(len(validate_x_y)):
    rmse_validate.append(float(RMSELoss(validate_hat[i],validate_real[i])))
for i in range(len(test_x_y)):
    rmse_test.append(float(RMSELoss(test_hat[i],test_real[i])))
print ("rmse_validate mean", np.mean(rmse_validate))
print ("rmse_test mean", np.mean(rmse_test))

#compute MAE
mae_validate = list()
mae_test = list()
for i in range(len(validate_x_y)):
    mae_validate.append(float(MAELoss(validate_hat[i],validate_real[i])))
for i in range(len(test_x_y)):
    mae_test.append(float(MAELoss(test_hat[i],test_real[i])))
    
print ("mae_validate mean", np.mean(mae_validate))
print ("mae_test mean", np.mean(mae_test))

#show RMSE and MAE together
mae_validate, rmse_validate, mae_test, rmse_test =np.array(mae_validate)*infection_normalize_ratio, np.array(rmse_validate)*infection_normalize_ratio,np.array(mae_test)*infection_normalize_ratio, np.array(rmse_test)*infection_normalize_ratio
print ("-----------------------------------------")
print ("mae_validate mean", round(np.mean(mae_validate),3), "     rmse_validate mean", round(np.mean(rmse_validate),3))
print ("mae_test mean", round(np.mean(mae_test),3), "         rmse_test mean", round(np.mean(rmse_test),3))
print ("-----------------------------------------")


# In[9]:


print (trained_model.v)


# In[10]:


print(validate_hat[0][Y_day-1])
print(torch.sum(validate_hat[0][Y_day-1]))
print(validate_real[0][Y_day-1])
print(torch.sum(validate_real[0][Y_day-1]))


# In[11]:


x = range(len(rmse_validate))
plt.figure(figsize=(8,2),dpi=300)
l1 = plt.plot(x, np.array(rmse_validate), 'ro-',linewidth=0.8, markersize=1.2, label='RMSE')
l2 = plt.plot(x, np.array(mae_validate), 'go-',linewidth=0.8, markersize=1.2, label='MAE')
plt.xlabel('Date from the first day of validation',fontsize=12)
plt.ylabel("RMSE/MAE daily new cases",fontsize=10)
my_y_ticks = np.arange(0,2100, 500)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
plt.grid()
plt.show()


# In[12]:


x = range(len(mae_test))
plt.figure(figsize=(8,2),dpi=300)
l1 = plt.plot(x, np.array(rmse_test), 'ro-',linewidth=0.8, markersize=1.2, label='RMSE')
l2 = plt.plot(x, np.array(mae_test), 'go-',linewidth=0.5, markersize=1.2, label='MAE')
plt.xlabel('Date from the first day of test',fontsize=12)
plt.ylabel("RMSE/MAE Daily new cases",fontsize=10)
my_y_ticks = np.arange(0,2100, 500)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
plt.grid()
plt.show()


# # Correlation

# In[13]:


from scipy import stats
#validate
y_days = Y_day
validate_hat_sum = [float(torch.sum(validate_hat[i][y_days-1])) for i in range(len(validate_hat))]
validate_real_sum = [float(torch.sum(validate_real[i][y_days-1])) for i in range(len(validate_real))]
print ("the correlation between validation: ", stats.pearsonr(validate_hat_sum, validate_real_sum)[0])
#test
test_hat_sum = [float(torch.sum(test_hat[i][y_days-1])) for i in range(len(test_hat))]
test_real_sum = [float(torch.sum(test_real[i][y_days-1])) for i in range(len(test_real))]
print ("the correlation between test: ", stats.pearsonr(test_hat_sum, test_real_sum)[0])
#train
train_result, train_hat, train_real = validate_test_process(trained_model, train_x_y)
train_hat_sum = [float(torch.sum(train_hat[i][0])) for i in range(len(train_hat))]
train_real_sum = [float(torch.sum(train_real[i][0])) for i in range(len(train_real))]
print ("the correlation between train: ", stats.pearsonr(train_hat_sum, train_real_sum)[0])


# # step 6. Visualization

# In[38]:


y1List = [np.sum(list(train_original[i+1][1][Y_day-1].values())) for i in range(len(train_original)-1)]
y2List = [np.sum(list(validate_original[i][1][Y_day-1].values())) for i in range(len(validate_original))]
y2List_hat = [float(torch.sum(validate_hat[i][Y_day-1])) for i in range(len(validate_hat))]
y3List = [np.sum(list(test_original[i][1][Y_day-1].values())) for i in range(len(test_original))]
y3List_hat = [float(torch.sum(test_hat[i][Y_day-1])) for i in range(len(test_hat))]

#x1 = np.array(range(len(y1List)))
#x2 = np.array([len(y1List)+j for j in range(len(y2List))])
x1 = train_list
x2 = validation_list
x3 = np.array([len(y1List)+len(y2List)+j for j in range(len(y3List))])

plt.figure(figsize=(8,2),dpi=300)
l1 = plt.plot(x1[0: len(y1List)], np.array(y1List)*infection_normalize_ratio, 'ro-',linewidth=2, markersize=0.8, label='Training: real cases')
l2 = plt.plot(x2, np.array(y2List)*infection_normalize_ratio, 'bo-',linewidth=2, markersize=0.8, label='Validation: real cases')
l3 = plt.plot(x2, np.array(y2List_hat)*infection_normalize_ratio, 'b--',linewidth=2, markersize=0.1, label='Validation: predicted cases')
l4 = plt.plot(x3, np.array(y3List)*infection_normalize_ratio, 'go-',linewidth=2, markersize=0.8, label='Testing: real cases')
l5 = plt.plot(x3, np.array(y3List_hat)*infection_normalize_ratio, 'g--',linewidth=2, markersize=0.1, label='Testing: predicted cases')
#plt.xlabel('Date from the first day of 2020/4/1',fontsize=12)
plt.ylabel("Daily new cases",fontsize=14)
plt.xlabel("Days from July 20, 2020",fontsize=16)
my_y_ticks = np.arange(0,1700, 500)
my_x_ticks = list()
summary = 0 
my_x_ticks.append(summary) 
for i in range(5):
    summary += 60
    my_x_ticks.append(summary) 
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks) 
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title("SAB-GNN",fontsize=16)
plt.legend()
os.makedirs('result/figure_supp/', exist_ok=True)
plt.savefig('result/figure_supp/prediction_curve.pdf',bbox_inches = 'tight')
plt.show()


# # step 7: Regional visualization

# In[15]:


cityList = ['Chiyoda','Chuo','Minato','Shinjuku','Bunkyo','Taito',
 'Sumida','Koto','Shinagawa','Meguro','Ota','Setagaya',
 'Shibuya','Nakano','Suginami','Toshima','Kita','Arakawa',
 'Itabashi','Nerima','Adachi','Katsushika','Edogawa']


# In[16]:

@torch.no_grad()
def getPredictionPlot(k):
    #location k
    x_k = [i for i in range(len(test_real))]
    real_k = [test_real[i][y_days-1][k] for i in range(len(test_real))]
    predict_k = [test_hat[i][y_days-1][k] for i in range(len(test_hat))]
    plt.figure(figsize=(4,2.5), dpi=300)
    l1 = plt.plot(x_k, np.array(real_k)*infection_normalize_ratio, 'ro-',linewidth=0.8, markersize=2.0, label='real cases',alpha = 0.8)
    l2 = plt.plot(x_k, np.array(predict_k)*infection_normalize_ratio, 'o-',color='black',linewidth=0.8, markersize=2.0, alpha = 0.8, label='predicted cases')
    #plt.xlabel('Date from the first day of 2020/4/1',fontsize=12)
    #plt.ylabel("Daily infection cases",fontsize=10)
    my_y_ticks = np.arange(0,51,25)
    my_x_ticks = list()
    summary = 0 
    my_x_ticks.append(summary) 
    for i in range(3):
        summary += 20
        my_x_ticks.append(summary) 
    plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks) 
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    plt.title("District "+str(k+1)+": "+cityList[k],fontsize= 16)
    #plt.legend(fontsize=20)
    plt.legend(loc=2,fontsize=16)
    #plt.grid()  
    os.makedirs('result/peak4/', exist_ok=True)
    plt.savefig('result/peak4/' + str(k+1) + '.pdf',bbox_inches = 'tight')
    plt.show()


for i in range(23):
    getPredictionPlot(i)


# # Visualize the infection error



zone_i_error = [[0.0 for i in range(len(test_hat))] for j in range(23)]
for j in range(23):
    for i in range(len(test_hat)):
        predict = float(test_hat[i][Y_day-1][j])
        real = float(test_real[i][Y_day-1][j])
        zone_i_error[j][i] = abs(predict-real)/real
average_error = [np.mean(zone_i_error[j]) for j in range(23)]




print (average_error)

