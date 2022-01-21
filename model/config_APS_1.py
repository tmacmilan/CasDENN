DATA_PATHA_pre = "../data/aps"
DATA_PATHA = "../data/aps/5years"
cascades  = DATA_PATHA_pre+"/APS_preprocessing.txt"
datasetName='aps_5years_'
cascade_train = DATA_PATHA+"/cascade_train.txt"
cascade_val = DATA_PATHA+"/cascade_val.txt"
cascade_test = DATA_PATHA+"/cascade_test.txt"
shortestpath_train = DATA_PATHA+"/shortestpath_train.txt"
shortestpath_val = DATA_PATHA+"/shortestpath_val.txt"
shortestpath_test = DATA_PATHA+"/shortestpath_test.txt"

observation = 5*365*24*60*60-1 ##5 years
observation_time=observation
pre_times = [20*365*24*60*60]
import math
train_pkl = DATA_PATHA+"/data_train.pkl"
val_pkl = DATA_PATHA+"/data_val.pkl"
test_pkl = DATA_PATHA+"/data_test.pkl"
information = DATA_PATHA+"/information.pkl"

#parameters
print ("observation time",observation)
n_time_interval = 12
print ("the number of time interval:",n_time_interval)
time_interval = math.ceil((observation+1)*1.0/n_time_interval)# 
print ("time interval:",time_interval)
lmax = 2
degree_interval_list = [0,1,2,3,5,8,10,15,20,30,40,50,70,90,110,120,150]
