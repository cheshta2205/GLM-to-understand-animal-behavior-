# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 14:26:47 2022

@author: cheshta, kiah
"""

# load relevant packages
import numpy as np
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
import csv
import os
import pandas as pd

# function to load csv files (these are made in matlab..made using modeling_dms_data_j7)
def load_data_from_csv(csv_file):
    file = open(csv_file)
    csvreader = csv.reader(file)
    header = next(csvreader)
    #print(header)
    rows = []
    for row in csvreader:
        rows.append(row)
    file.close()
    return rows

def create_y(y):

    y1 = []
    for row in y:
        if '1' not in row:
            y1.append(0)
        else:
            y1.append(row.index('1')+1)
    return y1

# function to load and split sessions
def load_split_sessions(prefix,count):
    
    # first, collect all the sessions
    for n in range(1,count):
        # get the X and y data from the session
        input_path = os.path.join(prefix,"inputtable_session_" + str(n) + ".csv")
        X = np.single(np.array(load_data_from_csv(input_path)))
        if n == 1:
            X_all = X
        else:
            X_all = np.append(X_all,X,axis=0)
            
        output_path = os.path.join(prefix,"outputtable_session_" + str(n) + ".csv")
        y = np.array(create_y(load_data_from_csv(output_path)))
        if n == 1:
            y_all = y
        else:
            y_all = np.append(y_all,y)
            
    # then, split into groups of 100
    stop = np.shape(X_all)[0] - np.remainder(np.shape(X_all)[0],100)
    X_all_mod100 = X_all[0:stop,:]
    num_blocks = int(np.shape(X_all_mod100)[0]/100)
    X_all = np.vsplit(X_all_mod100,num_blocks)
    y_all_mod100 = y_all[0:stop,]
    y_all = np.split(y_all_mod100,num_blocks)
            
    return X_all,y_all,num_blocks

# function to fit, test, generate predicted probabilities
def eval_model(model, X, y, kf):

    weights_all = []
    score = []
    count = 0
    for train_index, test_index in kf.split(X):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # ensure that y_train has at least one of every lever
        for i in range(1,4):
            if i not in y_train:
                y_train[-i]=i
        model.fit(X_train, y_train)

        weights_all.append(model.coef_)
        score.append(model.score(X_test,y_test))

        if count<1:
            yhat_all = model.predict_proba(X_test)
            ytest_all = y_test
        else:
            yhat_all = np.concatenate((yhat_all,model.predict_proba(X_test)))
            ytest_all = np.concatenate((ytest_all,y_test))

        count += 1

    weights = np.mean(weights_all,axis=0)

    return score, weights, yhat_all, ytest_all



####################### define the model ############################

# multinomial regression model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', penalty='l2', fit_intercept = 0, C=0.1, max_iter = 400)

# define the cross-validation split
kf = KFold(n_splits=10)

##################### load and run on the data ####################
#for j7_jasmine(dmslesion 1), no of sessions = 63
#prefix = r'W:\Lab\dms_lesion\updateddata_sep4_22\ALLSESSIONS_J7JASMINE'
#prefix1 = r'W:\Lab\dms_lesion\updateddata_sep4_22\MODELOP_J7JASMINE'


#for j5_joy (dmslesion2), no of sessions = 184
prefix = r'Z:\Lab\dms_lesion\antibiasing_model\data'
prefix1 = r'Z:\Lab\dms_lesion\antibiasing_model\model_output1'

#for t8_truffle, no of sessions = 105
#prefix = r'W:\Lab\dms_lesion\updateddata_sep4_22\ALLSESSIONS_T8TRUFFLE_1'
#prefix1 = r'W:\Lab\dms_lesion\updateddata_sep4_22\MODELOP_T8TRUFFLE1'

#D6_DAHLIA, 5 SESSIONS
#prefix = r'W:\Lab\dms_lesion\updateddata_sep4_22\ALLSESSIONS_D6DAHLIA'
#prefix1 = r'W:\Lab\dms_lesion\updateddata_sep4_22\MODELOP_D6DAHLIA'

files = os.listdir(prefix)
###this should be no of sessions+1
count=9
savedata = 0

score_all_avg = []
score_all_std = []
weights_all = []
yhat_all = []
yhat_cue1=[]
yhat_act1=[]
trial_num_all = []
score_null_avg = []
score_cues_avg = []
score_act_avg = []
score_prevcue_avg = []
score_rewact_avg = []
score_nonrewact_avg = []
score_rewt1_avg = []
score_rewt2_avg = []
score_rewt3_avg = []
score_nrewt3_avg = []
score_nrewt2_avg = []
score_nrewt1_avg = []
score_cues_std=[]
score_act_std=[]
score_ab_avg=[]

## load all of the sessions, concatenate, and then split into sets of 100
X_all,y_all,num_blocks = load_split_sessions(prefix,count)


for n in range(num_blocks):

    X = X_all[n]
    y = y_all[n]

    # fit the model with all data (past history+cue info+bias term)
    score, weights, yhat, ytest = eval_model(model, X, y, kf)
    score_all_avg.append(np.mean(score))
    score_all_std.append(np.std(score))
    weights_all.append(weights)
    yhat_all.append(yhat)

    # fit a model with just current cues (+bias)
    X_cue = X[:,(0,1,2,21)]
    score_cue, weights_cue, yhat_cue, ytest_cue = eval_model(model, X_cue, y, kf)
    score_cues_avg.append(np.mean(score_cue))
    score_cues_std.append(np.std(score_cue))
    yhat_cue1.append(yhat_cue)

    # fit a model with just the previous actions (rew+nonrew from t-3 to t-1) + bias term
    X_act = X[:,3:22] 
    score_act, weights_act, yhat_act, ytest_act = eval_model(model, X_act, y, kf)
    score_act_avg.append(np.mean(score_act))
    score_act_std.append(np.std(score_act))
    yhat_act1.append(yhat_act)
                                 
    # fit a model with just the rewarded previous actions (rew) + bias term
    X_rewact = X[:,(3,4,5,6,7,8,9,10,11,21)]
    score_rewact, weights_rewact, yhat_rewact, ytest_rewact = eval_model(model, X_rewact, y, kf)
    score_rewact_avg.append(np.mean(score_rewact))


    # fit a model with just nonrew prev actions  + bias term
    X_nonrewact = X[:,(12,13,14,15,16,17,18,19,20,21)]
    score_nonrewact, weights_nonrewact, yhat_nonrewact, ytest_nonrewact = eval_model(model, X_nonrewact, y, kf)
    score_nonrewact_avg.append(np.mean(score_nonrewact))


    # fit the null model (just the bias term, to see the influence of her intrinsic biases)
    X_null = np.column_stack((X[:,21],X[:,21])) #-1 coz -1 from reverse direction
    score_null, weights_null, yhat_null, ytest_null  = eval_model(model, X_null, y, kf)
    score_null_avg.append(np.mean(score_null))
    
    # fit a model with just rew prev actions of t-1 + bias term
    X_rewt1 = X[:,(5,8,11,21)]
    score_rewt1, weights_rewt1, yhat_rewt1, ytest_rewt1 = eval_model(model, X_rewt1, y, kf)
    score_rewt1_avg.append(np.mean(score_rewt1))
    
    # fit a model with just rew prev actions of t-2 + bias term
    X_rewt2 = X[:,(4,7,10,21)]
    score_rewt2, weights_rewt2, yhat_rewt2, ytest_rewt2 = eval_model(model, X_rewt2, y, kf)
    score_rewt2_avg.append(np.mean(score_rewt2))
    
    # fit a model with just rew prev actions of t-2 + bias term
    X_rewt3 = X[:,(3,6,9,21)]
    score_rewt3, weights_rewt3, yhat_rewt3, ytest_rewt3 = eval_model(model, X_rewt3, y, kf)
    score_rewt3_avg.append(np.mean(score_rewt3))
    
    # fit a model with just nonrew prev actions of t-1 + bias term
    X_nrewt1 = X[:,(14,17,20,21)]
    score_nrewt1, weights_nrewt1, yhat_nrewt1, ytest_nrewt1 = eval_model(model, X_nrewt1, y, kf)
    score_nrewt1_avg.append(np.mean(score_nrewt1))
    
    # fit a model with just nonrew prev actions of t-2 + bias term
    X_nrewt2 = X[:,(13,16,19,21)]
    score_nrewt2, weights_nrewt2, yhat_nrewt2, ytest_nrewt2 = eval_model(model, X_nrewt2, y, kf)
    score_nrewt2_avg.append(np.mean(score_nrewt2))
    
    # fit a model with just nonrew prev actions of t-2 + bias term
    X_nrewt3 = X[:,(12,15,18,21)]
    score_nrewt3, weights_nrewt3, yhat_nrewt3, ytest_nrewt3 = eval_model(model, X_nrewt3, y, kf)
    score_nrewt3_avg.append(np.mean(score_nrewt3))

# fit a model with ab agents + bias term
    X_ab = X[:,(21,22,23,24)]
    score_ab, weights_ab, yhat_ab, ytest_ab = eval_model(model, X_ab, y, kf)
    score_ab_avg.append(np.mean(score_ab))

    # write out yhat_all
    #if savedata:
    dfy =  pd.DataFrame(yhat)
    yhatpath = os.path.join(prefix1,"model_prediction" + str(n) + ".csv")
    dfy.to_csv(yhatpath)

    dfy =  pd.DataFrame(yhat_cue)
    yhatpath = os.path.join(prefix1,"cuemodel_prediction" + str(n) + ".csv")
    dfy.to_csv(yhatpath)  

    dfy =  pd.DataFrame(yhat_act)
    yhatpath = os.path.join(prefix1,"actmodel_prediction" + str(n) + ".csv")
    dfy.to_csv(yhatpath)

##################### export the rest of the data ####################
#if savedata:
weight_left = []
weight_center = []
weight_right = []
###put no. of sessions count as input in np.arange
for p in np.arange(num_blocks):
     weight_l = weights_all[p][0]
     weight_c = weights_all[p][1]
     weight_r = weights_all[p][2]
     weight_left.append(weight_l)
     weight_center.append(weight_c)
     weight_right.append(weight_r)
     weightsfor_lefttap = np.array(weight_left)
     weightsfor_centtap = np.array(weight_center)
     weightsfor_righttap = np.array(weight_right)

#lets export the arrays
df = pd.DataFrame(weightsfor_lefttap)
df1 = os.path.join(prefix1,"lefttapweights"+".csv")
df.to_csv(df1)
df2 =  pd.DataFrame(weightsfor_centtap)
df3 = os.path.join(prefix1,"centtapweights"+".csv")
df2.to_csv(df3)
df4 =  pd.DataFrame(weightsfor_righttap)
df5 = os.path.join(prefix1,"righttapweights"+".csv")
df4.to_csv(df5)

 # export the score
df6 =  pd.DataFrame(score_all_avg)
df7 =  os.path.join(prefix1,"score_full"+".csv")
df6.to_csv(df7)

 # export the score
df8 =  pd.DataFrame(score_cues_avg)
df9 =  os.path.join(prefix1,"score_cue"+".csv")
df8.to_csv(df9)

# export the score
df10 =  pd.DataFrame(score_act_avg)
df11 = os.path.join(prefix1,"score_act"+".csv")
df10.to_csv(df11)

 # export the score
df12 =  pd.DataFrame(score_null_avg)
df13 = os.path.join(prefix1,"score_null"+".csv")
df12.to_csv(df13)


 # export the score
df16 =  pd.DataFrame(score_rewact_avg)
df17 = os.path.join(prefix1,"score_rewardedactions"+".csv")
df16.to_csv(df17)

 # export the score
df18 =  pd.DataFrame(score_nonrewact_avg)
df19 = os.path.join(prefix1,"score_nonrewardedactions"+".csv")
df18.to_csv(df19)

# export the score
df20 =  pd.DataFrame(score_rewt1_avg)
df21 = os.path.join(prefix1,"score_rewt_1"+".csv")
df20.to_csv(df21)

# export the score
df22 =  pd.DataFrame(score_rewt2_avg)
df23 = os.path.join(prefix1,"score_rewt_2"+".csv")
df22.to_csv(df23)

# export the score
df24 =  pd.DataFrame(score_rewt3_avg)
df25 = os.path.join(prefix1,"score_rewt_3"+".csv")
df24.to_csv(df25)

# export the score
df26 =  pd.DataFrame(score_nrewt1_avg)
df27 = os.path.join(prefix1,"score_nonrewt_1"+".csv")
df26.to_csv(df27)

# export the score
df28 =  pd.DataFrame(score_nrewt2_avg)
df29 = os.path.join(prefix1,"score_nonrewt_2"+".csv")
df28.to_csv(df29)


# export the score
df30 =  pd.DataFrame(score_nrewt3_avg)
df31 = os.path.join(prefix1,"score_nonrewt_3"+".csv")
df30.to_csv(df31)
 
# export the score
df32 =  pd.DataFrame(score_all_std)
df33 =  os.path.join(prefix1,"score_std"+".csv")
df32.to_csv(df33)

# export the score
df34 =  pd.DataFrame(score_cues_std)
df35 =  os.path.join(prefix1,"score_cue_std"+".csv")
df34.to_csv(df35)


# export the score
df36 =  pd.DataFrame(score_act_std)
df37 =  os.path.join(prefix1,"score_actstd"+".csv")
df36.to_csv(df37)

# export the score
df38 =  pd.DataFrame(score_ab_avg)
df39 =  os.path.join(prefix1,"score_ab_avg"+".csv")
df38.to_csv(df39)



