import pandas as pd
import numpy as np
import math
import seaborn as sns
from matplotlib import pyplot as plt
import plotly.express as px  
import plotly.io as pio
import numpy as np
import plotly.graph_objects as go
import os

import warnings
warnings.filterwarnings("ignore")

import os
from sys import path
path.append(os.path.join(os.getcwd(), '')) 
from function import * 


from sklearn.metrics import accuracy_score, confusion_matrix, RocCurveDisplay
from sklearn.metrics import classification_report, auc, roc_curve
from sklearn.metrics import  f1_score, precision_score,recall_score, roc_auc_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression , RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

import pickle
import joblib


dir_read_data = "./data"




"""Read splited data for trian , test and validation set"""
x_train, x_test, x_val, y_train, y_val, y_test = read_data()





"""define some model for classification"""

model_pipline = []
#1
model_pipline.append(LogisticRegression())
#2
model_pipline.append(SVC())
#3
model_pipline.append(KNeighborsClassifier())
#4
model_pipline.append(GaussianNB())
#5
model_pipline.append(DecisionTreeClassifier(max_depth=None, min_samples_split=2,random_state=0))
#6
model_pipline.append(RandomForestClassifier(n_estimators=10, max_depth=None,   min_samples_split=2, random_state=0))
#7
model_pipline.append(ExtraTreesClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0))
#8
model_pipline.append(AdaBoostClassifier(n_estimators=100))
#9
model_pipline.append(GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0))
#10
model_pipline.append(RidgeClassifier())



model_list = ['LogisticRegression',
             ' SVC',
               'KNeighborsClassifier',
               'GaussianNB',
               'DecisionTreeClassifier',
               'RandomForestClassifier',
               'ExtraTreesClassifier',
               'AdaBoostClassifier',
               'GradientBoostingClassifier',
               'RidgeClassifier',
               ]

acc_train_list = []
acc_val_list = []

auc_train_list = []
auc_val_list = []

cm_train_list = []
cm_val_list = []

cros_val_ = []


for model in model_pipline:

    model.fit(x_train, y_train)
    y_pre_train = model.predict(x_train)
    y_pre_val = model.predict(x_val)

    acc_train_list.append(accuracy_score(y_pre_train,y_train, normalize = True ))
    acc_val_list.append(accuracy_score(y_val,y_pre_val, normalize = True ))

    fpr_train, tpr_train, _threshould_train = roc_curve(y_train,y_pre_train)
    fpr_val, tpr_val, _threshould_val = roc_curve(y_val,y_pre_val)


    auc_train_list.append(np.round(auc(fpr_train, tpr_train),2))
    auc_val_list.append(np.round(auc(fpr_val, tpr_val),2))

    cm_train_list.append(confusion_matrix(y_train,y_pre_train))
    cm_val_list.append(confusion_matrix(y_val,y_pre_val))



result_df = pd.DataFrame({
    'Model': model_list, 
    'Accurency_train': acc_train_list,
    'Accurency_val': acc_val_list ,
    'AUC_train': auc_train_list,
    'AUC_val':auc_val_list
    })


dir_save_model = "./results"

result_df.to_csv(dir_save_model + "/result_of_models.csv")


model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
model.fit(x_train, y_train)
filename = '/finalized_model.sav'
pickle.dump(model, open(dir_save_model + filename, 'wb'))



fig = plt.figure(figsize = (18, 10))

for i in range(len(cm_train_list)):
    cm = cm_val_list[i]
    model = model_list[i]
    sub =  fig.add_subplot(5,2,i+1).set_title(model)
    cm_plot = sns.heatmap(cm, annot = True, cmap='Blues_r')
    cm_plot.set_xlabel("prediction")
    cm_plot.set_ylabel("actual")

plt.savefig("./figures/confusion_matrix_of_all_model_on_val_set.png")