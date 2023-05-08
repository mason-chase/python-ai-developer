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



model = RidgeClassifier()
model.fit(x_train, y_train)
y_pre_test = model.predict(x_test)

acc_ = accuracy_score(y_pre_test,y_test, normalize = True )
fpr_, tpr_, _threshould_train = roc_curve(y_pre_test,y_test)
auc_ = np.round(auc(fpr_, tpr_),2)
cm = confusion_matrix(y_pre_test,y_test)

print("acc :", acc_)
print("auc :", auc_)



sns.heatmap(cm, annot = True, cmap='Blues_r')
plt.savefig("./results/confusion_matrix_of_final_model_on_test_set.png")

print(classification_report(y_test, y_pre_test, target_names= ['class 0', 'class 1']))