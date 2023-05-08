import pandas as pd
import numpy as np
import plotly.express as px  
import plotly.io as pio
import numpy as np
from scipy.stats import skew
import os
import pickle
import joblib

from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier



import pytest
import os
from sys import path
path.append(os.path.join(os.getcwd(), '')) 
from function import * 

import warnings
warnings.filterwarnings("ignore")


import logging
report_name = "log_test_train_model" 
logging.basicConfig(filename='logs/{}'.format(report_name),
                     format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)



#---------------------------------------------------------------------------------------------------------

def test_check_overfiting():
        dir_read_data = "./data"
        dir_read_model = "./results"

        x_train = pd.read_csv(dir_read_data + "/x_train.csv")
        x_train.set_index("customerID", inplace = True)

        y_train= pd.read_csv(dir_read_data + "/y_train.csv")


        loaded_model = joblib.load(dir_read_model + "/finalized_model.sav")

        y_train_pred = loaded_model.fit(x_train,y_train).predict(x_train)

        assert np.array_equal(y_train_pred,y_train["Churn"].to_numpy()) == False , "model are overfite"


        print("test for checking overfite pass!")

#---------------------------------------------------------------------------------------------------------
def test_model_evaluation():
    dir_read_data = "./data"
    dir_read_model = "./results"

    x_train = pd.read_csv(dir_read_data + "/x_train.csv")
    x_train.set_index("customerID", inplace = True)

    y_train= pd.read_csv(dir_read_data + "/y_train.csv")


    loaded_model = joblib.load(dir_read_model + "/finalized_model.sav")

    loaded_model.fit(x_train,y_train).predict(x_train)

    x_train, x_test, x_val, y_train, y_val, y_test  = read_data()


    pred_test = loaded_model.predict(x_test)
    pred_test_binary = np.round(pred_test)
    acc_test = accuracy_score(y_test, pred_test_binary)
    auc_test = roc_auc_score(y_test, pred_test)

    assert acc_test > 0.75, 'Accuracy on test should be > 0.75'
    assert auc_test > 0.70, 'AUC ROC on test should be > 0.70'

    print("test for model_evaluation pass!")





if __name__ == "__main__":
   x =  pytest.main(args=['-sv', os.path.abspath(__file__)])
   logger.info(x)