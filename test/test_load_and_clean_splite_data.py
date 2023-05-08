import pandas as pd
import numpy as np
import plotly.express as px  
import plotly.io as pio
import numpy as np
from scipy.stats import skew
import os
import pickle


import pytest
import os
from sys import path
path.append(os.path.join(os.getcwd(), '')) 
# from function import * 

import warnings
warnings.filterwarnings("ignore")

import os
from sys import path
path.append(os.path.join(os.getcwd(), '')) 
from function import * 



import logging
report_name = "log_test_load_and_clean_data" 
logging.basicConfig(filename='logs/{}'.format(report_name),
                     format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)

#---------------------------------------------------------------------------------------------------------

def test_load_data():
    """test correctness of reading csv file as our data"""

    dir_read_data = "./data"
    filename_csv = "/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = pd.read_csv(dir_read_data +filename_csv )
    numeric_data = df.select_dtypes(include=[np.number])
    categorical_data = df.select_dtypes(exclude=[np.number])


    assert  df.shape[1] == 21 , "Result does not match expected value"
    assert  df.shape[0] == 7043  , "Result does not match expected value"
    assert  df.drop_duplicates().shape[0] == 7043 ,"Result does not match expected value"
    assert  df.columns.tolist() == [
        'customerID', 'gender', 'SeniorCitizen', 'Partner', 
        'Dependents', 'tenure', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 
        'DeviceProtection', 'TechSupport', 'StreamingTV', 
        'StreamingMovies', 'Contract', 'PaperlessBilling',
        'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn'
        ],"Result does not match expected value"
    
    assert  numeric_data.columns.tolist() == [
        'SeniorCitizen', 'tenure', 'MonthlyCharges'
        ], "Result does not match expected value"
    assert categorical_data.columns.tolist() == [
        'customerID', 'gender', 'Partner', 'Dependents', 
        'PhoneService', 'MultipleLines', 'InternetService', 
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 
        'Contract', 'PaperlessBilling', 'PaymentMethod', 
        'TotalCharges', 'Churn'
        ] ,"Result does not match expected value"
    

    print("all test for loading data pass!", "\n")

#---------------------------------------------------------------------------------------------------------

def test_data_cleaning():
    """testing cleaning function"""
    dir_read_data = "./data"
    filename_csv = "/edited_df.csv"
    df = pd.read_csv(dir_read_data + filename_csv)

    assert df["TotalCharges"].dtypes == float , "TotalCharges's data type does not match data type"
    assert df["SeniorCitizen"].dtypes == 'object', "SeniorCitizen's data type does not match data type"
    assert df["tenure"].dtypes == int ,"Result does not match expected value"
    assert df["MonthlyCharges"].dtypes == float ,"Result does not match expected value"
    # assert df["TotalCharges"].dtypes == np.number ,"Result does not match expected value"

    assert df["gender"].unique().tolist() == ['Female', 'Male'],"Result does not match expected value"
    assert df["SeniorCitizen"].unique().tolist() == ['No', 'Yes'],"Result does not match expected value"
    assert df["Partner"].unique().tolist() == ['Yes', 'No'],"Result does not match expected value"
    assert df["Dependents"].unique().tolist() == ['No', 'Yes'],"Result does not match expected value"
    assert df["PhoneService"].unique().tolist() == ['No', 'Yes'],"Result does not match expected value"
    assert df["MultipleLines"].unique().tolist() == ['No phone service', 'No', 'Yes'],"Result does not match expected value"
    assert df["InternetService"].unique().tolist() == ['DSL', 'Fiber optic', 'No'],"Result does not match expected value"
    assert df["OnlineSecurity"].unique().tolist() == ['No', 'Yes', 'No internet service'],"Result does not match expected value"
    assert df["OnlineBackup"].unique().tolist() == ['Yes', 'No', 'No internet service'],"Result does not match expected value"
    assert df["DeviceProtection"].unique().tolist() == ['No', 'Yes', 'No internet service'],"Result does not match expected value"
    assert df["TechSupport"].unique().tolist() == ['No', 'Yes', 'No internet service'],"Result does not match expected value"
    assert df["StreamingTV"].unique().tolist() == ['No', 'Yes', 'No internet service'],"Result does not match expected value"
    assert df["StreamingMovies"].unique().tolist() == ['No', 'Yes', 'No internet service'],"Result does not match expected value"
    assert df["Contract"].unique().tolist() == ['Month-to-month', 'One year', 'Two year'],"Result does not match expected value"
    assert df["PaperlessBilling"].unique().tolist() == ['Yes', 'No'],"Result does not match expected value"
    assert df["PaymentMethod"].unique().tolist() == [
        'Electronic check', 'Mailed check', 
        'Bank transfer (automatic)', 'Credit card (automatic)'
        ],"Result does not match expected value"

    assert df["Churn"].unique().tolist() == ['No', 'Yes'],"Result does not match expected value"

    assert df["TotalCharges"].isna().sum() == 0,"Result does not match expected value"
    assert df["tenure"].isna().sum() == 0,"Result does not match expected value"
    assert df["MonthlyCharges"].isna().sum() == 0,"Result does not match expected value"

    print("all test for cleaning data pass!", "\n")

#---------------------------------------------------------------------------------------------------------

def test_onhot_encoding():
    """test correctness of onhot encoding function"""

    dir_read_data = "./data"
    filename_csv = '/encode_df.csv'
    df = pd.read_csv(dir_read_data + filename_csv )
    with open(dir_read_data + '/target_col.txt', 'rb') as f:
        target_col = pickle.load(f)
    with open(dir_read_data + '/onehote_encode_col.txt', 'rb') as f:
        onhote_encode_col = pickle.load(f)

    check_val = []
    for i in onhote_encode_col:
        x = df[i].unique().tolist()
        check_val = check_val + x

    check_val = list(set(check_val))

    assert df.shape[1] == 48 ,"Result does not match expected value"
    assert df[target_col[0]].unique().tolist() == [0 , 1 ],"Result does not match expected value"
    assert check_val ==  [0 , 1 ],"Result does not match expected value"

    print("all test for encoding categorical values pass!", "\n")

#---------------------------------------------------------------------------------------------------------

def test_split_data_for_model():
    x_train, x_test, x_val, y_train, y_val, y_test = read_data()


    assert x_train.shape == (4718, 46) ,"the values is wrong"
    assert x_val.shape == (1395, 46) ,"the values is wrong"
    assert x_test.shape == (930, 46) ,"the values is wrong"

    assert y_train.shape == (4718,1) ,"the values is wrong"
    assert y_val.shape == (1395,1) ,"the values is wrong"
    assert y_test.shape == (930,1) ,"the values is wrong"

    print("test train, val, test size pass!")
#---------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
   x =  pytest.main(args=['-sv', os.path.abspath(__file__)])
   logger.info(x)