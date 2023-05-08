import pandas as pd
import numpy as np
from scipy.stats import skew
import os
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")




"""Read data from resource"""
dir_read_data = "./data"
filename_csv =  "/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(dir_read_data  + filename_csv)

print("General information about data")
print("shape pf data:","\n" ,df.shape)
print("shape of data after drop duplicates: ","\n" ,df.drop_duplicates().shape)
print("columns of dataframe:","\n" , df.columns.tolist())




"""function for clenaing data, fixed data tuype, """
df["TotalCharges"].replace({' '}, None, inplace = True)
df["TotalCharges"] = df["TotalCharges"].astype(float)
df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)
df["SeniorCitizen"].replace({'0'}, 'No', inplace = True)
df["SeniorCitizen"].replace({'1'}, 'Yes', inplace = True)



print("numerical columns:")
print("tenure min, mean, max, std:" ,df["tenure"].min(),df["tenure"].mean() ,df["tenure"].max(), df["tenure"].std(), df[df["tenure"].isna()].shape)
print("MonthlyCharges min, mean, max, std:" ,df["MonthlyCharges"].min(),df["MonthlyCharges"].mean() ,df["MonthlyCharges"].max(), df["MonthlyCharges"].std(), df[df["MonthlyCharges"].isna()].shape)
print("TotalCharges min, mean, max, std:" ,df["TotalCharges"].min(),df["TotalCharges"].mean() ,df["TotalCharges"].max(), df["TotalCharges"].std(), df[df["TotalCharges"].isna()].shape)
print("\n")

target_col = ["Churn"]
numerical_col = ["tenure", "MonthlyCharges", "TotalCharges"]
categorical_val = [x for x in df.columns.tolist() if x not in target_col and x not in numerical_col and x != 'customerID']

print("categorical columns:")
for item in categorical_val:
    if item != 'customerID':
        print(item,"-",  len(df[item].unique().tolist()),  df[item].unique().tolist(), df[df[item].isna()].shape)
print("\n")
print("percent of No in Churn:", 100 *  df[df["Churn"]== "No"].shape[0]/df.shape[0])
print("percent of Yes in Churn:", 100 *  df[df["Churn"]== "Yes"].shape[0]/df.shape[0])





"""check null in numerical variable"""
for item in numerical_col:
    num_null =  df[df[item].isna()].shape[0]
    if num_null != 0:

        # approch one: according to distrubiutiona of data use mean or median to fill nan
        threshould = 0.5
        skew_column = skew(df[item].dropna())

        if skew_column > threshould or skew_column < threshould:
            filnan_val = df[item].dropna().median()
        else:
            filnan_val = df[item].dropna().mean()

        df[item].fillna(filnan_val, inplace = True)


        # approach two:create a model with other corrolated feature to predict nan value
        # corr_ = df[["MonthlyCharges", "TotalCharges","SeniorCitizen", "tenure"] ].corr()
        # sns.heatmap(corr_, annot=True)






"""Save the results"""
df.to_csv(dir_read_data + "/edited_df.csv")

with open(dir_read_data + "/numerical_col.txt", 'wb') as f:
    pickle.dump(numerical_col, f)

with open(dir_read_data + "/target_col.txt", 'wb') as f:
    pickle.dump(target_col,f)

with open(dir_read_data + "/categorical_val.txt", 'wb') as f:
    pickle.dump(categorical_val, f)




"""chne categirical value to numerical with one hot encoding"""
enc = OneHotEncoder( sparse=False).fit(df[categorical_val])
encoded = enc.transform(df[categorical_val])
onehote_encode_col = enc.get_feature_names_out()
encoded_df = pd.DataFrame( encoded,columns=enc.get_feature_names_out() )
encoded_df[numerical_col] = df[numerical_col]
encoded_df["customerID"] = df["customerID"]
encoded_df[target_col] = df[target_col]
encoded_df["Churn"].replace({"No"},0, inplace = True)
encoded_df["Churn"].replace({"Yes"},1, inplace = True)
encoded_df.set_index("customerID", inplace = True)

encoded_df.to_csv(dir_read_data + '/encode_df.csv')
with open(dir_read_data + "/onehote_encode_col.txt", 'wb') as f:
    pickle.dump(onehote_encode_col, f)



"""splite data to train, test, validation"""
data = encoded_df.reset_index()

x_train, x_test, y_train, y_test = train_test_split(data.drop(columns = target_col), data[target_col], test_size=0.33, random_state=42)
x_val, x_test, y_val, y_test  = train_test_split(x_test, y_test, test_size=0.4, random_state=42)

print("x_train shape", x_train.shape)
print("x_val shape", x_val.shape)
print("x_test shape",x_test.shape)

x_train.to_csv(dir_read_data + "/x_train.csv",index=False)
x_test.to_csv(dir_read_data + "/x_test.csv",index=False)
x_val.to_csv(dir_read_data + "/x_val.csv",index=False)

y_train.to_csv(dir_read_data + "/y_train.csv",index=False)
y_val.to_csv(dir_read_data + "/y_val.csv",index=False)
y_test.to_csv(dir_read_data + "/y_test.csv",index=False)
