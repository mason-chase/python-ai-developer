import pandas as pd
from PreprocessData import remove_missing_values, convert_data_types
from EncodDataset import preprocess_dataset
from sklearn.preprocessing import StandardScaler
#read-data
Dataset=pd.read_csv("/content/drive/MyDrive/AI_Developer/WA_Fn-UseC_-Telco-Customer-Churn.csv")
Dataset.drop('customerID', axis=1, inplace=True)

# clean-data
cleaned_dataset = remove_missing_values(Dataset)
cleaned_dataset = convert_data_types(cleaned_dataset)

#Encode data
Encodedata = preprocess_dataset(cleaned_dataset)
std_scaler = StandardScaler()
X = Encodedata.drop("Churn", axis=1)
X = std_scaler.fit_transform(X)
Y = Encodedata["Churn"].copy()