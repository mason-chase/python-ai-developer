import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
def preprocess_dataset(dataset):
    # Encoding Binary categorical columns with LabelEncoder
    binary_cat_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    label_encoder = LabelEncoder()
    for column in binary_cat_columns:
        dataset[column] = label_encoder.fit_transform(dataset[column])
    
    # Binning numerical columns using pd.cut
    dataset["Tenure_Cat"] = pd.cut(dataset["tenure"],
                                   bins=[0, 10, 20, 30, 40, 50, 60, np.inf],
                                   labels=['0 -> 10', '10 -> 20', '20 -> 30', '30 -> 40', '40 -> 50', '50 -> 60', '60 -> 72'])
    
    dataset["TotalCharges_Cat"] = pd.cut(dataset["TotalCharges"],
                                         bins=[0, 500, 1000, 2500, 4000, 5500, 7000, np.inf],
                                         labels=['0 -> 500', '500 -> 1000', '1000 -> 2500', '2500 -> 4000', '4000 -> 5500', '5500 -> 7000', '>7000'])
    
    # Copy dataset and drop unnecessary columns
    data_tel = dataset.copy()
    data_tel.drop(['Tenure_Cat', 'TotalCharges_Cat'], axis=1, inplace=True)
    
    # Perform one-hot encoding using pd.get_dummies
    df_dummies = pd.get_dummies(data_tel, dtype=int)
    
    return df_dummies