import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


class Preprocessing:
    def __init__(self, data):
        """Initialize a Preprocessing object.

        Args:
            data (DataFrame): The raw data to be preprocessed.

        Returns:
            None

        """
        self.data = data
        self.data_cleaning()
        self.preprocessing()
        self.data_splitting()

    def data_cleaning(self):
        """Clean the data by dropping unnecessary columns and converting TotalCharges to float.

        Args:
            None

        Returns:
            None

        """
        to_drop = ['customerID']

        self.data = self.data.drop(to_drop, axis=1)

        self.data['TotalCharges'] = self.data['TotalCharges'].replace('', None)
        self.data['TotalCharges'] = self.data['TotalCharges'].replace(' ', None)

        self.data['TotalCharges'] = self.data['TotalCharges'].astype(float)
        self.data = self.data.dropna()
        self.data = self.data.reset_index().drop('index', axis=1)

    def preprocessing(self):
        """Preprocess the data by factorizing binary columns and one-hot-encoding categorical columns.

        Args:
            None

        Returns:
            None

        """
        for column in self.data.columns:
            if len(np.unique(self.data[column])) == 2:
                self.data[column] = pd.factorize(self.data[column])[0]

        to_be_onehot = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                        'Contract', 'PaymentMethod']

        for column in to_be_onehot:
            # create an instance of OneHotEncoder
            ohe = OneHotEncoder()

            # use fit_transform() to convert the string variable to one-hot-encoded data
            data_encoded = ohe.fit_transform(self.data[[column]]).toarray()

            # create a new DataFrame with the one-hot-encoded data
            data_onehot = pd.DataFrame(data_encoded, columns=ohe.get_feature_names_out([column]))

            self.data = pd.concat([self.data, data_onehot], axis=1)
            self.data = self.data.drop(column, axis=1)

    def data_splitting(self, test_size=0.3):
        """Split the preprocessed data into training and testing sets.

        Args:
            test_size (float): The proportion of the data to be used for testing.

        Returns:
            None

        """
        X = self.data.drop('Churn', axis=1).values.astype(float)
        y = self.data['Churn'].values

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, stratify=y)
