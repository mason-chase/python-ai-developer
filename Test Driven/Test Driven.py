import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import unittest

def load_data(file_path):
    # Load the data from a CSV file
    data = pd.read_csv(file_path)
    
    # Split the data into features and labels
    X = data.drop('target', axis=1)
    y = data['target']
    
    return X, y

def preprocess_data(X_train, X_test):
    # Preprocess the data by scaling the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test

def train_model(X_train, y_train):
    # Train the logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    return model

def test_model(model, X_test, y_test):
    # Test the model on the test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy

class TestModel(unittest.TestCase):

    def setUp(self):
        # Load the data
        self.X, self.y = load_data('data.csv')
        
        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        
        # Preprocess the data
        self.X_train, self.X_test = preprocess_data(self.X_train, self.X_test)
        
        # Train the model
        self.model = train_model(self.X_train, self.y_train)
        
    def test_accuracy(self):
        # Test the accuracy of the model
        expected_accuracy = 0.9
        actual_accuracy = test_model(self.model, self.X_test, self.y_test)
        self.assertAlmostEqual(actual_accuracy, expected_accuracy, delta=0.05)

    def test_model_has_coefs(self):
        # Assert that the model has coefficients
        self.assertIsNotNone(self.model.coef_)

if __name__ == '__main__':
    unittest.main()