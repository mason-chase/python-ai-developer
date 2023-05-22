import numpy as np
import pandas as pd
from models import EnsembleModel
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score
import unittest

class TestModel(unittest.TestCase):

    def setUp(self):
        # Load the data from a CSV file
        self.data = pd.read_csv('../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
        
        self.estimators = [('rf', RandomForestClassifier()), ('gb', GradientBoostingClassifier()),
                    ('et', LogisticRegressionCV())]

        self.model = EnsembleModel(self.estimators, self.data)
        
        # Test the model on the test set
        self.y_pred = self.model.predict(self.model.X_test)
        
    def test_accuracy(self):
        # Assert
        expected_accuracy = 0.7
        actual_accuracy = accuracy_score(self.model.y_test, self.y_pred)
        assert acc > expected_accuracy, "Model accuracy is high."

    def test_model_has_coefs(self):
        # Assert
        self.assertIsNotNone(self.model.model.coef_)

if __name__ == '__main__':
    unittest.main()