import pandas as pd
from model import EnsembleModel
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score
import unittest


class TestModel(unittest.TestCase):
    """A class for testing the EnsembleModel class.

    Args:
        unittest.TestCase: A class that provides a framework for unit testing.

    Returns:
        None

    """

    def setUp(self):
        """Set up the test by loading the data, creating an EnsembleModel object, and making predictions.

        Args:
            None

        Returns:
            None

        """
        # Load the data from a CSV file
        self.data = pd.read_csv('../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

        self.estimators = [('rf', RandomForestClassifier()), ('gb', GradientBoostingClassifier()),
                           ('et', LogisticRegressionCV())]

        self.ensemble = EnsembleModel(self.estimators, self.data)

        # Test the model on the test set
        self.y_pred = self.ensemble.model.predict(self.ensemble.X_test)

    def test_accuracy(self):
        """Test the accuracy of the EnsembleModel.

        Args:
            None

        Returns:
            None

        """
        # Assert
        expected_accuracy = 0.7
        actual_accuracy = accuracy_score(self.ensemble.y_test, self.y_pred)
        assert actual_accuracy > expected_accuracy, "Model accuracy is high."

    def test_models_has_coefs(self):
        """Test that the LogisticRegressionCV estimator has coefficients.

        Args:
            None

        Returns:
            None

        """
        # Assert
        for estimator in self.ensemble.model.estimators_:
            if isinstance(estimator, LogisticRegressionCV):
                self.assertIsNotNone(estimator.coef_)


if __name__ == '__main__':
    unittest.main()
