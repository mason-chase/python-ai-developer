# Import necessary libraries
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier
from preprocessing import Preprocessing
from visualization import visualize_confusion_matrix, visualize_roc


class SingleModel:
    def __init__(self, model, model_type, params, data):
        """Initialize a SingleModel object.

        Args:
            model (estimator): The machine learning model to be trained.
            model_type (str): The type of model being trained.
            params (dict): The hyperparameters to be tuned using GridSearchCV.
            data (DataFrame): The preprocessed data to be used for training and testing.

        Returns:
            None

        """
        self.data = data
        self.model = model
        self.model_type = model_type
        self.params = params
        self.preprocess()
        self.train()
        self.visualize()

    def preprocess(self):
        """Preprocess the data using the Preprocessing class.

        Args:
            None

        Returns:
            None

        """
        preprocess = Preprocessing(self.data)
        self.data = preprocess.data
        self.X_train = preprocess.X_train
        self.X_test = preprocess.X_test
        self.y_train = preprocess.y_train
        self.y_test = preprocess.y_test

    def train(self):
        """Train the model using GridSearchCV to tune hyperparameters.

        Args:
            None

        Returns:
            None

        """
        grid_search = GridSearchCV(estimator=self.model, param_grid=self.params, cv=3)
        grid_search.fit(self.X_train, self.y_train)

        self.best_model = grid_search.best_estimator_

    def visualize(self):
        """Visualize the performance of the best model using confusion matrix and ROC curve plots.

        Args:
            None

        Returns:
            None

        """
        y_pred = self.best_model.predict(self.X_test)
        y_score = self.best_model.predict_proba(self.X_test)

        visualize_confusion_matrix(model_type=self.model_type, y_test=self.y_test, y_pred=y_pred)
        visualize_roc(model_type=self.model_type, y_test=self.y_test, y_score=y_score)


class EnsembleModel:
    def __init__(self, models, data, voting='soft'):
        """Initialize an EnsembleModel object.

        Args:
            models (list): A list of tuples containing the name and estimator of each model to be included in the ensemble.
            data (DataFrame): The preprocessed data to be used for training and testing.
            voting (str): The type of voting to be used by the VotingClassifier.

        Returns:
            None

        """
        self.data = data
        self.models = models
        self.voting = voting
        self.preprocess()
        self.train()
        self.visualize()

    def preprocess(self):
        """Preprocess the data using the Preprocessing class.

        Args:
            None

        Returns:
            None

        """
        preprocess = Preprocessing(self.data)
        self.data = preprocess.data
        self.X_train = preprocess.X_train
        self.X_test = preprocess.X_test
        self.y_train = preprocess.y_train
        self.y_test = preprocess.y_test

    def train(self):
        """Train the ensemble model using the VotingClassifier.

        Args:
            None

        Returns:
            None

        """
        self.model = VotingClassifier(estimators=self.models, voting=self.voting)
        self.model.fit(self.X_train, self.y_train)

    def visualize(self):
        """Visualize the performance of the ensemble model using confusion matrix and ROC curve plots.

        Args:
            None

        Returns:
            None

        """
        y_pred = self.model.predict(self.X_test)
        y_score = self.model.predict_proba(self.X_test)

        visualize_confusion_matrix(model_type='Ensemble', y_test=self.y_test, y_pred=y_pred)
        visualize_roc(model_type='Ensemble', y_test=self.y_test, y_score=y_score)
