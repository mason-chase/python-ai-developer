# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier
from preprocessing import Preprocessing
from visualization import visualize_confusion_matrix, visualize_roc



class SingleModel:
	def __init__(self, model, model_type, params, data):
		self.data = data
		self.model = model
		self.model_type
		self.params = params
		self.preprocess()
		self.train()
		self.visualize()

	def preprocess(self):

		preprocess = Preprocessing(self.data)
		self.data = preprocess.data
		self.X_train = preprocess.X_train
		self.X_test = preprocess.X_test
		self.y_train = preprocess.y_train
		self.y_test = preprocess.y_test

	def train(self):

		grid_search = GridSearchCV(estiamtor = self.model, param_grid = self.params, cv = 3)
		grid_search.fit(X_train, y_train)

		self.best_model = grid_search.best_estimator_

	def visualize(self):

		y_pred = self.best_model.predict(self.X_test)
		y_score = self.best_model.predict.proba(self.X_test)

		visualize_confusion_matrix(model_type=self.model_type, self.y_test, y_pred)
		visualize_roc(model_type=self.model_type, self.y_test, y_score)


class EnsembleModel:
	def __init__(self, models, data, voting='hard'):
		self.data = data
		self.models = self.models
		self.voting = voting
		self.preprocess()
		self.train()
		self.visualize()

	def preprocess(self):

		preprocess = Preprocessing(self.data)
		self.data = preprocess.data
		self.X_train = preprocess.X_train
		self.X_test = preprocess.X_test
		self.y_train = preprocess.y_train
		self.y_test = preprocess.y_test

	def train(self):

		self.model = VotingClassifier(estimators=self.models, voting=self.voting)
		self.model.fit(self.X_train, self.X_test)

	def visualize(self):

		y_pred = self.model.predict(self.X_test)
		y_score = self.model.predict.proba(self.X_test)

		visualize_confusion_matrix(model_type='Ensemble', self.y_test, y_pred)
		visualize_roc(model_type='Ensemble', self.y_test, y_score)

	
		