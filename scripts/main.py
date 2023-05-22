# Import necessary libraries
import pandas as pd
from model import EnsembleModel
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegressionCV

if __name__ == '__main__':
    # Load the data
    data = pd.read_csv('../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    # Train EnsembleModel
    estimators = [('rf', RandomForestClassifier()), ('gb', GradientBoostingClassifier()),
                ('et', LogisticRegressionCV())]

    model = EnsembleModel(estimators, data)
    
