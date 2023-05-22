# Import necessary libraries
import numpy as np
import pandas as pd
from models import EnsembleModel

if __name__ == '__main__':
    data = pd.read_csv('../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    estimators = [('rf', RandomForestClassifier()), ('gb', GradientBoostingClassifier()),
                ('et', LogisticRegressionCV())]

    model = EnsembleModel(estimators, data)
    