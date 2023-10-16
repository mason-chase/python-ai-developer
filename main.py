import pandas as pd
from PreprocessData import remove_missing_values, convert_data_types
from EncodDataset import preprocess_dataset
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier#
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier#
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
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

# models
models = [
    ('Linear Classifier', SGDClassifier(random_state=182)),
    ('Random Forest', RandomForestClassifier(random_state=1812)),
    ('Gradient Boost', GradientBoostingClassifier(n_estimators=100, learning_rate=0.01, max_depth=3, random_state=42)),
    ('XGBoost', XGBClassifier(random_state=1812)),
    ('KNN', KNeighborsClassifier(n_neighbors=5)),
    ('Decision Tree', DecisionTreeClassifier(random_state=1812)),
    ('Bagging Classifier', BaggingClassifier(n_estimators=150, random_state=1812)),
    ("Logistic Regression", LogisticRegression())
]