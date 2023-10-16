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
from sklearn.metrics import r2_score
import joblib
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from IPython.display import display
import os 

#read-data
Dataset=pd.read_csv("./Dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv")
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
    ('Linear Classifier', SGDClassifier(loss='log',random_state=182)),
    ('Random Forest', RandomForestClassifier(random_state=1812)),
    ('Gradient Boost', GradientBoostingClassifier(n_estimators=100, learning_rate=0.01, max_depth=3, random_state=42)),
    ('XGBoost', XGBClassifier(random_state=1812)),
    ('KNN', KNeighborsClassifier(n_neighbors=5)),
    ('Decision Tree', DecisionTreeClassifier(random_state=1812)),
    ('Bagging Classifier', BaggingClassifier(n_estimators=150, random_state=1812)),
    ("Logistic Regression", LogisticRegression())
]


def find_best_models(X, Y, models):
    # Split the data into training, validation, and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2,random_state=42)

    best_models = {}  # Dictionary to store the best models

    for name, model in models:
        best_accuracy = 0  # Variable to track the best accuracy
        best_model = None  # Variable to store the best model

        for _ in range(10):
            model.fit(X_train, y_train)
            y_pred_val = model.predict(X_val)
            accuracy = model.score(X_val, y_val)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model

        # Save the best model
        best_models[name] = best_model
        joblib.dump(best_model, f'{name}_best_model.pkl')

        # Test the best model
        y_pred_test = best_model.predict(X_test)
        accuracy_test = best_model.score(X_test, y_test)
        R2_score_test = r2_score(y_test, y_pred_test)
        cm_test = confusion_matrix(y_test, y_pred_test)

        print(f'Best {name} Model:')
        print(f'Validation Accuracy: {best_accuracy}')
        print(f'Test Accuracy: {accuracy_test}')
        print(f'R2 Score: {R2_score_test}')
        print('Confusion Matrix:')
        print(cm_test)
        print('----------------')

    # Plot and save the ROC curve of the best model
        if model == best_model:
            plot_roc_curve(best_model, X_train, y_train, X_test, y_test, name)
    # Create a DataFrame with the best model results
    best_models_list = []
    for name, model in best_models.items():
        accuracy_test = model.score(X_test, y_test)
        R2_score_test = r2_score(y_test, model.predict(X_test))
        best_models_list.append([name, accuracy_test, R2_score_test])

    df_best_models = pd.DataFrame(best_models_list, columns=['Model', 'Accuracy', 'R2_score'])

    # Select the best model for each algorithm
    df_best_model_per_algorithm = df_best_models.groupby('Model').apply(lambda x: x.loc[x['Accuracy'].idxmax()])

    # Highlight the best model
    df_styled_best = df_best_model_per_algorithm.style.highlight_max(subset=['Accuracy', 'R2_score'], color='green').highlight_min(subset=['Accuracy', 'R2_score'], color='red')

    # Return the styled DataFrame
    return df_styled_best

##Roc_curve-function
# split dataset to test and train
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1812)

roc_curve_folder = "roc_curve_plots"

def plot_roc_curve(model, X_train, y_train, X_test, y_test, model_name):
    # Train the model
    
    model.fit(X_train, y_train)

    # Obtain the predicted probabilities for the positive class
    y_scores = model.predict_proba(X_test)[:, 1]

    # Compute the false positive rate (FPR), true positive rate (TPR), and thresholds
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)

    # Calculate the area under the ROC curve (AUC)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % roc_auc, color='blue')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess', color='gray')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve - {}'.format(model_name))
    plt.legend(loc="lower right")
    
    # Save the ROC curve plot as an image file
    plt.savefig(os.path.join(roc_curve_folder, "roc_curve_{}.png".format(model_name.replace(" ", "_"))))



def main():
    if not os.path.exists(roc_curve_folder):
        os.makedirs(roc_curve_folder)
    df_styled_best = find_best_models(X, Y, models)



# Main
if __name__ == "__main__":
    main() 