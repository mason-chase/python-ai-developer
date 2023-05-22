from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def visualize_confusion_matrix(model_type, y_test, y_pred):
    # Calculate the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Calculate the total number of samples
    total_samples = np.sum(cm)

    # Calculate the percentage of each cell in the confusion matrix
    cm_percent = cm / total_samples * 100

    class_labels = ['Not Churn', 'Churn']

    # Create a pandas DataFrame from the confusion matrix
    cm_df = pd.DataFrame(cm_percent, index=class_labels, columns=class_labels)

    # Plot the confusion matrix with annotations using seaborn's heatmap function
    sns.heatmap(cm_df, annot=True, cmap='Blues')

    # Customize the plot by adding a title and axis labels
    plt.title(f'{model_type} Confusion Matrix (Percentages)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # Save the plot as an image file
    plt.savefig(f'../plots/{model_type}_cm.png')

    # Show the plot
    plt.show()

def visualize_roc(model_type, y_test, y_score):

    # Calculate the FPR, TPR, and classification thresholds using the roc_curve() function
    fpr, tpr, thresholds = roc_curve(y_test, y_score[:, 1])

    # Calculate the area under the ROC curve (AUC)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve using matplotlib
    plt.plot(fpr, tpr, color='darkorange', lw=2,
            label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    # Save the plot as an image file
    plt.savefig(f'../plots/{model_type}_roc.png')

    # Show the plot
    plt.show()